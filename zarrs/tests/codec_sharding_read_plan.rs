//! `ArrayPartialDecoderPlanned::read_plan` and the plans it returns.
//!
//! The point of planning is that a caller holding several decoders can collect
//! all of their reads, issue them together, and hand the bytes back. These
//! tests pin the two properties that makes possible: planning touches no
//! storage, and decoding from supplied bytes touches no storage either.

#![allow(missing_docs)]

use std::error::Error;
use std::num::NonZeroU64;
use std::sync::Arc;

use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::codec::array_to_bytes::sharding::{
    ShardingCodecBuilder, ShardingCodecOptions, SubchunkWriteOrder,
};
use zarrs::array::codec::{Crc32cCodec, GzipCodec};
use zarrs::array::{
    Array, ArrayBuilder, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView,
    ArrayPartialDecoderPlanned, ArrayPartialDecoderTraits, ArraySubset, BytesToBytesCodecTraits,
    CodecError, CodecOptions, DataPlan, IndexPlan, ReadPlan, UnboundArrayToBytesCodecTraits,
    data_type,
};
use zarrs::storage::byte_range::ByteRange;
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{Bytes, MaybeBytes, ReadableStorageTraits, StoreKey};

type TestStore = PerformanceMetricsStorageAdapter<MemoryStore>;
type TestArray = Result<(Arc<Array<TestStore>>, Arc<TestStore>), Box<dyn Error>>;

const fn nz(v: u64) -> NonZeroU64 {
    NonZeroU64::new(v).unwrap()
}

/// Do a plan's reads, in whatever order the caller likes.
///
/// What the store returns goes straight back to the decoder -- no copy, and no
/// conversion to talk it into the decoder's argument type. Every entry is a read.
fn fetch(
    store: &TestStore,
    key: &StoreKey,
    byte_ranges: &[ByteRange],
) -> Result<Vec<MaybeBytes>, Box<dyn Error>> {
    byte_ranges
        .iter()
        .map(|byte_range| Ok(store.get_partial(key, *byte_range)?))
        .collect()
}

/// The decoder as a planning one, which sharding always is.
fn planned_of(decoder: &Arc<dyn ArrayPartialDecoderTraits>) -> Arc<dyn ArrayPartialDecoderPlanned> {
    decoder.clone().into_planned().expect("sharding can plan")
}

fn expect_data(plan: ReadPlan) -> DataPlan {
    match plan {
        ReadPlan::Data(plan) => plan,
        ReadPlan::Indexes(_) => panic!("expected a data plan, got an index plan"),
    }
}

fn expect_indexes(plan: ReadPlan) -> IndexPlan {
    match plan {
        ReadPlan::Data(_) => panic!("expected an index plan, got a data plan"),
        ReadPlan::Indexes(plan) => plan,
    }
}

/// A `[16, 16]` `uint16` array in `[8, 8]` shards of `[2, 2]` inner chunks.
///
/// With `nested`, the shard instead holds `[4, 4]` inner chunks that are
/// themselves shards of `[2, 2]` chunks.
fn build(nested: bool) -> TestArray {
    build_opt(nested, ShardingCodecOptions::default())
}

/// Two arrays only hold their subchunks at the same offsets if both were written
/// in the same order, and the default order is deliberately not one. A test whose
/// premise is "same layout, same ranges" has to pin the order down or it is a test
/// of how the threads were scheduled.
fn build_ordered() -> TestArray {
    build_opt(
        false,
        ShardingCodecOptions::default().with_subchunk_write_order(SubchunkWriteOrder::C),
    )
}

fn build_opt(nested: bool, options: ShardingCodecOptions) -> TestArray {
    let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
        MemoryStore::default(),
    )));
    let data_type = data_type::uint16();
    let codec: Arc<dyn UnboundArrayToBytesCodecTraits> = if nested {
        let inner = ShardingCodecBuilder::new(vec![nz(2), nz(2)], &data_type).build();
        Arc::new(
            ShardingCodecBuilder::new(vec![nz(4), nz(4)], &data_type)
                .array_to_bytes_codec(Arc::new(inner))
                .build()
                .with_options(options),
        )
    } else {
        Arc::new(
            ShardingCodecBuilder::new(vec![nz(2), nz(2)], &data_type)
                .build()
                .with_options(options),
        )
    };
    let mut builder = ArrayBuilder::new(vec![16, 16], vec![8, 8], data_type, 0u16);
    builder.array_to_bytes_codec(codec);
    let array = builder.build_arc(store.clone(), "/array")?;
    let data = (0..256u16).collect::<Vec<_>>();
    array.store_array_subset(&array.subset_all(), &data)?;
    Ok((array, store))
}

/// Three levels: an `[8, 8]` shard of `[4, 4]` subchunks, each a shard of
/// `[2, 2]` inner shards, each holding `[1, 1]` chunks.
///
/// One index exchange cannot reach the innermost level of this, so it is what
/// the too-deep decline is tested against.
fn build_deep() -> TestArray {
    let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
        MemoryStore::default(),
    )));
    let data_type = data_type::uint16();
    let innermost = ShardingCodecBuilder::new(vec![nz(1), nz(1)], &data_type).build();
    let inner = ShardingCodecBuilder::new(vec![nz(2), nz(2)], &data_type)
        .array_to_bytes_codec(Arc::new(innermost))
        .build();
    let outer = ShardingCodecBuilder::new(vec![nz(4), nz(4)], &data_type)
        .array_to_bytes_codec(Arc::new(inner))
        .build();
    let mut builder = ArrayBuilder::new(vec![16, 16], vec![8, 8], data_type, 0u16);
    builder.array_to_bytes_codec(Arc::new(outer));
    let array = builder.build_arc(store.clone(), "/array")?;
    let data = (0..256u16).collect::<Vec<_>>();
    array.store_array_subset(&array.subset_all(), &data)?;
    Ok((array, store))
}

#[test]
fn plan_then_decode_from_bytes_matches_and_reads_nothing() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, store) = build(false)?;
    let key: StoreKey = array.chunk_key(&[0, 0]);
    // Shard-local: rows 2..6, columns 1..5, spanning six inner chunks.
    let subset = ArraySubset::new_with_ranges(&[2..6, 1..5]);

    let decoder = array.partial_decoder(&[0, 0])?;
    let expected = decoder.partial_decode(&subset, &options)?.into_fixed()?;
    let planned = planned_of(&decoder);

    let before = store.reads();
    let plan = planned
        .read_plan(&subset, &options)?
        .expect("sharding reports its reads");
    assert_eq!(store.reads(), before, "planning must not perform any reads");
    let plan = expect_data(plan);
    assert!(plan.num_entries() > 0);

    let fetched = fetch(&store, &key, plan.byte_ranges())?;

    let before = store.reads();
    let got = plan.decode(fetched, &options)?.into_fixed()?;
    assert_eq!(
        store.reads(),
        before,
        "decoding from supplied bytes must not touch storage"
    );
    assert_eq!(got, expected);

    Ok(())
}

/// A shard with no inner chunk present plans no reads at all: an absent chunk is
/// not a read, so it is not an entry. The whole selection is the fill value,
/// which `fill_absent_into` writes without any fetched data.
#[test]
fn plan_covers_a_missing_shard() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, store) = build(false)?;
    let subset = ArraySubset::new_with_ranges(&[0..4, 0..4]);

    // Chunk [1, 1] was written; erase it so the shard is absent.
    array.erase_chunk(&[1, 1])?;
    let decoder = array.partial_decoder(&[1, 1])?;
    let plan = expect_data(
        planned_of(&decoder)
            .read_plan(&subset, &options)?
            .expect("sharding reports its reads"),
    );
    assert_eq!(plan.num_entries(), 0, "nothing stored, so nothing to read");

    // The owned form fills its own buffer.
    let got = plan.decode(Vec::new(), &options)?.into_fixed()?;
    assert_eq!(got, vec![0u8; 4 * 4 * 2], "fill value");

    // The into form leaves filling to `fill_absent_into`, which reads nothing.
    let element_size = 2;
    let mut output = vec![0xAAu8; 4 * 4 * element_size];
    {
        let output_slice = UnsafeCellSlice::new(output.as_mut_slice());
        let mut view = unsafe {
            ArrayBytesFixedDisjointView::new(
                output_slice,
                element_size,
                &[4, 4],
                ArraySubset::new_with_shape(vec![4, 4]),
            )?
        };
        let before = store.reads();
        plan.fill_absent_into(ArrayBytesDecodeIntoTarget::Fixed(&mut view), &options)?;
        assert_eq!(store.reads(), before, "filling absent units reads nothing");
    }
    assert_eq!(output, vec![0u8; 4 * 4 * 2], "fill value");

    Ok(())
}

/// A nested selection is never reported as a one-level data plan.
///
/// One range per inner chunk would name whole inner shards rather than the bytes actually
/// wanted, over-reading by an unbounded factor. The first stage names the inner indexes
/// instead, and only the plan that comes back from refining is the data.
#[test]
fn nested_sharding_never_plans_whole_inner_shards() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, _) = build(true)?;
    let subset = ArraySubset::new_with_ranges(&[2..6, 1..5]);
    let decoder = array.partial_decoder(&[0, 0])?;

    let plan = planned_of(&decoder)
        .read_plan(&subset, &options)?
        .expect("a nested selection is planned, in stages");
    assert!(
        matches!(plan, ReadPlan::Indexes(_)),
        "the first stage must not be the data"
    );

    // The ordinary path is unaffected by any of this.
    assert!(
        !decoder
            .partial_decode(&subset, &options)?
            .into_fixed()?
            .is_empty(),
        "the ordinary path still decodes it"
    );

    Ok(())
}

/// A bytes-to-bytes codec outside the sharding codec means no plan.
///
/// The shard index's offsets are into whatever handle the decoder sits on. Put a
/// decompressor or a prefix-stripper in between and those offsets no longer name
/// the stored value, so a caller issuing them against the store reads the wrong
/// bytes -- of the right length, so neither the caller nor the decode notices.
/// The only safe answer is to decline.
#[test]
fn an_outer_bytes_codec_declines_planning() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let subset = ArraySubset::new_with_ranges(&[2..6, 1..5]);

    for (label, outer) in [
        (
            "gzip",
            Arc::new(GzipCodec::new(5)?) as Arc<dyn BytesToBytesCodecTraits>,
        ),
        (
            "crc32c",
            Arc::new(Crc32cCodec::new()) as Arc<dyn BytesToBytesCodecTraits>,
        ),
    ] {
        let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
            MemoryStore::default(),
        )));
        let data_type = data_type::uint16();
        let mut builder = ArrayBuilder::new(vec![8, 8], vec![8, 8], data_type.clone(), 0u16);
        builder.array_to_bytes_codec(Arc::new(
            ShardingCodecBuilder::new(vec![nz(2), nz(2)], &data_type).build(),
        ));
        builder.bytes_to_bytes_codecs(vec![outer]);
        let array = builder.build_arc(store, "/array")?;
        array.store_array_subset(&array.subset_all(), (0..64u16).collect::<Vec<_>>())?;

        let decoder = array.partial_decoder(&[0, 0])?;
        assert!(
            planned_of(&decoder).read_plan(&subset, &options)?.is_none(),
            "{label}: planned offsets that are not offsets into the stored value"
        );
        // Declining costs nothing but the plan: the ordinary path still reads it.
        assert_eq!(
            decoder
                .partial_decode(&subset, &options)?
                .into_fixed()?
                .len(),
            4 * 4 * 2,
            "{label}: the ordinary path must still decode"
        );
    }

    Ok(())
}

/// Nesting changes where bytes live, not what they decode to.
///
/// The outer level still goes through the shared subchunk geometry, so this is
/// what catches that path being wrong for nested shards specifically.
#[test]
fn nested_sharding_decodes_the_same_values() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (flat, _) = build(false)?;
    let (nested, _) = build(true)?;

    let flat_decoder = flat.partial_decoder(&[0, 0])?;
    let nested_decoder = nested.partial_decoder(&[0, 0])?;
    for ranges in [
        &[0..2, 0..2][..], // inside one inner shard
        &[1..3, 1..3][..], // straddling four inner chunks of the inner shard
        &[3..5, 2..7][..], // straddling inner shards
        &[0..8, 0..8][..], // the whole shard
    ] {
        let subset = ArraySubset::new_with_ranges(ranges);
        let want = flat_decoder
            .partial_decode(&subset, &options)?
            .into_fixed()?;
        let got = nested_decoder
            .partial_decode(&subset, &options)?
            .into_fixed()?;
        assert_eq!(got, want, "subset {ranges:?}");
    }

    Ok(())
}

/// Nesting plans every selection, and never in more than the one exchange the
/// types allow.
///
/// `refine` consuming an [`IndexPlan`] and returning a [`DataPlan`] is what
/// bounds the rounds -- this pins that every subset gets through it.
#[test]
fn nested_sharding_plans_every_subset() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, store) = build(true)?;
    let key: StoreKey = array.chunk_key(&[0, 0]);
    let decoder = array.partial_decoder(&[0, 0])?;
    let planned = planned_of(&decoder);

    for ranges in [
        &[0..2, 0..2][..],
        &[1..3, 1..3][..],
        &[3..5, 2..7][..],
        &[0..8, 0..8][..],
        &[4..8, 4..8][..], // exactly one inner shard
    ] {
        let subset = ArraySubset::new_with_ranges(ranges);
        let plan = planned
            .clone()
            .read_plan(&subset, &options)?
            .expect("every nested subset is planned");
        // One exchange at most, which the types state: refine returns a DataPlan.
        let _data: DataPlan = match plan {
            ReadPlan::Data(plan) => plan,
            ReadPlan::Indexes(plan) => {
                let fetched = fetch(&store, &key, plan.byte_ranges())?;
                plan.refine(fetched, &options)?
            }
        };
    }

    Ok(())
}

/// A hand-built plan pairing this decoder with another layout's ranges is
/// rejected on both entry points.
///
/// A plan produced by the API holds the decoder that made it, so it cannot be
/// paired with another decoder's bytes. Construction is public, though, so the
/// decoder still checks the ranges against the reads it would perform.
#[test]
fn a_plan_with_foreign_ranges_is_rejected() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (flat, store) = build(false)?;
    let (nested, _) = build(true)?;
    let key: StoreKey = flat.chunk_key(&[0, 0]);
    let subset = ArraySubset::new_with_ranges(&[2..6, 1..5]);

    let flat_decoder = flat.partial_decoder(&[0, 0])?;
    let flat_plan = expect_data(
        planned_of(&flat_decoder)
            .read_plan(&subset, &options)?
            .expect("the flat array plans"),
    );
    let fetched = fetch(&store, &key, flat_plan.byte_ranges())?;

    // The flat array's ranges, hand-paired with the nested array's decoder.
    let nested_decoder = nested.partial_decoder(&[0, 0])?;
    let forged = DataPlan::new(
        planned_of(&nested_decoder),
        subset.clone(),
        flat_plan.byte_ranges().to_vec(),
    );

    let err = forged
        .decode(fetched, &options)
        .expect_err("a nested decoder must not consume a flat plan's ranges");
    assert!(
        matches!(err, CodecError::ReadPlanMismatch),
        "unexpected error: {err}"
    );

    // The same must hold for the decode-into entry point, which has its own
    // validation path.
    let element_size = 2;
    let mut output = vec![0u8; 4 * 4 * element_size];
    let output_slice = UnsafeCellSlice::new(output.as_mut_slice());
    let mut view = unsafe {
        ArrayBytesFixedDisjointView::new(
            output_slice,
            element_size,
            &[4, 4],
            ArraySubset::new_with_shape(vec![4, 4]),
        )?
    };
    let err = forged
        .decode_into(
            fetch(&store, &key, flat_plan.byte_ranges())?,
            ArrayBytesDecodeIntoTarget::Fixed(&mut view),
            &options,
        )
        .expect_err("decode-into must reject it too");
    assert!(
        matches!(err, CodecError::ReadPlanMismatch),
        "unexpected error: {err}"
    );
    assert!(
        output.iter().all(|byte| *byte == 0),
        "a rejected plan must not have written anything"
    );

    Ok(())
}

/// A plan decodes to what `partial_decode` would have returned, for subsets that
/// sit inside one inner chunk, straddle several, and cover the whole shard.
///
/// The plan's contract is that entry `i` is the bytes for chunk `i`, which holds
/// only while planning and decoding agree on the order. Nothing in the types
/// enforces that, so it is pinned here.
#[test]
fn plan_matches_partial_decode_across_subsets() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, store) = build(false)?;
    let key: StoreKey = array.chunk_key(&[0, 0]);
    let decoder = array.partial_decoder(&[0, 0])?;
    let planned = planned_of(&decoder);

    for ranges in [
        &[0..2, 0..2][..], // exactly one inner chunk
        &[1..2, 1..2][..], // inside one inner chunk
        &[1..3, 1..3][..], // straddling four
        &[3..8, 0..1][..], // a column across several
        &[0..8, 0..8][..], // the whole shard
    ] {
        let subset = ArraySubset::new_with_ranges(ranges);
        let plan = expect_data(
            planned
                .clone()
                .read_plan(&subset, &options)?
                .expect("sharding reports its reads"),
        );
        let got = plan
            .decode(fetch(&store, &key, plan.byte_ranges())?, &options)?
            .into_fixed()?;
        let expected = decoder.partial_decode(&subset, &options)?.into_fixed()?;
        assert_eq!(got, expected, "subset {ranges:?}");
    }

    Ok(())
}

/// Bytes that do not match the plan are rejected rather than got.
///
/// The plan carries its own selection and decoder, so neither can be mismatched --
/// but the bytes fetched for it are still the caller's to get wrong.
///
/// Each of these decodes to plausible-looking data if only the entry count is
/// checked, so each must be an error instead.
#[test]
fn fetched_bytes_must_match_the_plan() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, store) = build(false)?;
    let key: StoreKey = array.chunk_key(&[0, 0]);
    let subset = ArraySubset::new_with_ranges(&[2..6, 1..5]);
    let decoder = array.partial_decoder(&[0, 0])?;
    let plan = expect_data(
        planned_of(&decoder)
            .read_plan(&subset, &options)?
            .expect("sharding reports its reads"),
    );
    assert!(
        plan.num_entries() > 0,
        "every inner chunk of this subset was written"
    );

    let reject = |fetched, what| {
        let err = plan.decode(fetched, &options).expect_err(what);
        assert!(
            matches!(err, CodecError::ReadPlanMismatch),
            "{what}: unexpected error: {err}"
        );
    };

    reject(vec![None; plan.num_entries() - 1], "too few entries");
    reject(vec![None; plan.num_entries() + 1], "too many entries");
    // Would otherwise decode to fill values -- what a shard erased between
    // planning and fetching looks like from here.
    reject(
        vec![None; plan.num_entries()],
        "nothing supplied for a read",
    );

    // Bytes of the wrong length, from the same shard.
    let mut fetched = fetch(&store, &key, plan.byte_ranges())?;
    let short = fetched[0].as_ref().expect("chunk was written").slice(1..);
    fetched[0] = Some(short);
    reject(fetched, "bytes of the wrong length");

    Ok(())
}

/// `decode_into` writes into the caller's view, at the view's own origin rather
/// than at the shard's.
///
/// The offset is the whole point: a caller assembling one array from several
/// decoders gives each a different corner of it, and a decoder that ignored the
/// view's start would overwrite its neighbours.
#[test]
fn decode_from_bytes_into_an_offset_view() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, store) = build(false)?;
    let key: StoreKey = array.chunk_key(&[0, 0]);
    let subset = ArraySubset::new_with_ranges(&[2..6, 1..5]);
    let decoder = array.partial_decoder(&[0, 0])?;
    let plan = expect_data(
        planned_of(&decoder)
            .read_plan(&subset, &options)?
            .expect("sharding reports its reads"),
    );
    let expected = plan
        .decode(fetch(&store, &key, plan.byte_ranges())?, &options)?
        .into_fixed()?
        .into_owned();

    // An [8, 8] output with the 4x4 result dropped at [3, 2], prefilled with a
    // sentinel so anything written outside the target subset shows up.
    let element_size = 2;
    let output_shape = [8u64, 8];
    let mut output = vec![0xAAu8; 8 * 8 * element_size];
    {
        let output_slice = UnsafeCellSlice::new(output.as_mut_slice());
        let mut view = unsafe {
            ArrayBytesFixedDisjointView::new(
                output_slice,
                element_size,
                &output_shape,
                ArraySubset::new_with_ranges(&[3..7, 2..6]),
            )?
        };
        plan.decode_into(
            fetch(&store, &key, plan.byte_ranges())?,
            ArrayBytesDecodeIntoTarget::Fixed(&mut view),
            &options,
        )?;
    }

    let mut want = vec![0xAAu8; 8 * 8 * element_size];
    for (row, chunk) in expected.chunks_exact(4 * element_size).enumerate() {
        let start = ((3 + row) * 8 + 2) * element_size;
        want[start..start + chunk.len()].copy_from_slice(chunk);
    }
    assert_eq!(output, want, "got into the wrong place, or spilled");

    Ok(())
}

/// A plan decodes with the decoder it holds, so two same-layout arrays cannot be
/// crossed by the API's own flow.
///
/// Two arrays written in the same order hold their inner chunks at the same
/// offsets, so their plans are byte-identical lists of ranges -- the case no
/// range comparison can catch. What prevents one array's bytes decoding as the
/// other's is that a plan is bound to its decoder at construction and every
/// decode goes through it.
#[test]
fn a_plan_decodes_with_the_decoder_it_holds() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array_a, store_a) = build_ordered()?;
    let (array_b, _) = build_ordered()?;
    let subset = ArraySubset::new_with_ranges(&[0..4, 0..4]);

    let decoder_a = array_a.partial_decoder(&[0, 0])?;
    let plan_a = expect_data(
        planned_of(&decoder_a)
            .read_plan(&subset, &options)?
            .expect("sharding reports its reads"),
    );
    let decoder_b = array_b.partial_decoder(&[0, 0])?;
    let plan_b = expect_data(
        planned_of(&decoder_b)
            .read_plan(&subset, &options)?
            .expect("sharding reports its reads"),
    );
    assert_eq!(
        plan_a.byte_ranges(),
        plan_b.byte_ranges(),
        "same layout, so the ranges alone cannot tell the two arrays apart"
    );

    // Decoding through plan_a can only ever use array A's decoder.
    let got = plan_a
        .decode(
            fetch(&store_a, &array_a.chunk_key(&[0, 0]), plan_a.byte_ranges())?,
            &options,
        )?
        .into_fixed()?;
    let want = decoder_a.partial_decode(&subset, &options)?.into_fixed()?;
    assert_eq!(got, want);

    Ok(())
}

/// A hand-built plan whose ranges are not this decoder's reads is rejected.
///
/// An erased shard's plan has nothing to read, so its ranges cannot match a
/// stored shard's -- pairing them by hand must fail rather than decode to fill.
#[test]
fn a_hand_built_plan_with_wrong_ranges_is_rejected() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, _) = build(false)?;
    let subset = ArraySubset::new_with_ranges(&[0..4, 0..4]);

    array.erase_chunk(&[1, 1])?;
    let erased_plan = expect_data(
        planned_of(&array.partial_decoder(&[1, 1])?)
            .read_plan(&subset, &options)?
            .expect("sharding reports its reads"),
    );

    let other = array.partial_decoder(&[0, 0])?;
    let forged = DataPlan::new(
        planned_of(&other),
        subset,
        erased_plan.byte_ranges().to_vec(),
    );
    let err = forged
        .decode(vec![None; forged.num_entries()], &options)
        .expect_err("ranges from another shard's index must fail");
    assert!(
        matches!(err, CodecError::ReadPlanMismatch),
        "unexpected error: {err}"
    );

    Ok(())
}

/// A selection reaching outside the shard is an error, not a panic.
///
/// Both entry points take a selection the caller chose, and a subset past the end
/// of the shard names chunks past the end of the shard index.
#[test]
fn an_out_of_bounds_selection_errors() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, _) = build(false)?;
    let oob = ArraySubset::new_with_ranges(&[0..99, 0..99]);

    for chunk_indices in [&[0, 0], &[1, 1]] {
        // `[1, 1]` erased: the shard index is absent, which used to skip the
        // bounds-dependent work entirely rather than reject it.
        if chunk_indices == &[1, 1] {
            array.erase_chunk(chunk_indices)?;
        }
        let decoder = array.partial_decoder(chunk_indices)?;
        let planned = planned_of(&decoder);

        let err = planned
            .clone()
            .read_plan(&oob, &options)
            .expect_err("planning an out-of-bounds subset must fail");
        assert!(
            matches!(err, CodecError::IncompatibleIndexer(_)),
            "{chunk_indices:?}: unexpected error: {err}"
        );

        // The same selection reaching the decode path as a hand-built plan. It is
        // rejected as hand-built -- carrying no decoder-minted state -- before its
        // bounds are ever looked at.
        let err = DataPlan::new(planned, oob.clone(), Vec::new())
            .decode(Vec::new(), &options)
            .expect_err("decoding an out-of-bounds plan must fail");
        assert!(
            matches!(err, CodecError::ReadPlanMismatch),
            "{chunk_indices:?}: unexpected error: {err}"
        );
    }

    Ok(())
}

/// A nested selection is planned in two stages, and the first one is computed.
///
/// The data cannot be named until the subchunk indexes are read, because that is where its
/// offsets are. What can be named without reading anything is where those indexes live.
#[test]
fn a_nested_plan_names_the_subchunk_indexes_first() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, store) = build(true)?;
    let key: StoreKey = array.chunk_key(&[0, 0]);
    let subset = ArraySubset::new_with_ranges(&[0..3, 0..3]);

    let decoder = array.partial_decoder(&[0, 0])?;

    let before = store.reads();
    let plan = planned_of(&decoder)
        .read_plan(&subset, &options)?
        .expect("a nested selection is planned");
    assert_eq!(store.reads(), before, "planning must not perform any reads");
    let index_plan = expect_indexes(plan);
    let index_reads = index_plan.reads().count();
    assert!(index_reads > 0, "part-wanted subchunks have indexes to read");

    let fetched = fetch(&store, &key, index_plan.byte_ranges())?;
    let data_plan = index_plan.refine(fetched, &options)?;
    assert_eq!(store.reads(), before + index_reads);
    assert!(
        data_plan.num_entries() > 0,
        "refining named the data itself"
    );

    Ok(())
}

/// The point of planning through the index: the reads are the innermost chunks, not the
/// subchunks that contain them.
///
/// A plan that stopped at the level-zero subchunks would have to name each one whole, since
/// it could not say where inside it the wanted bytes are. This compares the two.
#[test]
fn a_nested_plan_reads_less_than_the_subchunks_holding_it() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, store) = build(true)?;
    let key: StoreKey = array.chunk_key(&[0, 0]);
    // One innermost chunk out of each of the four subchunks of this shard.
    let subset = ArraySubset::new_with_ranges(&[0..6, 0..6]);

    let decoder = array.partial_decoder(&[0, 0])?;
    let index_plan = expect_indexes(
        planned_of(&decoder)
            .read_plan(&subset, &options)?
            .expect("a nested selection is planned"),
    );
    let fetched = fetch(&store, &key, index_plan.byte_ranges())?;
    let data_plan = index_plan.refine(fetched, &options)?;

    let planned_bytes: u64 = data_plan
        .reads()
        .map(|(_, range)| match range {
            ByteRange::FromStart(_, Some(len)) | ByteRange::Suffix(len) => len,
            ByteRange::FromStart(_, None) => 0,
        })
        .sum();

    // This selection touches every subchunk of the shard, so a plan that stopped at the
    // subchunks would have to read the whole shard bar its outer index.
    let shard_bytes = store.size_key(&key)?.expect("the shard is stored");

    println!("planned {planned_bytes} bytes of a {shard_bytes} byte shard");
    assert!(
        planned_bytes * 2 < shard_bytes,
        "planning through the index should read a fraction: {planned_bytes} of {shard_bytes}"
    );
    Ok(())
}

/// Refining is checked as strictly as decoding is.
#[test]
fn refining_rejects_bytes_that_are_not_the_indexes() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, _) = build(true)?;
    let subset = ArraySubset::new_with_ranges(&[0..2, 0..2]);

    let decoder = array.partial_decoder(&[0, 0])?;
    let planned = planned_of(&decoder);
    let plan = expect_indexes(
        planned
            .clone()
            .read_plan(&subset, &options)?
            .expect("a nested selection is planned"),
    );

    // Right count, wrong lengths.
    let short = vec![Some(Bytes::from_static(b"xy")); plan.num_entries()];
    let err = plan
        .refine(short, &options)
        .expect_err("bytes of the wrong length are not the indexes");
    assert!(
        matches!(err, CodecError::ReadPlanMismatch),
        "unexpected: {err}"
    );

    // An index plan a flat decoder would never have produced. Its own plans are
    // all data plans, so refining is not something it can be asked to do.
    let (flat_array, _) = build(false)?;
    let flat_decoder = flat_array.partial_decoder(&[0, 0])?;
    let forged = IndexPlan::new(planned_of(&flat_decoder), subset, Vec::new());
    let err = forged
        .refine(Vec::new(), &options)
        .expect_err("a flat decoder has no index plans to refine");
    assert!(
        matches!(err, CodecError::ReadPlanMismatch),
        "unexpected: {err}"
    );

    Ok(())
}

/// The whole nested path, against the answer the ordinary decode gives.
///
/// Plan, fetch the indexes, refine, fetch the data, decode. Nothing about the result should
/// depend on having gone that way round.
#[test]
fn a_nested_plan_decodes_to_the_same_bytes() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, store) = build(true)?;
    let key: StoreKey = array.chunk_key(&[0, 0]);
    let decoder = array.partial_decoder(&[0, 0])?;
    let planned = planned_of(&decoder);

    for ranges in [
        &[0..2, 0..2][..], // one innermost chunk
        &[1..3, 1..3][..], // straddling four of them
        &[2..6, 1..5][..], // straddling four subchunks
        &[3..5, 2..7][..],
        &[0..8, 0..8][..], // the whole shard
        &[4..8, 4..8][..], // exactly one subchunk
        &[0..1, 0..8][..], // one row
        &[7..8, 7..8][..], // one element, last chunk
    ] {
        let subset = ArraySubset::new_with_ranges(ranges);
        let want = decoder.partial_decode(&subset, &options)?.into_fixed()?;

        let data_plan = match planned
            .clone()
            .read_plan(&subset, &options)?
            .expect("a nested selection is planned")
        {
            ReadPlan::Data(plan) => plan,
            ReadPlan::Indexes(plan) => {
                let fetched = fetch(&store, &key, plan.byte_ranges())?;
                plan.refine(fetched, &options)?
            }
        };
        let fetched = fetch(&store, &key, data_plan.byte_ranges())?;
        let before = store.reads();
        let got = data_plan.decode(fetched, &options)?.into_fixed()?;
        assert_eq!(
            store.reads(),
            before,
            "decoding from bytes must not read, {ranges:?}"
        );

        assert_eq!(got, want, "subset {ranges:?}");
    }

    Ok(())
}

/// A selection that wants whole subchunks needs no second round.
///
/// The subchunk's extent is already in the outer index, so its own index would only say to
/// read all of it. Nesting costs an extra round exactly when part of a subchunk is wanted.
#[test]
fn wanting_whole_subchunks_needs_no_index_round() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, store) = build(true)?;
    let key: StoreKey = array.chunk_key(&[0, 0]);
    let decoder = array.partial_decoder(&[0, 0])?;
    let planned = planned_of(&decoder);

    for ranges in [
        &[0..8, 0..8][..], // the whole shard: four whole subchunks
        &[4..8, 4..8][..], // exactly one subchunk
        &[0..4, 0..8][..], // two whole subchunks
    ] {
        let subset = ArraySubset::new_with_ranges(ranges);
        let plan = planned
            .clone()
            .read_plan(&subset, &options)?
            .expect("planned in one go");
        let ReadPlan::Data(plan) = plan else {
            panic!("{ranges:?} wants whole subchunks, so it needs no index round");
        };
        let want = decoder.partial_decode(&subset, &options)?.into_fixed()?;
        let got = plan
            .decode(fetch(&store, &key, plan.byte_ranges())?, &options)?
            .into_fixed()?;
        assert_eq!(got, want, "subset {ranges:?}");
    }

    // And one that wants part of a subchunk still stages.
    assert!(
        matches!(
            planned
                .clone()
                .read_plan(&ArraySubset::new_with_ranges(&[0..3, 0..3]), &options)?
                .expect("planned"),
            ReadPlan::Indexes(_)
        ),
        "part of a subchunk needs its index"
    );

    Ok(())
}

/// File-adjacent inner chunks coalesce into single reads, and decode to exactly
/// what `partial_decode` returns.
///
/// The unit of a plan is a read: chunks written next to each other in the shard
/// are one byte range, however many of them there are. Write order `C` makes
/// grid-adjacent chunks file-adjacent, so the run structure here is exact.
#[test]
fn file_adjacent_chunks_coalesce_into_one_read() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, store) = build_ordered()?;
    let key: StoreKey = array.chunk_key(&[0, 0]);
    let decoder = array.partial_decoder(&[0, 0])?;
    let planned = planned_of(&decoder);

    // Chunk rows 1..3 x columns 0..3: six chunks, file-adjacent in two row runs
    // of three ([1,0][1,1][1,2] and [2,0][2,1][2,2]).
    let subset = ArraySubset::new_with_ranges(&[2..6, 1..5]);
    let plan = expect_data(
        planned
            .clone()
            .read_plan(&subset, &options)?
            .expect("sharding reports its reads"),
    );
    assert_eq!(
        plan.num_entries(),
        2,
        "six chunks in two file-adjacent runs are two reads"
    );

    let got = plan
        .decode(fetch(&store, &key, plan.byte_ranges())?, &options)?
        .into_fixed()?;
    let expected = decoder.partial_decode(&subset, &options)?.into_fixed()?;
    assert_eq!(got, expected);

    Ok(())
}

/// With no file adjacency, every run is one chunk and behaviour is identical to
/// planning one read per chunk.
///
/// A column of a C-order-written shard is chunks four apart in the file, so
/// nothing merges: the run structure degenerates to exactly the old plan.
#[test]
fn without_adjacency_every_run_is_one_chunk() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, store) = build_ordered()?;
    let key: StoreKey = array.chunk_key(&[0, 0]);
    let decoder = array.partial_decoder(&[0, 0])?;

    // Chunk column 0, all four rows: file positions 0, 4, 8, 12.
    let subset = ArraySubset::new_with_ranges(&[0..8, 0..2]);
    let plan = expect_data(
        planned_of(&decoder)
            .read_plan(&subset, &options)?
            .expect("sharding reports its reads"),
    );
    assert_eq!(plan.num_entries(), 4, "no adjacency, so one read per chunk");

    let got = plan
        .decode(fetch(&store, &key, plan.byte_ranges())?, &options)?
        .into_fixed()?;
    let expected = decoder.partial_decode(&subset, &options)?.into_fixed()?;
    assert_eq!(got, expected);

    Ok(())
}

/// A selection straddling stored and absent chunks: the reads cover only what is
/// stored, `fill_absent_into` covers the rest, and together they equal
/// `partial_decode`.
#[test]
fn absent_chunks_are_filled_not_read() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    // Store only the top half of the array: shard [0, 0]'s chunk rows 0..2 exist,
    // rows 2..4 are absent.
    let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
        MemoryStore::default(),
    )));
    let data_type = data_type::uint16();
    let codec = Arc::new(
        ShardingCodecBuilder::new(vec![nz(2), nz(2)], &data_type)
            .build()
            .with_options(
                ShardingCodecOptions::default().with_subchunk_write_order(SubchunkWriteOrder::C),
            ),
    );
    let mut builder = ArrayBuilder::new(vec![16, 16], vec![8, 8], data_type, 0u16);
    builder.array_to_bytes_codec(codec);
    let array = builder.build_arc(store.clone(), "/array")?;
    let top = ArraySubset::new_with_ranges(&[0..4, 0..16]);
    array.store_array_subset(&top, (0..64u16).collect::<Vec<_>>())?;

    let key: StoreKey = array.chunk_key(&[0, 0]);
    let decoder = array.partial_decoder(&[0, 0])?;
    // Rows 2..6: chunk row 1 is stored, chunk row 2 is absent.
    let subset = ArraySubset::new_with_ranges(&[2..6, 0..4]);
    let plan = expect_data(
        planned_of(&decoder)
            .read_plan(&subset, &options)?
            .expect("sharding reports its reads"),
    );
    assert!(
        plan.num_entries() > 0,
        "the stored chunk row is there to read"
    );

    let expected = decoder.partial_decode(&subset, &options)?.into_fixed()?;
    let element_size = 2;
    let mut output = vec![0xAAu8; 4 * 4 * element_size];
    {
        let output_slice = UnsafeCellSlice::new(output.as_mut_slice());
        // `ArrayBytesDecodeIntoTarget` borrows the view for its whole lifetime, so
        // each call gets a fresh view of the same output.
        let view = || unsafe {
            ArrayBytesFixedDisjointView::new(
                output_slice,
                element_size,
                &[4, 4],
                ArraySubset::new_with_shape(vec![4, 4]),
            )
        };
        // Fill before any read returns -- it needs no fetched data.
        let before = store.reads();
        plan.fill_absent_into(ArrayBytesDecodeIntoTarget::Fixed(&mut view()?), &options)?;
        assert_eq!(store.reads(), before, "filling absent units reads nothing");
        plan.decode_into(
            fetch(&store, &key, plan.byte_ranges())?,
            ArrayBytesDecodeIntoTarget::Fixed(&mut view()?),
            &options,
        )?;
    }
    assert_eq!(output, expected.into_owned(), "fill + decode = the answer");

    Ok(())
}

/// Only the decoder can mint a valid plan: hand-building one through the public
/// constructor with the decoder's *own correct ranges* is still rejected.
///
/// The run structure -- which units each read holds, where each decodes to -- is
/// trusted because it is provably the decoder's own work. A hand-built plan
/// carries no such work, so there is nothing to trust, however right its ranges
/// happen to be.
#[test]
fn a_plan_without_decoder_state_is_rejected() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, store) = build(false)?;
    let key: StoreKey = array.chunk_key(&[0, 0]);
    let subset = ArraySubset::new_with_ranges(&[2..6, 1..5]);

    let decoder = array.partial_decoder(&[0, 0])?;
    let genuine = expect_data(
        planned_of(&decoder)
            .read_plan(&subset, &options)?
            .expect("sharding reports its reads"),
    );
    let hand_built = DataPlan::new(
        planned_of(&decoder),
        subset.clone(),
        genuine.byte_ranges().to_vec(),
    );

    let fetched = fetch(&store, &key, hand_built.byte_ranges())?;
    let err = hand_built
        .decode(fetched, &options)
        .expect_err("the ranges are right, but nothing proves the structure");
    assert!(
        matches!(err, CodecError::ReadPlanMismatch),
        "unexpected error: {err}"
    );

    // The genuine plan, of course, still decodes.
    let got = genuine
        .decode(fetch(&store, &key, genuine.byte_ranges())?, &options)?
        .into_fixed()?;
    assert_eq!(
        got,
        decoder.partial_decode(&subset, &options)?.into_fixed()?
    );

    Ok(())
}

/// Each entry decodes as its read lands: per-entry decodes in any order, plus one
/// absent fill, equal `partial_decode` -- and read nothing.
///
/// Entries are independent, so nothing requires a chunk's reads to all be back
/// before any of them decodes. Reverse order is the arbitrary-arrival stand-in.
#[test]
fn entries_decode_one_at_a_time_in_any_order() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    for (array, store) in [build(false)?, build_ordered()?] {
        let key: StoreKey = array.chunk_key(&[0, 0]);
        let decoder = array.partial_decoder(&[0, 0])?;
        let subset = ArraySubset::new_with_ranges(&[2..6, 1..5]);
        let plan = expect_data(
            planned_of(&decoder)
                .read_plan(&subset, &options)?
                .expect("sharding reports its reads"),
        );
        let expected = decoder.partial_decode(&subset, &options)?.into_fixed()?;

        let element_size = 2;
        let mut output = vec![0xAAu8; 4 * 4 * element_size];
        {
            let output_slice = UnsafeCellSlice::new(output.as_mut_slice());
            let view = || unsafe {
                ArrayBytesFixedDisjointView::new(
                    output_slice,
                    element_size,
                    &[4, 4],
                    ArraySubset::new_with_shape(vec![4, 4]),
                )
            };
            plan.fill_absent_into(ArrayBytesDecodeIntoTarget::Fixed(&mut view()?), &options)?;
            let fetched = fetch(&store, &key, plan.byte_ranges())?;
            let before = store.reads();
            for (entry, bytes) in fetched.into_iter().enumerate().rev() {
                plan.decode_entry_into(
                    entry,
                    bytes,
                    ArrayBytesDecodeIntoTarget::Fixed(&mut view()?),
                    &options,
                )?;
            }
            assert_eq!(
                store.reads(),
                before,
                "per-entry decoding must not touch storage"
            );
        }
        assert_eq!(output, expected.into_owned());
    }
    Ok(())
}

/// A per-entry decode is checked exactly as strictly as the whole-plan one,
/// narrowed to its entry.
#[test]
fn a_per_entry_decode_rejects_what_does_not_match() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, store) = build(false)?;
    let key: StoreKey = array.chunk_key(&[0, 0]);
    let subset = ArraySubset::new_with_ranges(&[2..6, 1..5]);
    let decoder = array.partial_decoder(&[0, 0])?;
    let plan = expect_data(
        planned_of(&decoder)
            .read_plan(&subset, &options)?
            .expect("sharding reports its reads"),
    );
    let fetched = fetch(&store, &key, plan.byte_ranges())?;

    let element_size = 2;
    let mut output = vec![0u8; 4 * 4 * element_size];
    let output_slice = UnsafeCellSlice::new(output.as_mut_slice());
    let view = || unsafe {
        ArrayBytesFixedDisjointView::new(
            output_slice,
            element_size,
            &[4, 4],
            ArraySubset::new_with_shape(vec![4, 4]),
        )
    };

    let reject = |entry, bytes: MaybeBytes, what: &str| {
        let err = plan
            .decode_entry_into(
                entry,
                bytes,
                ArrayBytesDecodeIntoTarget::Fixed(&mut view().unwrap()),
                &options,
            )
            .expect_err(what);
        assert!(
            matches!(err, CodecError::ReadPlanMismatch),
            "{what}: unexpected error: {err}"
        );
    };

    reject(plan.num_entries(), fetched[0].clone(), "entry out of bounds");
    reject(0, None, "nothing supplied for a read");
    let short = fetched[0].as_ref().expect("stored").slice(1..);
    reject(0, Some(short), "bytes of the wrong length");

    // A hand-built plan has no per-entry path either.
    let hand_built = DataPlan::new(
        planned_of(&decoder),
        subset.clone(),
        plan.byte_ranges().to_vec(),
    );
    let err = hand_built
        .decode_entry_into(
            0,
            fetched[0].clone(),
            ArrayBytesDecodeIntoTarget::Fixed(&mut view()?),
            &options,
        )
        .expect_err("no decoder-minted state, so nothing to trust");
    assert!(matches!(err, CodecError::ReadPlanMismatch));

    Ok(())
}

/// The nested path decodes per entry too: refine, then decode each read as it
/// lands, against the ordinary decode's answer.
#[test]
fn nested_entries_decode_one_at_a_time() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, store) = build(true)?;
    let key: StoreKey = array.chunk_key(&[0, 0]);
    let decoder = array.partial_decoder(&[0, 0])?;

    // Straddles subchunks wanted in part and whole, so the refined plan mixes
    // innermost chunks with whole subchunks.
    let subset = ArraySubset::new_with_ranges(&[2..8, 1..8]);
    let plan = match planned_of(&decoder)
        .read_plan(&subset, &options)?
        .expect("a nested selection is planned")
    {
        ReadPlan::Data(plan) => plan,
        ReadPlan::Indexes(plan) => {
            let fetched = fetch(&store, &key, plan.byte_ranges())?;
            plan.refine(fetched, &options)?
        }
    };
    let expected = decoder.partial_decode(&subset, &options)?.into_fixed()?;

    let element_size = 2;
    let mut output = vec![0xAAu8; 6 * 7 * element_size];
    {
        let output_slice = UnsafeCellSlice::new(output.as_mut_slice());
        let view = || unsafe {
            ArrayBytesFixedDisjointView::new(
                output_slice,
                element_size,
                &[6, 7],
                ArraySubset::new_with_shape(vec![6, 7]),
            )
        };
        plan.fill_absent_into(ArrayBytesDecodeIntoTarget::Fixed(&mut view()?), &options)?;
        for (entry, bytes) in fetch(&store, &key, plan.byte_ranges())?
            .into_iter()
            .enumerate()
            .rev()
        {
            plan.decode_entry_into(
                entry,
                bytes,
                ArrayBytesDecodeIntoTarget::Fixed(&mut view()?),
                &options,
            )?;
        }
    }
    assert_eq!(output, expected.into_owned());

    Ok(())
}

/// Nesting deeper than one exchange declines part-wanted selections instead of
/// planning them badly.
///
/// One index round cannot reach the innermost level of a three-level shard. A
/// data plan at the subchunk level would name whole nested shards -- reading far
/// more than was asked, with nothing for the caller to notice -- so the only
/// honest answers are a whole-subchunk plan when that is what was asked, and no
/// plan at all otherwise.
#[test]
fn too_deep_nesting_declines_part_wanted_selections() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, store) = build_deep()?;
    let key: StoreKey = array.chunk_key(&[0, 0]);
    let decoder = array.partial_decoder(&[0, 0])?;
    let planned = planned_of(&decoder);

    // Wants part of a subchunk: cannot be read minimally, so it is not planned.
    for ranges in [
        &[1..3, 1..3][..], // inside one subchunk
        &[2..6, 1..5][..], // straddling subchunks, all in part
        &[0..4, 0..6][..], // one whole subchunk, one in part
    ] {
        let subset = ArraySubset::new_with_ranges(ranges);
        assert!(
            planned.clone().read_plan(&subset, &options)?.is_none(),
            "{ranges:?} wants part of a too-deep subchunk and must be declined"
        );
        // Declining costs nothing but the plan.
        let got = decoder.partial_decode(&subset, &options)?.into_fixed()?;
        assert!(!got.is_empty(), "{ranges:?}: ordinary path still decodes");
    }

    // Whole subchunks are their extents in the outer index: still plannable.
    let subset = ArraySubset::new_with_ranges(&[0..4, 0..4]);
    let plan = expect_data(
        planned
            .clone()
            .read_plan(&subset, &options)?
            .expect("a whole subchunk needs no index round at any depth"),
    );
    let want = decoder.partial_decode(&subset, &options)?.into_fixed()?;
    let got = plan
        .decode(fetch(&store, &key, plan.byte_ranges())?, &options)?
        .into_fixed()?;
    assert_eq!(got, want);

    Ok(())
}
