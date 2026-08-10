//! `ArrayPartialDecoderPlanned::read_plan` / `partial_decode_from_bytes`.
//!
//! The point of the pair is that a caller holding several decoders can collect
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
    Array, ArrayBuilder, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArraySubset,
    BytesToBytesCodecTraits, CodecError, CodecOptions, ReadPlan, UnboundArrayToBytesCodecTraits,
    data_type,
};
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{MaybeBytes, ReadableStorageTraits, StoreKey};

type TestStore = PerformanceMetricsStorageAdapter<MemoryStore>;
type TestArray = Result<(Arc<Array<TestStore>>, Arc<TestStore>), Box<dyn Error>>;

const fn nz(v: u64) -> NonZeroU64 {
    NonZeroU64::new(v).unwrap()
}

/// Do the plan's reads, in whatever order the caller likes.
///
/// What the store returns goes straight back to the decoder -- no copy, and no
/// conversion to talk it into the decoder's argument type. Entries with nothing to
/// read keep their place without being visited, which is what `reads` is for.
fn fetch(
    store: &TestStore,
    key: &StoreKey,
    plan: &ReadPlan,
) -> Result<Vec<MaybeBytes>, Box<dyn Error>> {
    let mut fetched = vec![None; plan.num_entries()];
    for (entry, byte_range) in plan.reads() {
        fetched[entry] = store.get_partial(key, byte_range)?;
    }
    Ok(fetched)
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

#[test]
fn plan_then_decode_from_bytes_matches_and_reads_nothing() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, store) = build(false)?;
    let key: StoreKey = array.chunk_key(&[0, 0]);
    // Shard-local: rows 2..6, columns 1..5, spanning six inner chunks.
    let subset = ArraySubset::new_with_ranges(&[2..6, 1..5]);

    let decoder = array.partial_decoder(&[0, 0])?;
    let expected = decoder.partial_decode(&subset, &options)?.into_fixed()?;
    let planned = decoder.as_planned().expect("sharding can plan");

    let before = store.reads();
    let plan = planned
        .read_plan(&subset, &options)?
        .expect("sharding reports its reads");
    assert_eq!(store.reads(), before, "planning must not perform any reads");
    assert!(!plan.is_empty());

    let fetched = fetch(&store, &key, &plan)?;

    let before = store.reads();
    let decoded = planned
        .partial_decode_from_bytes(&plan, fetched, &options)?
        .into_fixed()?;
    assert_eq!(
        store.reads(),
        before,
        "decoding from supplied bytes must not touch storage"
    );
    assert_eq!(decoded, expected);

    Ok(())
}

/// A shard with no inner chunk present still reports one entry per chunk, so
/// the plan stays one-to-one with what `partial_decode_from_bytes` expects.
#[test]
fn plan_covers_a_missing_shard() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, _) = build(false)?;
    let subset = ArraySubset::new_with_ranges(&[0..4, 0..4]);

    // Chunk [1, 1] was written; erase it so the shard is absent.
    array.erase_chunk(&[1, 1])?;
    let decoder = array.partial_decoder(&[1, 1])?;
    let planned = decoder.as_planned().expect("sharding can plan");
    let plan = planned
        .read_plan(&subset, &options)?
        .expect("sharding reports its reads");
    assert_eq!(
        plan.num_entries(),
        4,
        "one entry per inner chunk in the subset"
    );
    assert_eq!(plan.reads().count(), 0, "but nothing to read");
    assert!(!plan.is_empty(), "entries without reads are still entries");

    let decoded = planned
        .partial_decode_from_bytes(&plan, vec![None; plan.num_entries()], &options)?
        .into_fixed()?;
    assert_eq!(decoded, vec![0u8; 4 * 4 * 2], "fill value");

    Ok(())
}

/// Nested sharding reports no plan.
///
/// One range per inner chunk would name whole inner shards rather than the
/// bytes actually wanted. Rather than hand back a plan that over-reads by an
/// unbounded factor, report nothing and let the caller use `partial_decode`,
/// which walks the index levels itself.
#[test]
fn nested_sharding_reports_no_plan() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, _) = build(true)?;
    let subset = ArraySubset::new_with_ranges(&[2..6, 1..5]);
    let decoder = array.partial_decoder(&[0, 0])?;

    // `as_planned` is about the decoder, not the selection: sharding plans
    // some indexers, so it answers yes and `read_plan` declines this one.
    let planned = decoder.as_planned().expect("sharding can plan");
    assert!(
        planned.read_plan(&subset, &options)?.is_none(),
        "nested sharding must not report a one-level plan"
    );

    // With no plan there is nothing to hand to `partial_decode_from_bytes`, so
    // the caller takes the ordinary path.
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
        array.store_array_subset(&array.subset_all(), &(0..64u16).collect::<Vec<_>>())?;

        let decoder = array.partial_decoder(&[0, 0])?;
        let planned = decoder.as_planned().expect("sharding can plan");
        assert!(
            planned.read_plan(&subset, &options)?.is_none(),
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
/// what catches that path being wrong for nested shards specifically -- the
/// planned tests cannot cover it, because nesting declines to plan.
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

/// Nesting declines every selection, not just the one the other test uses.
///
/// `as_planned` answers for the decoder and `read_plan` for the selection, so a
/// nested decoder saying yes to the first and no to the second is the shape the
/// caller has to handle for every subset it asks about.
#[test]
fn nested_sharding_declines_every_subset() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, _) = build(true)?;
    let decoder = array.partial_decoder(&[0, 0])?;
    let planned = decoder.as_planned().expect("sharding can plan");

    for ranges in [
        &[0..2, 0..2][..],
        &[1..3, 1..3][..],
        &[3..5, 2..7][..],
        &[0..8, 0..8][..],
        &[4..8, 4..8][..], // exactly one inner shard
    ] {
        let subset = ArraySubset::new_with_ranges(ranges);
        assert!(
            planned.read_plan(&subset, &options)?.is_none(),
            "planned {ranges:?}, which would name whole inner shards"
        );
    }

    Ok(())
}

/// A plan from a flat array is rejected by a nested decoder, on both entry points.
///
/// The two arrays have the same shape and shard shape, so the selection is valid
/// for either. Nothing but the nesting check stands between the caller and inner
/// shard bytes decoded as if they were inner chunks.
#[test]
fn a_flat_plan_is_rejected_by_a_nested_decoder() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (flat, store) = build(false)?;
    let (nested, _) = build(true)?;
    let key: StoreKey = flat.chunk_key(&[0, 0]);
    let subset = ArraySubset::new_with_ranges(&[2..6, 1..5]);

    let flat_decoder = flat.partial_decoder(&[0, 0])?;
    let plan = flat_decoder
        .as_planned()
        .expect("sharding can plan")
        .read_plan(&subset, &options)?
        .expect("the flat array plans");
    let fetched = fetch(&store, &key, &plan)?;

    let nested_decoder = nested.partial_decoder(&[0, 0])?;
    let nested_planned = nested_decoder.as_planned().expect("sharding can plan");

    let err = nested_planned
        .partial_decode_from_bytes(&plan, fetched, &options)
        .expect_err("a nested decoder must not consume a flat plan");
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
    let err = nested_planned
        .partial_decode_from_bytes_into(
            &plan,
            fetch(&store, &key, &plan)?,
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
    let planned = decoder.as_planned().expect("sharding can plan");

    for ranges in [
        &[0..2, 0..2][..], // exactly one inner chunk
        &[1..2, 1..2][..], // inside one inner chunk
        &[1..3, 1..3][..], // straddling four
        &[3..8, 0..1][..], // a column across several
        &[0..8, 0..8][..], // the whole shard
    ] {
        let subset = ArraySubset::new_with_ranges(ranges);
        let plan = planned
            .read_plan(&subset, &options)?
            .expect("sharding reports its reads");
        let decoded = planned
            .partial_decode_from_bytes(&plan, fetch(&store, &key, &plan)?, &options)?
            .into_fixed()?;
        let expected = decoder.partial_decode(&subset, &options)?.into_fixed()?;
        assert_eq!(decoded, expected, "subset {ranges:?}");
    }

    Ok(())
}

/// Bytes that do not match the plan are rejected rather than decoded.
///
/// The plan carries its own selection, so a plan cannot be paired with a
/// different one -- but the bytes fetched for it are still the caller's to get
/// wrong.
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
    let planned = decoder.as_planned().expect("sharding can plan");
    let plan = planned
        .read_plan(&subset, &options)?
        .expect("sharding reports its reads");
    assert!(
        plan.byte_ranges().iter().all(Option::is_some),
        "every inner chunk of this subset was written"
    );

    let reject = |fetched, what| {
        let err = planned
            .partial_decode_from_bytes(&plan, fetched, &options)
            .expect_err(what);
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
    let mut fetched = fetch(&store, &key, &plan)?;
    let short = fetched[0].as_ref().expect("chunk was written").slice(1..);
    fetched[0] = Some(short);
    reject(fetched, "bytes of the wrong length");

    Ok(())
}

/// `partial_decode_from_bytes_into` writes into the caller's view, at the view's
/// own origin rather than at the shard's.
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
    let planned = decoder.as_planned().expect("sharding can plan");
    let plan = planned
        .read_plan(&subset, &options)?
        .expect("sharding reports its reads");
    let expected = planned
        .partial_decode_from_bytes(&plan, fetch(&store, &key, &plan)?, &options)?
        .into_fixed()?;

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
        planned.partial_decode_from_bytes_into(
            &plan,
            fetch(&store, &key, &plan)?,
            ArrayBytesDecodeIntoTarget::Fixed(&mut view),
            &options,
        )?;
    }

    let mut want = vec![0xAAu8; 8 * 8 * element_size];
    for (row, chunk) in expected.chunks_exact(4 * element_size).enumerate() {
        let start = ((3 + row) * 8 + 2) * element_size;
        want[start..start + chunk.len()].copy_from_slice(chunk);
    }
    assert_eq!(output, want, "decoded into the wrong place, or spilled");

    Ok(())
}

/// A plan from a *different array* with the same layout is still rejected.
///
/// This is the case comparing byte ranges cannot catch: two arrays written the
/// same way hold their inner chunks at the same offsets, so both plans are the
/// same list of ranges. Only the identity of the decoder that produced the plan
/// tells them apart -- without it, one array's bytes decode as the other's and
/// are returned as correct.
#[test]
fn a_plan_from_another_array_is_rejected() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array_a, store_a) = build_ordered()?;
    let (array_b, _) = build_ordered()?;
    let subset = ArraySubset::new_with_ranges(&[0..4, 0..4]);

    let decoder_a = array_a.partial_decoder(&[0, 0])?;
    let plan_a = decoder_a
        .as_planned()
        .expect("sharding can plan")
        .read_plan(&subset, &options)?
        .expect("sharding reports its reads");
    let decoder_b = array_b.partial_decoder(&[0, 0])?;
    let plan_b = decoder_b
        .as_planned()
        .expect("sharding can plan")
        .read_plan(&subset, &options)?
        .expect("sharding reports its reads");
    assert_eq!(
        plan_a.byte_ranges(),
        plan_b.byte_ranges(),
        "same layout, so the ranges alone cannot tell the two arrays apart"
    );

    // Bytes really fetched from A, handed to B's decoder.
    let fetched = fetch(&store_a, &array_a.chunk_key(&[0, 0]), &plan_a)?;
    let err = decoder_b
        .as_planned()
        .expect("sharding can plan")
        .partial_decode_from_bytes(&plan_a, fetched, &options)
        .expect_err("another array's plan must not be accepted");
    assert!(
        matches!(err, CodecError::ReadPlanMismatch),
        "unexpected error: {err}"
    );

    Ok(())
}

/// A plan from a shard whose index differs is rejected.
///
/// The check compares byte ranges, and shard index offsets are shard-relative, so
/// two shards with equally sized inner chunks produce identical plans and this
/// cannot fire. Erasing a shard is what makes the indexes differ. Pairing a plan
/// with the right shard is the caller's side of the contract.
#[test]
fn a_plan_from_a_differing_shard_is_rejected() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, _) = build(false)?;
    let subset = ArraySubset::new_with_ranges(&[0..4, 0..4]);

    array.erase_chunk(&[1, 1])?;
    let plan = array
        .partial_decoder(&[1, 1])?
        .as_planned()
        .expect("sharding can plan")
        .read_plan(&subset, &options)?
        .expect("sharding reports its reads");

    let other = array.partial_decoder(&[0, 0])?;
    let err = other
        .as_planned()
        .expect("sharding can plan")
        .partial_decode_from_bytes(&plan, vec![None; plan.num_entries()], &options)
        .expect_err("a plan from another shard must fail");
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
        let planned = decoder.as_planned().expect("sharding can plan");

        let err = planned
            .read_plan(&oob, &options)
            .expect_err("planning an out-of-bounds subset must fail");
        assert!(
            matches!(err, CodecError::IncompatibleIndexer(_)),
            "{chunk_indices:?}: unexpected error: {err}"
        );

        // The same selection reaching the decode path as a hand-built plan.
        let err = planned
            .partial_decode_from_bytes(
                &ReadPlan::new(oob.clone(), Vec::new(), 0),
                Vec::new(),
                &options,
            )
            .expect_err("decoding an out-of-bounds plan must fail");
        assert!(
            matches!(err, CodecError::IncompatibleIndexer(_)),
            "{chunk_indices:?}: unexpected error: {err}"
        );
    }

    Ok(())
}
