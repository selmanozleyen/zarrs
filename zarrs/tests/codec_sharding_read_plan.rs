//! `ArrayPartialDecoderTraits::read_plan` / `partial_decode_prefetched`.
//!
//! The point of the pair is that a caller holding several decoders can collect
//! all of their reads, issue them together, and hand the bytes back. These
//! tests pin the two properties that makes possible: planning touches no
//! storage, and decoding from prefetched bytes touches no storage either.

#![allow(missing_docs)]

use std::error::Error;
use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
use zarrs::array::{
    Array, ArrayBuilder, ArraySubset, CodecOptions, UnboundArrayToBytesCodecTraits, data_type,
};
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{ReadableStorageTraits, StoreKey};

type TestStore = PerformanceMetricsStorageAdapter<MemoryStore>;
type TestArray = Result<(Arc<Array<TestStore>>, Arc<TestStore>), Box<dyn Error>>;

const fn nz(v: u64) -> NonZeroU64 {
    NonZeroU64::new(v).unwrap()
}

/// A `[16, 16]` `uint16` array in `[8, 8]` shards of `[2, 2]` inner chunks.
///
/// With `nested`, the shard instead holds `[4, 4]` inner chunks that are
/// themselves shards of `[2, 2]` chunks.
fn build(nested: bool) -> TestArray {
    let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
        MemoryStore::default(),
    )));
    let data_type = data_type::uint16();
    let codec: Arc<dyn UnboundArrayToBytesCodecTraits> = if nested {
        let inner = ShardingCodecBuilder::new(vec![nz(2), nz(2)], &data_type).build();
        Arc::new(
            ShardingCodecBuilder::new(vec![nz(4), nz(4)], &data_type)
                .array_to_bytes_codec(Arc::new(inner))
                .build(),
        )
    } else {
        Arc::new(ShardingCodecBuilder::new(vec![nz(2), nz(2)], &data_type).build())
    };
    let mut builder = ArrayBuilder::new(vec![16, 16], vec![8, 8], data_type, 0u16);
    builder.array_to_bytes_codec(codec);
    let array = builder.build_arc(store.clone(), "/array")?;
    let data = (0..256u16).collect::<Vec<_>>();
    array.store_array_subset(&array.subset_all(), &data)?;
    Ok((array, store))
}

#[test]
fn plan_then_prefetched_decode_matches_and_reads_nothing() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, store) = build(false)?;
    let key: StoreKey = array.chunk_key(&[0, 0]);
    // Shard-local: rows 2..6, columns 1..5, spanning six inner chunks.
    let subset = ArraySubset::new_with_ranges(&[2..6, 1..5]);

    let decoder = array.partial_decoder(&[0, 0])?;
    let expected = decoder.partial_decode(&subset, &options)?.into_fixed()?;

    let before = store.reads();
    let plan = decoder
        .read_plan(&subset, &options)?
        .expect("sharding reports its reads");
    assert_eq!(store.reads(), before, "planning must not perform any reads");
    assert!(!plan.is_empty());

    // The caller does the I/O itself, in whatever order it likes.
    let fetched = plan
        .iter()
        .map(|byte_range| {
            byte_range
                .map(|byte_range| store.get_partial(&key, byte_range))
                .transpose()
                .map(|bytes| bytes.flatten().map(|bytes| bytes.to_vec().into()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let before = store.reads();
    let decoded = decoder
        .partial_decode_prefetched(&subset, fetched, &options)?
        .into_fixed()?;
    assert_eq!(
        store.reads(),
        before,
        "prefetched decode must not touch storage"
    );
    assert_eq!(decoded, expected);

    Ok(())
}

/// A shard with no inner chunk present still reports one entry per chunk, so
/// the plan stays one-to-one with what `partial_decode_prefetched` expects.
#[test]
fn plan_covers_a_missing_shard() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, _) = build(false)?;
    let subset = ArraySubset::new_with_ranges(&[0..4, 0..4]);

    // Chunk [1, 1] was written; erase it so the shard is absent.
    array.erase_chunk(&[1, 1])?;
    let decoder = array.partial_decoder(&[1, 1])?;
    let plan = decoder
        .read_plan(&subset, &options)?
        .expect("sharding reports its reads");
    assert_eq!(plan.len(), 4, "one entry per inner chunk in the subset");
    assert!(plan.iter().all(Option::is_none), "nothing to read");

    let decoded = decoder
        .partial_decode_prefetched(&subset, vec![None; plan.len()], &options)?
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

    assert!(
        decoder.read_plan(&subset, &options)?.is_none(),
        "nested sharding must not report a one-level plan"
    );

    // And the fallback stays correct: with no plan, prefetched decode defers
    // to `partial_decode` and ignores whatever it was handed.
    let expected = decoder.partial_decode(&subset, &options)?.into_fixed()?;
    let decoded = decoder
        .partial_decode_prefetched(&subset, Vec::new(), &options)?
        .into_fixed()?;
    assert_eq!(decoded, expected);

    Ok(())
}
