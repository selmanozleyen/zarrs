//! Coverage for a `sharding_indexed` codec nested inside another.
//!
//! The inner codec chain of a shard may itself be a sharding codec, so an
//! inner chunk is a shard with its own index. Nothing else in the test suite
//! exercises that, and the read behaviour it produces is not obvious: each
//! level of nesting shrinks the index that must be read to locate a chunk, but
//! adds a dependent read to walk down to it.

#![allow(missing_docs)]

use std::error::Error;
use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::codec::array_to_bytes::sharding::{ShardingCodecBuilder, ShardingCodecOptions};
use zarrs::array::{
    Array, ArrayBuilder, ArraySubset, CodecOptions, UnboundArrayToBytesCodecTraits, data_type,
};
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
use zarrs::storage::store::MemoryStore;

type TestStore = PerformanceMetricsStorageAdapter<MemoryStore>;
type TestArray = Result<(Arc<Array<TestStore>>, Arc<TestStore>), Box<dyn Error>>;

const fn nz(v: u64) -> NonZeroU64 {
    NonZeroU64::new(v).unwrap()
}

/// A `[16, 16]` `uint16` array in `[8, 8]` shards, filled with `0..256`.
///
/// The innermost chunk is `[2, 2]` either way, so the two layouts differ only
/// in how many index levels stand between the shard and that chunk:
///
/// - flat: one index over 16 `[2, 2]` chunks
/// - nested: one index over 4 `[4, 4]` inner shards, each with its own index
///   over 4 `[2, 2]` chunks
fn build(nested: bool) -> TestArray {
    build_opt(nested, ShardingCodecOptions::default())
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
fn nested_sharding_round_trips() -> Result<(), Box<dyn Error>> {
    let (flat, _) = build(false)?;
    let (nested, _) = build(true)?;

    assert_eq!(
        nested.retrieve_array_subset::<Vec<u16>>(&nested.subset_all())?,
        (0..256u16).collect::<Vec<_>>()
    );

    // Agreement with the unnested layout, over selections that variously fall
    // inside one innermost chunk, straddle inner shards, and straddle shards.
    for subset in [
        &[4..6, 4..6],
        &[3..5, 3..5],
        &[0..16, 7..9],
        &[7..9, 7..9],
        &[1..2, 14..15],
    ] {
        assert_eq!(
            nested.retrieve_array_subset::<Vec<u16>>(subset)?,
            flat.retrieve_array_subset::<Vec<u16>>(subset)?,
            "nested and flat disagree on {subset:?}"
        );
    }

    Ok(())
}

#[test]
fn nested_sharding_partial_writes_round_trip() -> Result<(), Box<dyn Error>> {
    let (nested, _) = build(true)?;

    // Overwrite a region straddling two inner shards within one shard.
    let subset = &[3..5, 3..5];
    nested.store_array_subset(subset, vec![900u16, 901, 902, 903])?;
    assert_eq!(
        nested.retrieve_array_subset::<Vec<u16>>(subset)?,
        vec![900, 901, 902, 903]
    );

    // Neighbours are untouched.
    assert_eq!(
        nested.retrieve_array_subset::<Vec<u16>>(&[2..3, 2..6])?,
        vec![34, 35, 36, 37]
    );

    Ok(())
}

/// Nesting trades index bytes for round trips.
///
/// Locating a chunk means reading one index per level, so the nested layout
/// performs one more read. Each index is correspondingly smaller, which for
/// this geometry more than pays for the extra read -- 16 index entries flat
/// against 4 + 4 nested.
#[test]
fn nested_sharding_reads_one_index_per_level() -> Result<(), Box<dyn Error>> {
    let mut measured = Vec::new();
    for nested in [false, true] {
        let (array, store) = build(nested)?;
        let before = (store.bytes_read(), store.reads());
        assert_eq!(
            array.retrieve_array_subset::<Vec<u16>>(&[4..6, 4..6])?,
            vec![68, 69, 84, 85]
        );
        measured.push((store.bytes_read() - before.0, store.reads() - before.1));
    }
    let (flat_bytes, flat_reads) = measured[0];
    let (nested_bytes, nested_reads) = measured[1];
    println!("flat:   {flat_bytes} bytes / {flat_reads} reads");
    println!("nested: {nested_bytes} bytes / {nested_reads} reads");

    assert_eq!(
        nested_reads,
        flat_reads + 1,
        "one extra level should cost exactly one extra read"
    );
    assert!(
        nested_bytes < flat_bytes,
        "nested indexes are smaller: {nested_bytes} vs {flat_bytes}"
    );

    Ok(())
}

/// A partial decoder reads each inner shard's index once, not once per access.
///
/// The extra read nesting costs is per inner shard, not per read: a decoder
/// keeps the inner decoders it has built, and each of those holds its decoded
/// index. Two reads landing in the same inner shard should therefore differ by
/// exactly that index read.
#[test]
fn nested_sharding_reads_each_inner_index_once() -> Result<(), Box<dyn Error>> {
    let options = CodecOptions::default();
    let (array, store) = build(true)?;
    let decoder = array.partial_decoder(&[0, 0])?;

    // Both subsets live in inner shard [0, 0], in different innermost chunks.
    let before = store.reads();
    decoder.partial_decode(&ArraySubset::new_with_ranges(&[0..2, 0..2]), &options)?;
    let first = store.reads() - before;

    let before = store.reads();
    decoder.partial_decode(&ArraySubset::new_with_ranges(&[2..4, 2..4]), &options)?;
    let second = store.reads() - before;

    println!("first: {first} reads, second: {second} reads");
    assert_eq!(
        second,
        first - 1,
        "the second read into the same inner shard should skip its index"
    );

    Ok(())
}

/// Reading the subchunk indexes up front moves their cost, it does not remove
/// it.
///
/// Eager pays for every inner shard at construction; lazy pays per inner shard
/// on first touch. A reader that eventually touches them all does the same
/// total work either way, so the case for eager is that afterwards every read
/// resolves without going to storage -- and the case against is the inner
/// shards it reads that are never used.
#[test]
fn eager_prefetch_moves_index_reads_to_construction() -> Result<(), Box<dyn Error>> {
    let mut measured = Vec::new();
    for eager in [false, true] {
        let (array, store) = build_opt(
            true,
            ShardingCodecOptions::default().with_prefetch_subchunk_indexes(eager),
        )?;
        let options = CodecOptions::default();

        let before = store.reads();
        let decoder = array.partial_decoder_opt(&[0, 0], &options)?;
        let build_reads = store.reads() - before;

        // Touch all four inner shards of this shard, one small region each.
        let before = store.reads();
        for (row, col) in [(0, 0), (0, 4), (4, 0), (4, 4)] {
            decoder.partial_decode(
                &ArraySubset::new_with_ranges(&[row..row + 2, col..col + 2]),
                &options,
            )?;
        }
        let read_reads = store.reads() - before;

        println!(
            "eager={eager}: {build_reads} reads to build, {read_reads} reads to decode, \
             {} total",
            build_reads + read_reads
        );
        measured.push((build_reads, read_reads));
    }

    let (lazy_build, lazy_read) = measured[0];
    let (eager_build, eager_read) = measured[1];

    assert!(
        eager_build > lazy_build,
        "eager should read the inner indexes at construction: {eager_build} vs {lazy_build}"
    );
    assert!(
        eager_read < lazy_read,
        "and so decode should not read them again: {eager_read} vs {lazy_read}"
    );
    assert_eq!(
        eager_build + eager_read,
        lazy_build + lazy_read,
        "touching every inner shard costs the same either way; eager only moves it"
    );

    Ok(())
}
