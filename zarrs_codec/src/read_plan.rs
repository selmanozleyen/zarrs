//! The reads a partial decode would perform.

use zarrs_chunk_grid::ArraySubset;
use zarrs_storage::byte_range::ByteRange;

/// The reads a partial decode would perform, and the selection they were computed for.
///
/// Produced by
/// [`read_plan`](crate::ArrayPartialDecoderPlanned::read_plan) and consumed by
/// [`partial_decode_from_bytes`](crate::ArrayPartialDecoderPlanned::partial_decode_from_bytes),
/// which reads the selection back off the plan rather than taking it again. The contract
/// is *entry `i` corresponds to fetched bytes `i`*.
///
/// A plan is not a value to pass around: it belongs to the decoder that produced it, and
/// carries a [`source`](Self::source) so that decoder can recognise it. What the type
/// itself guarantees is only that a plan cannot be paired with a *selection* other than
/// its own; everything else is checked by the decoder when the plan comes back.
///
/// A [`None`] entry marks a unit with nothing to read, which decodes to the fill value.
/// Entries are never omitted, because their positions are the only thing tying the plan to
/// the bytes returned for it.
///
/// The selection is an [`ArraySubset`] rather than an
/// [`Indexer`](zarrs_chunk_grid::Indexer) because only subsets are planned today.
#[derive(Clone, Debug)]
pub struct ReadPlan {
    subset: ArraySubset,
    byte_ranges: Vec<Option<ByteRange>>,
    source: u64,
}

impl ReadPlan {
    /// Create a read plan for `subset` from one byte range per unit of encoded input.
    ///
    /// For implementors of
    /// [`ArrayPartialDecoderPlanned`](crate::ArrayPartialDecoderPlanned), which have to
    /// return one. A caller has no reason to build a plan: a decoder will reject one it
    /// would not have produced itself.
    ///
    /// `source` identifies the state the byte ranges were computed from, and is what a
    /// decoder checks to know the plan is its own. Comparing the ranges is not enough:
    /// two shards with equally sized chunks hold them at the same offsets, so their plans
    /// are identical and each would otherwise accept the other's bytes.
    #[must_use]
    pub const fn new(
        subset: ArraySubset,
        byte_ranges: Vec<Option<ByteRange>>,
        source: u64,
    ) -> Self {
        Self {
            subset,
            byte_ranges,
            source,
        }
    }

    /// The state the byte ranges were computed from, as the producing decoder reported it.
    ///
    /// For implementors, to reject a plan that is not their own.
    #[must_use]
    pub const fn source(&self) -> u64 {
        self.source
    }

    /// The selection this plan describes the reads for.
    ///
    /// For implementors, which read the selection back off the plan in
    /// [`partial_decode_from_bytes`](crate::ArrayPartialDecoderPlanned::partial_decode_from_bytes)
    /// rather than taking it again.
    #[must_use]
    pub const fn subset(&self) -> &ArraySubset {
        &self.subset
    }

    /// One entry per unit of encoded input, in the order the bytes must be handed back,
    /// [`None`] where there is nothing to read.
    ///
    /// Use [`reads`](Self::reads) to visit only the entries with something to read.
    #[must_use]
    pub fn byte_ranges(&self) -> &[Option<ByteRange>] {
        &self.byte_ranges
    }

    /// The reads to perform, each with the index of the entry it belongs to.
    ///
    /// Entries with nothing to read are skipped, so this is what a caller issuing the
    /// reads wants. The index is what puts the bytes back in the right place, and is why
    /// it is yielded rather than left implicit.
    pub fn reads(&self) -> impl Iterator<Item = (usize, ByteRange)> + '_ {
        self.byte_ranges
            .iter()
            .enumerate()
            .filter_map(|(index, byte_range)| byte_range.map(|byte_range| (index, byte_range)))
    }

    /// The number of entries in the plan, and so the number of byte ranges expected back.
    ///
    /// Not the number of reads: an entry with nothing to read still holds a place, since
    /// positions are what tie the plan to the bytes fetched for it.
    #[must_use]
    pub const fn num_entries(&self) -> usize {
        self.byte_ranges.len()
    }

    /// Returns true if the plan has no entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.byte_ranges.is_empty()
    }
}
