//! The reads a partial decode would perform.

use zarrs_chunk_grid::ArraySubset;
use zarrs_storage::byte_range::ByteRange;

/// The reads a partial decode would perform, and the selection they were computed for.
///
/// Produced by
/// [`read_plan`](crate::ArrayPartialDecoderPlanned::read_plan) and consumed by
/// [`partial_decode_from_bytes`](crate::ArrayPartialDecoderPlanned::partial_decode_from_bytes),
/// which reads the selection back off the plan rather than taking it again. That is the
/// point of the type: the contract is *entry `i` corresponds to fetched bytes `i`*, and a
/// plan that carries its own selection cannot be paired with a different one.
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
}

impl ReadPlan {
    /// Create a read plan for `subset` from one byte range per unit of encoded input.
    ///
    /// For decoders producing a plan. Public only because a decoder generally lives in
    /// another crate; callers have no reason to build one.
    #[doc(hidden)]
    #[must_use]
    pub const fn new(subset: ArraySubset, byte_ranges: Vec<Option<ByteRange>>) -> Self {
        Self {
            subset,
            byte_ranges,
        }
    }

    /// The selection this plan describes the reads for.
    ///
    /// For the decoder consuming the plan, which reads the selection back off it rather
    /// than taking it again.
    #[doc(hidden)]
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
