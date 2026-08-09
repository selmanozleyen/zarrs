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
    #[must_use]
    pub const fn new(subset: ArraySubset, byte_ranges: Vec<Option<ByteRange>>) -> Self {
        Self {
            subset,
            byte_ranges,
        }
    }

    /// The selection this plan describes the reads for.
    #[must_use]
    pub const fn subset(&self) -> &ArraySubset {
        &self.subset
    }

    /// The byte ranges to fetch, in the order the bytes must be handed back.
    #[must_use]
    pub fn byte_ranges(&self) -> &[Option<ByteRange>] {
        &self.byte_ranges
    }

    /// The number of entries in the plan, and so the number of byte ranges expected back.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.byte_ranges.len()
    }

    /// Returns true if the plan has no entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.byte_ranges.is_empty()
    }
}
