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
/// One entry per subchunk the selection touches, in the order the bytes must be handed
/// back. A [`None`] entry marks a subchunk with nothing to read, which decodes to the fill
/// value. Entries are never omitted, because their positions are the only thing tying the
/// plan to the bytes returned for it.
///
/// Where subchunks are themselves subchunked, a plan reaches the innermost level rather
/// than naming whole nested subchunks -- which would read far more than was asked for. That
/// takes two rounds, since the nested indexes have to be read before the data they locate
/// can be named: see [`PlanStage`] and [`is_final`](Self::is_final).
///
/// The selection is an [`ArraySubset`] rather than an
/// [`Indexer`](zarrs_chunk_grid::Indexer) because only subsets are planned today.
#[derive(Clone, Debug)]
pub struct ReadPlan {
    subset: ArraySubset,
    byte_ranges: Vec<Option<ByteRange>>,
    source: u64,
    stage: PlanStage,
}

/// What a [`ReadPlan`]'s reads are for.
///
/// Locating some data takes a read of its own: a subchunk nested inside a subchunk has its
/// own index, and where that index lives is computable but what it says is not. So a plan
/// may name those indexes instead of the data, and be exchanged for the data plan once they
/// have been fetched.
///
/// The staging is what keeps the reads batchable. The alternative is discovering each
/// nested index when its subchunk is first touched, which serialises exactly the reads a
/// plan exists to issue together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanStage {
    /// Encoded array data. Hand the bytes to
    /// [`partial_decode_from_bytes`](crate::ArrayPartialDecoderPlanned::partial_decode_from_bytes).
    Data,
    /// Subchunk indexes. Hand the bytes to
    /// [`refine_read_plan`](crate::ArrayPartialDecoderPlanned::refine_read_plan), which
    /// returns the plan for the data they locate.
    SubchunkIndexes,
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
            stage: PlanStage::Data,
        }
    }

    /// Create a plan whose reads are subchunk indexes rather than data.
    ///
    /// For a decoder whose subchunks are themselves subchunked: the reads locate each
    /// touched subchunk's own index, and
    /// [`refine_read_plan`](crate::ArrayPartialDecoderPlanned::refine_read_plan) turns
    /// those bytes into the plan for the data.
    #[must_use]
    pub const fn new_subchunk_indexes(
        subset: ArraySubset,
        byte_ranges: Vec<Option<ByteRange>>,
        source: u64,
    ) -> Self {
        Self {
            subset,
            byte_ranges,
            source,
            stage: PlanStage::SubchunkIndexes,
        }
    }

    /// What this plan's reads are for.
    #[must_use]
    pub const fn stage(&self) -> PlanStage {
        self.stage
    }

    /// Whether the fetched bytes can go straight to `partial_decode_from_bytes`.
    ///
    /// [`false`] means one more round: fetch, then
    /// [`refine_read_plan`](crate::ArrayPartialDecoderPlanned::refine_read_plan).
    #[must_use]
    pub const fn is_final(&self) -> bool {
        matches!(self.stage, PlanStage::Data)
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

    /// One entry per level-zero subchunk, in the order the bytes must be handed back,
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
    /// Not the number of reads: a subchunk with nothing to read still holds a place, since
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
