//! The reads a partial decode would perform.

use std::sync::Arc;

use zarrs_chunk_grid::ArraySubset;
use zarrs_storage::MaybeBytes;
use zarrs_storage::byte_range::ByteRange;

use core::fmt;

use crate::codec_traits::array_partial_sync::ArrayPartialDecoderPlanned;
use crate::{ArrayBytes, ArrayBytesDecodeIntoTarget, CodecError, CodecOptions};

/// The reads a partial decode would perform, and the selection they were computed for.
///
/// Produced by [`read_plan`](ArrayPartialDecoderPlanned::read_plan). A plan holds the
/// decoder that produced it, so it cannot be paired with another decoder's bytes, and it
/// can outlive the call that made it -- a caller may hold plans for many chunks, or for
/// batches it has not started fetching yet, and issue all their reads together.
///
/// The two stages are two types. [`Data`](Self::Data) reads are the encoded data itself:
/// fetch them and [`decode_into`](DataPlan::decode_into). [`Indexes`](Self::Indexes)
/// reads locate data rather than being it -- a subchunk nested inside a subchunk has its
/// own index, at a computable place with uncomputable contents. Fetch those and exchange
/// them through [`refine`](IndexPlan::refine), which returns the [`DataPlan`] -- exactly
/// one exchange, which the types state: `refine` does not return another [`IndexPlan`],
/// and a selection that would need more than one round is not planned at all.
///
/// Staging is what keeps the reads batchable: the alternative is discovering each nested
/// index when its subchunk is first touched, inside the decode, which serialises exactly
/// the reads a plan exists to issue together.
///
/// The selection is an [`ArraySubset`] rather than an
/// [`Indexer`](zarrs_chunk_grid::Indexer) because only subsets are planned today.
#[derive(Clone)]
pub enum ReadPlan {
    /// The reads are the encoded data. Fetch, then [`DataPlan::decode_into`].
    Data(DataPlan),
    /// The reads are subchunk indexes. Fetch, then [`IndexPlan::refine`].
    Indexes(IndexPlan),
}

impl ReadPlan {
    /// The reads shared by either stage: one entry per unit, in the order the bytes must
    /// come back, [`None`] where there is nothing to read.
    #[must_use]
    pub fn byte_ranges(&self) -> &[Option<ByteRange>] {
        match self {
            Self::Data(plan) => plan.byte_ranges(),
            Self::Indexes(plan) => plan.byte_ranges(),
        }
    }

    /// The reads to perform, each with the index of the entry it belongs to.
    pub fn reads(&self) -> impl Iterator<Item = (usize, ByteRange)> + '_ {
        match self {
            Self::Data(plan) => plan.reads.reads(),
            Self::Indexes(plan) => plan.reads.reads(),
        }
    }

    /// The number of entries, and so the number of byte ranges expected back.
    #[must_use]
    pub fn num_entries(&self) -> usize {
        self.byte_ranges().len()
    }
}

/// The entries of one plan stage: the selection and one optional read per unit.
///
/// The contract is *entry `i` corresponds to fetched bytes `i`*. A [`None`] entry marks a
/// unit with nothing to read; entries are never omitted, because their positions are the
/// only thing tying a plan to the bytes returned for it.
#[derive(Clone, Debug)]
struct PlanReads {
    subset: ArraySubset,
    byte_ranges: Vec<Option<ByteRange>>,
}

impl PlanReads {
    fn reads(&self) -> impl Iterator<Item = (usize, ByteRange)> + '_ {
        self.byte_ranges
            .iter()
            .enumerate()
            .filter_map(|(index, byte_range)| byte_range.map(|byte_range| (index, byte_range)))
    }
}

/// A plan whose reads are the encoded data a selection wants.
///
/// Fetch the [`reads`](Self::reads) and hand the bytes to [`decode_into`](Self::decode_into)
/// or [`decode`](Self::decode). One entry per subchunk (or innermost chunk, where
/// subchunks are nested), in order; a [`None`] entry decodes to the fill value.
#[derive(Clone)]
pub struct DataPlan {
    decoder: Arc<dyn ArrayPartialDecoderPlanned>,
    reads: PlanReads,
}

impl DataPlan {
    /// Create a data plan, for implementors of [`ArrayPartialDecoderPlanned`].
    ///
    /// `decoder` must be the decoder whose reads these are: the plan's methods hand the
    /// fetched bytes back to it. A decoder still checks the ranges against the reads it
    /// would perform -- construction is public, so holding the right decoder does not
    /// prove the ranges are its.
    #[must_use]
    pub fn new(
        decoder: Arc<dyn ArrayPartialDecoderPlanned>,
        subset: ArraySubset,
        byte_ranges: Vec<Option<ByteRange>>,
    ) -> Self {
        Self {
            decoder,
            reads: PlanReads {
                subset,
                byte_ranges,
            },
        }
    }

    /// The selection this plan describes the reads for.
    #[must_use]
    pub fn subset(&self) -> &ArraySubset {
        &self.reads.subset
    }

    /// One entry per unit, in the order the bytes must be handed back, [`None`] where
    /// there is nothing to read.
    #[must_use]
    pub fn byte_ranges(&self) -> &[Option<ByteRange>] {
        &self.reads.byte_ranges
    }

    /// The reads to perform, each with the index of the entry it belongs to.
    pub fn reads(&self) -> impl Iterator<Item = (usize, ByteRange)> + '_ {
        self.reads.reads()
    }

    /// The number of entries, and so the number of byte ranges expected back.
    #[must_use]
    pub fn num_entries(&self) -> usize {
        self.reads.byte_ranges.len()
    }

    /// Decode the fetched bytes into a preallocated output.
    ///
    /// `fetched` must correspond one-to-one, and in order, with the plan's entries --
    /// [`MaybeBytes`], exactly what a store hands back, [`None`] for an entry the plan
    /// said there was nothing to read for. Performs no I/O.
    ///
    /// # Errors
    /// Returns [`CodecError::ReadPlanMismatch`] if the plan or `fetched` do not match the
    /// reads the decoder would perform, or [`CodecError`] if a codec fails.
    pub fn decode_into(
        &self,
        fetched: Vec<MaybeBytes>,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        self.decoder
            .partial_decode_from_bytes_into(self, fetched, output_target, options)
    }

    /// Decode the fetched bytes into freshly allocated array bytes.
    ///
    /// See [`decode_into`](Self::decode_into).
    ///
    /// # Errors
    /// Returns [`CodecError::ReadPlanMismatch`] if the plan or `fetched` do not match the
    /// reads the decoder would perform, or [`CodecError`] if a codec fails.
    pub fn decode(
        &self,
        fetched: Vec<MaybeBytes>,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        self.decoder
            .partial_decode_from_bytes(self, fetched, options)
    }
}

/// A plan whose reads locate data rather than being it: the indexes of subchunks nested
/// inside the decoder's subchunks.
///
/// Fetch the [`reads`](Self::reads) and exchange them through [`refine`](Self::refine)
/// for the [`DataPlan`]. One entry per subchunk the selection wants only part of.
#[derive(Clone)]
pub struct IndexPlan {
    decoder: Arc<dyn ArrayPartialDecoderPlanned>,
    reads: PlanReads,
}

impl IndexPlan {
    /// Create an index plan, for implementors of [`ArrayPartialDecoderPlanned`].
    ///
    /// See [`DataPlan::new`] for what `decoder` must be and what it still checks.
    #[must_use]
    pub fn new(
        decoder: Arc<dyn ArrayPartialDecoderPlanned>,
        subset: ArraySubset,
        byte_ranges: Vec<Option<ByteRange>>,
    ) -> Self {
        Self {
            decoder,
            reads: PlanReads {
                subset,
                byte_ranges,
            },
        }
    }

    /// The selection this plan describes the reads for.
    #[must_use]
    pub fn subset(&self) -> &ArraySubset {
        &self.reads.subset
    }

    /// One entry per unit, in the order the bytes must be handed back, [`None`] where
    /// there is nothing to read.
    #[must_use]
    pub fn byte_ranges(&self) -> &[Option<ByteRange>] {
        &self.reads.byte_ranges
    }

    /// The reads to perform, each with the index of the entry it belongs to.
    pub fn reads(&self) -> impl Iterator<Item = (usize, ByteRange)> + '_ {
        self.reads.reads()
    }

    /// The number of entries, and so the number of byte ranges expected back.
    #[must_use]
    pub fn num_entries(&self) -> usize {
        self.reads.byte_ranges.len()
    }

    /// Exchange the fetched index bytes for the plan of the data they locate.
    ///
    /// Consumes the plan: an index round happens once, and the result is a [`DataPlan`]
    /// rather than another [`IndexPlan`] -- one exchange is the whole state machine.
    /// Performs no I/O.
    ///
    /// # Errors
    /// Returns [`CodecError::ReadPlanMismatch`] if the plan or `fetched` do not match the
    /// reads the decoder would perform, or [`CodecError`] if a codec fails.
    pub fn refine(
        self,
        fetched: Vec<MaybeBytes>,
        options: &CodecOptions,
    ) -> Result<DataPlan, CodecError> {
        self.decoder
            .clone()
            .refine_index_plan(self, fetched, options)
    }
}

// The decoder is a trait object without `Debug`, so these print what a plan says
// rather than what it holds.
impl fmt::Debug for DataPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DataPlan")
            .field("reads", &self.reads)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for IndexPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IndexPlan")
            .field("reads", &self.reads)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for ReadPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Data(plan) => f.debug_tuple("Data").field(plan).finish(),
            Self::Indexes(plan) => f.debug_tuple("Indexes").field(plan).finish(),
        }
    }
}
