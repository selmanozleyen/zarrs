//! The reads a partial decode would perform.

use std::any::Any;
use std::sync::Arc;

use zarrs_chunk_grid::ArraySubset;
use zarrs_plugin::{MaybeSend, MaybeSync};
use zarrs_storage::MaybeBytes;
use zarrs_storage::byte_range::ByteRange;

use core::fmt;

use crate::codec_traits::array_partial_sync::ArrayPartialDecoderPlanned;
use crate::{ArrayBytes, ArrayBytesDecodeIntoTarget, CodecError, CodecOptions};

/// Decoder-private state a plan carries: the walk as the decoder built it.
///
/// A plan's entries are reads, and one read may cover several units whose
/// subdivision only the decoder that planned it knows -- where each unit's bytes sit
/// within the read, and which part of the output it decodes into. That knowledge rides
/// along here as an opaque value the decoder downcasts on the way back in, instead of
/// being re-derived (and re-validated against a rebuilt walk) on every decode.
///
/// Opaque on purpose: the caller's surface is the byte ranges, and the state is the
/// decoder's own. A plan built through the public constructors carries none, which is
/// how a decoder distinguishes plans it minted from plans assembled by hand.
pub trait PlanState: Any + MaybeSend + MaybeSync {
    /// The state as [`Any`], for the decoder that minted it to downcast.
    fn as_any(&self) -> &dyn Any;
}

impl<T: Any + MaybeSend + MaybeSync> PlanState for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The reads a partial decode would perform, and the selection they were computed for.
///
/// Produced by [`read_plan`](ArrayPartialDecoderPlanned::read_plan). A plan holds the
/// decoder that produced it, so it cannot be paired with another decoder's bytes, and it
/// can outlive the call that made it -- a caller may hold plans for many chunks, or for
/// batches it has not started fetching yet, and issue all their reads together.
///
/// **The unit of a plan is a read.** Every entry is a byte range to fetch; units that
/// need nothing read -- absent chunks, decoding to the fill value -- are not entries.
/// One read may cover several stored units when they are adjacent in the stored value,
/// which is the decoder's business: the caller fetches ranges and hands bytes back,
/// however many units each range happens to hold. What was skipped as absent is filled
/// separately through [`DataPlan::fill_absent_into`], which needs no fetched data.
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
    /// The reads shared by either stage: one entry per read, in the order the bytes must
    /// come back.
    #[must_use]
    pub fn byte_ranges(&self) -> &[ByteRange] {
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

/// The entries of one plan stage: the selection and one read per entry.
///
/// The contract is *entry `i` corresponds to fetched bytes `i`*. Every entry is a read;
/// units with nothing to read are not entries.
#[derive(Clone, Debug)]
struct PlanReads {
    subset: ArraySubset,
    byte_ranges: Vec<ByteRange>,
}

impl PlanReads {
    fn reads(&self) -> impl Iterator<Item = (usize, ByteRange)> + '_ {
        self.byte_ranges.iter().copied().enumerate()
    }
}

/// A plan whose reads are the encoded data a selection wants.
///
/// Fetch the [`reads`](Self::reads) and hand the bytes to [`decode_into`](Self::decode_into)
/// or [`decode`](Self::decode), and fill what the plan skipped as absent through
/// [`fill_absent_into`](Self::fill_absent_into) -- once, at any point, since it needs no
/// fetched data. [`decode`](Self::decode) allocates its own output, so it does the fill
/// itself.
#[derive(Clone)]
pub struct DataPlan {
    decoder: Arc<dyn ArrayPartialDecoderPlanned>,
    reads: PlanReads,
    state: Option<Arc<dyn PlanState>>,
}

impl DataPlan {
    /// Create a data plan, for implementors of [`ArrayPartialDecoderPlanned`].
    ///
    /// `decoder` must be the decoder whose reads these are: the plan's methods hand the
    /// fetched bytes back to it. A plan built this way carries no
    /// [state](Self::new_with_state), and a decoder that requires its own state to
    /// decode rejects it.
    #[must_use]
    pub fn new(
        decoder: Arc<dyn ArrayPartialDecoderPlanned>,
        subset: ArraySubset,
        byte_ranges: Vec<ByteRange>,
    ) -> Self {
        Self {
            decoder,
            reads: PlanReads {
                subset,
                byte_ranges,
            },
            state: None,
        }
    }

    /// [`new`](Self::new), carrying decoder-private state.
    ///
    /// For implementors: `state` is whatever the decoder needs to consume the plan
    /// without re-deriving its walk -- see [`PlanState`]. It is the decoder's job to
    /// verify on the way back in that the state is one it minted, and that it matches
    /// the plan it arrived with.
    #[must_use]
    pub fn new_with_state(
        decoder: Arc<dyn ArrayPartialDecoderPlanned>,
        subset: ArraySubset,
        byte_ranges: Vec<ByteRange>,
        state: Arc<dyn PlanState>,
    ) -> Self {
        Self {
            decoder,
            reads: PlanReads {
                subset,
                byte_ranges,
            },
            state: Some(state),
        }
    }

    /// The decoder-private state, for the decoder that minted it.
    #[must_use]
    pub fn state(&self) -> Option<&dyn PlanState> {
        self.state.as_deref()
    }

    /// The selection this plan describes the reads for.
    #[must_use]
    pub fn subset(&self) -> &ArraySubset {
        &self.reads.subset
    }

    /// One entry per read, in the order the bytes must be handed back.
    #[must_use]
    pub fn byte_ranges(&self) -> &[ByteRange] {
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

    /// Fill the parts of the output whose units have nothing to read.
    ///
    /// A unit that is not stored decodes to the fill value, and saying so takes no
    /// fetched data -- so this is separate from [`decode_into`](Self::decode_into),
    /// which writes only what was read, and can run before any read returns. Call it
    /// once per plan, on the same output the decodes write into. Performs no I/O.
    ///
    /// # Errors
    /// Returns [`CodecError::ReadPlanMismatch`] if the plan is not one this decoder
    /// produced, or [`CodecError`] if the output is invalid for the plan's selection.
    pub fn fill_absent_into(
        &self,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        self.decoder.fill_absent_into(self, output_target, options)
    }

    /// Decode the fetched bytes into a preallocated output.
    ///
    /// `fetched` must correspond one-to-one, and in order, with the plan's entries --
    /// [`MaybeBytes`], exactly what a store hands back. Every entry is a read, so
    /// [`None`] never matches the plan; a store answering [`None`] means the value
    /// changed since planning, and the decode reports the mismatch rather than guessing.
    ///
    /// Writes only what was read: the parts of the output belonging to units with
    /// nothing to read are [`fill_absent_into`](Self::fill_absent_into)'s, called once
    /// by the caller. Performs no I/O.
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
    /// See [`decode_into`](Self::decode_into). The output is complete: this allocates
    /// the buffer itself, so it also does what [`fill_absent_into`](Self::fill_absent_into)
    /// would.
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
/// for the [`DataPlan`]. Entries are reads: one may cover several subchunks' indexes
/// when they are adjacent in the stored value.
#[derive(Clone)]
pub struct IndexPlan {
    decoder: Arc<dyn ArrayPartialDecoderPlanned>,
    reads: PlanReads,
    state: Option<Arc<dyn PlanState>>,
}

impl IndexPlan {
    /// Create an index plan, for implementors of [`ArrayPartialDecoderPlanned`].
    ///
    /// See [`DataPlan::new`] for what `decoder` must be and what a stateless plan means.
    #[must_use]
    pub fn new(
        decoder: Arc<dyn ArrayPartialDecoderPlanned>,
        subset: ArraySubset,
        byte_ranges: Vec<ByteRange>,
    ) -> Self {
        Self {
            decoder,
            reads: PlanReads {
                subset,
                byte_ranges,
            },
            state: None,
        }
    }

    /// [`new`](Self::new), carrying decoder-private state. See [`DataPlan::new_with_state`].
    #[must_use]
    pub fn new_with_state(
        decoder: Arc<dyn ArrayPartialDecoderPlanned>,
        subset: ArraySubset,
        byte_ranges: Vec<ByteRange>,
        state: Arc<dyn PlanState>,
    ) -> Self {
        Self {
            decoder,
            reads: PlanReads {
                subset,
                byte_ranges,
            },
            state: Some(state),
        }
    }

    /// The decoder-private state, for the decoder that minted it.
    #[must_use]
    pub fn state(&self) -> Option<&dyn PlanState> {
        self.state.as_deref()
    }

    /// The selection this plan describes the reads for.
    #[must_use]
    pub fn subset(&self) -> &ArraySubset {
        &self.reads.subset
    }

    /// One entry per read, in the order the bytes must be handed back.
    #[must_use]
    pub fn byte_ranges(&self) -> &[ByteRange] {
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
