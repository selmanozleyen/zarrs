use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use itertools::izip;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_chunk_grid::{ArraySubset, ChunkGridTraits};

use super::{
    ShardingCodecOptions, ShardingIndexLocation, calculate_chunks_per_shard,
    nested_local_subchunk_grids,
};
use crate::array::array_bytes_internal::merge_chunks_vlen;
use crate::array::chunk_grid::RegularChunkGrid;
use crate::array::{
    ArrayBytes, ArrayBytesFixedDisjointView, ArrayBytesOffsets, ArrayBytesRaw, ArrayIndices,
    ArrayIndicesTinyVec, ArraySubsetTraits, ChunkGrid, ChunkShape, ChunkShapeTraits,
    CodecChainBound, DataType, DataTypeSize, IncompatibleDimensionalityError, Indexer,
    IndexerError, ravel_indices,
};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayCodecTraits, ArrayPartialDecoderPlanned,
    ArrayPartialDecoderTraits, ArrayToBytesCodecTraits, ByteIntervalPartialDecoder,
    BytesPartialDecoderTraits, CodecError, CodecOptions, ExpectedFixedLengthBytesError,
    InvalidNumberOfElementsError, ReadPlan, decode_into_array_bytes_target,
};
use zarrs_plugin::ExtensionAliasesV3;
use zarrs_storage::byte_range::{ByteLength, ByteOffset, ByteRange};
use zarrs_storage::{Bytes, MaybeBytes, StorageError};

/// Partial decoder for the sharding codec.
pub struct ShardingPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    shard_shape: ChunkShape,
    subchunk_shape: ChunkShape,
    inner_codecs: Arc<CodecChainBound>,
    shard_index: Option<Vec<u64>>,
    #[expect(dead_code)] // TODO: Remove when sharding-specific options are added
    sharding_options: ShardingCodecOptions,
    /// Whether an inner chunk is itself a shard, resolved once at construction.
    ///
    /// Answering it walks the codec chain, and it decides both whether inner decoders
    /// are worth keeping and whether a selection can be planned -- the latter on every
    /// planned call. Resolved with the options this decoder was built with, as the
    /// decoder cache below already was.
    nested: bool,
    /// Whether the shard index's byte ranges are offsets into the stored value.
    ///
    /// They are offsets into `input_handle`, which is the stored value only when nothing
    /// sits in between. A bytes-to-bytes codec *outside* the sharding codec puts a
    /// decompressor or a prefix-stripper there, and a range reported to a caller would
    /// then name the wrong bytes of the stored value -- of the right length, so neither
    /// the caller nor the decode would notice. Planning is declined in that case.
    plannable_input: bool,
    /// Identifies this decoder, so a plan can be recognised as its own.
    ///
    /// Neither the byte ranges nor the shard index distinguish shards: equally sized
    /// inner chunks sit at the same offsets in every shard, so two shards of one array
    /// have byte-identical indexes and byte-identical plans. Only identity separates
    /// them, and a decoder is the thing that holds a shard's index, so identity is
    /// per decoder.
    plan_source: u64,
}

/// Hands each decoder an identity, so a plan can be recognised as its own.
///
/// A counter rather than a hash of anything: what has to be told apart is which shard
/// the bytes came from, and two shards of one array are indistinguishable by content.
/// Rebuilding a decoder therefore invalidates plans it did not make, which is the safe
/// direction.
static PLAN_SOURCE: AtomicU64 = AtomicU64::new(0);

impl ShardingPartialDecoder {
    /// Create a new partial decoder for the sharding codec.
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shard_shape: ChunkShape,
        subchunk_shape: ChunkShape,
        inner_codecs: Arc<CodecChainBound>,
        index_codecs: &CodecChainBound,
        index_location: ShardingIndexLocation,
        options: &CodecOptions,
        sharding_options: ShardingCodecOptions,
    ) -> Result<Self, CodecError> {
        let shard_index = super::decode_shard_index_partial_decoder(
            &*input_handle,
            index_codecs,
            index_location,
            &shard_shape,
            &subchunk_shape,
            options,
        )?;

        let plannable_input = input_handle.byte_ranges_are_stored_offsets();
        let plan_source = PLAN_SOURCE.fetch_add(1, Ordering::Relaxed);
        let mut decoder = Self {
            input_handle,
            shard_shape,
            subchunk_shape,
            inner_codecs,
            shard_index,
            sharding_options,
            nested: false,
            plannable_input,
            plan_source,
        };

        // More than one level in the local grid hierarchy means the inner chunk is
        // itself a shard carrying its own index.
        decoder.nested = decoder.local_subchunk_grids(options)?.len() > 1;
        Ok(decoder)
    }

    /// Retrieve the byte range of an encoded subchunk.
    ///
    /// The `chunk_indices` are relative to the start of the shard.
    pub fn subchunk_byte_range(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ByteRange>, CodecError> {
        super::subchunk_byte_range(
            self.shard_index.as_deref(),
            &self.shard_shape,
            &self.subchunk_shape,
            chunk_indices,
        )
    }

    /// Retrieve the encoded bytes of a subchunk.
    ///
    /// The `chunk_indices` are relative to the start of the shard.
    pub fn retrieve_subchunk_encoded(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayBytesRaw<'_>>, CodecError> {
        let byte_range = self.subchunk_byte_range(chunk_indices)?;
        if let Some(byte_range) = byte_range {
            self.input_handle
                .partial_decode(byte_range, &CodecOptions::default())
        } else {
            Ok(None)
        }
    }
}

pub(crate) fn partial_decode(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    indexer: &dyn crate::array::Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let data_type = inner_codecs.data_type();
    if indexer.dimensionality() != shard_shape.len() {
        return Err(IndexerError::new_incompatible_dimensionality(
            indexer.dimensionality(),
            shard_shape.len(),
        )
        .into());
    }

    if data_type.is_optional() {
        return Err(CodecError::UnsupportedDataType(
            data_type.clone(),
            super::ShardingCodec::aliases_v3().default_name.to_string(),
        ));
    }

    match data_type.size() {
        DataTypeSize::Fixed(data_type_size) => {
            if let Some(subset) = indexer.as_array_subset() {
                let array_shape = subset.shape();
                let array_subset_size = subset.num_elements_usize() * data_type_size;
                let mut out_array_subset = vec![0; array_subset_size];
                let out_array_subset_slice = UnsafeCellSlice::new(out_array_subset.as_mut_slice());
                let mut output_view = unsafe {
                    ArrayBytesFixedDisjointView::new(
                        out_array_subset_slice,
                        data_type_size,
                        &array_shape,
                        ArraySubset::new_with_shape(array_shape.to_vec()),
                    )?
                };
                partial_decode_fixed_array_subset_into(
                    input_handle,
                    shard_shape,
                    subchunk_shape,
                    inner_codecs,
                    shard_index,
                    subset,
                    options,
                    &mut output_view,
                )?;
                Ok(ArrayBytes::from(out_array_subset))
            } else {
                partial_decode_fixed_indexer(
                    input_handle,
                    shard_shape,
                    subchunk_shape,
                    inner_codecs,
                    shard_index,
                    indexer,
                    options,
                )
            }
        }
        DataTypeSize::Variable => {
            if let Some(subset) = indexer.as_array_subset() {
                partial_decode_variable_array_subset(
                    input_handle,
                    shard_shape,
                    subchunk_shape,
                    inner_codecs,
                    shard_index,
                    subset,
                    options,
                )
            } else {
                partial_decode_variable_indexer(
                    input_handle,
                    shard_shape,
                    subchunk_shape,
                    inner_codecs,
                    shard_index,
                    indexer,
                    options,
                )
            }
        }
    }
}

impl ArrayPartialDecoderTraits for ShardingPartialDecoder {
    fn data_type(&self) -> &DataType {
        self.inner_codecs.data_type()
    }

    fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
            + self.shard_index.as_ref().map_or(0, Vec::len) * size_of::<u64>()
    }

    fn partial_decode(
        &self,
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        partial_decode(
            &self.input_handle,
            &self.shard_shape,
            &self.subchunk_shape,
            &self.inner_codecs,
            self.shard_index.as_deref(),
            indexer,
            options,
        )
    }

    fn local_subchunk_grids(
        &self,
        _options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        let shard_shape = bytemuck::must_cast_slice(&self.shard_shape).to_vec();
        let subchunk_grid = ChunkGrid::new(
            RegularChunkGrid::new(shard_shape, self.subchunk_shape.clone())
                .map_err(|err| CodecError::Other(err.to_string()))?,
        );
        nested_local_subchunk_grids(subchunk_grid, &self.inner_codecs)
    }

    fn partial_decode_into(
        &self,
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        if indexer.len() != output_target.num_elements() {
            return Err(InvalidNumberOfElementsError::new(
                indexer.len(),
                output_target.num_elements(),
            )
            .into());
        }
        if let DataTypeSize::Fixed(_data_type_size) = self.inner_codecs.data_type().size()
            && let Some(subset) = indexer.as_array_subset()
            && let ArrayBytesDecodeIntoTarget::Fixed(output_view) = output_target
        {
            partial_decode_fixed_array_subset_into(
                &self.input_handle,
                &self.shard_shape,
                &self.subchunk_shape,
                &self.inner_codecs,
                self.shard_index.as_deref(),
                subset,
                options,
                output_view,
            )
        } else {
            let decoded_value = self.partial_decode(indexer, options)?;
            decode_into_array_bytes_target(&decoded_value, output_target)
        }
    }

    fn as_planned(&self) -> Option<&dyn ArrayPartialDecoderPlanned> {
        Some(self)
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

impl ArrayPartialDecoderPlanned for ShardingPartialDecoder {
    fn read_plan(
        &self,
        indexer: &dyn Indexer,
        // Whether a selection can be planned was resolved at construction.
        _options: &CodecOptions,
    ) -> Result<Option<ReadPlan>, CodecError> {
        let Some((subset, _)) = self.planned_subset(indexer) else {
            return Ok(None);
        };
        let planned = plan_subchunk_tasks(
            &self.shard_shape,
            &self.subchunk_shape,
            self.shard_index.as_deref(),
            subset,
        )?;
        Ok(Some(ReadPlan::new(
            subset.to_array_subset(),
            planned.tasks.iter().map(SubchunkTask::byte_range).collect(),
            self.plan_source,
        )))
    }

    fn partial_decode_from_bytes(
        &self,
        plan: &ReadPlan,
        fetched: Vec<MaybeBytes>,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        let (subset, data_type_size, planned) = self.checked_tasks(plan, &fetched)?;

        let array_shape = subset.shape();
        let mut out = vec![0; subset.num_elements_usize() * data_type_size];
        let out_slice = UnsafeCellSlice::new(out.as_mut_slice());
        let mut output_view = unsafe {
            ArrayBytesFixedDisjointView::new(
                out_slice,
                data_type_size,
                &array_shape,
                ArraySubset::new_with_shape(array_shape.to_vec()),
            )?
        };
        partial_decode_fixed_array_subset_from_bytes_into(
            &self.subchunk_shape,
            &self.inner_codecs,
            planned,
            subset,
            fetched,
            options,
            &mut output_view,
        )?;
        Ok(ArrayBytes::from(out))
    }

    fn partial_decode_from_bytes_into(
        &self,
        plan: &ReadPlan,
        fetched: Vec<MaybeBytes>,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        // Checked in the same order as the default implementation, so the two are
        // substitutable: a caller gets the same error whichever one runs.
        if plan.subset().num_elements() != output_target.num_elements() {
            return Err(InvalidNumberOfElementsError::new(
                plan.subset().num_elements(),
                output_target.num_elements(),
            )
            .into());
        }
        // Only the fixed path is ever planned. The plan is not what is wrong here, so
        // this does not report a plan mismatch.
        let ArrayBytesDecodeIntoTarget::Fixed(output_view) = output_target else {
            return Err(ExpectedFixedLengthBytesError.into());
        };
        let (_, _, planned) = self.checked_tasks(plan, &fetched)?;
        let subset = plan.subset();
        // Straight into the caller's view: the inner chunks already decode into subdivisions
        // of whatever view they are given, so there is nothing for an owned buffer to do.
        partial_decode_fixed_array_subset_from_bytes_into(
            &self.subchunk_shape,
            &self.inner_codecs,
            planned,
            subset,
            fetched,
            options,
            output_view,
        )
    }
}

impl ShardingPartialDecoder {
    /// The geometry a plan describes, once the plan and the bytes fetched for it have been
    /// checked against the reads this decoder would perform.
    ///
    /// Rebuilding the plan is cheap -- the shard index is resident, and the decode needs the
    /// geometry anyway -- so both decode entry points check: one entry per inner chunk, the
    /// same range for each, and bytes of the length that range asks for.
    ///
    /// What this cannot catch is a permutation of entries whose ranges are all the same
    /// length, which is the usual case for uncompressed inner chunks. Order is the caller's
    /// side of the contract.
    fn checked_tasks<'a>(
        &self,
        plan: &'a ReadPlan,
        fetched: &[MaybeBytes],
    ) -> Result<(&'a dyn ArraySubsetTraits, usize, SubchunkTasks), CodecError> {
        // A selection this decoder does not plan -- nested sharding, a variable-size
        // type -- cannot have produced this plan, so something else did.
        let Some((subset, data_type_size)) = self.planned_subset(plan.subset()) else {
            return Err(CodecError::ReadPlanMismatch);
        };
        let planned = plan_subchunk_tasks(
            &self.shard_shape,
            &self.subchunk_shape,
            self.shard_index.as_deref(),
            subset,
        )?;
        let tasks = &planned.tasks;
        if plan.source() != self.plan_source
            || fetched.len() != tasks.len()
            || plan.num_entries() != tasks.len()
            || izip!(plan.byte_ranges(), fetched, tasks).any(|(range, bytes, task)| {
                *range != task.byte_range() || bytes.as_ref().map(Bytes::len) != task.fetched_len()
            })
        {
            return Err(CodecError::ReadPlanMismatch);
        }
        Ok((subset, data_type_size, planned))
    }

    /// The array subset a read plan can be built for and the size of one of its
    /// elements, or [`None`] if this indexer takes a path that does not read one
    /// inner chunk per range.
    fn planned_subset<'a>(
        &self,
        indexer: &'a dyn Indexer,
    ) -> Option<(&'a dyn ArraySubsetTraits, usize)> {
        // A byte range is only worth reporting if the caller can issue it against the
        // stored value and get the same bytes back.
        if !self.plannable_input {
            return None;
        }
        // Only the fixed-size array subset path decodes one inner chunk per read.
        // Returning the size is what lets the decode path have it without asking
        // again and finding a case this rejected.
        let data_type = self.inner_codecs.data_type();
        let DataTypeSize::Fixed(data_type_size) = data_type.size() else {
            return None;
        };
        if data_type.is_optional() {
            return None;
        }
        let subset = indexer.as_array_subset()?;
        if subset.dimensionality() != self.shard_shape.len() {
            return None;
        }

        // A plan covers level-zero subchunks only. When a subchunk is itself
        // subchunked -- sharding nested inside sharding -- a range per level-zero
        // subchunk names a whole nested shard rather than the bytes actually
        // wanted. Report nothing rather than something misleading; the caller
        // falls back to `partial_decode`, which walks the levels itself.
        if self.nested {
            return None;
        }

        Some((subset, data_type_size))
    }
}

/// One inner chunk's contribution to a fixed-size array subset decode.
///
/// Deliberately cheap to build: the subsets it decodes into come from
/// [`subchunk_subsets`], called from the decode closures so that work stays on the
/// worker threads instead of in a serial pre-pass.
struct SubchunkTask {
    /// Where the encoded chunk lives, or [`None`] if it is absent and decodes to
    /// the fill value.
    encoded: Option<(ByteOffset, ByteLength)>,
    /// Which inner chunk this is, relative to the shard.
    chunk_indices: ArrayIndicesTinyVec,
}

impl SubchunkTask {
    /// The read this task performs, in the form a [`ReadPlan`] reports it.
    fn byte_range(&self) -> Option<ByteRange> {
        self.encoded
            .map(|(offset, size)| ByteRange::FromStart(offset, Some(size)))
    }

    /// How many bytes must have been fetched for this task, if any.
    ///
    /// Sharding only ever plans an exact range, so this is the whole check on what came
    /// back: bytes of another length are not the bytes this task asked for.
    fn fetched_len(&self) -> Option<usize> {
        // A size beyond `usize` cannot have been fetched into memory, so nothing matches.
        self.encoded
            .map(|(_, size)| usize::try_from(size).unwrap_or(usize::MAX))
    }
}

/// The inner chunks a fixed-size array subset decode touches, and the grid they are
/// indexed in.
struct SubchunkTasks {
    grid: RegularChunkGrid,
    chunks_per_shard: Vec<u64>,
    tasks: Vec<SubchunkTask>,
}

/// The inner chunks a fixed-size array subset decode touches, in order.
///
/// Every path that consumes a shard subset goes through here: planning, decoding
/// from supplied bytes, and decoding from the input handle. The plan's contract is
/// that entry `i` corresponds to fetched bytes `i`, which holds only if all three
/// agree on the order -- so the order exists once, here.
///
/// A missing shard, or a chunk absent from a present one, still gets an entry: the
/// positions are the only thing tying a plan to the bytes fetched for it.
///
/// Pure computation, no reads: the shard index is already resident. Kept to a shard
/// index lookup per chunk, because this runs on one thread ahead of the decode --
/// [`subchunk_subsets`] holds the part that is worth doing in parallel.
///
/// # Errors
/// Returns [`CodecError::IncompatibleIndexer`] if `array_subset` reaches outside the
/// shard. That check is what keeps the rest of this function panic-free, since a subset
/// past the end of the shard yields chunk indices past the end of the shard index.
fn plan_subchunk_tasks(
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
) -> Result<SubchunkTasks, CodecError> {
    // Callers reach here with a selection they chose, so this is the trust boundary. It
    // also rejects a mismatched dimensionality.
    let shard_shape_u64 = bytemuck::must_cast_slice(shard_shape);
    if !array_subset.inbounds_shape(shard_shape_u64) {
        return Err(IndexerError::new_oob(array_subset.end_exc(), shard_shape_u64.to_vec()).into());
    }

    let chunks_per_shard =
        calculate_chunks_per_shard(shard_shape, subchunk_shape)?.to_array_shape();
    let grid = RegularChunkGrid::new(shard_shape_u64.to_vec(), subchunk_shape.to_vec())
        .map_err(Into::<IncompatibleDimensionalityError>::into)?;
    let chunks = grid
        .chunks_in_array_subset(array_subset)?
        // Only `None` for a zero-sized grid, and a shard shape is `NonZeroU64`.
        .expect("subchunks always within shard");

    let tasks = chunks
        .indices()
        .into_iter()
        .map(|chunk_indices| {
            // In-bounds by the check above, so both of these hold.
            let entry = ravel_indices(&chunk_indices, &chunks_per_shard).expect("inbounds chunk");
            let index = usize::try_from(entry).expect("index fits in usize");
            let encoded = shard_index.and_then(|shard_index| {
                let offset = shard_index[index * 2];
                let size = shard_index[index * 2 + 1];
                // The De Morgan dual of the `&&` a decode path would write. The
                // polarity must stay paired or plan and decode disagree about
                // which chunks exist.
                (offset != u64::MAX || size != u64::MAX).then_some((offset, size))
            });
            SubchunkTask {
                encoded,
                chunk_indices,
            }
        })
        .collect();
    Ok(SubchunkTasks {
        grid,
        chunks_per_shard,
        tasks,
    })
}

/// Where one inner chunk's contribution comes from and goes to: the subset to decode
/// from the chunk, relative to the chunk's own start, and where those elements land,
/// relative to `array_subset_start`.
///
/// Called per chunk from the decode closures rather than hoisted into
/// [`plan_subchunk_tasks`], so its allocations happen on whichever thread is about to
/// use them.
///
/// # Errors
/// Returns [`CodecError`] if the chunk does not overlap `array_subset`, which cannot
/// happen for a chunk that function reported.
fn subchunk_subsets(
    grid: &RegularChunkGrid,
    array_subset: &dyn ArraySubsetTraits,
    array_subset_start: &[u64],
    chunk_indices: &[u64],
) -> Result<(ArraySubset, ArraySubset), CodecError> {
    let chunk_subset = grid
        .subset(chunk_indices)
        .expect("matching dimensionality")
        .expect("subchunk always within shard");
    let overlap = array_subset.overlap(&chunk_subset)?;
    Ok((
        overlap.relative_to(chunk_subset.start())?,
        overlap.relative_to(array_subset_start)?,
    ))
}

/// The supplied-bytes twin of [`partial_decode_fixed_array_subset_into`]: the same
/// tasks, but each inner chunk decodes from bytes the caller supplied rather than
/// from a byte interval of the input handle.
///
/// `fetched` must be one-to-one with `planned.tasks`; the caller has already checked that.
fn partial_decode_fixed_array_subset_from_bytes_into(
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    planned: SubchunkTasks,
    array_subset: &dyn ArraySubsetTraits,
    fetched: Vec<MaybeBytes>,
    options: &CodecOptions,
    output_view: &mut ArrayBytesFixedDisjointView<'_>,
) -> Result<(), CodecError> {
    let fill_value = inner_codecs.fill_value();
    let SubchunkTasks {
        grid,
        chunks_per_shard,
        tasks,
    } = planned;
    let (subchunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
        inner_codecs,
        subchunk_shape,
        &chunks_per_shard,
        options,
    )?;

    let array_subset_start = array_subset.start();
    let decode_subchunk = |(task, encoded): (SubchunkTask, MaybeBytes)| {
        let (decode_subset, output_subset) = subchunk_subsets(
            &grid,
            array_subset,
            &array_subset_start,
            &task.chunk_indices,
        )?;
        let output_subset = output_subset.offset(output_view.subset().start())?;
        // SAFETY: chunks represent disjoint array subsets
        let mut subchunk_view: ArrayBytesFixedDisjointView<'_> =
            unsafe { output_view.subdivide(output_subset)? };
        let Some(encoded) = encoded else {
            return subchunk_view
                .fill(fill_value.as_ne_bytes())
                .map_err(CodecError::from);
        };
        // The bytes are already here, so the inner decoder reads from memory. They go in
        // as the store returned them: `Bytes` is a handle, and cloning it into the
        // decoder does not copy the buffer.
        let inner_partial_decoder =
            inner_codecs
                .clone()
                .partial_decoder(Arc::new(encoded), subchunk_shape, &options)?;
        inner_partial_decoder.partial_decode_into(
            &decode_subset,
            ArrayBytesDecodeIntoTarget::Fixed(&mut subchunk_view),
            &options,
        )
    };

    crate::iter_concurrent_limit!(
        subchunk_concurrent_limit,
        tasks.into_iter().zip(fetched).collect::<Vec<_>>(),
        try_for_each,
        decode_subchunk
    )?;
    Ok(())
}

fn get_subchunk_partial_decoder(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    options: &CodecOptions,
    byte_offset: ByteOffset,
    byte_length: ByteLength,
) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
    inner_codecs
        .clone()
        .partial_decoder(
            Arc::new(ByteIntervalPartialDecoder::new(
                input_handle.clone(),
                byte_offset,
                byte_length,
            )),
            subchunk_shape,
            options,
        )
        .map_err(|err| {
            if let CodecError::InvalidByteRangeError(_) = err {
                CodecError::Other(
                    "The shard index references out-of-bounds bytes. The chunk may be corrupted."
                        .to_string(),
                )
            } else {
                err
            }
        })
}

#[expect(clippy::too_many_arguments)]
fn partial_decode_fixed_array_subset_into(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
    output_view: &mut ArrayBytesFixedDisjointView<'_>,
) -> Result<(), CodecError> {
    let fill_value = inner_codecs.fill_value();
    if array_subset.len() != output_view.num_elements() {
        return Err(InvalidNumberOfElementsError::new(
            array_subset.len(),
            output_view.num_elements(),
        )
        .into());
    }
    let Some(shard_index) = shard_index else {
        return output_view
            .fill(fill_value.as_ne_bytes())
            .map_err(CodecError::from);
    };
    let SubchunkTasks {
        grid,
        chunks_per_shard,
        tasks,
    } = plan_subchunk_tasks(shard_shape, subchunk_shape, Some(shard_index), array_subset)?;
    let (subchunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
        inner_codecs,
        subchunk_shape,
        &chunks_per_shard,
        options,
    )?;

    let array_subset_start = array_subset.start();
    let decode_subchunk_subset_into_slice = |task: SubchunkTask| {
        let (decode_subset, output_subset) = subchunk_subsets(
            &grid,
            array_subset,
            &array_subset_start,
            &task.chunk_indices,
        )?;
        // Calculate the chunk's position in the output view coordinate space
        let output_subset = output_subset.offset(output_view.subset().start())?;
        // SAFETY: chunks represent disjoint array subsets
        let mut subchunk_view: ArrayBytesFixedDisjointView<'_> =
            unsafe { output_view.subdivide(output_subset)? };
        let Some((offset, size)) = task.encoded else {
            return subchunk_view
                .fill(fill_value.as_ne_bytes())
                .map_err(CodecError::from);
        };
        // Partially decode the subchunk
        let inner_partial_decoder = get_subchunk_partial_decoder(
            input_handle,
            subchunk_shape,
            inner_codecs,
            &options,
            offset,
            size,
        )?;
        inner_partial_decoder.partial_decode_into(
            &decode_subset,
            ArrayBytesDecodeIntoTarget::Fixed(&mut subchunk_view),
            &options,
        )
    };

    crate::iter_concurrent_limit!(
        subchunk_concurrent_limit,
        tasks,
        try_for_each,
        decode_subchunk_subset_into_slice
    )?;
    Ok(())
}

fn partial_decode_variable_array_subset(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let data_type = inner_codecs.data_type();
    let fill_value = inner_codecs.fill_value();
    let Some(shard_index) = &shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, array_subset);
    };
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_shape, subchunk_shape)?.to_array_shape();
    let (subchunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
        inner_codecs,
        subchunk_shape,
        &chunks_per_shard,
        options,
    )?;
    let options = &options;

    let shard_chunk_grid = RegularChunkGrid::new(
        bytemuck::must_cast_slice(shard_shape).to_vec(),
        subchunk_shape.to_vec(),
    )
    .expect("matching dimensionality");

    let array_subset_start = array_subset.start();
    let decode_subchunk_subset = |chunk_indices: ArrayIndicesTinyVec| {
        let shard_index_idx =
            ravel_indices(&chunk_indices, &chunks_per_shard).expect("inbounds chunk");
        let shard_index_idx = usize::try_from(shard_index_idx).unwrap();
        let offset = shard_index[shard_index_idx * 2];
        let size = shard_index[shard_index_idx * 2 + 1];

        // Get the subset of bytes from the chunk which intersect the array
        let chunk_subset = shard_chunk_grid
            .subset(&chunk_indices)
            .expect("matching dimensionality")
            .expect("subchunk always within shard");
        let chunk_subset_overlap = array_subset.overlap(&chunk_subset)?;

        let chunk_subset_bytes = if offset == u64::MAX && size == u64::MAX {
            ArrayBytes::new_fill_value(data_type, chunk_subset_overlap.num_elements(), fill_value)?
                .into_variable()?
        } else {
            // Partially decode the subchunk
            let inner_partial_decoder = get_subchunk_partial_decoder(
                input_handle,
                subchunk_shape,
                inner_codecs,
                options,
                offset,
                size,
            )?;
            inner_partial_decoder
                .partial_decode(
                    &chunk_subset_overlap
                        .relative_to(chunk_subset.start())
                        .unwrap(),
                    options,
                )?
                .into_owned()
                .into_variable()?
        };
        Ok::<_, CodecError>((
            chunk_subset_bytes,
            chunk_subset_overlap
                .relative_to(&array_subset_start)
                .unwrap(),
        ))
    };
    // Decode the subchunk subsets
    let chunks = shard_chunk_grid
        .chunks_in_array_subset(array_subset)?
        .expect("subchunks always within shard");
    let chunk_bytes_and_subsets = crate::iter_concurrent_limit!(
        subchunk_concurrent_limit,
        chunks.indices(),
        map,
        decode_subchunk_subset
    )
    .collect::<Result<Vec<_>, _>>()?;

    // Convert into an array
    let out_array_subset = merge_chunks_vlen(chunk_bytes_and_subsets, &array_subset.shape());
    Ok(ArrayBytes::Variable(out_array_subset))
}

fn partial_decode_fixed_indexer(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    indexer: &dyn Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let data_type = inner_codecs.data_type();
    let fill_value = inner_codecs.fill_value();
    let data_type_size = data_type.fixed_size().expect("called on fixed data type");
    let Some(shard_index) = &shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, indexer);
    };
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_shape, subchunk_shape)?.to_array_shape();
    // let (subchunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
    //     &inner_codecs,
    //     &chunk_representation,
    //     &chunks_per_shard,
    //     options,
    // )?;
    let options = &options;

    let output_len = usize::try_from(indexer.len() * data_type_size as u64).unwrap();
    let mut output: Vec<u8> = Vec::with_capacity(output_len);

    #[cfg(not(target_arch = "wasm32"))]
    let subchunk_partial_decoders = moka::sync::Cache::new(chunks_per_shard.iter().product());
    #[cfg(target_arch = "wasm32")]
    let subchunk_partial_decoders = quick_cache::sync::Cache::new(
        usize::try_from(chunks_per_shard.iter().product::<u64>()).unwrap(),
    );

    for indices in indexer.iter_indices() {
        // Get intersected index
        if indices.len() != shard_shape.len() {
            return Err(IndexerError::new_incompatible_dimensionality(
                indices.len(),
                shard_shape.len(),
            )
            .into());
        }
        let chunk_index: ArrayIndices = indices
            .iter()
            .zip(subchunk_shape)
            .map(|(&i, &cs)| i / cs)
            .collect();
        let chunk_index_1d = ravel_indices(&chunk_index, &chunks_per_shard)
            .ok_or_else(|| IndexerError::new_oob(chunk_index, chunks_per_shard.clone()))?;

        // Get the partial decoder
        let shard_index_idx: usize = usize::try_from(chunk_index_1d).unwrap();
        let offset = shard_index[shard_index_idx * 2];
        let size = shard_index[shard_index_idx * 2 + 1];

        #[cfg(not(target_arch = "wasm32"))]
        let inner_partial_decoder = subchunk_partial_decoders
            .entry(chunk_index_1d)
            .or_try_insert_with(|| {
                get_subchunk_partial_decoder(
                    input_handle,
                    subchunk_shape,
                    inner_codecs,
                    options,
                    offset,
                    size,
                )
            })
            .map_err(Arc::unwrap_or_clone)?
            .into_value();
        #[cfg(target_arch = "wasm32")]
        let inner_partial_decoder =
            subchunk_partial_decoders.get_or_insert_with(&chunk_index_1d, || {
                get_subchunk_partial_decoder(
                    input_handle,
                    subchunk_shape,
                    inner_codecs,
                    options,
                    offset,
                    size,
                )
            })?;

        // Get the element index
        let indices_in_subchunk: ArrayIndices = indices
            .iter()
            .zip(subchunk_shape)
            .map(|(&i, &cs)| i - (i / cs) * cs.get())
            .collect();

        let element_bytes = inner_partial_decoder
            .partial_decode(&[indices_in_subchunk], options)?
            .into_fixed()
            .expect("fixed data");
        output.extend_from_slice(&element_bytes);
    }

    debug_assert_eq!(output.len(), output_len);

    Ok(output.into())
}

fn partial_decode_variable_indexer(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    indexer: &dyn Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let data_type = inner_codecs.data_type();
    let fill_value = inner_codecs.fill_value();
    let Some(shard_index) = &shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, indexer);
    };
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_shape, subchunk_shape)?.to_array_shape();
    // let (subchunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
    //     &inner_codecs,
    //     &chunk_representation,
    //     &chunks_per_shard,
    //     options,
    // )?;
    let options = &options;

    let offsets_len = usize::try_from(indexer.len() + 1).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    let mut offsets: Vec<usize> = Vec::with_capacity(offsets_len);
    offsets.push(0);

    #[cfg(not(target_arch = "wasm32"))]
    let subchunk_partial_decoders = moka::sync::Cache::new(chunks_per_shard.iter().product());
    #[cfg(target_arch = "wasm32")]
    let subchunk_partial_decoders = quick_cache::sync::Cache::new(
        usize::try_from(chunks_per_shard.iter().product::<u64>()).unwrap(),
    );

    for indices in indexer.iter_indices() {
        // Get intersected index
        if indices.len() != shard_shape.len() {
            return Err(IndexerError::new_incompatible_dimensionality(
                indices.len(),
                shard_shape.len(),
            )
            .into());
        }
        let chunk_index: ArrayIndices = indices
            .iter()
            .zip(subchunk_shape)
            .map(|(&i, &cs)| i / cs)
            .collect();
        let chunk_index_1d = ravel_indices(&chunk_index, &chunks_per_shard)
            .ok_or_else(|| IndexerError::new_oob(chunk_index, chunks_per_shard.clone()))?;

        // Get the partial decoder
        let shard_index_idx: usize = usize::try_from(chunk_index_1d).unwrap();
        let offset = shard_index[shard_index_idx * 2];
        let size = shard_index[shard_index_idx * 2 + 1];

        #[cfg(not(target_arch = "wasm32"))]
        let inner_partial_decoder = subchunk_partial_decoders
            .entry(chunk_index_1d)
            .or_try_insert_with(|| {
                get_subchunk_partial_decoder(
                    input_handle,
                    subchunk_shape,
                    inner_codecs,
                    options,
                    offset,
                    size,
                )
            })
            .map_err(Arc::unwrap_or_clone)?
            .into_value();
        #[cfg(target_arch = "wasm32")]
        let inner_partial_decoder =
            subchunk_partial_decoders.get_or_insert_with(&chunk_index_1d, || {
                get_subchunk_partial_decoder(
                    input_handle,
                    subchunk_shape,
                    inner_codecs,
                    options,
                    offset,
                    size,
                )
            })?;

        // Get the element index
        let indices_in_subchunk: ArrayIndices = indices
            .iter()
            .zip(subchunk_shape)
            .map(|(&i, &cs)| i - (i / cs) * cs.get())
            .collect();

        let (element_bytes, element_offsets) = inner_partial_decoder
            .partial_decode(&[indices_in_subchunk], options)?
            .into_variable()?
            .into_parts();
        debug_assert_eq!(element_offsets.len(), 2);
        bytes.extend_from_slice(&element_bytes);
        offsets.push(bytes.len());
    }

    Ok(ArrayBytes::new_vlen(
        bytes,
        ArrayBytesOffsets::new(offsets)?,
    )?)
}
