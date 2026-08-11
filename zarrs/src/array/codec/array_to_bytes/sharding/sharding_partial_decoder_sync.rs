use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex, Weak};

use itertools::izip;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_chunk_grid::{ArraySubset, ChunkGridTraits};
use zarrs_codec::{
    ArrayToBytesCodecSubchunkingTraits, DataPlan, IndexPlan, PlanState, SubchunkGeometry,
};

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
    /// Where `input_handle` begins within the stored value, if it is part of it.
    ///
    /// The shard index gives offsets into `input_handle`, and a plan has to give offsets
    /// into the stored value, so this is the difference between the two. [`Some(0)`] for a
    /// decoder reading a whole stored key, and the interval's start for one reading a
    /// shard nested inside another shard.
    ///
    /// [`None`] when the handle transforms what it sits on -- a bytes-to-bytes codec
    /// *outside* the sharding codec puts a decompressor or a prefix-stripper there, and a
    /// range reported to a caller would then name the wrong bytes of the stored value, of
    /// the right length, so neither the caller nor the decode would notice. Planning is
    /// declined in that case.
    plan_base: Option<ByteOffset>,
    /// Decoded indexes of subchunks that are themselves subchunked, keyed by the
    /// subchunk's linear entry in this shard.
    ///
    /// A nested plan cannot be built without them, and they are read a stage before the
    /// data they locate, so they have to outlive that stage. Indexes rather than decoders:
    /// an index is a value, sized exactly `2 * chunks_per_subchunk` u64s, where a decoder
    /// would also hold the handle it was built on and a clone of its codec chain.
    ///
    /// Empty, and costing nothing, for a decoder whose subchunks are not subchunked or that
    /// is never asked to plan. Bounded by the subchunks per shard, and shares this
    /// decoder's snapshot semantics: like [`shard_index`](Self::shard_index) it reflects
    /// the shard as it was read, so a concurrent writer invalidates the decoder as a
    /// whole, not this cache in particular.
    subchunk_indexes: Mutex<HashMap<u64, Arc<[u64]>>>,
}

/// Whether the selection wants all of a subchunk, so that reading it whole reads nothing
/// that was not asked for.
fn wants_whole_subchunk(subchunk_subset: &ArraySubset, subchunk_shape: &[NonZeroU64]) -> bool {
    subchunk_subset.shape() == bytemuck::must_cast_slice::<_, u64>(subchunk_shape)
}

/// One stored unit a plan's reads cover, named by where it decodes to.
///
/// The unit of a plan is a read, and a read may hold several of these; each one knows
/// which disjoint subdivision of the output it decodes into (or, when absent, is filled
/// as).
#[derive(Clone, Debug)]
enum RunUnit {
    /// A level-zero subchunk, decoded whole by the inner codec chain -- which, for a
    /// subchunk that is itself a shard, is that shard's own partial decoder reading
    /// from memory.
    Subchunk(ArrayIndicesTinyVec),
    /// An innermost chunk of a subchunk the selection wants only part of. Reaching it
    /// took the subchunk's own index, so this only appears in plans refined from an
    /// index stage.
    Innermost {
        subchunk: ArrayIndicesTinyVec,
        inner: ArrayIndicesTinyVec,
    },
}

/// One stored unit and where its bytes live, before runs are built.
struct Leaf {
    offset: ByteOffset,
    length: ByteLength,
    unit: RunUnit,
}

/// One unit within a run: its bytes sit at `offset..offset + length` of the run's
/// fetched bytes.
#[derive(Debug)]
struct RunMember {
    offset: ByteOffset,
    length: ByteLength,
    unit: RunUnit,
}

/// One read: a contiguous byte range covering one or more units.
#[derive(Debug)]
struct Run {
    offset: ByteOffset,
    length: ByteLength,
    members: Vec<RunMember>,
}

impl Run {
    /// The read this run is, in the form a plan reports it.
    fn range(&self) -> ByteRange {
        ByteRange::FromStart(self.offset, Some(self.length))
    }
}

/// Merge units that are adjacent in the stored value into single reads.
///
/// **This is where the order a plan is walked in exists.** Planning reports the runs'
/// ranges, decoding consumes their members, and neither re-derives the walk -- so
/// sorting here is the whole ordering story. Strict adjacency only: a gap, however
/// small, ends the run.
fn coalesce_leaves(mut leaves: Vec<Leaf>) -> Vec<Run> {
    leaves.sort_unstable_by_key(|leaf| leaf.offset);
    let mut runs: Vec<Run> = Vec::new();
    for leaf in leaves {
        match runs.last_mut() {
            Some(run) if run.offset + run.length == leaf.offset => {
                run.members.push(RunMember {
                    offset: leaf.offset - run.offset,
                    length: leaf.length,
                    unit: leaf.unit,
                });
                run.length += leaf.length;
            }
            _ => runs.push(Run {
                offset: leaf.offset,
                length: leaf.length,
                members: vec![RunMember {
                    offset: 0,
                    length: leaf.length,
                    unit: leaf.unit,
                }],
            }),
        }
    }
    runs
}

/// What a plan's entries are, as the decoder built them: [`PlanState`] for this decoder.
///
/// Minted only here, and bound to the decoder that minted it: `minted_by` is checked by
/// identity on the way back in, so a plan assembled by hand -- even one pairing this
/// decoder with plausible ranges, or carrying another decoder's state -- is rejected
/// without rebuilding anything. The payload is still checked, because byte lengths come
/// from the caller.
struct ShardingPlanState {
    /// The decoder that built this state. [`Weak`]: the plan holds the decoder strongly
    /// already, and this only needs to answer "is it the same one".
    minted_by: Weak<ShardingPartialDecoder>,
    /// The selection the runs were computed for.
    subset: ArraySubset,
    stage: PlanStage,
}

enum PlanStage {
    /// The runs of a data plan, and the units with nothing to read.
    Data { runs: Vec<Run>, absent: Vec<RunUnit> },
    /// The runs of an index plan: each member is a subchunk whose index the read holds.
    Indexes { runs: Vec<Run> },
}

impl ShardingPlanState {
    fn runs(&self) -> &[Run] {
        match &self.stage {
            PlanStage::Data { runs, .. } | PlanStage::Indexes { runs } => runs,
        }
    }
}

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

        let plan_base = input_handle.stored_offset_base();
        Ok(Self {
            input_handle,
            shard_shape,
            subchunk_shape,
            inner_codecs,
            shard_index,
            sharding_options,
            plan_base,
            subchunk_indexes: Mutex::new(HashMap::new()),
        })
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

    fn into_planned(self: Arc<Self>) -> Option<Arc<dyn ArrayPartialDecoderPlanned>> {
        Some(self)
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

impl ShardingPartialDecoder {
    /// The geometry one level in, if the subchunks are subchunked exactly once more and
    /// nothing about them stands in the way of planning.
    ///
    /// Everything here is computed from metadata: where a subchunk's index sits within it
    /// has a fixed size at a known place, so it is known before anything is read.
    fn nested_geometry(&self) -> Option<Arc<SubchunkGeometry>> {
        let geometry = self.inner_codecs.subchunk_geometry(&self.subchunk_shape)?;
        // One extra level only. A subchunk of a subchunk that is subchunked again would need
        // its index read after this stage, and there is only one exchange.
        if geometry
            .codecs()
            .subchunk_geometry(geometry.shape())
            .is_some()
        {
            return None;
        }
        Some(geometry)
    }

    /// Where the index lives for each subchunk the selection wants only part of.
    ///
    /// Nothing is reported for the others: a subchunk wanted whole is read whole, since its
    /// extent is already in the outer index, and an absent one is not read at all. So an
    /// empty result means no stage is needed -- the reads are already nameable.
    ///
    /// The outer index says where the subchunk is; `index_within` says where inside it its
    /// index sits. A [`Suffix`](ByteRange::Suffix) counts from the subchunk's end, which is
    /// why its length is needed to make the range absolute.
    fn subchunk_index_leaves(
        &self,
        tasks: &SubchunkTasks,
        subset: &dyn ArraySubsetTraits,
        base: ByteOffset,
        index_within: ByteRange,
    ) -> Result<Vec<Leaf>, CodecError> {
        let subset_start = subset.start();
        let mut leaves = Vec::new();
        for task in &tasks.tasks {
            let Some((offset, size)) = task.encoded else {
                continue;
            };
            let (subchunk_subset, _) =
                subchunk_subsets(&tasks.grid, subset, &subset_start, &task.chunk_indices)?;
            if wants_whole_subchunk(&subchunk_subset, &self.subchunk_shape) {
                continue;
            }
            let (index_offset, index_length) = match index_within {
                ByteRange::FromStart(within, Some(len)) => (base + offset + within, len),
                ByteRange::FromStart(within, None) => (base + offset + within, size - within),
                ByteRange::Suffix(len) => (base + offset + size - len, len),
            };
            leaves.push(Leaf {
                offset: index_offset,
                length: index_length,
                unit: RunUnit::Subchunk(task.chunk_indices.clone()),
            });
        }
        Ok(leaves)
    }

    /// Whether the selection wants only part of any subchunk that is stored.
    ///
    /// The gate for planning a selection this decoder cannot reach the inside of:
    /// absent subchunks decode to fill and whole-wanted ones are read whole either
    /// way, so those plan fine -- a stored subchunk wanted in part is the only case
    /// that would force a read past what was asked.
    fn wants_any_stored_subchunk_in_part(
        &self,
        tasks: &SubchunkTasks,
        subset: &dyn ArraySubsetTraits,
    ) -> Result<bool, CodecError> {
        let subset_start = subset.start();
        for task in &tasks.tasks {
            if task.encoded.is_none() {
                continue;
            }
            let (subchunk_subset, _) =
                subchunk_subsets(&tasks.grid, subset, &subset_start, &task.chunk_indices)?;
            if !wants_whole_subchunk(&subchunk_subset, &self.subchunk_shape) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// The stored units a nested selection wants, from indexes already read, and the
    /// units with nothing to read.
    ///
    /// Every index it needs must be cached, which it is once `refine_index_plan` has run
    /// for this selection. Without them there is nothing to walk and no read can be
    /// performed here to get them, so this reports a mismatch rather than reading.
    fn nested_leaves(
        &self,
        subset: &dyn ArraySubsetTraits,
        base: ByteOffset,
    ) -> Result<(Vec<Leaf>, Vec<RunUnit>), CodecError> {
        let geometry = self.nested_geometry().ok_or(CodecError::ReadPlanMismatch)?;
        let outer = plan_subchunk_tasks(
            &self.shard_shape,
            &self.subchunk_shape,
            self.shard_index.as_deref(),
            subset,
        )?;
        let subset_start = subset.start();
        let mut leaves = Vec::new();
        let mut absent = Vec::new();
        for task in &outer.tasks {
            let (subchunk_subset, _) =
                subchunk_subsets(&outer.grid, subset, &subset_start, &task.chunk_indices)?;
            match task.encoded {
                None => absent.push(RunUnit::Subchunk(task.chunk_indices.clone())),
                // Wanted whole, so its own index would only tell us to read all of it.
                Some((offset, size))
                    if wants_whole_subchunk(&subchunk_subset, &self.subchunk_shape) =>
                {
                    leaves.push(Leaf {
                        offset: base + offset,
                        length: size,
                        unit: RunUnit::Subchunk(task.chunk_indices.clone()),
                    });
                }
                Some((offset, _)) => {
                    let entry = ravel_indices(&task.chunk_indices, &outer.chunks_per_shard)
                        .expect("inbounds chunk");
                    let index = self
                        .subchunk_indexes
                        .lock()
                        .unwrap()
                        .get(&entry)
                        .cloned()
                        .ok_or(CodecError::ReadPlanMismatch)?;
                    let inner_tasks = plan_subchunk_tasks(
                        &self.subchunk_shape,
                        geometry.shape(),
                        Some(&index),
                        &subchunk_subset,
                    )?;
                    for inner_task in &inner_tasks.tasks {
                        match inner_task.encoded {
                            None => absent.push(RunUnit::Innermost {
                                subchunk: task.chunk_indices.clone(),
                                inner: inner_task.chunk_indices.clone(),
                            }),
                            // The base composes: this subchunk starts `offset` into
                            // whatever the outer base already accounts for.
                            Some((inner_offset, inner_size)) => leaves.push(Leaf {
                                offset: base + offset + inner_offset,
                                length: inner_size,
                                unit: RunUnit::Innermost {
                                    subchunk: task.chunk_indices.clone(),
                                    inner: inner_task.chunk_indices.clone(),
                                },
                            }),
                        }
                    }
                }
            }
        }
        Ok((leaves, absent))
    }

    /// A subchunk's decoded index, decoding it only the first time it is asked for.
    ///
    /// Kept for the decoder's lifetime, because the data plan and the decode that consumes
    /// it are separate calls and both need it. Decoding is cheap; reading it again is not.
    fn cached_subchunk_index(
        &self,
        entry: u64,
        encoded: &[u8],
        options: &CodecOptions,
    ) -> Result<Arc<[u64]>, CodecError> {
        if let Some(index) = self.subchunk_indexes.lock().unwrap().get(&entry) {
            return Ok(index.clone());
        }
        // Decoded outside the lock: holding it across a decode would serialise every
        // subchunk behind whichever was decoded first.
        let index: Arc<[u64]> = self
            .inner_codecs
            .decode_subchunk_index(&self.subchunk_shape, encoded, options)?
            .ok_or(CodecError::ReadPlanMismatch)?
            .into();
        Ok(self
            .subchunk_indexes
            .lock()
            .unwrap()
            .entry(entry)
            .or_insert(index)
            .clone())
    }
}

impl ArrayPartialDecoderPlanned for ShardingPartialDecoder {
    fn read_plan(
        self: Arc<Self>,
        indexer: &dyn Indexer,
        // Whether a selection can be planned was resolved at construction.
        _options: &CodecOptions,
    ) -> Result<Option<ReadPlan>, CodecError> {
        let Some((subset, _, base)) = self.planned_subset(indexer) else {
            return Ok(None);
        };
        let planned = plan_subchunk_tasks(
            &self.shard_shape,
            &self.subchunk_shape,
            self.shard_index.as_deref(),
            subset,
        )?;
        // Where subchunks are subchunked, a subchunk wanted only in part cannot be named
        // yet: the offsets inside it are in an index that has not been read. Name those
        // indexes instead. A selection that wants whole subchunks needs none of this and
        // falls through -- the reads below are exactly those subchunks' extents.
        if self
            .inner_codecs
            .subchunk_geometry(&self.subchunk_shape)
            .is_some()
        {
            match self.nested_geometry() {
                Some(geometry) => {
                    let index_leaves = self.subchunk_index_leaves(
                        &planned,
                        subset,
                        base,
                        geometry.index_within(),
                    )?;
                    if !index_leaves.is_empty() {
                        let runs = coalesce_leaves(index_leaves);
                        let byte_ranges = runs.iter().map(Run::range).collect();
                        let subset = subset.to_array_subset();
                        let state = Arc::new(ShardingPlanState {
                            minted_by: Arc::downgrade(&self),
                            subset: subset.clone(),
                            stage: PlanStage::Indexes { runs },
                        });
                        return Ok(Some(ReadPlan::Indexes(IndexPlan::new_with_state(
                            self,
                            subset,
                            byte_ranges,
                            state,
                        ))));
                    }
                }
                // Subchunked deeper than the one exchange refining performs. A subchunk
                // wanted whole is still just its extent, so the flat plan below serves --
                // but one wanted in part cannot be reached, and naming it whole would
                // read far more than was asked with nothing for the caller to notice.
                // Decline instead; the ordinary decode path reads it minimally.
                None => {
                    if self.wants_any_stored_subchunk_in_part(&planned, subset)? {
                        return Ok(None);
                    }
                }
            }
        }
        let mut leaves = Vec::new();
        let mut absent = Vec::new();
        for task in &planned.tasks {
            match task.encoded {
                None => absent.push(RunUnit::Subchunk(task.chunk_indices.clone())),
                Some((offset, size)) => leaves.push(Leaf {
                    offset: base + offset,
                    length: size,
                    unit: RunUnit::Subchunk(task.chunk_indices.clone()),
                }),
            }
        }
        let runs = coalesce_leaves(leaves);
        let byte_ranges = runs.iter().map(Run::range).collect();
        let subset = subset.to_array_subset();
        let state = Arc::new(ShardingPlanState {
            minted_by: Arc::downgrade(&self),
            subset: subset.clone(),
            stage: PlanStage::Data { runs, absent },
        });
        Ok(Some(ReadPlan::Data(DataPlan::new_with_state(
            self,
            subset,
            byte_ranges,
            state,
        ))))
    }

    fn refine_index_plan(
        self: Arc<Self>,
        plan: IndexPlan,
        fetched: Vec<MaybeBytes>,
        options: &CodecOptions,
    ) -> Result<DataPlan, CodecError> {
        let state = self.verified_state(plan.subset(), plan.state(), plan.byte_ranges())?;
        let PlanStage::Indexes { runs } = &state.stage else {
            return Err(CodecError::ReadPlanMismatch);
        };
        check_fetched_lengths(runs, &fetched)?;
        let Some((subset, _, base)) = self.planned_subset(plan.subset()) else {
            return Err(CodecError::ReadPlanMismatch);
        };
        let chunks_per_shard =
            calculate_chunks_per_shard(&self.shard_shape, &self.subchunk_shape)?.to_array_shape();

        // Decode each subchunk's index out of the run that holds it, into the cache the
        // data walk reads from.
        for (run, bytes) in izip!(runs, &fetched) {
            let bytes = bytes.as_ref().ok_or(CodecError::ReadPlanMismatch)?;
            for member in &run.members {
                let RunUnit::Subchunk(chunk_indices) = &member.unit else {
                    return Err(CodecError::ReadPlanMismatch);
                };
                // In-range: the members subdivide the run, and `bytes` has its length.
                let start = usize::try_from(member.offset).expect("member within fetched run");
                let end = usize::try_from(member.offset + member.length)
                    .expect("member within fetched run");
                let entry =
                    ravel_indices(chunk_indices, &chunks_per_shard).expect("inbounds chunk");
                self.cached_subchunk_index(entry, &bytes[start..end], options)?;
            }
        }

        // Built here and consumed by the decode as-is: the walk exists once.
        let (leaves, absent) = self.nested_leaves(subset, base)?;
        let runs = coalesce_leaves(leaves);
        let byte_ranges = runs.iter().map(Run::range).collect();
        let subset = subset.to_array_subset();
        let state = Arc::new(ShardingPlanState {
            minted_by: Arc::downgrade(&self),
            subset: subset.clone(),
            stage: PlanStage::Data { runs, absent },
        });
        Ok(DataPlan::new_with_state(self, subset, byte_ranges, state))
    }

    fn fill_absent_into(
        &self,
        plan: &DataPlan,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        _options: &CodecOptions,
    ) -> Result<(), CodecError> {
        // Checked in the same order as the decode entry points, so the caller gets the
        // same error whichever runs first.
        if plan.subset().num_elements() != output_target.num_elements() {
            return Err(InvalidNumberOfElementsError::new(
                plan.subset().num_elements(),
                output_target.num_elements(),
            )
            .into());
        }
        let ArrayBytesDecodeIntoTarget::Fixed(output_view) = output_target else {
            return Err(ExpectedFixedLengthBytesError.into());
        };
        let state = self.verified_state(plan.subset(), plan.state(), plan.byte_ranges())?;
        let PlanStage::Data { absent, .. } = &state.stage else {
            return Err(CodecError::ReadPlanMismatch);
        };
        self.fill_absent_units(absent, plan.subset(), output_view)
    }

    fn partial_decode_from_bytes(
        &self,
        plan: &DataPlan,
        fetched: Vec<MaybeBytes>,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        let state = self.verified_state(plan.subset(), plan.state(), plan.byte_ranges())?;
        let PlanStage::Data { runs, absent } = &state.stage else {
            return Err(CodecError::ReadPlanMismatch);
        };
        check_fetched_lengths(runs, &fetched)?;
        let (_, data_type_size, _) = self
            .planned_subset(plan.subset())
            .ok_or(CodecError::ReadPlanMismatch)?;

        let subset = plan.subset();
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
        // The buffer is this call's own, so nothing else will fill the absent units.
        self.fill_absent_units(absent, subset, &mut output_view)?;
        self.decode_runs_into(runs, subset, fetched, options, &mut output_view)?;
        Ok(ArrayBytes::from(out))
    }

    fn partial_decode_from_bytes_into(
        &self,
        plan: &DataPlan,
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
        let state = self.verified_state(plan.subset(), plan.state(), plan.byte_ranges())?;
        let PlanStage::Data { runs, .. } = &state.stage else {
            return Err(CodecError::ReadPlanMismatch);
        };
        check_fetched_lengths(runs, &fetched)?;
        // Straight into the caller's view, and only what was read: the absent units are
        // `fill_absent_into`'s, called once by the caller.
        self.decode_runs_into(runs, plan.subset(), fetched, options, output_view)
    }
}

/// Whether each run's bytes are exactly as long as the run asked for.
///
/// Sharding only ever plans exact ranges, so this is the whole check on what came back:
/// bytes of another length are not the bytes the run asked for, and every entry is a
/// read, so [`None`] never matches.
fn check_fetched_lengths(runs: &[Run], fetched: &[MaybeBytes]) -> Result<(), CodecError> {
    if fetched.len() != runs.len()
        || izip!(runs, fetched).any(|(run, bytes)| {
            bytes.as_ref().map(Bytes::len)
                != Some(usize::try_from(run.length).unwrap_or(usize::MAX))
        })
    {
        return Err(CodecError::ReadPlanMismatch);
    }
    Ok(())
}

impl ShardingPartialDecoder {
    /// The plan's state, if it is state this decoder minted and the plan is unaltered.
    ///
    /// The seam validation splits along: the *structure* -- which units each read holds,
    /// and where each decodes to -- is trusted because it is provably this decoder's own
    /// work, checked by identity rather than by rebuilding the walk. The *payload* is
    /// still checked, because byte lengths come from the caller. The identity check is
    /// what makes a hand-built plan impossible: state is minted only by planning, a plan
    /// built through the public constructors carries none, and state lifted from another
    /// decoder's plan upgrades to that decoder, not this one.
    ///
    /// The ranges are compared too -- cheap, both sides resident -- so genuine state
    /// cannot be re-paired with edited public ranges, which are what the caller fetched.
    ///
    /// What none of this can catch is a permutation of fetched bytes whose runs are all
    /// the same length. Order is the caller's side of the contract.
    fn verified_state<'a>(
        &self,
        subset: &ArraySubset,
        state: Option<&'a dyn PlanState>,
        byte_ranges: &[ByteRange],
    ) -> Result<&'a ShardingPlanState, CodecError> {
        let state = state
            .and_then(|state| state.as_any().downcast_ref::<ShardingPlanState>())
            .ok_or(CodecError::ReadPlanMismatch)?;
        let minted_here = state
            .minted_by
            .upgrade()
            // The plan itself keeps its decoder alive, so if this state was minted by
            // the decoder now consuming it, the upgrade cannot fail.
            .is_some_and(|minter| std::ptr::eq(Arc::as_ptr(&minter), self));
        let runs = state.runs();
        if !minted_here
            || state.subset != *subset
            || byte_ranges.len() != runs.len()
            || izip!(byte_ranges, runs).any(|(range, run)| *range != run.range())
        {
            return Err(CodecError::ReadPlanMismatch);
        }
        Ok(state)
    }

    /// Fill the output subdivisions of units with nothing to read.
    fn fill_absent_units(
        &self,
        absent: &[RunUnit],
        subset: &ArraySubset,
        output_view: &mut ArrayBytesFixedDisjointView<'_>,
    ) -> Result<(), CodecError> {
        let fill_value = self.inner_codecs.fill_value();
        for unit in absent {
            let output_subset = self.unit_output_subset(unit, subset)?.1;
            let output_subset = output_subset.offset(output_view.subset().start())?;
            // SAFETY: units represent disjoint array subsets
            let mut unit_view = unsafe { output_view.subdivide(output_subset)? };
            unit_view.fill(fill_value.as_ne_bytes())?;
        }
        Ok(())
    }

    /// Where one unit decodes from and to: the subset to decode from the unit, in the
    /// unit's own coordinates, and where those elements land, relative to `subset`'s
    /// start.
    fn unit_output_subset(
        &self,
        unit: &RunUnit,
        subset: &ArraySubset,
    ) -> Result<(ArraySubset, ArraySubset), CodecError> {
        let subset_start = subset.start();
        let outer_grid = RegularChunkGrid::new(
            bytemuck::must_cast_slice(&self.shard_shape).to_vec(),
            self.subchunk_shape.to_vec(),
        )
        .map_err(Into::<IncompatibleDimensionalityError>::into)?;
        match unit {
            RunUnit::Subchunk(chunk_indices) => {
                subchunk_subsets(&outer_grid, subset, &subset_start, chunk_indices)
            }
            RunUnit::Innermost { subchunk, inner } => {
                let geometry = self.nested_geometry().ok_or(CodecError::ReadPlanMismatch)?;
                let (subchunk_subset, subchunk_output) =
                    subchunk_subsets(&outer_grid, subset, &subset_start, subchunk)?;
                let inner_grid = RegularChunkGrid::new(
                    bytemuck::must_cast_slice(&self.subchunk_shape).to_vec(),
                    geometry.shape().to_vec(),
                )
                .map_err(Into::<IncompatibleDimensionalityError>::into)?;
                let (decode_subset, inner_output) = subchunk_subsets(
                    &inner_grid,
                    &subchunk_subset,
                    &subchunk_subset.start(),
                    inner,
                )?;
                Ok((
                    decode_subset,
                    inner_output.offset(subchunk_output.start())?,
                ))
            }
        }
    }

    /// Decode fetched runs into a view of the output, each member from its slice of the
    /// run's bytes.
    ///
    /// `fetched` has already been checked one-to-one against `runs`. Members decode
    /// concurrently across runs -- they are disjoint subdivisions of the output, exactly
    /// as the per-chunk path's chunks are.
    fn decode_runs_into(
        &self,
        runs: &[Run],
        subset: &ArraySubset,
        fetched: Vec<MaybeBytes>,
        options: &CodecOptions,
        output_view: &mut ArrayBytesFixedDisjointView<'_>,
    ) -> Result<(), CodecError> {
        let chunks_per_shard =
            calculate_chunks_per_shard(&self.shard_shape, &self.subchunk_shape)?.to_array_shape();
        let (concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
            &self.inner_codecs,
            &self.subchunk_shape,
            &chunks_per_shard,
            options,
        )?;

        // A member decodes from its slice of the run's bytes: `Bytes` is a handle, so
        // slicing shares the fetched buffer rather than copying it.
        let members: Vec<(&RunMember, Bytes)> = izip!(runs, fetched)
            .flat_map(|(run, bytes)| {
                let bytes = bytes.expect("checked against the run");
                run.members.iter().map(move |member| {
                    // In-range: the members subdivide the run, and `bytes` has its length.
                    let start =
                        usize::try_from(member.offset).expect("member within fetched run");
                    let end = usize::try_from(member.offset + member.length)
                        .expect("member within fetched run");
                    (member, bytes.slice(start..end))
                })
            })
            .collect();

        let geometry = self.nested_geometry();
        let decode_member = |(member, encoded): (&RunMember, Bytes)| {
            let (decode_subset, output_subset) = self.unit_output_subset(&member.unit, subset)?;
            let output_subset = output_subset.offset(output_view.subset().start())?;
            // SAFETY: units represent disjoint array subsets
            let mut unit_view = unsafe { output_view.subdivide(output_subset)? };
            let unit_decoder = match &member.unit {
                RunUnit::Subchunk(_) => self.inner_codecs.clone().partial_decoder(
                    Arc::new(encoded),
                    &self.subchunk_shape,
                    &options,
                )?,
                RunUnit::Innermost { .. } => {
                    let geometry = geometry.as_ref().ok_or(CodecError::ReadPlanMismatch)?;
                    geometry.codecs().clone().partial_decoder(
                        Arc::new(encoded),
                        geometry.shape(),
                        &options,
                    )?
                }
            };
            unit_decoder.partial_decode_into(
                &decode_subset,
                ArrayBytesDecodeIntoTarget::Fixed(&mut unit_view),
                &options,
            )
        };

        crate::iter_concurrent_limit!(concurrent_limit, members, try_for_each, decode_member)?;
        Ok(())
    }

    /// The array subset a read plan can be built for and the size of one of its
    /// elements, or [`None`] if this indexer takes a path that does not read one
    /// inner chunk per range.
    fn planned_subset<'a>(
        &self,
        indexer: &'a dyn Indexer,
    ) -> Option<(&'a dyn ArraySubsetTraits, usize, ByteOffset)> {
        // A byte range is only worth reporting if the caller can issue it against the
        // stored value and get the same bytes back, which is what having a base means.
        let base = self.plan_base?;
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

        Some((subset, data_type_size, base))
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
