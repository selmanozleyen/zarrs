use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

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
    ArrayBytesDecodeIntoTarget, ArrayCodecTraits, ArrayPartialDecoderTraits,
    ArrayToBytesCodecTraits, ByteIntervalPartialDecoder, BytesPartialDecoderTraits, CodecError,
    CodecOptions, InvalidNumberOfElementsError, decode_into_array_bytes_target,
};
use zarrs_plugin::ExtensionAliasesV3;
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::{ByteLength, ByteOffset, ByteRange};

/// Partial decoder for the sharding codec.
pub struct ShardingPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    shard_shape: ChunkShape,
    subchunk_shape: ChunkShape,
    inner_codecs: Arc<CodecChainBound>,
    shard_index: Option<Vec<u64>>,
    #[expect(dead_code)] // TODO: Remove when sharding-specific options are added
    sharding_options: ShardingCodecOptions,
    /// Inner-chunk decoders kept across accesses, keyed by shard index entry.
    ///
    /// Building one of these decodes that chunk's own index when the inner
    /// chunk is itself a shard — a read plus a decode. Without keeping them,
    /// that repeats every time the same inner shard is touched, so a scattered
    /// read over one shard pays it per access rather than per inner shard.
    ///
    /// [`None`] when inner chunks are not themselves shards: construction is
    /// then cheap and a shared map would only add contention.
    subchunk_decoders: Option<Mutex<HashMap<u64, Arc<dyn ArrayPartialDecoderTraits>>>>,
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

        let mut decoder = Self {
            input_handle,
            shard_shape,
            subchunk_shape,
            inner_codecs,
            shard_index,
            sharding_options,
            subchunk_decoders: None,
        };

        // Only worth keeping decoders when building one is expensive, which is
        // when the inner chunk is itself a shard carrying its own index. More
        // than one level in the local grid hierarchy is exactly that.
        if decoder.local_subchunk_grids(options)?.len() > 1 {
            decoder.subchunk_decoders = Some(Mutex::new(HashMap::new()));
        }
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

#[expect(clippy::too_many_arguments)]
pub(crate) fn partial_decode(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    subchunk_decoders: Option<&SubchunkDecoderCache>,
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
                    subchunk_decoders,
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
            self.subchunk_decoders.as_ref(),
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
                self.subchunk_decoders.as_ref(),
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

    fn read_plan(
        &self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
    ) -> Result<Option<Vec<Option<ByteRange>>>, CodecError> {
        let Some(subset) = self.planned_subset(indexer, options)? else {
            return Ok(None);
        };
        plan_fixed_array_subset(
            &self.shard_shape,
            &self.subchunk_shape,
            self.shard_index.as_deref(),
            subset,
        )
        .map(Some)
    }

    fn partial_decode_prefetched(
        &self,
        indexer: &dyn Indexer,
        fetched: Vec<Option<ArrayBytesRaw<'static>>>,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        let Some(subset) = self.planned_subset(indexer, options)? else {
            // No plan was offered for this indexer, so `fetched` cannot belong
            // to one. Decode normally.
            return self.partial_decode(indexer, options);
        };
        let data_type_size = match self.inner_codecs.data_type().size() {
            DataTypeSize::Fixed(size) => size,
            DataTypeSize::Variable => unreachable!("planned_subset rejects variable sizes"),
        };

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
        partial_decode_fixed_array_subset_prefetched_into(
            &self.shard_shape,
            &self.subchunk_shape,
            &self.inner_codecs,
            subset,
            fetched,
            options,
            &mut output_view,
        )?;
        Ok(ArrayBytes::from(out))
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

impl ShardingPartialDecoder {
    /// The array subset a read plan can be built for, or [`None`] if this
    /// indexer takes a path that does not read one inner chunk per range.
    fn planned_subset<'a>(
        &self,
        indexer: &'a dyn Indexer,
        options: &CodecOptions,
    ) -> Result<Option<&'a dyn ArraySubsetTraits>, CodecError> {
        // Only the fixed-size array subset path decodes one inner chunk per read.
        let data_type = self.inner_codecs.data_type();
        if data_type.is_optional() || matches!(data_type.size(), DataTypeSize::Variable) {
            return Ok(None);
        }
        let Some(subset) = indexer.as_array_subset() else {
            return Ok(None);
        };
        if subset.dimensionality() != self.shard_shape.len() {
            return Ok(None);
        }

        // A plan describes one level of reads. When an inner chunk is itself
        // subchunked -- sharding nested inside sharding -- a range per inner
        // chunk names whole inner shards rather than the bytes actually
        // wanted. Report nothing rather than something misleading; the caller
        // falls back to `partial_decode`, which walks the levels itself.
        if self.local_subchunk_grids(options)?.len() > 1 {
            return Ok(None);
        }

        Ok(Some(subset))
    }
}

/// The reads a fixed-size array subset decode performs, in subchunk iteration
/// order. Pure computation: the shard index is already resident.
fn plan_fixed_array_subset(
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    shard_index: Option<&[u64]>,
    array_subset: &dyn ArraySubsetTraits,
) -> Result<Vec<Option<ByteRange>>, CodecError> {
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_shape, subchunk_shape)?.to_array_shape();
    let shard_chunk_grid = RegularChunkGrid::new(
        bytemuck::must_cast_slice(shard_shape).to_vec(),
        subchunk_shape.to_vec(),
    )
    .map_err(Into::<IncompatibleDimensionalityError>::into)?;
    let chunks = shard_chunk_grid
        .chunks_in_array_subset(array_subset)?
        .expect("subchunks always within shard");
    // A missing shard reads nothing at all, but still reports one entry per
    // inner chunk so the plan stays one-to-one with the prefetched bytes.
    Ok(chunks
        .indices()
        .into_iter()
        .map(|chunk_indices| {
            shard_index.and_then(|shard_index| {
                subchunk_encoded_range(shard_index, &chunks_per_shard, &chunk_indices)
            })
        })
        .collect())
}

/// The byte range of one encoded inner chunk, or [`None`] if it is absent and
/// decodes to the fill value.
fn subchunk_encoded_range(
    shard_index: &[u64],
    chunks_per_shard: &[u64],
    chunk_indices: &[u64],
) -> Option<ByteRange> {
    let shard_index_idx =
        usize::try_from(ravel_indices(chunk_indices, chunks_per_shard).expect("inbounds chunk"))
            .expect("index fits in usize");
    let offset = shard_index[shard_index_idx * 2];
    let size = shard_index[shard_index_idx * 2 + 1];
    (offset != u64::MAX || size != u64::MAX).then_some(ByteRange::FromStart(offset, Some(size)))
}

/// The prefetched twin of [`partial_decode_fixed_array_subset_into`]: identical
/// geometry, but each inner chunk decodes from bytes the caller supplied rather
/// than from a byte interval of the input handle.
fn partial_decode_fixed_array_subset_prefetched_into(
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    array_subset: &dyn ArraySubsetTraits,
    fetched: Vec<Option<ArrayBytesRaw<'static>>>,
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
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_shape, subchunk_shape)?.to_array_shape();
    let (subchunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
        inner_codecs,
        subchunk_shape,
        &chunks_per_shard,
        options,
    )?;
    let shard_chunk_grid = RegularChunkGrid::new(
        bytemuck::must_cast_slice(shard_shape).to_vec(),
        subchunk_shape.to_vec(),
    )
    .map_err(Into::<IncompatibleDimensionalityError>::into)?;

    let chunks = shard_chunk_grid
        .chunks_in_array_subset(array_subset)?
        .expect("subchunks always within shard");
    let chunk_indices = chunks.indices().into_iter().collect::<Vec<_>>();
    if fetched.len() != chunk_indices.len() {
        return Err(CodecError::Other(format!(
            "fetched bytes ({}) do not match the read plan ({})",
            fetched.len(),
            chunk_indices.len()
        )));
    }

    let array_subset_start = array_subset.start();
    let decode_subchunk =
        |(chunk_indices, encoded): (ArrayIndicesTinyVec, Option<ArrayBytesRaw>)| {
            let chunk_subset = shard_chunk_grid
                .subset(&chunk_indices)
                .expect("matching dimensionality")
                .expect("subchunk always within shard");
            let chunk_subset_overlap = array_subset.overlap(&chunk_subset)?;
            let chunk_relative = chunk_subset_overlap.relative_to(&array_subset_start)?;
            let chunk_output_overlap_subset =
                chunk_relative.offset(output_view.subset().start())?;
            // SAFETY: chunks represent disjoint array subsets
            let mut subchunk_view: ArrayBytesFixedDisjointView<'_> =
                unsafe { output_view.subdivide(chunk_output_overlap_subset)? };
            let Some(encoded) = encoded else {
                return subchunk_view
                    .fill(fill_value.as_ne_bytes())
                    .map_err(CodecError::from);
            };
            // The bytes are already here, so the inner decoder reads from memory.
            let inner_partial_decoder = inner_codecs.clone().partial_decoder(
                Arc::new(encoded.into_owned()),
                subchunk_shape,
                &options,
            )?;
            inner_partial_decoder.partial_decode_into(
                &chunk_subset_overlap
                    .relative_to(chunk_subset.start())
                    .unwrap(),
                ArrayBytesDecodeIntoTarget::Fixed(&mut subchunk_view),
                &options,
            )
        };

    crate::iter_concurrent_limit!(
        subchunk_concurrent_limit,
        chunk_indices.into_iter().zip(fetched).collect::<Vec<_>>(),
        try_for_each,
        decode_subchunk
    )?;
    Ok(())
}

/// Inner-chunk decoders kept across accesses, keyed by shard index entry.
type SubchunkDecoderCache = Mutex<HashMap<u64, Arc<dyn ArrayPartialDecoderTraits>>>;

/// [`get_subchunk_partial_decoder`], reusing an already-built decoder for the
/// same inner chunk when the caller is keeping them.
#[expect(clippy::too_many_arguments)]
fn cached_subchunk_partial_decoder(
    cache: Option<&SubchunkDecoderCache>,
    entry: u64,
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    options: &CodecOptions,
    byte_offset: ByteOffset,
    byte_length: ByteLength,
) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
    let Some(cache) = cache else {
        return get_subchunk_partial_decoder(
            input_handle,
            subchunk_shape,
            inner_codecs,
            options,
            byte_offset,
            byte_length,
        );
    };
    if let Some(decoder) = cache.lock().unwrap().get(&entry) {
        return Ok(decoder.clone());
    }
    // Built outside the lock: this reads and decodes the inner index, and
    // holding the lock across it would serialise every inner shard behind
    // whichever one happens to be read first.
    let decoder = get_subchunk_partial_decoder(
        input_handle,
        subchunk_shape,
        inner_codecs,
        options,
        byte_offset,
        byte_length,
    )?;
    Ok(cache
        .lock()
        .unwrap()
        .entry(entry)
        .or_insert(decoder)
        .clone())
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
    subchunk_decoders: Option<&SubchunkDecoderCache>,
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
    let chunks_per_shard =
        calculate_chunks_per_shard(shard_shape, subchunk_shape)?.to_array_shape();
    let (subchunk_concurrent_limit, options) = super::get_concurrent_target_and_codec_options(
        inner_codecs,
        subchunk_shape,
        &chunks_per_shard,
        options,
    )?;
    let shard_chunk_grid = RegularChunkGrid::new(
        bytemuck::must_cast_slice(shard_shape).to_vec(),
        subchunk_shape.to_vec(),
    )
    .map_err(Into::<IncompatibleDimensionalityError>::into)?;

    let array_subset_start = array_subset.start();
    let decode_subchunk_subset_into_slice = |chunk_indices: ArrayIndicesTinyVec| {
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
        // Calculate the chunk's position in the output view coordinate space
        let chunk_relative = chunk_subset_overlap.relative_to(&array_subset_start)?;
        let chunk_output_overlap_subset = chunk_relative.offset(output_view.subset().start())?;
        // SAFETY: chunks represent disjoint array subsets
        let mut subchunk_view: ArrayBytesFixedDisjointView<'_> =
            unsafe { output_view.subdivide(chunk_output_overlap_subset)? };
        if offset == u64::MAX && size == u64::MAX {
            subchunk_view
                .fill(fill_value.as_ne_bytes())
                .map_err(CodecError::from)
        } else {
            // Partially decode the subchunk
            let inner_partial_decoder = cached_subchunk_partial_decoder(
                subchunk_decoders,
                shard_index_idx as u64,
                input_handle,
                subchunk_shape,
                inner_codecs,
                &options,
                offset,
                size,
            )?;
            inner_partial_decoder.partial_decode_into(
                &chunk_subset_overlap
                    .relative_to(chunk_subset.start())
                    .unwrap(),
                ArrayBytesDecodeIntoTarget::Fixed(&mut subchunk_view),
                &options,
            )
        }
    };

    let chunks = shard_chunk_grid
        .chunks_in_array_subset(array_subset)?
        .expect("subchunks always within shard");
    crate::iter_concurrent_limit!(
        subchunk_concurrent_limit,
        chunks.indices(),
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
