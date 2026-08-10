use std::any::Any;

use zarrs_chunk_grid::{ChunkGrid, Indexer};
use zarrs_data_type::DataType;
use zarrs_plugin::{MaybeSend, MaybeSync};
use zarrs_storage::{MaybeBytes, StorageError};

use crate::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, CodecError, CodecOptions, InvalidNumberOfElementsError,
    ReadPlan, decode_into_array_bytes_target,
};

/// Partial array decoder traits.
pub trait ArrayPartialDecoderTraits: Any + MaybeSend + MaybeSync {
    /// Return the data type of the partial decoder.
    fn data_type(&self) -> &DataType;

    /// Returns whether the chunk exists.
    ///
    /// # Errors
    /// Returns [`StorageError`] if a storage operation fails.
    fn exists(&self) -> Result<bool, StorageError>;

    /// Returns the size of chunk bytes held by the partial decoder.
    ///
    /// Intended for use by size-constrained partial decoder caches.
    fn size_held(&self) -> usize;

    /// Return the chunk-local subchunk grid hierarchy for this decoder.
    ///
    /// Grids are ordered from outermost to innermost and are relative to the decoded
    /// chunk handled by this partial decoder, not to the full array. A `None` entry
    /// preserves a level that cannot be resolved in this decoder's local context.
    ///
    /// # Errors
    /// Returns [`CodecError`] if the local grid cannot be resolved.
    fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError>;

    /// Return the outermost chunk-local subchunk grid for this decoder, if available.
    ///
    /// This is a compatibility wrapper around [`local_subchunk_grids`](Self::local_subchunk_grids).
    ///
    /// # Errors
    /// Returns [`CodecError`] if the local grid hierarchy cannot be resolved.
    fn local_subchunk_grid(&self, options: &CodecOptions) -> Result<Option<ChunkGrid>, CodecError> {
        Ok(self
            .local_subchunk_grids(options)?
            .into_iter()
            .next()
            .flatten())
    }

    /// Partially decode a chunk.
    ///
    /// If the inner `input_handle` is a bytes decoder and partial decoding returns [`None`], then the array subsets have the fill value.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or an array subset is invalid.
    fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError>;

    /// Partially decode into a preallocated output.
    ///
    /// This method is intended for internal use by Array.
    /// It currently only works for fixed length data types.
    ///
    /// The `indexer` shape and dimensionality does not need to match `output_subset`, but the number of elements must match.
    /// Extracted elements from the `indexer` are written as ordered by the indexer.
    /// For an [`ArraySubset`](zarrs_chunk_grid::ArraySubset), that is C order.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the number of elements in `indexer` does not match the number of elements in `output_view`,
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

        let decoded_value = self.partial_decode(indexer, options)?;
        decode_into_array_bytes_target(&decoded_value, output_target)
    }

    /// Return this decoder as one that can describe its reads, if it can.
    ///
    /// Most decoders cannot: only a format that already holds a map from array regions to
    /// byte ranges, such as a shard index, can say where bytes live without reading them.
    /// The default answers [`None`], so a decoder opts in by implementing
    /// [`ArrayPartialDecoderPlanned`] and overriding this.
    ///
    /// Answering [`Some`] means only that this decoder plans *some* indexers -- whether it
    /// can plan a particular one is [`read_plan`](ArrayPartialDecoderPlanned::read_plan)'s
    /// answer, since that depends on the selection.
    fn as_planned(&self) -> Option<&dyn ArrayPartialDecoderPlanned> {
        None
    }

    /// Returns whether this decoder supports partial decoding.
    ///
    /// If this returns `true`, the decoder can efficiently handle partial decoding operations.
    /// If this returns `false`, partial decoding will fall back to a full decode operation.
    fn supports_partial_decode(&self) -> bool;
}

/// A partial decoder that can describe its reads before performing them.
///
/// Separate from [`ArrayPartialDecoderTraits`] because almost nothing can do this, and a
/// capability every implementor declines is not a capability the base trait should claim.
/// Keeping the pair here also means they cannot be half-implemented: a decoder that hands
/// out byte ranges must be able to decode the bytes that come back.
///
/// Reach it through [`as_planned`](ArrayPartialDecoderTraits::as_planned).
///
/// Deliberately not a subtrait of [`ArrayPartialDecoderTraits`]: neither method performs
/// I/O, so nothing here is sync-specific, and an async decoder can implement this same
/// trait rather than needing a duplicate of it.
pub trait ArrayPartialDecoderPlanned: Any + MaybeSend + MaybeSync {
    /// Report the reads [`partial_decode`](ArrayPartialDecoderTraits::partial_decode) would
    /// perform, without performing them.
    ///
    /// This lets a caller holding several decoders issue all of their reads together and
    /// schedule them as a whole, rather than one decoder at a time. It is worthwhile when
    /// a read costs far more than the decode it feeds, which is the usual case for a
    /// sharded array on network or parallel storage.
    ///
    /// Planning performs no reads: it is computed from state the decoder already holds.
    /// The plan covers the decoder's level-zero subchunks, one entry each; subchunks
    /// nested inside subchunks are not planned.
    ///
    /// Returns [`None`] when *this indexer* cannot be planned even though the decoder
    /// plans others -- use [`partial_decode`](ArrayPartialDecoderTraits::partial_decode)
    /// then. A decoder should also decline whenever anything sits between it and the
    /// stored bytes, such as a bytes-to-bytes codec applied outside it: the byte ranges
    /// it holds are then offsets into a decoded stream rather than into the stored value,
    /// and a caller issuing them against the store would read the wrong bytes at the
    /// right lengths, with nothing to notice it by.
    ///
    /// # Errors
    /// Returns [`CodecError`] if the indexer is invalid for this decoder.
    fn read_plan(
        &self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
    ) -> Result<Option<ReadPlan>, CodecError>;

    /// Partially decode a chunk from encoded bytes the caller already fetched.
    ///
    /// `fetched` must correspond one-to-one, and in order, with `plan`. The selection comes
    /// from the plan, so there is no second selection to keep matched to it. The call
    /// performs no I/O.
    ///
    /// Entries are [`MaybeBytes`] -- exactly what a store hands back -- so the caller does
    /// not have to convert or copy them to hand them over. A [`None`] entry is one the plan
    /// said there was nothing to read for.
    ///
    /// The implementation checks `plan` and `fetched` against the reads it would have
    /// performed itself: the number of entries, the range of each, and the length of the
    /// bytes supplied for it. It cannot check the *order*, since entries of equal length
    /// are indistinguishable, so handing back bytes in plan order is the caller's side of
    /// the contract.
    ///
    /// # Errors
    /// Returns [`CodecError::ReadPlanMismatch`] if `plan` is not one this decoder would
    /// produce or `fetched` does not match it, [`CodecError::IncompatibleIndexer`] if the
    /// plan's selection is invalid for this decoder, or [`CodecError`] if a codec fails.
    fn partial_decode_from_bytes(
        &self,
        plan: &ReadPlan,
        fetched: Vec<MaybeBytes>,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError>;

    /// [`partial_decode_from_bytes`](Self::partial_decode_from_bytes) into a preallocated
    /// output.
    ///
    /// A caller assembling one output from several decoders wants this rather than the
    /// owned form, which has to allocate a buffer per call and copy it into place. The
    /// default does exactly that; an implementor that can decode straight into
    /// `output_target` should override it.
    ///
    /// # Errors
    /// Returns [`InvalidNumberOfElementsError`] if the plan's selection and
    /// `output_target` hold different numbers of elements,
    /// [`CodecError::ExpectedFixedLengthBytes`] if `output_target` is a kind this decoder
    /// does not plan, [`CodecError::ReadPlanMismatch`] if `plan` is not one this decoder
    /// would produce or `fetched` does not match it, or [`CodecError`] if a codec fails.
    fn partial_decode_from_bytes_into(
        &self,
        plan: &ReadPlan,
        fetched: Vec<MaybeBytes>,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        if plan.subset().num_elements() != output_target.num_elements() {
            return Err(InvalidNumberOfElementsError::new(
                plan.subset().num_elements(),
                output_target.num_elements(),
            )
            .into());
        }
        let decoded = self.partial_decode_from_bytes(plan, fetched, options)?;
        decode_into_array_bytes_target(&decoded, output_target)
    }
}

/// Partial array encoder traits.
pub trait ArrayPartialEncoderTraits:
    ArrayPartialDecoderTraits + Any + MaybeSend + MaybeSync
{
    /// Erase the chunk.
    ///
    /// # Errors
    /// Returns an error if there is an underlying store error.
    fn erase(&self) -> Result<(), CodecError>;

    /// Partially encode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or an array subset is invalid.
    fn partial_encode(
        &self,
        indexer: &dyn Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError>;

    /// Returns whether this encoder supports partial encoding.
    ///
    /// If this returns `true`, the encoder can efficiently handle partial encoding operations.
    /// If this returns `false`, partial encoding will fall back to a full decode and encode operation.
    fn supports_partial_encode(&self) -> bool;
}
