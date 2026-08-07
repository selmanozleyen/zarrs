use std::any::Any;

use zarrs_chunk_grid::{ChunkGrid, Indexer};
use zarrs_data_type::DataType;
use zarrs_plugin::{MaybeSend, MaybeSync};
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::ByteRange;

use crate::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayBytesRaw, CodecError, CodecOptions,
    InvalidNumberOfElementsError, decode_into_array_bytes_target,
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
pub trait ArrayPartialDecoderPlanned: ArrayPartialDecoderTraits {
    /// Report the reads [`partial_decode`](ArrayPartialDecoderTraits::partial_decode) would
    /// perform, without performing them.
    ///
    /// Returns one entry per unit of encoded input, in the order
    /// [`partial_decode_from_bytes`](Self::partial_decode_from_bytes) expects them back.
    /// A [`None`] entry marks a unit with nothing to read, which decodes to the fill value.
    /// Entries are never omitted, because their positions are the only thing tying the
    /// plan to the bytes returned for it.
    ///
    /// This lets a caller holding several decoders issue all of their reads together and
    /// schedule them as a whole, rather than one decoder at a time. It is worthwhile when
    /// a read costs far more than the decode it feeds, which is the usual case for a
    /// sharded array on network or parallel storage.
    ///
    /// Planning performs no reads: it is computed from state the decoder already holds.
    /// Returns [`None`] when *this indexer* cannot be planned even though the decoder
    /// plans others -- use [`partial_decode`](ArrayPartialDecoderTraits::partial_decode)
    /// then.
    ///
    /// # Errors
    /// Returns [`CodecError`] if the indexer is invalid for this decoder.
    fn read_plan(
        &self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
    ) -> Result<Option<Vec<Option<ByteRange>>>, CodecError>;

    /// Partially decode a chunk from encoded bytes the caller already fetched.
    ///
    /// `fetched` must correspond one-to-one, and in order, with the plan returned by
    /// [`read_plan`](Self::read_plan) for the same `indexer`. The call performs no I/O.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails, an array subset is invalid, or `fetched`
    /// does not match the plan.
    fn partial_decode_from_bytes(
        &self,
        indexer: &dyn Indexer,
        fetched: Vec<Option<ArrayBytesRaw<'static>>>,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError>;
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
