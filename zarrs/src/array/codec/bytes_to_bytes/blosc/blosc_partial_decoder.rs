use std::borrow::Cow;
use std::sync::Arc;

use super::{
    blosc_blocksize, blosc_decompress_bytes, blosc_decompress_bytes_partial, blosc_typesize,
    blosc_validate,
};
use crate::array::ArrayBytesRaw;
use crate::array::codec::bytes_to_bytes::blosc::blosc_nbytes;
#[cfg(feature = "async")]
use zarrs_codec::AsyncBytesPartialDecoderTraits;
use zarrs_codec::{BytesPartialDecoderTraits, CodecError, CodecOptions};
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::{ByteRange, ByteRangeIterator};

/// The blocks a set of regions is charged for, stopping once it exceeds `n_blocks`.
///
/// `blosc1_getitem` decompresses every block the requested range touches, whatever the
/// size of that range: a one element read pays for a whole block, and two regions
/// landing in the same block pay for it twice. So the cost of serving regions one by
/// one is the sum over regions of the blocks each spans, counting repeats.
///
/// The count stops early because a scattered selection can carry millions of regions
/// and the answer is already decided the moment it passes `n_blocks`.
fn blocks_charged(
    regions: &[ByteRange],
    nbytes: usize,
    blocksize: usize,
    n_blocks: usize,
) -> usize {
    let mut charged: usize = 0;
    for byte_range in regions {
        let start = usize::try_from(byte_range.start(nbytes as u64)).unwrap();
        let end = usize::try_from(byte_range.end(nbytes as u64)).unwrap();
        charged = charged
            .saturating_add(end.div_ceil(blocksize).saturating_sub(start / blocksize));
        if charged > n_blocks {
            break;
        }
    }
    charged
}

/// Serve `decoded_regions` from one `blosc` buffer, decoding it whole when that is cheaper.
///
/// Decoding the whole buffer costs every block exactly once; serving the regions one by
/// one costs what `blocks_charged` counts. Both are counted in blocks, so which is
/// cheaper is a comparison and not a threshold -- there is no constant to tune, and the
/// worst case for guessing wrong is bounded by one buffer's worth of blocks.
///
/// This matters where a selection is dense in the blocks it touches. A strided read is
/// the extreme: every index is its own region, they all land in the same few blocks, and
/// each one re-decompresses a block that the previous region had already decompressed.
fn blosc_partial_decode<'a>(
    encoded_value: &[u8],
    decoded_regions: ByteRangeIterator<'_>,
) -> Result<Vec<ArrayBytesRaw<'a>>, CodecError> {
    let invalid = || CodecError::from("blosc encoded value is invalid");
    let Some(destsize) = blosc_validate(encoded_value) else {
        return Err(invalid());
    };
    let (Some(nbytes), Some(typesize)) = (blosc_nbytes(encoded_value), blosc_typesize(encoded_value))
    else {
        return Err(invalid());
    };
    let regions: Vec<ByteRange> = decoded_regions.collect();

    let decode_whole = blosc_blocksize(encoded_value).is_some_and(|blocksize| {
        let n_blocks = nbytes.div_ceil(blocksize);
        blocks_charged(&regions, nbytes, blocksize, n_blocks) > n_blocks
    });

    if decode_whole {
        // One thread, matching `BloscCodec::decode`: the concurrency belongs to the
        // caller, and a nested pool here would oversubscribe it.
        let decoded = blosc_decompress_bytes(encoded_value, destsize, 1)
            .map_err(|err| CodecError::from(err.to_string()))?;
        return regions
            .iter()
            .map(|byte_range| {
                let start = usize::try_from(byte_range.start(nbytes as u64)).unwrap();
                let end = usize::try_from(byte_range.end(nbytes as u64)).unwrap();
                decoded
                    .get(start..end)
                    .map(|region| Cow::Owned(region.to_vec()))
                    .ok_or_else(|| {
                        CodecError::from(format!(
                            "zarrs-blosc-block-dedup: region {start}..{end} is out of \
                             bounds of the {} decoded bytes",
                            decoded.len()
                        ))
                    })
            })
            .collect();
    }

    regions
        .iter()
        .map(|byte_range| {
            let start = usize::try_from(byte_range.start(nbytes as u64)).unwrap();
            let end = usize::try_from(byte_range.end(nbytes as u64)).unwrap();
            blosc_decompress_bytes_partial(encoded_value, start, end - start, typesize)
                .map(Cow::Owned)
                .map_err(|err| CodecError::from(err.to_string()))
        })
        .collect()
}

/// Partial decoder for the `blosc` codec.
pub(crate) struct BloscPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
}

impl BloscPartialDecoder {
    pub(crate) fn new(input_handle: Arc<dyn BytesPartialDecoderTraits>) -> Self {
        Self { input_handle }
    }
}

impl BytesPartialDecoderTraits for BloscPartialDecoder {
    fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'_>>>, CodecError> {
        let encoded_value = self.input_handle.decode(options)?;
        let Some(encoded_value) = encoded_value else {
            return Ok(None);
        };
        blosc_partial_decode(&encoded_value, decoded_regions).map(Some)
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}

#[cfg(feature = "async")]
/// Asynchronous partial decoder for the `blosc` codec.
pub(crate) struct AsyncBloscPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
}

#[cfg(feature = "async")]
impl AsyncBloscPartialDecoder {
    pub(crate) fn new(input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>) -> Self {
        Self { input_handle }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncBytesPartialDecoderTraits for AsyncBloscPartialDecoder {
    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'a>>>, CodecError> {
        let encoded_value = self.input_handle.decode(options).await?;
        let Some(encoded_value) = encoded_value else {
            return Ok(None);
        };
        blosc_partial_decode(&encoded_value, decoded_regions).map(Some)
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::codec::bytes_to_bytes::blosc::{
        BloscCompressionLevel, BloscCompressor, BloscShuffleMode, blosc_compress_bytes,
    };

    /// A buffer whose blocks are small enough that a selection can be dense in one.
    fn compressed(elements: usize, blocksize: usize) -> (Vec<u8>, Vec<u8>) {
        let raw: Vec<u8> = (0..elements * 4)
            .map(|i| u8::try_from((i * 7 + i / 13) % 251).unwrap())
            .collect();
        let encoded = blosc_compress_bytes(
            &raw,
            BloscCompressionLevel::try_from(5u8).unwrap(),
            BloscShuffleMode::Shuffle,
            4,
            BloscCompressor::LZ4,
            blocksize,
            1,
        )
        .unwrap();
        (raw, encoded)
    }

    fn served(encoded: &[u8], regions: &[ByteRange]) -> Vec<Vec<u8>> {
        blosc_partial_decode(encoded, Box::new(regions.to_vec().into_iter()))
            .unwrap()
            .into_iter()
            .map(std::borrow::Cow::into_owned)
            .collect()
    }

    /// Both branches of the gate must return the same bytes as a plain slice of the
    /// original: a selection dense enough to trip the whole-buffer decode, and a single
    /// region that stays on the per-region path.
    #[test]
    fn either_branch_returns_the_same_bytes() {
        let (raw, encoded) = compressed(1 << 20, 4096);
        let blocksize = blosc_blocksize(&encoded).unwrap();
        let nbytes = blosc_nbytes(&encoded).unwrap();
        let n_blocks = nbytes.div_ceil(blocksize);
        assert!(n_blocks > 1, "a one block buffer cannot show repeated decompression");

        // One 4 byte region every 8 bytes across the first block: every region after the
        // first lands in a block an earlier region already paid for, which is the shape
        // the gate exists for.
        let dense: Vec<ByteRange> = (0..blocksize as u64 / 8)
            .map(|i| ByteRange::FromStart(i * 8, Some(4)))
            .collect();
        assert!(blocks_charged(&dense, nbytes, blocksize, n_blocks) > n_blocks);
        let got = served(&encoded, &dense);
        assert_eq!(got.len(), dense.len());
        for (i, region) in got.iter().enumerate() {
            assert_eq!(region[..], raw[i * 8..i * 8 + 4], "dense region {i}");
        }

        // A single region cannot beat decoding every block once, so this stays partial.
        let sparse = vec![ByteRange::FromStart(64, Some(16))];
        assert!(blocks_charged(&sparse, nbytes, blocksize, n_blocks) <= n_blocks);
        assert_eq!(served(&encoded, &sparse)[0][..], raw[64..80]);
    }

    /// The count is in blocks and counts repeats, which is the whole basis of the gate.
    #[test]
    fn repeats_in_one_block_are_charged_repeatedly() {
        let (_raw, encoded) = compressed(1 << 20, 4096);
        let blocksize = blosc_blocksize(&encoded).unwrap();
        let nbytes = blosc_nbytes(&encoded).unwrap();
        assert!(
            nbytes >= blocksize * 3,
            "this test needs at least three blocks, got {nbytes} bytes in blocks of {blocksize}"
        );

        // Ten regions inside one block are charged ten times: that is the repeated
        // decompression `blosc1_getitem` performs and the gate is meant to notice.
        let same_block: Vec<ByteRange> = (0..10)
            .map(|i| ByteRange::FromStart(i * 4, Some(4)))
            .collect();
        assert_eq!(blocks_charged(&same_block, nbytes, blocksize, usize::MAX), 10);
        assert_eq!(blocks_charged(&same_block[..1], nbytes, blocksize, usize::MAX), 1);

        // A region spanning three blocks is charged three, not one.
        let spanning = vec![ByteRange::FromStart(0, Some(blocksize as u64 * 2 + 1))];
        assert_eq!(blocks_charged(&spanning, nbytes, blocksize, usize::MAX), 3);

        // The early exit must not change the verdict, only how long it takes to reach it.
        assert!(blocks_charged(&same_block, nbytes, blocksize, 3) > 3);
    }
}
