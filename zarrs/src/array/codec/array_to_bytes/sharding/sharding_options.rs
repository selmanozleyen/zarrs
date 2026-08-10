//! Runtime options for the sharding codec.

/// Write order for subchunks within a shard
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum SubchunkWriteOrder {
    /// An alias for `Unordered`. Soft deprecated.
    ///
    /// `Random` is a misnomer and this variant will be removed in a future release.
    // TODO: Remove in 0.24
    Random,
    /// C order i.e., row-major
    C,
    /// No order guarantee.
    ///
    /// Because subchunk writing is parallelized, it will often appear that subchunks are written at random with this setting although this is dependent on the parallelizable workload.
    /// For example in the degenerate case of one thread, you may observe (mostly) ordered chunks.
    Unordered,
    // TODO: Morton order - depend on https://docs.rs/morton-encoding/latest/morton_encoding/?
}

/// Runtime options for the [`ShardingCodec`](super::ShardingCodec).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ShardingCodecOptions {
    subchunk_write_order: SubchunkWriteOrder,
    subchunk_decoder_cache: bool,
}

impl Default for ShardingCodecOptions {
    fn default() -> Self {
        Self {
            subchunk_write_order: SubchunkWriteOrder::Unordered,
            subchunk_decoder_cache: false,
        }
    }
}

impl ShardingCodecOptions {
    /// Set the subchunk ordering.
    #[must_use]
    pub fn with_subchunk_write_order(mut self, subchunk_write_order: SubchunkWriteOrder) -> Self {
        self.subchunk_write_order = subchunk_write_order;
        self
    }

    /// Set the subchunk ordering.
    pub fn set_subchunk_write_order(
        &mut self,
        subchunk_write_order: SubchunkWriteOrder,
    ) -> &mut Self {
        self.subchunk_write_order = subchunk_write_order;
        self
    }

    /// Return the subchunk ordering.
    #[must_use]
    pub fn subchunk_write_order(&self) -> SubchunkWriteOrder {
        self.subchunk_write_order
    }

    /// Keep inner-chunk partial decoders for the lifetime of a shard's partial
    /// decoder. Disabled by default.
    ///
    /// Only has an effect when inner chunks are themselves shards. Building such
    /// a decoder reads and decodes that inner shard's own index, so without this
    /// a scattered read over one shard pays that cost per access rather than per
    /// inner shard. Where inner chunks are not shards, construction is cheap and
    /// no decoders are kept regardless of this setting.
    ///
    /// Off by default because the decoders hold a shard index that a concurrent
    /// write would invalidate, and because they are held for as long as the shard's
    /// decoder is, with no bound on how many. A reader that keeps decoders around
    /// and does not write is the case this is for.
    #[must_use]
    pub fn with_subchunk_decoder_cache(mut self, subchunk_decoder_cache: bool) -> Self {
        self.subchunk_decoder_cache = subchunk_decoder_cache;
        self
    }

    /// Keep inner-chunk partial decoders. See
    /// [`with_subchunk_decoder_cache`](Self::with_subchunk_decoder_cache).
    pub fn set_subchunk_decoder_cache(&mut self, subchunk_decoder_cache: bool) -> &mut Self {
        self.subchunk_decoder_cache = subchunk_decoder_cache;
        self
    }

    /// Whether inner-chunk partial decoders are kept.
    #[must_use]
    pub fn subchunk_decoder_cache(&self) -> bool {
        self.subchunk_decoder_cache
    }
}

#[cfg(test)]
mod tests {
    use crate::array::codec::array_to_bytes::sharding::sharding_options::SubchunkWriteOrder;

    use super::ShardingCodecOptions;
    use zarrs_codec::CodecSpecificOptions;

    #[test]
    fn sharding_options_not_set_by_default() {
        let opts = CodecSpecificOptions::default();
        assert!(opts.get_option::<ShardingCodecOptions>().is_none());
    }

    #[test]
    fn sharding_options_present_after_set() {
        let opts = CodecSpecificOptions::default().with_option(ShardingCodecOptions::default());
        assert!(opts.get_option::<ShardingCodecOptions>().is_some());
    }

    #[test]
    fn sharding_has_option() {
        let opts = CodecSpecificOptions::default().with_option(
            ShardingCodecOptions::default().with_subchunk_write_order(SubchunkWriteOrder::C),
        );
        assert!(matches!(
            opts.get_option::<ShardingCodecOptions>()
                .unwrap()
                .subchunk_write_order(),
            SubchunkWriteOrder::C
        ));
    }
}
