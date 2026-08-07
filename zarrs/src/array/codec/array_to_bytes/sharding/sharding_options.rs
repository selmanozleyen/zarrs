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
    prefetch_subchunk_indexes: bool,
}

impl Default for ShardingCodecOptions {
    fn default() -> Self {
        Self {
            subchunk_write_order: SubchunkWriteOrder::Unordered,
            prefetch_subchunk_indexes: false,
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

    /// Read every subchunk index when a partial decoder is created.
    ///
    /// Only has an effect where inner chunks are themselves shards. Such an
    /// inner chunk carries its own index, and a partial decoder keeps each one
    /// it has read, so the cost is paid once per inner shard either way. This
    /// moves it to construction, in one pass, instead of letting it land on
    /// whichever read touches each inner shard first.
    ///
    /// Worth setting when a decoder is reused across reads that between them
    /// touch most inner shards — repeated scattered reads of one shard, say —
    /// because afterwards every read is planned without going to storage.
    /// Wasteful when a decoder is built to read one small region once, since
    /// it reads indexes for inner shards that are never touched.
    #[must_use]
    pub fn with_prefetch_subchunk_indexes(mut self, prefetch_subchunk_indexes: bool) -> Self {
        self.prefetch_subchunk_indexes = prefetch_subchunk_indexes;
        self
    }

    /// Set whether subchunk indexes are read up front.
    pub fn set_prefetch_subchunk_indexes(&mut self, prefetch_subchunk_indexes: bool) -> &mut Self {
        self.prefetch_subchunk_indexes = prefetch_subchunk_indexes;
        self
    }

    /// Whether subchunk indexes are read up front.
    ///
    /// See [`with_prefetch_subchunk_indexes`](Self::with_prefetch_subchunk_indexes).
    #[must_use]
    pub fn prefetch_subchunk_indexes(&self) -> bool {
        self.prefetch_subchunk_indexes
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
