# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add `CodecCreateError` for codec creation, reconfiguration, and binding failures
- Add `UnboundArrayTo{Array,Bytes}CodecTraits`
- Implement `[Async]BytesPartial{Encoder,Decoder}Traits` for `(Tstorage: *StorageTraits, StoreKey)`
- Add `ChunkGrid{Encoded,Decoded}Ref` and `[Async]ArrayPartialDecoderTraits::local_subchunk_grid[s]` for chunk-local subchunk grids
- Add `ArrayPartialDecoderPlanned` for decoders that can report their reads before performing them, reached through `ArrayPartialDecoderTraits::as_planned`
  - Add `ReadPlan`, which carries the selection its byte ranges were computed for, and `CodecError::ReadPlanMismatch`
  - A plan's unit is a read: one entry may cover several stored units that are adjacent in the stored value, and absent units are not entries -- `DataPlan::fill_absent_into` fills them without fetched data
  - Add `PlanState`, decoder-private state carried by plans built with `{Data,Index}Plan::new_with_state`; only plans the decoder minted validate, so decoding never re-derives the walk
  - Add `DataPlan::decode_entry_into` to decode one entry as its read lands, with per-entry (constant-work) validation; `decode_into` remains for callers that already hold everything
- Implement `BytesPartialDecoderTraits` for `Bytes`, so bytes a store returned can be decoded without being copied into a `Vec` first

### Changed
- **Breaking**: Refactor `ArrayTo{Array,Bytes}CodecTraits`
  - These traits are now associated with codecs that are _bound_ to a data type and fill value and validated at array creation time
  - **Breaking**: Add `data_type()`, `fill_value()`, `encoded_chunk_grid()` and `decoded_subchunk_grid[s]()` methods
  - **Breaking**: Remove `decoded_shape()` and `partial_decode_granularity()` methods
  - **Breaking**: Remove `data_type` and `fill_value` parameters from various methods
  - **Breaking**: Add `ArrayTo{Array,Bytes}CodecSubchunkingTraits` supertraits for resolving subchunk grids
    - `ArrayToArrayCodecSubchunkingIdentityTraits` and `ArrayToBytesCodecNoSubchunkingTraits` marker traits are available for common codecs

### Removed
- **Breaking**: Remove `ArrayCodecTraits::partial_decode_granularity`
- **Breaking**: Remove `[Async]StoragePartial{Encoder,Decoder}`
- **Breaking**: Remove `[Async]ArrayPartialEncoderTraits::into_dyn_decoder()`

## [0.2.1] - 2026-03-21

### Added
- Add `CodecSpecificOptions` for codec-specific runtime configuration
- Add `with_codec_specific_options` default method to `ArrayToArrayCodecTraits`, `ArrayToBytesCodecTraits`, and `BytesToBytesCodecTraits`

## [0.2.0] - 2026-02-02

### Added
- Add `CodecTraitsV2` and  `CodecTraitsV3` traits
- Add `CodecError::UnsupportedDataTypeCodec` variant for data type codec support errors
- Add `ExpectedFixedLengthBytesError`, `ExpectedVariableLengthBytesError`, and `ExpectedOptionalBytesError` error types

### Changed
- **Breaking**: Remove `create_fn` parameter from `CodecPluginV2::create()` and add `T: CodecTraitsV2` bound
- **Breaking**: Remove `create_fn` parameter from `CodecPluginV3::create()` and add `T: CodecTraitsV3` bound
- **Breaking**: Rename `ArrayRawBytesOffsetsOutOfBoundsError` to `ArrayBytesRawOffsetsOutOfBoundsError`
- **Breaking**: Rename `ArrayRawBytesOffsetsCreateError` to `ArrayBytesRawOffsetsCreateError`
- **Breaking**: `ArrayBytes::into_fixed()` now returns `Result<_, ExpectedFixedLengthBytesError>` instead of `Result<_, CodecError>`
- **Breaking**: `ArrayBytes::into_variable()` now returns `Result<_, ExpectedVariableLengthBytesError>` instead of `Result<_, CodecError>`
- **Breaking**: `ArrayBytes::into_optional()` now returns `Result<_, ExpectedOptionalBytesError>` instead of `Result<_, CodecError>`
- **Breaking**: `CodecError::ExpectedFixedLengthBytes`, `CodecError::ExpectedVariableLengthBytes`, and `CodecError::ExpectedOptionalBytes` now wrap their respective dedicated error types

### Removed
- **Breaking**: Remove `CodecError::ExpectedNonOptionalBytes` (replaced with `CodecError::ExpectedOptionalBytes`)
- **Breaking**: Remove `ArrayBytes::into_optional_bytes()` method (use `into_optional()` instead)
- **Breaking**: Remove `optional_nesting_depth`, `build_nested_optional_target`, `merge_chunks_vlen`, `merge_chunks_vlen_optional`, and `extract_decoded_regions_vlen` (moved to `zarrs` as private functions)

## [0.1.0] - 2026-01-14

### Added
- Split from the `zarrs::array::codec` module of `zarrs` 0.23.0-beta.5

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_codec-v0.2.1...HEAD
[0.2.1]: https://github.com/zarrs/zarrs/releases/tag/zarrs_codec-v0.2.1
[0.2.0]: https://github.com/zarrs/zarrs/releases/tag/zarrs_codec-v0.2.0
[0.1.0]: https://github.com/zarrs/zarrs/releases/tag/zarrs_codec-v0.1.0
