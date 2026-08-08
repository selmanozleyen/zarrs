# Read-plan cleanup, before upstreaming

Findings from a fresh review of `perf/sharding-read-plan`. Ordered by whether
they block a PR to `zarrs/zarrs`.

The feature: `ArrayPartialDecoderPlanned` lets a caller ask a decoder which
byte ranges a read would perform, issue them itself, and hand the bytes back.
Measured 6.9x on scattered CSR gathers over Lustre, because a read there costs
far more than the decode it feeds.

Files: `zarrs_codec/src/codec_traits/array_partial_sync.rs`,
`zarrs/src/array/codec/array_to_bytes/sharding/sharding_partial_decoder_sync.rs`.

---

## 1. Geometry is computed three times and nothing keeps the copies in step

**Blocking. This is the correctness risk, not a tidiness point.**

Three functions independently do `calculate_chunks_per_shard` →
`RegularChunkGrid::new` → `chunks_in_array_subset` → `.indices()`:

- `plan_fixed_array_subset` — builds the plan
- `partial_decode_fixed_array_subset_from_bytes_into` — consumes it
- `partial_decode_fixed_array_subset_into` — the ordinary path

The whole contract is *plan entry `i` corresponds to `fetched[i]`*, and it holds
only because all three iterate in the same order. Nothing enforces that: no
shared helper, no test pinning iteration order. Change the iteration in one
place and bytes decode into the wrong chunk — silently, no error, wrong data.

**Do:** extract one `fn plan_subchunk_tasks(shard_shape, subchunk_shape,
shard_index, subset) -> Result<Vec<SubchunkTask>, CodecError>` returning per
chunk: index entry, byte range, chunk subset, and output subset. All three call
it. Order then exists once.

**Done when:** the three call sites share it, and a test asserts a plan built
for a subset decodes to the same bytes as `partial_decode` of that subset for
several subset shapes, including ones straddling chunk boundaries.

## 2. A plan is not bound to the indexer that produced it

**Blocking.**

The only check is a length comparison:

```rust
if fetched.len() != chunk_indices.len() { ...error... }
```

Plan subset A, decode subset B with the same chunk count, get wrong data and no
error. The API invites it, because `read_plan(indexer)` and
`partial_decode_from_bytes(indexer, fetched)` take the indexer separately and
nothing requires them to match.

**Do:** a `ReadPlan` newtype holding the ranges plus enough to identify the
selection it came from. `partial_decode_from_bytes` takes the `ReadPlan`
instead of a bare `Vec`, and rejects a mismatch. Folds naturally into item 1,
since the newtype is where the shared geometry output lives.

Also collapses `Option<Vec<Option<ByteRange>>>` — two nested `Option`s meaning
different things — into `Option<ReadPlan>`.

**Done when:** feeding a plan from one subset into a decode of another fails
with a typed error, pinned by a test.

## 3. No async counterpart

**Blocking, or explicitly deferred with the maintainer's agreement.**

`AsyncArrayPartialDecoderTraits` and `sharding_partial_decoder_async.rs` exist
and get nothing. The justification for this feature is high-latency storage,
which is object storage, which is where the async API is used. Shipping the
latency optimisation only for the sync path invites the obvious question, and
"we benchmarked a filesystem" is a weak answer to it.

**Do:** settle this before writing more. Either add
`AsyncArrayPartialDecoderPlanned` in the same PR, or agree it is follow-up —
but decide it *before* items 1 and 2, because the shape they land on should not
have to be redesigned to accommodate async later.

---

## 4. `ArrayBytesRaw<'static>` forces the caller to allocate

`partial_decode_from_bytes` takes `Vec<Option<ArrayBytesRaw<'static>>>`, so
bytes cannot be borrowed from a buffer pool, an mmap, or a store that hands
back a view. zarrs-python currently does `bytes.to_vec()` purely to satisfy
this — a copy per chunk in a path whose entire purpose is I/O throughput.

**Do:** take a lifetime, or generic over `Into<ArrayBytesRaw<'_>>`. Check
against how the sharding decoder already stores borrowed bytes internally.

## 5. `unreachable!()` behind an invariant held elsewhere

```rust
DataTypeSize::Variable => unreachable!("planned_subset rejects variable sizes"),
```

True today. It is a panic in a library guarded by a condition in a *different*
function; relax `planned_subset` and this crashes rather than errors.

**Do:** return a `CodecError`, or restructure so `planned_subset` returns the
size it already checked and the branch disappears.

## 6. Untyped errors next to typed ones

Two `CodecError::Other(format!(...))` sit three lines from an
`InvalidNumberOfElementsError`. Callers cannot match on them.

**Do:** typed variants for "bytes supplied without a plan" and "bytes do not
match the plan". Item 2 likely absorbs both.

---

## Lower priority

- **`.expect()` on caller-influenced paths.** `"subchunks always within shard"`,
  `"inbounds chunk"`, `"index fits in usize"` mirror existing house style, but
  this API consumes caller-supplied data, so a panic is a worse failure mode
  here than in the code they were copied from. Audit which are reachable from a
  mismatched plan; item 2 removes most of the exposure.
- **`subchunk_decoders` is unbounded**, with a `Mutex` taken per inner-chunk
  access. `size_held()` exists to bound it. Nested-sharding only, so it does not
  affect the flat path.
- **`local_subchunk_grids(options)?` in `new()`** allocates two `Vec`s and a
  `ChunkGrid` per decoder construction, including flat arrays that never use the
  result. Measured as noise beside the shard-index read `new()` already does, so
  not worth fixing on its own — but if a cheaper nesting predicate appears, take
  it.

## Not problems — do not "fix" these in review

- **`Option` per plan entry, never omitted.** A missing inner chunk must keep
  its position or every later entry misaligns with `fetched`. Pinned by
  `plan_covers_a_missing_shard`.
- **`(offset != u64::MAX || size != u64::MAX)`** in `subchunk_encoded_range` is
  the De Morgan dual of the `&&` the decode path uses. The polarity must stay
  paired or plan and decode disagree about which chunks exist.
- **Nested sharding returns no plan.** One range per inner chunk would name
  whole inner shards rather than the bytes wanted. Declining is correct until
  planning can descend levels.
- **The capability trait.** `read_plan` keeps its `Option` because sharding
  plans some indexers and not others; `as_planned` answers for the decoder, the
  `Option` answers for the selection. Two different questions.

---

## Sequencing

Settle **3** with the maintainer first — it decides the API shape. Then **1**,
because it is the correctness risk and it creates the natural home for **2**.
Then **2**, **5**, **6** together. **4** last, since it touches the caller.

Do not bundle the inner-shard index cache (`3773412d`) into this PR. It is
nested-only, contributed nothing to the 6.9x, and has its own justification:
repeat access to an inner shard re-reads its index, where main does 2 reads and
1 suffices.
