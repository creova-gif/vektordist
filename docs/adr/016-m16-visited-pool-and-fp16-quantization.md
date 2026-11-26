# ADR-016: Zero-Allocation HNSW `VisitedPool` and IEEE 754 `FP16` Half-Precision Quantization
- **Status:** Accepted
- **Owner:** Justin Mafie

## Context
Two hot-path memory bottlenecks constrained single-shard latency and RAM density in `vektordist-core`:
1. **Per-Query `O(N)` `visited` Allocation & Memset:** Both `HnswIndex::search_layer` and `HnswIndex::search_layer_filtered` allocated and zero-initialized `vec![false; total_slots]` on every beam traversal. At `1M` slots per shard, even when HNSW explores only `~200–400` graph hops (`O(log N)`), every query paid a `1 MB` heap allocation/deallocation and cache-polluting memset across concurrent reader threads.
2. **Lossless-Grade `2.0x` Compression Tier:** While `Sq8HnswIndex` (`4.0x` compression) and `PqHnswIndex` / `BqHnswIndex` (`16x–32x` compression) trade coordinate precision or require codebook calibration and optional `f32` reranking arenas, embeddings workloads (`768d–3072d`) frequently require calibration-free `2.0x` RAM reduction with `>= 99%` lossless-grade `recall@10` directly from graph traversal.

## Decision
1. **Generation-Stamped `VisitedTable` Pool (`crates/vektordist-core/src/hnsw.rs`):**
   - Replace `vec![false; total_slots]` with `VisitedTable { marks: Vec<u32>, gen: u32 }` managed by a thread-safe `Mutex<Vec<VisitedTable>>` (`visited_pool`) and an RAII `VisitedGuard`.
   - `advance_and_prepare(required_slots)` grows `marks` only when index slot count exceeds current table length and increments `gen = gen.wrapping_add(1)` in `O(1)`. On 32-bit wraparound (`gen == 0`, once every `4,294,967,295` traversals), `marks` is zero-filled and `gen` resets to `1`.
   - Implement `Clone` manually on `HnswIndex` (initializing a fresh empty `Mutex::new(Vec::new())`) so `HnswIndex: Debug + Clone + Send + Sync` remains 100% source-compatible with all callers while bounding pooled buffers to peak reader concurrency.
2. **First-Principles IEEE 754 `binary16` (`FP16`) Engine (`crates/vektordist-core/src/fp16.rs`):**
   - Implement dependency-free, panic-free `f32_to_f16_bits(f32) -> u16` (round-to-nearest-ties-to-even across signed zeros, subnormals, subnormal-to-normal promotion, normal numbers, overflow, `+-Inf`, and quiet `NaN`) and exact `f16_bits_to_f32(u16) -> f32` widening.
   - Implement 4-lane unrolled (`acc0..acc3`) asymmetric (`dot_product_f32_f16`, `l2_squared_f32_f16`) and symmetric (`dot_product_f16_f16`, `l2_squared_f16_f16`) distance kernels.
   - Provide `Fp16Quantizer`, `Fp16Arena` (`slot_count * dims * 2` payload bytes = exact `2.0x` compression vs `f32`), and `Fp16HnswIndex` backed by the same generation-stamped `VisitedTable` pool, achieving `>= 99%` `recall@10` without external `f32` reranking.

## Alternatives Considered
- **`thread_local!` Visited Buffers:** A global `thread_local!` buffer avoids locking a `Mutex`, but leaks peak capacity across unrelated indices on long-lived Tokio worker threads and complicates deterministic memory accounting per shard. A brief checkout/return lock on `Mutex<Vec<VisitedTable>>` runs only twice per query (outside the inner neighbor loop), incurs negligible contention, and ties pooled memory lifecycle strictly to the owning `HnswIndex`.
- **`bf16` (`bfloat16`) Truncation Instead of IEEE 754 `binary16` (`FP16`):** `bf16` retains only `7` explicit mantissa bits (`8`-bit dynamic range), causing measurably higher cosine similarity noise on unit-normalized vectors (`|x_i| <= 1.0`) than `FP16`'s `10` explicit mantissa bits (`2^-11` relative precision).

## Consequences & Failure Modes
- **Concurrency-Proportional Pool Footprint:** If `C` reader threads concurrently query the same `HnswIndex` of `N` slots, up to `C` `VisitedTable` instances (`4 * N` bytes each) remain parked in `visited_pool` (`memory_bytes()` explicitly accounts for pooled table capacity).
- **Finite Range Clamping in `Fp16Quantizer`:** Unnormalized `L2` coordinates with `|x_i| > 65,504.0` exceed `binary16`'s maximum finite magnitude; `Fp16Quantizer::quantize_into` clamps finite inputs to `[-65,504.0, 65,504.0]` so stored database vectors never introduce `+-Inf` into distance accumulations (`NaN`/`+-Inf` inputs continue to be rejected upfront by `validate_vector`).
