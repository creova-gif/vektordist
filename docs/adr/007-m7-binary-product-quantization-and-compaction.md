# ADR-007: Binary Quantization (BQ 32x), Product Quantization (PQ ADC), and Tiered Segment Compaction
- **Status:** Accepted
- **Owner:** Justin Mafie

## Context
Even with 8-bit uniform scalar quantization (`SQ8`, `4x` reduction from `3,072` bytes to `768` bytes per 768-d vector), multi-tenanted or edge deployments holding tens of millions of high-dimensional vectors require higher compression ratios (`16x`–`32x`) to keep working sets resident in L2/L3 cache and RAM. Simultaneously, append-only `VectorArena` segments accumulate tombstoned slots under heavy overwrite and delete workloads, inflating memory usage and degrading sequential scan locality unless periodically compacted.

## Decision
1. **Binary Quantization (`BqQuantizer`, `BqArena`, `BqIndex` in `crates/vektordist-core/src/pq_bq.rs`):**
   - Quantize each `f32` coordinate into a single sign bit (`>= 0.0 -> 1`, `< 0.0 -> 0`) packed into contiguous `u64` words (`dims.div_ceil(64)` words, cutting a 768-d `f32` vector from `3,072` bytes to `96` bytes — an exact **32x** payload compression).
   - Compute Hamming distances via a 4-wide unrolled bitwise `XOR + POPCNT` (`u64::count_ones`) kernel (`hamming_distance_u64`), paired with two-stage oversampled reranking (`search_reranked`) against an exact `VectorArena` (which can be `mmap`-backed on disk) to maintain `recall@10 >= 0.95`.
2. **Product Quantization (`PqCodebook`, `PqDistanceTable`, `PqIndex` in `crates/vektordist-core/src/pq_bq.rs`):**
   - Decompose `D`-dimensional vectors into `M` orthogonal subspaces of dimension `D / M` (`dims.is_multiple_of(m_subspaces)`).
   - Train `K <= 256` centroids per subspace deterministically using `SeededRng` k-means++ seeding and iterative Lloyd's updates, encoding each vector into `M` `u8` centroid codes.
   - Evaluate queries via **Asymmetric Distance Computation (`ADC`)**: precompute a `PqDistanceTable` of size `M * K` for each query so candidate scoring requires only `M` table lookups and additions (`O(M)` instead of `O(D)`), optionally followed by exact reranking.
3. **LSM-Style Tiered Segment Compaction (`TieredCompactor`, `CompactionStats`):**
   - Merge ordered `VectorArena` segments (`oldest -> newest`) in a single streaming pass.
   - Deduplicate overwritten `VectorId`s (`newest segment wins`), propagate in-segment and cross-segment delete tombstones, and emit a compacted `VectorArena` with zero tombstones (`slot_count() == live_len()`) alongside detailed `CompactionStats` (`input_segments`, `input_slots`, `output_live_vectors`, `reclaimed_tombstones`, `reclaimed_bytes`).

## Alternatives Considered
- **Symmetric PQ distance tables (`K * K` per subspace):** Adds quantization error on the query vector as well as the database vector; Asymmetric Distance Computation (`ADC`) keeps the query in full `f32` precision at negligible per-query table construction cost (`O(D * K)` once per query).
- **In-place slot compaction via swap-remove:** Mutating slot indices in-place invalidates external HNSW layer adjacency lists and requires global lock contention; LSM-style tiered segment compaction produces an immutable, densely packed `VectorArena` cleanly off the hot path.

## Consequences & Failure Modes
- **Sign-bit sensitivity on non-centered embeddings:** Binary quantization assumes coordinates are centered near `0.0` (standard for unit-normalized cosine embeddings). If raw uncentered embeddings are used with `BQ`, increasing the reranking `oversample` factor (`8x`–`16x`) or using `PqIndex` preserves recall.
- **Compaction transient memory:** `TieredCompactor` allocates a new `VectorArena` sized to `output_live_vectors` before old segments are dropped, requiring transient headroom proportional to the live vector count of the merged tier.
