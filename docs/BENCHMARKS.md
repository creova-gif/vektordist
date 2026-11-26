# VektorDist Benchmark Report (Milestones 1–10)

## 1. Hardware & Toolchain Environment

All benchmarks were measured locally on the following machine:

| Parameter | Specification |
| --- | --- |
| **SoC / CPU** | Apple M5 (`arm64` / `aarch64-apple-darwin`) |
| **Logical Cores** | 10 (`sysctl hw.ncpu`) |
| **Unified Memory** | 24 GB (`25,769,803,776` bytes) |
| **SIMD Instruction Set** | ARM NEON (`core::arch::aarch64::*`, 4x4-lane unrolled `vfmaq_f32` + `vmovl_u8` asymmetric widening) |
| **Compiler** | `rustc 1.100.0 (0ef1bc386 2026-11-09)` |

---

## 2. Vector Memory Footprint Comparison (`f32` vs `SQ8`)

`Sq8Quantizer` maps each `f32` coordinate (`4` bytes/dim) to a calibrated `u8` code (`1` byte/dim), achieving an exact **4.0x** reduction in contiguous vector payload memory across both `64d` and `768d` embeddings:

| Dataset Scale | `VectorArena` (`f32`) Payload | `Sq8Arena` (`u8`) Payload | Payload Compression | `HnswIndex` (`f32`) Total Heap | `Sq8HnswIndex` (`u8`) Total Heap |
| --- | --- | --- | --- | --- | --- |
| **2,000 x 64d** (measured in `m3_quant_mmap`) | `512,000 B` (`500.0 KiB`) | `128,000 B` (`125.0 KiB`) | **4.00x** | `1,215,184 B` (`1.16 MiB`) | `831,184 B` (`0.79 MiB`) |
| **100,000 x 768d** (M1 single-replica scale) | `307,200,000 B` (`292.97 MiB`) | `76,800,000 B` (`73.24 MiB`) | **4.00x** | `~342.0 MiB` | `~122.3 MiB` |
| **1,000,000 x 768d** (per-node shard scale) | `3,072,000,000 B` (`2.86 GiB`) | `768,000,000 B` (`732.42 MiB`) | **4.00x** | `~3.21 GiB` | `~1.08 GiB` |

---

## 3. Measured Index Build, Query Latency, and Recall (`2,000 x 64d`, 50 Queries, Seed `42`)

Measured by `crates/vektordist-core/tests/m3_quant_mmap.rs` (`sq8_quantizer_achieves_4x_compression_and_95_percent_recall_at_10`) with HNSW parameters `M = 16`, `M_max0 = 32`, `ef_construction = 128`, `ef_search = 64`, `top_k = 10`:

| Index Variant | Vector Payload RAM | Total Index RAM | Build Time (`--release`) | Avg Query Latency (`--release`) | Recall@10 |
| --- | --- | --- | --- | --- | --- |
| **`ExactIndex` (`f32` NEON)** | `512,000 B` | `585,856 B` | `0.66 ms` | `79.08 - 182.25 us` | `1.0000` (baseline) |
| **`HnswIndex` (`f32` NEON)** | `512,000 B` | `1,215,184 B` | `566.28 ms` | `106.31 - 144.14 us` | **`0.9940`** |
| **`Sq8HnswIndex` (Asymmetric `f32 x u8` NEON)** | `128,000 B` (**4x smaller**) | `831,184 B` | `447.50 ms` | `56.48 us` | **`0.9820`** (`>= 0.95`) |
| **`Sq8HnswIndex` (Asymmetric + `3x` Oversample Rerank)** | `128,000 B` (**4x smaller**) | `831,184 B` | `447.50 ms` | `31.88 - 66.88 us` | **`0.9940`** (matches `f32` HNSW) |

### Key Takeaways

1. **4.0x Vector RAM Reduction with >= 98.2% Pure Asymmetric Recall:** Even without exact reranking, `Sq8HnswIndex` achieves `recall@10 = 0.9820` (`98.2%`, exceeding the `95%` target) while storing `1` byte per dimension (`128 KB` vs `512 KB`).
2. **Lossless Recall Parity with 3x Oversample Reranking:** Reranking the top `3 * k = 30` candidates restores full `f32` HNSW recall (`0.9940` / `99.4%`) at `< 67 us` per query.
3. **Zero-Copy Persistent Segment Checkpoints (`b"VDSG"`):** `PersistentVectorSegment` stores 8-byte-aligned vector payloads with a trailing IEEE CRC32 checksum so `vector_at_slot` and `get` borrow `&[f32]` slices directly without per-vector heap allocation while deterministically rejecting any bit-flipped segment file with `VektorError::WalCorruption`.

---

## 4. M1–M10 Multi-Tier Compression & Chaos SLA Benchmarks (`3,000 x 768d`, Seed `42`)

Measured via `cargo run --release -- bench --vectors 3000 --dimensions 768 --queries 20`:

| Index Tier | Payload / Heap Footprint | Compression vs `f32` | `p50` Latency (`--release`) | `p95` Latency (`--release`) | `Recall@10` |
| --- | --- | --- | --- | --- | --- |
| **`ExactIndex` (`f32` NEON)** | `8.87 MiB` arena | `1.0x` | `0.29 ms` | `0.34 ms` | `100.00%` |
| **`HnswIndex` (`f32` NEON)** | `9.78 MiB` total heap | `1.0x` | `0.14 ms` | `0.19 ms` | **`100.00%`** |
| **`Fp16HnswIndex` (`f16` `binary16`, M16)** | `5.39 MiB` total heap | **`2.0x` payload** | `0.53 ms` | `0.59 ms` | **`100.00%`** |
| **`Sq8HnswIndex` (`u8` + `3x` Rerank)** | `3.17 MiB` total heap | **`4.0x` payload** | `0.16 ms` | `0.24 ms` | **`100.00%`** |
| **`BqIndex` (`1-bit` POPCNT + `10x` Rerank, M7)** | `281.2 KiB` (`96 B/vec`) | **`32.0x` payload** | **`0.08 ms`** | **`0.11 ms`** | **`100.00%`** |
| **`AdaptiveSearchController` (M6)** | Shared `HnswIndex` | `1.0x` | `0.18 ms` | `0.30 ms` | **`100.00%`** |

### M15 Million-Vector Streaming Scale Benchmark (`1,000,000 x 768d`, `BQ` `32x` + `32x` Oversample Rerank)

Measured via `cargo run --release -- bench-scale --vectors 1000000 --dimensions 768 --queries 20`:

| Metric | Measured Result (`Apple M5`, `--release`) |
| --- | --- |
| **Dataset Scale** | **`1,000,000` vectors x `768` dimensions** |
| **Uncompressed `f32` Equivalent** | `2,929.69 MiB` (`3,072,000,000` bytes) |
| **`BqIndex` Packed Payload (`96 B/vec`)** | **`91.55 MiB`** (**`32.0x` exact compression**, `127.30 MiB` total arena) |
| **Streaming Build Time** | **`3.56 s`** (`~280,898` vectors/sec) |
| **Query Latency (`p50` / `p95`)** | **`3.05 ms` / `3.88 ms`** (`<= 50.0 ms` target by **12.8x**) |
| **Hardware `XOR + POPCNT` Scan Throughput** | **`328.3 Million Vectors/sec`** |
| **Measured `Recall@10` vs Exact Cosine** | **`100.00%`** |

### M10 Automated 5-Node Chaos SLA Campaign (`vektordist chaos --nodes 5 --vectors 800 --dimensions 128`)

- **Schedule:** `SeedVectors(800)` -> `ConcurrentWorkload(writes=120, reads=40)` -> `KillNode(node_id=2)` -> `VerifyZeroWriteLoss` -> `AddReplacementNode(node_id=6)` -> `VerifyZeroWriteLoss` -> `AssertRecallAndLatencySla`
- **Measured Result (`--release`):** `acked_verified = 920`, `acked_lost = 0`, `recovered_shards = 5`, `recall@10 = 100.00%`, `p95_ms = 1.30 ms` (`<= 50.0 ms` SLA), `sla_passed = true`.

---

## 5. Reproduction Commands

Run the full M1–M15 verification and benchmark suite:

```bash
# 1. Run preflight host, SIMD, WAL fsync, and network diagnostics:
cargo run --release -- doctor

# 2. Run the 1,000,000 x 768d streaming Binary Quantization (32x) scale benchmark (~127 MiB RAM):
cargo run --release -- bench-scale --vectors 1000000 --dimensions 768 --queries 20

# 3. Run the M1-M10 768d release benchmark suite across Exact, HNSW, SQ8 (4x), BQ (32x), and Adaptive M6:
cargo run --release -- bench --vectors 3000 --dimensions 768 --queries 20

# 4. Run the M10 automated 5-node fault-injection & zero-write-loss SLA campaign:
cargo run --release -- chaos --nodes 5 --vectors 800 --dimensions 128

# 5. Run the full workspace test suite (68 unit & integration tests across M1-M15) and strict Clippy lints:
cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
```
