# VektorDist Benchmark Report (Milestone 3: SIMD, SQ8 Quantization, and Zero-Copy Mmap Segments)

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

## 4. Reproduction Commands

Run the full Milestone 3 verification and benchmark suite:

```bash
# 1. Verify SIMD vs scalar parity, SQ8 4x compression + recall@10 >= 0.95, and VDSG CRC32 integrity (release mode):
cargo test -p vektordist-core --release --test m3_quant_mmap -- --nocapture

# 2. Run all vektordist-core unit and integration tests (WAL recovery, HNSW recall, SIMD, SQ8, Mmap):
cargo test -p vektordist-core -- --nocapture

# 3. Verify formatting and strict Clippy lints (zero unwrap/expect in production code):
cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings
```
