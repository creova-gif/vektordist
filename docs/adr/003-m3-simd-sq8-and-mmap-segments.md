# ADR-002: Milestone 3 SIMD Distance Kernels, SQ8 Scalar Quantization, and Zero-Copy Segment Storage

- **Status:** Accepted
- **Owner:** Justin Mafie

## Context

Milestone 3 scales VektorDist's single-node and per-shard storage efficiency so high-dimensional embeddings (`768d`) can fit 4x more vectors in laptop RAM while accelerating distance evaluation and enabling zero-copy persistent segment checkpoints alongside the WAL.

## Decision

1. **Architecture-Specific SIMD Distance Kernels (`crates/vektordist-core/src/simd.rs`):**
   - Implement explicit `#[cfg(target_arch = "aarch64")]` ARM NEON intrinsics (`vfmaq_f32`, `vld1_u8`, `vmovl_u8`, `vcvtq_f32_u32`, `vaddvq_f32`) with 4x4-lane unrolling for `dot_product_simd`, `l2_squared_simd`, `dot_product_u8_asymmetric`, and `l2_squared_u8_asymmetric`.
   - Provide an 8-wide unrolled portable fallback on `x86_64` and other targets that auto-vectorizes under LLVM AVX2/SSE2, verified against scalar reference kernels across odd (`1, 5, 13, 19, 33, 769`) and even (`64, 128, 384, 768, 1536`) dimensions.
2. **SQ8 Uniform Scalar Quantization (`crates/vektordist-core/src/quant.rs`):**
   - `Sq8Quantizer` maps `f32` coordinates (`4` bytes/dim) to `u8` codes (`1` byte/dim) over calibrated `[min_val, max_val]` bounds, reducing contiguous vector memory in `Sq8Arena` by strictly `4.0x`.
   - `Sq8HnswIndex` performs asymmetric graph traversal (`f32` query vs `u8` database codes) using NEON widening FMA instructions and supports optional top-`oversample` reranking against either dequantized unit vectors or an exact `VectorArena` / `PersistentVectorSegment`, maintaining `recall@10 >= 95%`.
3. **Zero-Copy Aligned Persistent Vector Segments (`crates/vektordist-core/src/mmap.rs`):**
   - Define the `b"VDSG"` binary segment format with a 32-byte header, 8-byte-aligned `slot_ids` (`u64`), 8-byte-aligned `tombstones` (`u64`), 8-byte-aligned contiguous `f32` vector payload, and a trailing 4-byte IEEE CRC32 footer.
   - Load segment bytes into a 64-bit aligned backing buffer after verifying the full-file CRC32 checksum so `vector_at_slot` and `get` return zero-copy `&[f32]` slices with zero per-vector heap allocations.

## Alternatives Considered

- **Symmetric `u8 x u8` integer dot products only:** Slightly faster raw integer throughput, but quantizing the query vector introduces double quantization error on small coordinate ranges; asymmetric `f32` query vs `u8` database codes preserves higher recall (`>= 98%`) with negligible overhead in NEON.
- **Product Quantization (PQ):** Higher compression ratios (`8x-32x`), but requires k-means codebook training and lookup tables that degrade recall on small/medium dynamic shards and add training latency during online ingestion.
- **Unchecked raw `mmap` without full CRC32 verification:** Faster initial open on multi-GB files, but risks silent bit-rot or `SIGBUS` if underlying segment files are truncated; verifying the IEEE CRC32 footer into an 8-byte-aligned segment buffer guarantees deterministic corruption detection (`VektorError::WalCorruption`).

## Consequences & Failure Modes

- **Memory footprint:** Vector payload memory drops from `3,072` bytes/vector (`768 * 4`) to `768` bytes/vector (`768 * 1`), a `4.0x` reduction.
- **Integrity:** Any bit flip or truncation in a `.vdsg` segment file is caught deterministically on `PersistentVectorSegment::open` via the trailing IEEE CRC32 checksum.
