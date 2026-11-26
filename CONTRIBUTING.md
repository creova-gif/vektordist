# Contributing to VektorDist

Thank you for your interest in contributing to **VektorDist**! Every subsystem—from SIMD inner-product kernels and quantized HNSW storage to quorum WAL replication, Raft topology consensus, and the zero-trust HTTP gateway—is engineered from first principles with deterministic verification.

This guide covers everything you need to build, test, benchmark, and submit changes cleanly.

---

## 1. Prerequisites

- **Rust Toolchain (`1.85+`):** Install via [rustup](https://rustup.rs/) with `rustfmt` and `clippy`:
  ```bash
  rustup update stable
  rustup component add rustfmt clippy
  ```
- **Protocol Buffers Compiler (`protoc`, Optional):**
  Generated Tonic/Prost gRPC stubs (`crates/vektordist-proto/src/vektordist.v1.rs`) are checked into version control so `cargo build` and `cargo test` work out of the box without external system dependencies. You only need `protoc` installed if you edit `crates/vektordist-proto/proto/vektordist.proto` and rebuild with `--features vektordist-proto/codegen`:
  ```bash
  # macOS
  brew install protobuf

  # Ubuntu / Debian
  sudo apt-get update && sudo apt-get install -y protobuf-compiler libprotobuf-dev
  ```
- **Cargo Audit (Recommended):** Used in CI to check `Cargo.lock` against the RustSec advisory database:
  ```bash
  cargo install cargo-audit
  ```

---

## 2. Workspace Architecture

VektorDist is organized as a 4-member Cargo workspace:

| Crate / Path | Responsibility |
| --- | --- |
| **`vektordist` (`src/`)** | Root CLI binary (`demo`, `node`, `coord`, `gateway`, `keygen`, `snapshot`, `bench`, `chaos`), `VektorNodeConfig` loader, multi-process `DemoClusterSupervisor`, `LiveRecallEvaluator`, `WriteIdLedger`, and interactive `ratatui` TUI (`src/tui.rs`). |
| **`vektordist-core` (`crates/vektordist-core`)** | Contiguous `VectorArena`, ARM NEON & portable SIMD kernels (`simd.rs`), exact brute-force oracle (`exact.rs`), layered HNSW with tombstones & one-stage metadata filtering (`hnsw.rs`), closed-loop `AdaptiveSearchController` (`adaptive.rs`), `SQ8` 4x scalar quantization (`quant.rs`), `BQ` 32x binary & `PQ` product quantization + `TieredCompactor` (`pq_bq.rs`), zero-copy `PersistentVectorSegment` (`mmap.rs`), multi-tenant `OrgId` + BM25 + RRF hybrid search (`hybrid.rs`), programmable `AgentMemoryStore` + `ColbertIndex` + `SemanticCache` (`agentic.rs`), and IEEE CRC32 write-ahead log (`wal.rs`). |
| **`vektordist-proto` (`crates/vektordist-proto`)** | Packed little-endian `f32` wire codecs and checked-in Tonic/Prost gRPC service definitions (`TopologyRpc`, `ShardReplicaRpc`). |
| **`vektordist-cluster` (`crates/vektordist-cluster`)** | `TopologyStore` trait, authoritative coordinator (`coord.rs`), Raft consensus & online shard rebalancer (`raft.rs`), per-shard `ShardEngine` & `NodeService` (`node.rs`), direct-to-primary `ClusterClient` (`client.rs`), point-in-time `SnapshotManager` (`backup.rs`), Power-of-Two-Choices `AdaptiveRouter` (`placement.rs`), zero-trust `SecurityGate` (`security.rs`), and authenticated HTTP/1.1 REST + Prometheus `/metrics` gateway + `ChaosCampaign` runner (`gateway.rs`). |

---

## 3. Coding Standards & Static Analysis Gates

All production crates enforce strict error handling and lint hygiene:

1. **No Panics on Production Paths:**
   Every crate root declares:
   ```rust
   #![deny(clippy::unwrap_used, clippy::expect_used)]
   #![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
   ```
   Never introduce `.unwrap()`, `.expect()`, `panic!()`, or unchecked slice indexing in non-test code. Propagate typed errors via `VektorError` or `SecurityError`.
2. **Formatting, Linting & Security Audit:**
   Run the exact CI commands locally before opening a pull request:
   ```bash
   cargo fmt --all -- --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   cargo audit --deny warnings
   ```
3. **Unsafe Code Policy:**
   `unsafe` is restricted to hardware SIMD intrinsics (`crates/vektordist-core/src/simd.rs`) and alignment-verified segment slices where every invariant is explicitly checked beforehand and accompanied by scalar parity unit tests.

---

## 4. Architecture Decision Records (ADR) Workflow

Every non-trivial design choice—including changes to wire protocols, WAL/segment binary layouts, consensus/fencing rules, quantization schemes, or security controls—must be documented in `docs/adr/`:

1. Copy [`docs/adr/000-template.md`](docs/adr/000-template.md) to `docs/adr/NNN-short-slug.md` (incrementing the next available 3-digit number).
2. Fill out **Context**, **Decision**, **Alternatives Considered**, and **Consequences & Failure Modes** (explicitly noting trade-offs, memory bounds, and failure behavior).
3. Include the ADR in the same pull request as the implementation and link it in `README.md`.

---

## 5. Deterministic Seeded Testing & Benchmarks

Flaky tests are treated as bugs. Follow these verification rules:

- **Always Use `SeededRng`:** Generate synthetic embeddings, centroids, queries, and fault schedules with `vektordist_core::exact::SeededRng` (SplitMix64) and fixed seeds (e.g., `SeededRng::new(42)`). Do not use unseeded OS randomness in tests.
- **Assert Falsifiable Invariants:**
  - Durability tests must verify `acknowledged_writes_lost == 0` against an independent `WriteId` ledger across failovers and WAL replays.
  - Storage & snapshot tests must flip payload/header bytes and assert deterministic `VektorError::WalCorruption` rejection.
  - Multi-tenant and security tests must verify zero cross-tenant leakage (`SecurityError::CrossTenantDenied`) and `AuditLog::verify_integrity()`.
- **Measure Before Claiming Performance:** Never claim latency, throughput, or compression numbers without running `--release` benchmarks (`cargo run --release -- bench ...` or `cargo run --release -- chaos ...`) and recording the exact hardware and reproduction command in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

---

## 6. Pull Request Checklist

Before requesting review, verify all of the following:

- [ ] `cargo fmt --all -- --check` exits cleanly.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` reports zero warnings across library, binary, example, and test targets.
- [ ] `cargo test --workspace` passes deterministically.
- [ ] `cargo audit --deny warnings` reports zero vulnerable dependencies.
- [ ] No `.unwrap()` or `.expect()` calls were added to production code paths.
- [ ] Deterministic unit and/or integration tests cover both happy paths and failure/corruption edge cases.
- [ ] An ADR in `docs/adr/` is included if the change alters architecture, consensus, storage formats, or public protocols.
- [ ] No `.env` files, secrets, WAL/segment artifacts (`*.wal`, `*.vdsg`), or temporary directories are staged.
