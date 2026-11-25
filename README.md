# VektorDist

**VektorDist** is a laptop-first, distributed sharded HNSW vector search engine written in Rust from first principles.

It combines an interactive 5-process cluster chaos demo (`SIGKILL` failover, `W=2` of `RF=3` CRC32 WAL quorum, replacement replica catch-up, and independent write-ID verification) with Raft topology consensus & online shard rebalancing (M2), ARM NEON SIMD + `SQ8` 4x scalar quantization + zero-copy persistent vector segments (M3), and strict multi-tenant `org_id` isolation + one-stage in-graph metadata filtering + BM25 Reciprocal Rank Fusion hybrid search (M4).

## Quickstart

```bash
cargo install --path .

# Interactive 5-process cluster with ratatui dashboard ([k] Kill node, [a] Add node, [r] Reset, [q] Quit)
vektordist demo \
  --nodes 5 \
  --vectors 100000 \
  --dimensions 768 \
  --replication-factor 3 \
  --seed 42

# Local exact & HNSW benchmark
vektordist bench --vectors 100000 --dimensions 768 --seed 42 --queries 20
```

## Workspace Structure

- `src/` — Root `vektordist` CLI binary (`demo`, `node`, `coord`, `bench`), multi-process `DemoClusterSupervisor`, `LiveRecallEvaluator`, `WriteIdLedger`, and `ratatui` TUI
- `crates/vektordist-core` — Contiguous `VectorArena`, ARM NEON / portable SIMD kernels (`simd.rs`), exact cosine/L2 search (`exact.rs`), layered HNSW + tombstones + one-stage filtered traversal (`hnsw.rs`), `SQ8` 4x scalar quantization (`quant.rs`), zero-copy `PersistentVectorSegment` (`mmap.rs`), multi-tenant `org_id` + BM25 + RRF hybrid search (`hybrid.rs`), and CRC32 WAL (`wal.rs`)
- `crates/vektordist-proto` — Packed little-endian `f32` wire codecs and checked-in Tonic/Prost gRPC stubs
- `crates/vektordist-cluster` — `TopologyStore` trait, M1 authoritative coordinator (`coord.rs`), M2 Raft topology consensus & online shard rebalancer (`raft.rs`), per-shard `ShardEngine` (`node.rs`), and direct-to-primary `ClusterClient` (`client.rs`)

## Documentation & Architecture

- **Engineering Standards:** [AGENTS.md](AGENTS.md)
- **Milestone 1 Design & Eng Review:** [docs/designs/vektordist-milestone-1.md](docs/designs/vektordist-milestone-1.md)
- **Failure Model & Consistency Guarantees:** [docs/FAILURE_MODEL.md](docs/FAILURE_MODEL.md)
- **Benchmarks & Hardware Reproduction:** [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
- **Threat Model:** [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md)
- **Architecture Decision Records:**
  - [ADR-001: M1 Workspace, Direct-to-Primary Routing & Coordinator](docs/adr/001-m1-workspace-and-coordinator.md)
  - [ADR-002: M2 Raft Topology Consensus, Lease Fencing & Online Rebalancing](docs/adr/002-m2-raft-and-rebalancing.md)
  - [ADR-003: M3 SIMD Kernels, SQ8 Scalar Quantization & Zero-Copy Segments](docs/adr/003-m3-simd-sq8-and-mmap-segments.md)
  - [ADR-004: M4 Multi-Tenant `org_id` Isolation, One-Stage Filtering & Hybrid RRF](docs/adr/004-m4-multi-tenant-and-hybrid-search.md)
