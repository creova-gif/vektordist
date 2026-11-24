# ADR-001: Milestone 1 Workspace Layout, Direct-to-Primary Routing, and Single Authoritative Coordinator

- **Status:** Accepted
- **Owner:** Justin Mafie

## Context

VektorDist Milestone 1 targets a laptop-first (macOS ARM64 and Linux), zero-Docker, single-command interactive distributed vector search demonstration:

```bash
cargo install --path .
vektordist demo --nodes 5 --vectors 100000 --dimensions 768 --replication-factor 3 --seed 42
```

The system must launch 5 real OS processes, build a from-scratch sharded HNSW index with `RF=3` / `W=2` durable WAL replication, sustain `recall@10 >= 95%` and `p95 <= 50ms` at 10 concurrent clients, survive `SIGKILL` of a live data node with `acknowledged_writes_lost == 0` verified by an external write-ID ledger, and support replacement-node WAL catch-up. At the same time, Milestone 1 must provide a clean architectural runway into M2 (Raft topology consensus and online shard rebalancing), M3 (SQ8 quantization, mmap, SIMD, 10M vectors), M4 (multi-tenant `org_id` isolation, hybrid search), and M5 (GCP/Kubernetes/SDKs).

## Decision

1. **Hybrid 3-Member-Crate Cargo Workspace + Root CLI Binary:**
   - Root `Cargo.toml` defines both `[workspace]` and `[package] name = "vektordist"`, owning the CLI/TUI binary (`src/main.rs`, `src/cli/`, `src/tui/`) so `cargo install --path .` works directly.
   - `crates/vektordist-core`: Pure Rust indexing and storage primitives (`VectorArena` flat `Vec<f32>`, exact cosine/L2 search, HNSW with tombstone bitset, CRC32-framed WAL with crash replay, shared domain types).
   - `crates/vektordist-proto`: Protobuf schemas (`ShardReplicaRpc`, `TopologyRpc`) with packed little-endian `bytes` vector encoding and checked-in Tonic/Prost Rust stubs (so `cargo install --path .` requires no `protoc`).
   - `crates/vektordist-cluster`: Combined `node/` (per-shard `ShardEngine`, `W=2` quorum replication, WAL catch-up) and `coord/` (`TopologyStore` trait + M1 single authoritative coordinator, heartbeat monitor, union WAL reconciliation) plus `ClusterClient`.
2. **Shape 2: Direct-to-Primary Smart Client with `shard_epoch` Fencing:**
   - `ClusterClient` caches `TopologySnapshot(epoch, shards)` and sends writes and reads directly to shard primaries/replicas, eliminating an extra coordinator hop from the data plane.
   - Every shard assignment carries a monotonic `epoch: u64`. Primaries reject stale-epoch writes with `WRONG_EPOCH`, and followers reject `ReplicateWal` RPCs with `epoch < local_epoch`.
3. **Honest Single Authoritative Coordinator in M1:**
   - M1 implements `TopologyStore` with a single authoritative coordinator (running in-process inside `vektordist demo` or standalone via `vektordist coord`). Coordinator HA via Raft is explicitly deferred to M2 and documented in `docs/FAILURE_MODEL.md`.
4. **Durability & Code Safety Policies:**
   - WAL replay truncates a partial/corrupt final record (from mid-write `SIGKILL`) with a warning while hard-failing on mid-file CRC32 corruption.
   - All production crates enforce `#![deny(clippy::unwrap_used, clippy::expect_used)]` (relaxed only under `#[cfg(test)]`).

## Alternatives Considered

- **Single-crate framed-TCP monolith (Approach A):** Faster initial bring-up, but custom `bincode`/TCP wire protocol lacks schema evolution and SDK support for M4/M5 and weakens compile-time boundaries around `vektordist-core`.
- **5-member-crate workspace (`node`, `coord`, `cli` separated):** Adds cross-crate manifest and error-conversion boilerplate in M1 before M2 Raft justifies splitting `coord` into its own crate.
- **Coordinator-proxied data plane (Shape 1):** Adds an avoidable gRPC hop on every write and search RPC, hurting p95 latency.
- **Full deterministic simulation runtime in M1 (Approach C):** High upfront async trait/lifetime complexity that would delay the real 5-process laptop failover demo.

## Consequences & Failure Modes

- **Supported M1 failures:** Single data-node `SIGKILL` (detected in `~1.5s`, union WAL reconciliation across 2 survivors before primary promotion, `acknowledged_writes_lost == 0`), double data-node `SIGKILL` (under-quorum shards remain read-only while rejecting writes with `QUORUM_UNAVAILABLE`), mid-write WAL tail truncation, stale-primary `WRONG_EPOCH` rejection, and `RECOVERING` dual-stream catch-up.
- **Unsupported M1 failures (deferred to M2):** Coordinator process crash, asymmetric network partitions between coordinator and nodes, and multi-node simultaneous loss of `>= 2` replicas of the same shard for write availability.
- **Tradeoff:** Combining `node/` and `coord/` into `vektordist-cluster` accelerates M1 delivery while keeping `TopologyStore` trait boundaries clean for a future `vektordist-raft` split in M2.
