# VektorDist

**VektorDist** is a laptop-first, distributed sharded HNSW vector search engine written in Rust from first principles.

It spans the full engineering lifecycle across **10 Milestones (M1–M10)**:
- **M1 — Interactive Cluster & Quorum WAL:** 5-process laptop cluster chaos demo (`SIGKILL` failover, `W=2` of `RF=3` CRC32 WAL quorum, replacement replica catch-up, and independent write-ID verification).
- **M2 — Distributed Consensus & Rebalancing:** Raft topology state machine, epoch-fenced primary leases, and zero-downtime online shard migration.
- **M3 — Performance & Scale:** ARM NEON SIMD distance kernels, `SQ8` 4x scalar quantization, and zero-copy memory-mapped `PersistentVectorSegment` storage.
- **M4 — Enterprise Retrieval:** Strict multi-tenant `org_id` isolation (`Zero` cross-tenant leakage), one-stage in-graph `MetadataFilter` HNSW traversal, and BM25 + Reciprocal Rank Fusion (`RRF`) hybrid search.
- **M5 — Production Infrastructure:** Point-in-time cluster snapshots, CRC32 restore verifier, and Docker / Kubernetes `StatefulSet` / GCP Terraform deployment manifests.
- **M6 — Adaptive Resource Control:** Closed-loop `AdaptiveEfController` latency-budget tuner and workload-aware Power-of-Two-Choices (`P2C`) replica routing & hot-primary rebalancing.
- **M7 — Extreme Compression & LSM Compaction:** Binary Quantization (`BQ` 32x compression via `XOR + POPCNT` with two-stage exact reranking), Product Quantization (`PQ` subspace k-means codebooks with `ADC` lookup tables), and `TieredCompactor` 100% tombstone reclamation.
- **M8 — Zero-Trust Security, Quotas & Audit:** Constant-time HMAC-SHA256 bearer token authentication, scoped RBAC (`Admin`, `TenantWriter`, `TenantReader`, `MetricsObserver`), per-tenant token-bucket rate limiting & vector capacity quotas, and tamper-evident SHA-256 hash-chained audit logs.
- **M9 — Programmable AI Agent Memory & ColBERT:** Time-decayed episodic/semantic/procedural `AgentMemoryStore`, multi-vector ColBERT late-interaction `MaxSim` scoring (`ColBertIndex`), and cosine-thresholded LRU `SemanticQueryCache`.
- **M10 — Cloud-Native HTTP/JSON Gateway, Prometheus & Chaos:** Embedded HTTP/1.1 JSON REST gateway (`/v1/upsert`, `/v1/search`, `/v1/health`, `/metrics`), lock-free Prometheus text-0.0.4 metrics exporter, and deterministic `JepsenChaosRunner` fault-injection matrix.

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

# Local exact, HNSW, SQ8 (4x), BQ (32x) & adaptive M1-M10 benchmark suite
vektordist bench --vectors 100000 --dimensions 768 --seed 42 --queries 20

# Automated multi-step fault-injection & zero-data-loss SLA campaign
vektordist chaos --nodes 5 --vectors 2000 --dimensions 128 --seed 42

# Launch HTTP/1.1 REST + Prometheus /metrics gateway in front of a coordinator
vektordist gateway --coord 127.0.0.1:50051 --listen 127.0.0.1:8080

# Point-in-time cluster snapshot creation, verification & restore
vektordist snapshot create --node-dir /tmp/vektordist-node --snapshot-dir ./snapshots/snap-01 --dimensions 768
vektordist snapshot verify --snapshot-dir ./snapshots/snap-01
```

## Workspace & Infrastructure Structure

- `src/` — Root `vektordist` CLI binary (`demo`, `node`, `coord`, `bench`, `snapshot`), multi-process `DemoClusterSupervisor`, `LiveRecallEvaluator`, `WriteIdLedger`, and `ratatui` TUI
- `crates/vektordist-core` — Contiguous `VectorArena`, ARM NEON / portable SIMD kernels (`simd.rs`), exact cosine/L2 search (`exact.rs`), layered HNSW + tombstones + one-stage filtered traversal (`hnsw.rs`), adaptive `ef_search` controller (`adaptive.rs`), `SQ8` 4x scalar quantization (`quant.rs`), `BQ` 32x binary quantization + `PQ` product quantization + `TieredCompactor` (`pq_bq.rs`), zero-copy `PersistentVectorSegment` (`mmap.rs`), multi-tenant `org_id` + BM25 + RRF hybrid search (`hybrid.rs`), programmable agent memory + ColBERT `MaxSim` + `SemanticQueryCache` (`agentic.rs`), and CRC32 WAL (`wal.rs`)
- `crates/vektordist-proto` — Packed little-endian `f32` wire codecs and checked-in Tonic/Prost gRPC stubs
- `crates/vektordist-cluster` — `TopologyStore` trait, M1 authoritative coordinator (`coord.rs`), M2 Raft topology consensus & online shard rebalancer (`raft.rs`), M5 point-in-time cluster snapshots & restore verifier (`backup.rs`), M6 workload-aware Power-of-Two-Choices router & hot-primary rebalancer (`placement.rs`), M8 HMAC-SHA256 RBAC + tenant quotas + hash-chained audit log (`security.rs`), M10 HTTP/1.1 JSON REST gateway + Prometheus `/metrics` exporter + `JepsenChaosRunner` (`gateway.rs`), per-shard `ShardEngine` (`node.rs`), and direct-to-primary `ClusterClient` (`client.rs`)
- `deploy/` — Multi-stage distroless `Dockerfile`, Kubernetes 5-replica `StatefulSet` (`deploy/k8s/statefulset.yaml`), and GCP Compute Engine `pd-ssd` Terraform module (`deploy/gcp/main.tf`)

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
  - [ADR-005: M5 Point-in-Time Cluster Snapshots & Production Deployment](docs/adr/005-m5-snapshots-and-production-infra.md)
  - [ADR-006: M6 Adaptive `ef_search` Controller & Workload-Aware Placement](docs/adr/006-m6-adaptive-ef-and-workload-placement.md)
  - [ADR-007: M7 Binary Quantization (32x), Product Quantization & LSM Tiered Compaction](docs/adr/007-m7-binary-product-quantization-and-compaction.md)
  - [ADR-008: M8 Zero-Trust RBAC, Tenant Rate/Storage Quotas & Tamper-Evident Audit Log](docs/adr/008-m8-enterprise-security-rbac-quotas-and-audit.md)
  - [ADR-009: M9 Agentic Memory Hierarchy, ColBERT MaxSim Late Interaction & Semantic Cache](docs/adr/009-m9-agentic-memory-colbert-and-semantic-cache.md)
  - [ADR-010: M10 HTTP/JSON REST Gateway, Prometheus Metrics & Jepsen-Style Chaos Runner](docs/adr/010-m10-http-gateway-prometheus-and-chaos-runner.md)
