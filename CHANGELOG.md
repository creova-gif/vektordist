# Changelog

All notable changes to **VektorDist** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-11-26

Production hardening and scale-readiness release of **VektorDist**, spanning **Milestones M12 through M15**: bounded WAL storage via atomic binary checkpointing, pooled HTTP/2 gRPC transport & bulk ingestion, graceful `/readyz` gateway draining & ergonomic Rust client SDK, Raft-replicated zero-trust key/quota management with durable audit logs, `1M`-vector scale verification, and enterprise Kubernetes SRE operations.

### Added

- **Bounded Storage, Atomic WAL Checkpointing & Pooled Bulk Ingestion (M12):**
  - `ShardCheckpointManager` pairing CRC32-verified checkpoint metadata (`b"VDCK"`) with 8-byte-aligned zero-copy `PersistentVectorSegment` (`b"VDSG"`) snapshots and crash-safe atomic WAL compaction (`write-tmp -> fsync -> rename`), reclaiming disk space and bounding cold-start replay latency.
  - `GrpcChannelPool` multiplexing persistent HTTP/2 tonic channels across shard primaries and followers, eliminating per-RPC TCP connection churn on hot paths.
  - High-throughput bulk ingestion via shard-grouped `ClusterClient::upsert_batch` and `POST /v1/upsert_batch` with `W=2` quorum durability.
- **Graceful Gateway Draining, DoS Guards & Rust Client SDK (M13):**
  - `GET /readyz` Kubernetes readiness & graceful shutdown drain endpoint (`503 Service Unavailable` while draining or unready) decoupled from `GET /healthz` process liveness, plus `413 Content Too Large` request-body DoS protection.
  - Ergonomic `HttpVektorClient` Rust SDK supporting bearer-token authentication, tenant scoping (`X-Org-Id`), single/batch upserts, search, quorum deletes, topology/audit inspection, and `traceparent` propagation.
- **Distributed Zero-Trust Security & Durable Audit Log (M14):**
  - Raft-replicated `SecurityGate` state machine synchronizing dynamic API key issuance (`IssueApiKey`), instant cluster-wide key revocation (`RevokeApiKey`), and live `TenantQuotaPolicy` updates across stateless gateway replicas.
  - Crash-safe, append-only `PersistentAuditLog` persisting hash-chained `AuditEntry` records to disk with `fsync` durability and genesis-to-head tamper verification (`GET /v1/audit`).
- **Pre-Flight Diagnostics, `1M`-Vector Scale Harness & Enterprise SRE Operations (M15):**
  - `vektordist doctor` pre-flight environment, configuration, `fsync`, and security diagnostics CLI, alongside streaming `1M`-vector `vektordist bench-scale` memory and latency verification.
  - Hardened multi-document Kubernetes manifest (`deploy/k8s/statefulset.yaml`) with `PodDisruptionBudget` (`minAvailable: 4` preserving `W=2` of `RF=3` quorum), host/zone `PodAntiAffinity`, zero-trust non-root (`65532:65532`) read-only rootfs `securityContext`, `vektordist-gateway` HA deployment with `/readyz` + `/healthz` probes & Prometheus scrape annotations, updated distroless `deploy/Dockerfile`, and authoritative production runbook (`docs/OPERATIONS.md`).

## [0.1.0] - 2026-11-26

Initial open-source release of **VektorDist**, spanning **Milestones M1 through M11** from core sharded HNSW storage and quorum WAL failover through distributed Raft consensus, multi-tier quantization, hybrid multi-tenant retrieval, zero-trust RBAC/audit security, programmable AI agent memory, and authenticated cloud-native observability.

### Added

- **Core HNSW, Contiguous Arenas & Quorum WAL (M1):**
  - Contiguous cache-aligned `VectorArena`, exact brute-force cosine/L2 search oracle (`ExactIndex`), and layered HNSW graph (`HnswIndex`) with tombstone deletion.
  - Length-prefixed, IEEE CRC32-checksummed write-ahead log (`WalLog`) with `fsync` durability, safe torn-EOF truncation after `SIGKILL`, and fail-closed `WalCorruption` detection on mid-file bit-rot.
  - 16-shard SplitMix64 partitioning, direct-to-primary `W=2` of `RF=3` quorum writes, epoch fencing (`STATUS_WRONG_EPOCH`), union WAL anti-entropy failover reconciliation, replacement replica catch-up (`RECOVERING -> HEALTHY`), `WriteIdLedger` zero-loss verifier, and `LiveRecallEvaluator`.
- **Raft Topology Consensus & Online Rebalancing (M2):**
  - Raft-replicated topology state machine (`RaftNode`) eliminating coordinator single-point-of-failure, lease-fenced primary promotions, and zero-downtime online shard migration across cluster nodes.
- **SIMD Acceleration, SQ8 Quantization & Zero-Copy Segments (M3):**
  - 4x4-lane unrolled ARM NEON (`aarch64`) and portable SIMD inner-product/L2 kernels with asymmetric `f32 x u8` widening dot products.
  - `Sq8Quantizer` and `Sq8HnswIndex` achieving **4.0x** vector payload compression with `>= 98.2%` pure asymmetric `recall@10` and lossless `f32` recall parity under `3x` oversample reranking.
  - 8-byte-aligned `PersistentVectorSegment` (`VDSG`) checkpoints with trailing IEEE CRC32 verification and zero-copy `&[f32]` slice borrowing.
- **Multi-Tenant Hybrid Search & One-Stage Filtering (M4):**
  - Structural `OrgId` data-plane isolation guaranteeing zero cross-tenant leakage.
  - One-stage in-graph `MetadataFilter` HNSW traversal (`Eq`, `In`, `Range`, `And`, `Or`) avoiding post-filter recall collapse.
  - Lexical BM25 inverted index fused with dense HNSW retrieval via Reciprocal Rank Fusion (`RRF`).
- **Point-in-Time Snapshots & Production Deployment Infrastructure (M5):**
  - `SnapshotManager` (`vektordist snapshot create | verify | restore`) with manifest and per-shard IEEE CRC32 integrity verification.
  - Multi-stage distroless `Dockerfile` (`gcr.io/distroless/cc-debian12:nonroot`), Kubernetes 5-replica `StatefulSet` (`deploy/k8s/statefulset.yaml`), and GCP Compute Engine `pd-ssd` Terraform module (`deploy/gcp/main.tf`).
- **Adaptive Latency Control & Workload-Aware Placement (M6):**
  - Closed-loop `AdaptiveSearchController` dynamically tuning `ef_search` to meet p95 latency budgets while maximizing recall.
  - Power-of-Two-Choices (`P2C`) latency/queue-aware replica routing (`AdaptiveRouter`) and hot-primary shard rebalancing.
- **Binary/Product Quantization & LSM Tiered Compaction (M7):**
  - `BqIndex` 1-bit Binary Quantization (**32.0x** payload compression via `XOR + POPCNT` Hamming distance with two-stage exact reranking).
  - `PqCodebook` / `PqIndex` subspace k-means Product Quantization (**16.0x** compression) with Asymmetric Distance Computation (`ADC`) lookup tables.
  - `TieredCompactor` background segment compaction reclaiming 100% of tombstoned vector slots and HNSW graph edges.
- **Zero-Trust Security, Tenant Quotas & Tamper-Evident Audit Log (M8):**
  - `SecurityGate` issuing scoped `vdk.<key_id>.<hex_tag>` bearer tokens verified in constant time (`constant_time_eq_32`) with monotonic RBAC roles (`Reader`, `Writer`, `Admin`).
  - Per-tenant token-bucket QPS rate limiting and atomic vector/memory storage ceilings (`TenantQuotaTracker`).
  - Append-only SHA/CRC-chained `AuditLog` (`verify_integrity()`) detecting any historical mutation, deletion, or reordering.
- **Programmable AI Agent Memory, ColBERT & Semantic Cache (M9):**
  - `AgentMemoryStore` combining cosine similarity, exponential recency decay, and intrinsic importance with greedy `max_token_budget` context-window packing.
  - Multi-vector late-interaction `ColbertIndex` (`MaxSim` SIMD token scoring) for standalone search and two-stage candidate reranking.
  - `(OrgId, filter_hash)`-partitioned `SemanticCache` with TTL expiration, FIFO capacity bounds, and instant tenant invalidation.
- **Authenticated HTTP/1.1 Gateway, Prometheus Metrics, CLI/TUI & Chaos Suite (M10–M11):**
  - Zero-dependency HTTP/1.1 REST gateway (`/v1/upsert`, `/v1/search`, `/v1/delete`, `/v1/topology`, `/v1/audit`, `/healthz`, `/metrics`) integrated with `SecurityGate` (`Authorization: Bearer vdk...`, `X-Org-Id`), W3C `traceparent` distributed trace spans, and Prometheus `0.0.4` metrics exposition.
  - Declarative `VektorNodeConfig` (`vektordist node --config` and `VEKTORDIST_*` environment overrides), `vektordist keygen` token provisioning CLI, enriched `ratatui` cluster dashboard (`[k]` kill, `[a]` add, `[p]` pause/resume, `[r]` reset, latency sparkline, shard health summary), `vektordist chaos` automated SLA runner, and `quickstart_hybrid_agent_rag` example.
  - Full open-source governance suite (`LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, GitHub issue/PR templates, CI + multi-arch release tarballs with SHA-256 checksums).

[Unreleased]: https://github.com/creova-gif/vektordist/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/creova-gif/vektordist/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/creova-gif/vektordist/releases/tag/v0.1.0
