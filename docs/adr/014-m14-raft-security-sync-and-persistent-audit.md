# ADR-014: Raft-Replicated Cluster Security State and `fsync`-Backed Tamper-Evident `PersistentAuditLog`
- **Status:** Accepted
- **Owner:** Justin Mafie

## Context
In Milestone 8 and Milestone 11, `SecurityGate` stored scoped API key digests, revocations, per-tenant token-bucket quotas, and hash-chained `AuditLog` entries purely in process memory on a single gateway instance. In horizontally scaled enterprise deployments:
1. Horizontally scaled `HttpGatewayServer` replicas must enforce consistent cluster-wide API keys, instant key revocations, and tenant QPS/storage quota policies without split-brain drift.
2. Compliance and forensic audit trails must survive process restarts via `fsync` durability and deterministically detect any post-hoc disk tampering, frame reordering, or truncation.

## Decision
1. **Raft-Replicated Security Control Plane (`crates/vektordist-cluster/src/raft.rs` & `security.rs`):**
   - Extended `TopologyCommand` with three deterministic, `Eq`-safe control-plane commands:
     - `UpsertApiKey { record: ScopedApiKey }`
     - `RevokeApiKey { key_id: String }`
     - `SetTenantQuota { org_id: OrgId, max_qps_milli: u64, burst_milli: u64, max_vectors: u64, max_memory_bytes: u64 }` (representing rates in integer milli-ops/sec so `TopologyCommand` preserves total equality `Eq` across Raft log comparisons).
   - Materialized `api_keys`, `revoked_keys`, and `tenant_quotas` inside `RaftNode::apply_command`, and added `SecurityGate::upsert_key_record`, `SecurityGate::get_key_record`, and `SecurityGate::sync_from_raft(&mut self, raft: &RaftNode, now_ms: u64)` so stateless gateways converge deterministically on committed Raft state while preserving live token-bucket balances across sync ticks.
2. **`fsync`-Backed `PersistentAuditLog` (`crates/vektordist-cluster/src/security.rs`):**
   - Implemented `PersistentAuditLog` wrapping `AuditLog` and an append-only `std::fs::File` using binary `VAUD`-magic frames (`magic[4] | payload_len: u32 | crc32: u32 | payload`).
   - `append_and_sync` computes the chained `entry_hash` (`prev_hash` + sequence + timestamp + actor + tenant + action + outcome), writes the frame, and executes `File::sync_data()` before updating in-memory state.
   - `open` and `reopen_and_verify` validate every frame header, IEEE CRC32 checksum, contiguous 1-based `seq_no`, `prev_hash` linkage, and recomputed `entry_hash` from genesis (`0`), returning `Err(VektorError::WalCorruption { offset })` if any byte on disk was mutated, reordered, or truncated.

## Alternatives Considered
- **Floating-Point Quota Fields in `TopologyCommand`:** Storing `f64` directly in `TopologyCommand` breaks `Eq` due to `NaN` semantics. Fixed-point `max_qps_milli: u64` and `burst_milli: u64` provide sub-unit (`0.001 QPS`) precision while keeping `TopologyCommand: Eq`.
- **Plain Newline-Delimited Text Audit Logs:** Vulnerable to delimiter injection if denial reasons or action strings contain control characters, and ambiguous under partial tail writes. Length-prefixed binary frames with CRC32 + cryptographic hash chaining eliminate framing ambiguity and pinpoint exact byte offsets on corruption.

## Consequences & Failure Modes
- **Fail-Closed Audit Recovery:** Unlike data-plane shard WALs that truncate incomplete trailing crash frames, `PersistentAuditLog::reopen_and_verify` treats any truncated or modified byte on disk as potential security log tampering and fails closed with `VektorError::WalCorruption { offset }` for operator forensic inspection.
- **Stateless Gateway Horizontal Scaling:** Any number of `HttpGatewayServer` instances sharing the cluster master secret can call `sync_from_raft` against a local Raft voter/learner to enforce identical authentication, quotas, and revocations without sharing mutable memory.
