# ADR-008: M8 Enterprise Security — Scoped API Keys, RBAC, Per-Tenant Quotas, and Tamper-Evident Audit Log
- **Status:** Accepted
- **Owner:** Justin Mafie

## Context
Multi-tenant production deployments of VektorDist require defense-in-depth beyond the storage-engine `OrgId` data-plane isolation introduced in M6:
1. **Authentication & RBAC:** Requests must be authenticated with tenant-scoped credentials bound to a single `OrgId` and privilege tier (`Reader`, `Writer`, `Admin`) without leaking secret key material through timing side-channels.
2. **Noisy-Neighbor Isolation:** A single tenant issuing unbounded query bursts or bulk vector upserts must not exhaust CPU or memory capacity for co-located tenants on the same shard replicas.
3. **Tamper-Evident Compliance Auditing:** Every security-relevant access decision (`Allowed` or `Denied`) must be recorded in an append-only audit log that detects any historical modification, deletion, or reordering.

## Decision
1. **Scoped API-Key Authenticator & RBAC (`crates/vektordist-cluster/src/security.rs`):**
   - `SecurityGate::issue_api_key` issues tokens formatted as `vdk.<key_id>.<hex_tag>` and stores only a 32-byte keyed digest (`ScopedApiKey::secret_digest`) derived from the gate's 256-bit master secret.
   - `SecurityGate::authenticate` and `constant_time_eq_32` compare all 32 digest bytes using a branchless bitwise XOR accumulator (`diff |= a[i] ^ b[i]`) and execute a dummy comparison when `key_id` is unknown, preventing timing side-channels.
   - `SecurityGate::authorize` enforces mandatory tenant binding (`&key.org_id == target_org`, returning `SecurityError::CrossTenantDenied` on mismatch) and monotonic RBAC role checks (`Reader < Writer < Admin`, returning `SecurityError::InsufficientRole`).
2. **Per-Tenant Token-Bucket Rate Limiter & Resource Quotas (`TenantQuotaPolicy`, `TenantQuotaTracker`):**
   - Each `OrgId` maintains an independent token bucket (`max_qps`, `burst_capacity`) refilled deterministically from monotonic millisecond timestamps, returning `SecurityError::RateLimited { org_id, retry_after_ms }` when a tenant bursts past its allocation.
   - `TenantQuotaTracker::reserve_storage` and `release_storage` enforce per-tenant `max_vectors` and `max_memory_bytes` ceilings atomically before admitting new vectors.
3. **Tamper-Evident Hash-Chained Audit Log (`AuditLog`, `AuditEntry`, `AuditOutcome`):**
   - Each `AuditEntry` records `(timestamp_ms, seq_no, actor_key_id, org_id, action, outcome, prev_hash, entry_hash)` where `entry_hash` chains `prev_hash` and all entry fields via domain-separated SplitMix64 + dual-lane CRC32 mixing.
   - `AuditLog::verify_integrity()` recomputes the chain from genesis (`prev_hash = 0`) and returns `Err(corrupted_seq_no)` if any historical record is modified, re-hashed without updating the full suffix, reordered, or deleted.

## Alternatives Considered
- **Plain String Equality for API Keys:** Rejected because short-circuit byte comparisons leak prefix match length over high-resolution network timing measurements.
- **Global Node-Wide Rate Limiting Only:** Rejected because a single runaway tenant could exhaust the global token bucket and cause denial-of-service for well-behaved tenants.
- **Unchained Append-Only Audit Records:** Rejected because an attacker or misconfigured operator with write access to the log could alter or delete individual historical entries without invalidating subsequent entries.

## Consequences & Failure Modes
- **Pros:** Zero external cryptographic crate bloat, deterministic constant-time key verification, strict per-tenant blast-radius containment for QPS and memory, and O(N) verifiable tamper detection across the entire audit history.
- **Cons / Failure Modes:** In-memory token bucket and audit state on a single `SecurityGate` instance must be periodically checkpointed or replicated if cross-coordinator audit persistence is required across process restarts.
