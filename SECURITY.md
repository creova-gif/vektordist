# Security Policy

VektorDist is built for multi-tenant, zero-trust production retrieval workloads. We treat authentication bypasses, cross-tenant data leakage, audit-chain forgery, memory-safety bugs in SIMD/segment paths, and silent durability corruption as high-severity security issues.

---

## Supported Versions

Security patches are published for the latest minor release series:

| Version | Supported |
| --- | --- |
| `0.3.x` | Yes |
| `0.2.x` | Security fixes only |
| `0.1.x` | No |
| `< 0.1.0` | No |

---

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues, discussions, or pull requests.**

Instead, use **GitHub Private Vulnerability Reporting** to submit a confidential advisory directly to the maintainers:

- **Report URL:** <https://github.com/creova-gif/vektordist/security/advisories/new>

### What to Include in Your Report

To help us reproduce and remediate the issue quickly, please include:

1. **Affected Component:** Crate and module path (e.g., `vektordist_cluster::security`, `vektordist_cluster::gateway`, `vektordist_core::simd`, `vektordist_core::mmap`).
2. **Impact Classification:** Cross-tenant read/write bypass, RBAC privilege escalation, constant-time MAC side-channel, DoS/quota bypass, memory safety (`unsafe`), or silent data corruption.
3. **Deterministic Reproduction:** Exact CLI command, HTTP request (`curl`), or minimal Rust test case (with `--seed` if applicable), plus OS/CPU architecture (`aarch64` / `x86_64`) and `rustc --version`.
4. **Suggested Mitigation (Optional):** Any proposed fix or architectural hardening.

### Response & Disclosure Timeline

- **Initial Acknowledgement:** Within **48 hours** of submission.
- **Triage & Severity Assessment:** Within **7 calendar days**, including confirmation of affected versions and a remediation timeline.
- **Coordinated Release:** Once a fix and regression test are verified, we publish a patched release, credit reporters (unless anonymity is requested), and publish the GitHub Security Advisory.

---

## Security Architecture Summary

VektorDist enforces defense-in-depth across its gateway, cluster control plane, and storage engine. For full trust-boundary analysis and STRIDE mitigations, see [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) and [`ADR-008`](docs/adr/008-m8-enterprise-security-rbac-quotas-and-audit.md).

### 1. Constant-Time Scoped Authentication & RBAC (`SecurityGate`)
- **Keyed Digest Tokens (`vdk.<key_id>.<hex_tag>`):** Raw bearer tokens are returned only once at issuance (`SecurityGate::issue_api_key` / `vektordist keygen`). Server memory stores only a 32-byte keyed digest derived from the 256-bit master secret.
- **Side-Channel Resistance (`constant_time_eq_32`):** Digest verification accumulates XOR differences across all 32 bytes without short-circuit branching and performs an identical dummy digest comparison when `key_id` is unknown, preventing network timing attacks.
- **Monotonic RBAC (`TenantRole`):** Every API key is bound to an explicit role (`Reader < Writer < Admin`) enforced prior to executing search, mutation, or administrative endpoints.

### 2. Strict `OrgId` Multi-Tenant Isolation & Blast-Radius Quotas
- **Tenant Binding:** Every authenticated key is cryptographically bound to a single `OrgId`. Requests targeting a mismatched `X-Org-Id` fail closed with `SecurityError::CrossTenantDenied`.
- **Partitioned Caching & Retrieval:** Dense/hybrid indices and `SemanticCache` buckets (`(OrgId, filter_hash)`) isolate tenant embeddings structurally so neither ANN traversal nor cache lookups can cross tenant boundaries.
- **Per-Tenant Rate & Storage Quotas (`TenantQuotaTracker`):** Independent per-tenant token buckets (`max_qps`, `burst_capacity`) and atomic storage reservations (`max_vectors`, `max_memory_bytes`) contain noisy-neighbor bursts (`429 Too Many Requests` / `StorageQuotaExceeded`).

### 3. Tamper-Evident Hash-Chained Audit Log (`AuditLog`)
- Every `Allowed` and `Denied` authorization check appends an `AuditEntry` whose 64-bit `entry_hash` chains `prev_hash` (anchored at genesis `0`) with `(timestamp_ms, seq_no, actor_key_id, org_id, action, outcome)`.
- `AuditLog::verify_integrity()` recomputes the full chain from genesis and pinpoints the exact `corrupted_seq_no` if any historical entry is modified, deleted, truncated, or reordered.

### 4. Fail-Closed Storage Integrity (`WalLog`, `PersistentVectorSegment`, `SnapshotManager`)
- Every write-ahead log record (`*.wal`), persistent zero-copy vector segment (`*.vdsg`), and point-in-time cluster snapshot carries IEEE CRC32 checksums.
- Mid-file bit-rot or tampered snapshot payloads fail closed with `VektorError::WalCorruption` rather than silently serving corrupt embeddings.

### 5. Code & Container Hardening
- `#![deny(clippy::unwrap_used, clippy::expect_used)]` across all production targets prevents untrusted inputs from triggering panics.
- Continuous `cargo audit --deny warnings` runs on every push and pull request in CI.
- Production container images (`deploy/Dockerfile`) run as `nonroot:nonroot` (`UID/GID 65532`) on `gcr.io/distroless/cc-debian12:nonroot` with no shell or package manager.
