# ADR-011: Zero-Trust Authenticated HTTP/1.1 Gateway, Quorum REST Delete, and Audit Chain Verification
- **Status:** Accepted
- **Owner:** Justin Mafie

## Context
Milestone 8 introduced `SecurityGate` (constant-time scoped API-key authentication, `Reader`/`Writer`/`Admin` RBAC, per-tenant token-bucket rate limiting, and a tamper-evident hash-chained `AuditLog`), while Milestone 10 introduced `HttpGatewayServer` and `ClusterClient` over HTTP/1.1 and gRPC. To complete the enterprise zero-trust data plane without breaking existing unauthenticated internal deployments:
1. **Zero-Trust Ingress Enforcement:** Protected HTTP/1.1 endpoints must authenticate `Authorization: Bearer <token>`, isolate tenants via `X-Org-Id`, enforce least-privilege RBAC roles and QPS token buckets, and record every allowed or denied call in the tamper-evident `AuditLog`, while keeping `GET /healthz` and `GET /metrics` open for Kubernetes probes and Prometheus scrapers.
2. **Quorum-Durable Vector Deletion:** Application clients require a direct-to-primary `W=2` quorum `delete` path in `ClusterClient` (`WIRE_OP_DELETE`) exposed via `POST /v1/delete` with automatic `WRONG_EPOCH` topology refresh and retry.
3. **Live Control-Plane & Security Introspection:** Operators and automated compliance checks require `GET /v1/topology` (`Reader`) for cluster epoch/quorum inspection and `GET /v1/audit` (`Admin`) to cryptographically verify the live hash chain from genesis.

## Decision
1. **Quorum Delete in `ClusterClient` (`crates/vektordist-cluster/src/client.rs`):**
   - Added `ClusterClient::delete(id, write_id)` routing `WIRE_OP_DELETE` (empty `vector_f32_le`) directly to `shard_for(id)`'s primary with `UnderQuorum` pre-checks, `WRITE_QUORUM_TIMEOUT`, and a 2-attempt `WRONG_EPOCH`/transport-error topology-refresh retry loop matching `upsert`.
2. **Optional Zero-Trust `SecurityGate` in `HttpGatewayServer` (`crates/vektordist-cluster/src/gateway.rs`):**
   - Preserved 100% backward compatibility when `security_gate` is `None` (`HttpGatewayServer::new` and `HttpGatewayServer::with_metrics`) and added `HttpGatewayServer::with_security_gate(Arc<Mutex<SecurityGate>>)`.
   - Added `POST /v1/delete` (`TenantRole::Writer`), `GET /v1/topology` (`TenantRole::Reader`), and `GET /v1/audit` (`TenantRole::Admin`), alongside existing `POST /v1/upsert` (`TenantRole::Writer`) and `POST /v1/search` (`TenantRole::Reader`).
   - Mapped `SecurityError` exhaustively to HTTP semantics: `401 Unauthorized`, `403 Forbidden` (cross-tenant or insufficient role), `429 Too Many Requests` (with RFC 9110 `Retry-After` header in integer seconds `>= 1`), `507 Insufficient Storage`, and `500 Internal Server Error` (on `AuditIntegrityViolation`).

## Alternatives Considered
- **Enforcing Authentication on `/healthz` and `/metrics`:** Breaks unauthenticated kubelet TCP/HTTP readiness probes and standard Prometheus scrape configs without sidecar token injection; keeping health and telemetry endpoints read-only and credential-free follows cloud-native operational practice.
- **Holding `tokio::sync::Mutex` Across Shard RPCs:** `SecurityGate` authorization and `AuditLog` verification are pure in-memory CPU operations (`< 1 us`). Acquiring a synchronous `std::sync::Mutex` and dropping the guard strictly before any `.await` avoids async lock overhead and preserves `Send` futures across multi-threaded Tokio workers.

## Consequences & Failure Modes
- **Audit Log Growth:** Every authenticated or rejected HTTP request appends one `AuditEntry` to the in-memory hash chain (`SecurityGate::audit_log`); long-running gateways should periodically checkpoint and rotate verified audit segments to durable storage.
- **Logical Vector Counter Estimation:** `POST /v1/delete` decrements the gateway's `indexed_vectors` gauge via `saturating_sub(1)`, and subsequent `/healthz` or `/metrics` scrapes reconcile `indexed_vectors` from authoritative node heartbeats (`live_vectors / replication_factor`).
