# ADR-020: Gateway HTTP Request Smuggling Defense, Hardened JSON/Path Parsing, and Durable FP16 Collections
- **Status:** Accepted
- **Owner:** Justin Mafie

## Context
As VektorDist advances from `v0.3.0` into enterprise internet-facing deployments behind L7 load balancers, reverse proxies, and browser-based control consoles, four critical security and storage durability gaps must be closed in `vektordist-cluster`:

1. **HTTP/1.1 Desynchronization & Request Smuggling (`RFC 9112 §6.1 / §6.3`):** Reverse proxies and backend HTTP/1.1 parsers that disagree on `Transfer-Encoding: chunked` vs `Content-Length` framing (`CL.TE` / `TE.CL`) or duplicate `Content-Length` headers enable attackers to smuggle unauthorized requests across connection boundaries. Similarly, percent-encoded path segments (`/v1/collections/%2e%2e%2fadmin` or `%00` null bytes) must be decoded and validated before collection routing.
2. **String-Shadowed JSON Keys & Non-Finite Float Payloads:** Substring-based JSON field lookup (`json.find("\"id\"")`) allows an earlier untrusted string value (e.g., `{"note": "\"id\": 999999", "id": 42}`) to shadow the real object key, while non-finite float tokens (`NaN`, `Infinity`, `1e999`) must be rejected at the HTTP wire boundary before reaching HNSW distance kernels.
3. **Browser Security Posture & CORS Preflight:** Browser-hosted operator consoles (`/` and `/console`) require deterministic `OPTIONS` preflight support, configurable `Access-Control-Allow-Origin` headers, and defense-in-depth response headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store`).
4. **Durable `FP16` Multi-Collection Persistence:** While `Fp16HnswIndex` (M16) provides calibration-free `2.0x` vector RAM compression with `>= 99%` `recall@10`, `CollectionRegistry` (M18) previously exposed only `F32`, `Sq8`, and `Bq` in-memory profiles without crash-safe disk persistence across gateway restarts.

## Decision
1. **Strict RFC 9112 Framing, Percent-Decode Traversal Guard, and CORS (`crates/vektordist-cluster/src/gateway.rs`):**
   - Reject any HTTP request carrying a `Transfer-Encoding` header (`400 Bad Request`: `"Transfer-Encoding is not supported; use Content-Length"`).
   - Reject requests with non-numeric, negative, overflowing, or conflicting duplicate `Content-Length` headers (`400 Bad Request`: `"invalid or conflicting Content-Length header"`), as well as malformed HTTP request lines missing an `HTTP/1.x` version token or containing ASCII control characters.
   - Decode `{name}` path segments via zero-dependency `percent_decode_path_segment` (rejecting malformed `%XX` hex sequences and `%00` null bytes) and enforce `validate_collection_name` (`[A-Za-z0-9_-]{1,128}`) prior to dispatching collection routes.
   - Support `OPTIONS` preflight (`204 No Content`) universally and expose `HttpGatewayServer::with_cors_origin` + `HttpGatewayServer::with_collections_dir`.
2. **String-Aware JSON Scanner, Finite-Float Gate, and Default Security Headers (`crates/vektordist-cluster/src/telemetry.rs`):**
   - Replace naive substring key lookup in `find_json_key_value` with a lexical string-literal-aware scanner tracking `in_string` and backslash `escaped` states so embedded quotes inside string values can never impersonate object keys.
   - Enforce `val.is_finite()` inside `extract_json_f32_array` so `NaN`, `+-Infinity`, and exponent overflow (`1e999`) deterministically return `400 Bad Request` while preserving standard scientific notation (`1.25e-2`).
   - Emit `x-content-type-options: nosniff`, `x-frame-options: DENY`, `referrer-policy: no-referrer`, and `cache-control: no-store` on every gateway HTTP response unless explicitly overridden, and update `/openapi.json` to `v0.4.0` with `/` and `/console` endpoints.
3. **`CollectionQuantization::Fp16` & Atomic CRC32 Snapshot Persistence (`crates/vektordist-cluster/src/collections.rs`):**
   - Wire `CollectionQuantization::Fp16` (`"fp16" | "f16" | "half"`) into `TenantCollectionIndex::Fp16(Fp16HnswIndex)`.
   - Retain canonical `f32` tenant coordinates alongside quantized HNSW graphs and implement `CollectionRegistry::save_to_dir` / `load_from_dir` / `with_persistence_dir` using binary `collections.vdcl` (`magic: b"VDCL"`, `version: 1`, collection schemas, `next_seq`, live `(org_id, vector_id, f32_coords)` entries, and trailing `crc32fast` IEEE CRC32) written via `.vdcl.tmp` -> `sync_all()` -> atomic `std::fs::rename`.
4. **Control-Plane Modularization (`raft.rs` + `rebalance.rs`):**
   - Extract `ShardMove`, `RebalancePlan`, `compute_rebalance_plan`, `execute_online_rebalance`, and shard/WAL gRPC transfer helpers into `crates/vektordist-cluster/src/rebalance.rs` while re-exporting public types in `raft.rs` for 100% API compatibility.

## Alternatives Considered
- **Adding `hyper` + `serde_json` to `vektordist-cluster`:** Rejected to preserve VektorDist's zero-bloat dependency footprint and full control over socket Slowloris deadlines, bounded body draining, and deterministic error framing.
- **Persisting Quantized `u16` / `u8` / `1-bit` Codes Instead of Canonical `f32` Vectors in `collections.vdcl`:** Rejected because storing exact canonical `f32` coordinates guarantees lossless deterministic HNSW reconstruction across restarts without cumulative re-quantization drift and allows future schema inspection/export.

## Consequences & Failure Modes
- **Full-Snapshot Write Amplification on Auto-Persist:** `CollectionRegistry::with_persistence_dir` rewrites `collections.vdcl.tmp` + `fsync` + atomic `rename` on collection mutations. This guarantees `O(1)` single-file crash consistency (either the previous valid CRC32 snapshot or the new CRC32 snapshot is observed after `SIGKILL`) and is well suited for multi-collection namespace workloads up to hundreds of thousands of vectors; multi-million-vector distributed workloads continue to use sharded `W=2` WAL + `VDCK` checkpointing (`ShardEngine`).
- **Strict `Transfer-Encoding` Rejection:** Clients attempting chunked uploads without a known `Content-Length` receive an explicit `400 Bad Request` instructing them to send `Content-Length`, eliminating `CL.TE`/`TE.CL` ambiguity by construction.
