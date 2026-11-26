# ADR-019: OpenAPI 3.1 Contract, Zero-Dependency Python SDK, and Turnkey Docker Compose Stack
- **Status:** Accepted
- **Owner:** Justin Mafie

## Context
With VektorDist `v0.3.0` delivering `VisitedPool` zero-allocation HNSW search & IEEE 754 `FP16` 2x quantization (M16), bounded gateway concurrency/Slowloris protection & `WriteIdDeduplicator` idempotency (M17), and named Multi-Collection `/v1/collections/*` management (M18), AI/RAG application engineers and platform teams need three ecosystem pillars beyond the Rust `HttpVektorClient`:

1. **Authoritative Machine-Readable API Contract:** An OpenAPI `3.1.0` specification (`docs/openapi.yaml`) documenting every gateway route, RBAC requirement, W3C `traceparent` header, `X-Org-Id` tenant boundary, and error contract (`401`, `403`, `408`, `413`, `429`, `503`).
2. **Frictionless Python Client Ecosystem:** Python dominates embedding generation, RAG pipelines, and agentic orchestration, yet SDKs with compiled Rust/C++ bindings or heavy HTTP client dependencies frequently cause dependency conflicts in ML container images.
3. **Single-Command Local Multi-Container Cluster:** Operators evaluating VektorDist before deploying the 5-node Kubernetes `StatefulSet` (`deploy/k8s/statefulset.yaml`) need a turnkey `docker compose up` topology pairing `coord`, 3 persistent `node` replicas, `gateway`, and live Prometheus `5s` scraping.

## Decision
1. **Checked-In OpenAPI 3.1.0 Specification (`docs/openapi.yaml`):**
   - Document all gateway endpoints (`/healthz`, `/readyz`, `/metrics`, `/openapi.json`, `/v1/upsert`, `/v1/upsert_batch`, `/v1/search`, `/v1/delete`, `/v1/topology`, `/v1/audit`, and `/v1/collections` + `/v1/collections/{name}` + `/upsert`, `/search`, `/delete`), `BearerAuth` (`vdk.<key_id>.<hex_tag>`), `X-Org-Id`, W3C `traceparent`, `Retry-After` rate-limit headers, and strict JSON request/response schemas.
2. **Zero-Dependency Strictly Typed Python SDK (`clients/python/`):**
   - Implement `VektorDistClient` (`clients/python/vektordist_client.py` + `py.typed`) using only Python 3.11+ / 3.12+ standard-library modules (`urllib.request`, `urllib.error`, `urllib.parse`, `json`, `os`, `time`, `dataclasses`, `typing`), verified under `mypy --strict`.
   - Automatically inject `Authorization: Bearer ...`, `X-Org-Id`, and cryptographically random 55-character W3C `traceparent` (`00-<32hex>-<16hex>-01`) headers when callers do not pass an upstream parent span.
   - Implement bounded exponential backoff for `429 Too Many Requests` (honoring `Retry-After` capped to `2.0s`) and `503 Service Unavailable`, mapping terminal failures to typed exceptions (`AuthenticationError`, `AuthorizationError`, `PayloadTooLargeError`, `RateLimitError`, `VektorDistError`).
   - Verify header propagation, retry semantics, and exception mapping deterministically against a live ephemeral `ThreadingHTTPServer` (`127.0.0.1:0`) in `clients/python/test_vektordist_client.py`.
3. **Turnkey Docker Compose + Prometheus Stack (`deploy/docker-compose.yml` & `deploy/prometheus.yml`):**
   - Provision `coord` (`0.0.0.0:50051`), three shard replicas (`node-1..3` on `0.0.0.0:50052` backed by named persistent volumes `node_1_data..node_3_data`, read-only rootfs, and `cap_drop: [ALL]`), `gateway` (`8080:8080` probing `/readyz`), and `prom/prometheus:v2.54.1` (`9090:9090`) scraping `gateway:8080/metrics` every `5s`.

## Alternatives Considered
- **`httpx` or `requests` Runtime Dependency in Python SDK:** Rejected for the core client because zero external dependencies allow instant vendoring or import inside air-gapped, serverless, and pinned PyTorch/CUDA environments without version solver conflicts.
- **gRPC-Direct Python Client (`grpcio` + `protobuf`):** Rejected as the primary SDK surface because `grpcio` native wheel/ABI mismatches are a recurring friction point across Python minor releases, whereas the stateless zero-trust HTTP/1.1 gateway already handles `GrpcChannelPool` multiplexing, `W=2` quorum fan-out, RBAC, rate limiting, and audit logging centrally.

## Consequences & Failure Modes
- **Synchronous `urllib.request` Connection Model:** Each Python `VektorDistClient` call opens a bounded-timeout HTTP/1.1 request suitable for RAG retrieval and `upsert_batch` bulk ingestion; ultra-high-QPS async Python callers can wrap calls via `asyncio.to_thread` or batch writes with `upsert_batch`.
- **Bounded `Retry-After` Cap:** Capping `429` sleep duration (`MAX_RETRY_BACKOFF_CAP_SECS = 2.0`) prevents an aggressive quota window from stalling caller worker threads indefinitely while still exposing the server's exact `retry_after_secs` on `RateLimitError` when `max_retries` exhausts.
