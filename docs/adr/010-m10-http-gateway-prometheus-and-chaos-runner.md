# ADR-010: HTTP/1.1 REST & Prometheus `/metrics` Gateway, W3C `traceparent` Distributed Trace Spans, and Automated Chaos Campaign Runner
- **Status:** Accepted
- **Owner:** Justin Mafie

## Context
While VektorDist's internal control and data planes communicate over low-overhead binary gRPC (`TopologyRpc` and `ShardReplicaRpc`), production operators, cloud load balancers, and observability pipelines require three additional capabilities:
1. **Standardized HTTP/1.1 Ingress & Health Probes:** Kubernetes readiness/liveness probes (`GET /healthz`) and language-agnostic application clients (`POST /v1/upsert`, `POST /v1/search`) require an HTTP/1.1 REST gateway without forcing callers to compile Protobuf stubs.
2. **Prometheus Telemetry & W3C Distributed Tracing:** Scraping live cluster health (`active_nodes`, `indexed_vectors`, `search_latency_p50/p95/p99_ms`, `recall_at_10`, `acked_writes_total`, `acked_writes_lost_total`) in standard Prometheus text exposition format (`0.0.4`) and propagating W3C Trace Context (`traceparent: 00-<trace_id>-<span_id>-01`) across gateway requests are essential for tail-latency attribution and SLA alerting.
3. **Falsifiable Multi-Step Chaos Verification:** Validating `W=2` quorum durability (`0` acknowledged writes lost) and `recall@10 >= 0.95` across concurrent workloads, abrupt node failure (`SIGKILL`), union WAL reconciliation, and replacement node catch-up (`RECOVERING` -> `HEALTHY`) requires a deterministic, programmable `ChaosCampaign` runner.

## Decision
1. **Zero-Dependency HTTP/1.1 & Prometheus Gateway (`vektordist_cluster::gateway::HttpGatewayServer`):**
   - Implemented directly on `tokio::net::TcpListener` (`spawn_http_gateway`) with framing-aware `Content-Length` request parsing and zero additional HTTP framework crates.
   - Exposes `GET /healthz` (JSON cluster status), `GET /metrics` (`text/plain; version=0.0.4` Prometheus exposition via `PrometheusMetrics`), `POST /v1/upsert`, and `POST /v1/search`.
   - Parses incoming W3C `traceparent` headers (`TraceSpan::parse_traceparent`), generates child spans (`TraceSpan::child_span`) preserving the caller's 128-bit `trace_id`, and attaches `x-trace-id`, `traceparent`, and `x-latency-ms` headers to every HTTP response.
2. **Programmable Chaos Campaign Runner (`vektordist_cluster::gateway::ChaosCampaign`):**
   - Executes a declarative sequence of `ChaosStep` variants (`SeedVectors`, `ConcurrentWorkload`, `KillNode`, `VerifyZeroWriteLoss`, `AddReplacementNode`, `AssertRecallAndLatencySla`) against live `CoordinatorService` and `NodeService` gRPC instances.
   - Maintains an external acknowledged-write ledger (`WriteId`) and an `ExactIndex` brute-force oracle to produce a falsifiable `ChaosCampaignReport` (`steps_executed`, `acked_writes_verified`, `acked_writes_lost`, `measured_recall_at_10`, `measured_p95_ms`, `sla_passed`).

## Alternatives Considered
- **Adding Heavy Third-Party Web Frameworks (`axum` / `actix-web` / `opentelemetry`):** Introduces dozens of transitive dependencies and increases compile times for four concise endpoints (`/healthz`, `/metrics`, `/v1/upsert`, `/v1/search`) and W3C `traceparent` header parsing.
- **External Shell-Based Chaos Scripts:** Bash scripts orchestrating `kill -9` processes cannot deterministically track every in-flight `WriteId` at the exact millisecond a quorum acknowledgment returns or compute exact brute-force `recall@10` in-memory.

## Consequences & Failure Modes
- **Lightweight JSON Parser Scope:** The built-in JSON parser in `gateway.rs` is purpose-built for flat `/v1/upsert` and `/v1/search` numeric payloads (`id`, `write_id`, `top_k`, `ef_search`, `vector`, `query`); complex nested metadata filters continue to use the typed gRPC/Rust API.
- **Bounded Latency Window:** Gateway latency percentile calculations (`p50`, `p95`, `p99`) sort observed search latencies recorded by the gateway instance; high-throughput deployments can periodically reset or bound the sample buffer.
