# ADR-017: Gateway Modularity, Bounded Telemetry, Slowloris/Connection Hardening, OpenAPI 3.1, and `WriteId` Idempotency
- **Status:** Accepted
- **Owner:** Justin Mafie

## Context
As `vektordist-cluster` evolved through Milestones 10–15, three production engineering gaps emerged:
1. **Gateway Modularity & Unbounded Telemetry Memory:** `crates/vektordist-cluster/src/gateway.rs` grew to `1,940` lines (combining HTTP transport, Prometheus exposition, W3C trace propagation, and the `ChaosCampaign` fault-injection harness) and stored every historical search latency sample in an unbounded `Vec<f64>`, causing linear memory growth and `O(N log N)` sort overhead on long-lived gateways.
2. **TCP Connection Exhaustion & Slowloris Stalls:** `spawn_http_gateway` spawned an unbounded Tokio task per accepted socket without a read deadline, exposing the gateway to file-descriptor/task exhaustion under connection floods and Slowloris partial-header/partial-body stalls. External API consumers also lacked a machine-readable OpenAPI specification.
3. **Duplicate WAL Frames on Client Retries:** Although every `WalRecord` already carried a caller-supplied `write_id: u64`, `NodeService::append_quorum` unconditionally assigned a new `seq_no` and appended a duplicate WAL frame whenever a client retried an `upsert`, `upsert_batch`, or `delete` after a transient network timeout or `503` retry.

## Decision
1. **Modular Split (`telemetry.rs`, `chaos.rs`, `gateway.rs`):**
   - Extracted `PrometheusMetrics`, `TraceSpan`, `BoundedLatencyWindow` (fixed `1,024`-sample `VecDeque<f64>` ring buffer), and `openapi_spec_json()` into `crates/vektordist-cluster/src/telemetry.rs`.
   - Extracted `ChaosStep`, `ChaosCampaignReport`, and `ChaosCampaign` into `crates/vektordist-cluster/src/chaos.rs`.
   - Preserved `pub use` re-exports in `gateway.rs` so 100% of existing callers compile unchanged while keeping `gateway.rs`, `telemetry.rs`, and `chaos.rs` each well below `1,000` lines.
2. **Connection Concurrency Cap, Slowloris Timeout & OpenAPI `3.1.0`:**
   - Added `max_connections: usize` (default `512`, `with_max_connections`) enforced via an `Arc<tokio::sync::Semaphore>` permit acquired before spawning each connection handler in `spawn_http_gateway`.
   - Added `read_timeout: Duration` (default `5s`, `with_read_timeout`) wrapping `read_full_http_request` in `tokio::time::timeout`; stalled connections receive `408 Request Timeout` (`{"status":"error","error":"request read timeout"}`) and close cleanly.
   - Added public `GET /openapi.json` (`HttpVektorClient::openapi_raw()`) serving an OpenAPI `3.1.0` JSON specification covering all health, metrics, data-plane, control-plane, audit, and collection routes.
3. **Bounded `WriteIdDeduplicator` in `ShardEngine` (`node.rs`):**
   - Added `WriteIdDeduplicator` (`capacity = 4,096`, `HashMap<WriteId, SeqNo>` + FIFO `VecDeque<WriteId>`) inside `ShardEngine`, populated during `ShardEngine::open` from replayed WAL records (`write_id != 0`) and updated in `ShardEngine::apply_records` on both primaries and followers.
   - Updated `NodeService::append_quorum` to short-circuit already-committed non-zero `write_id`s: if every record in an `AppendQuorumRequest` was already committed, the primary returns `STATUS_OK` with the original `assigned_seq_nos` immediately without writing duplicate WAL frames; mixed batches deduplicate per record while preserving 1-to-1 `assigned_seq_nos` alignment.

## Alternatives Considered
- **HDR Histogram External Crate for Latency Percentiles:** Adding a third-party dependency violates VektorDist's zero-bloat workspace discipline. A `1,024`-entry `VecDeque<f64>` caps memory at `8 KiB` per gateway and computes exact window `p50`/`p95`/`p99` in microseconds.
- **Unbounded `HashMap<WriteId, SeqNo>` per Shard:** An unbounded map leaks `16+` bytes per write indefinitely across millions of mutations. A `4,096`-entry FIFO cache per shard (`65,536` entries across 16 shards) covers realistic retry windows in `< 2 MiB` total RAM while treating `write_id == 0` as explicitly non-idempotent.

## Consequences & Failure Modes
- **Retry Window Eviction After `4,096` Subsequent Shard Writes:** If a client retries an ancient `write_id` after more than `4,096` newer writes have landed on that specific shard, the entry will have been FIFO-evicted and the retry will execute as a fresh idempotent overwrite on `VectorId` with a new `seq_no`. Within normal retry backoff windows (`ms` to `s`), retries are guaranteed zero-WAL-amplification idempotency.
- **Accept-Loop Backpressure at `max_connections`:** When `512` concurrent requests are active, additional sockets queue in the OS TCP accept backlog until an in-flight request completes or hits `read_timeout` (`408`), preventing out-of-memory task storms.
