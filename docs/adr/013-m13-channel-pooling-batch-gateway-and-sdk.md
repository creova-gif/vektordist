# ADR-013: HTTP/2 gRPC Channel Pooling, Batched REST Quorum Ingestion, Gateway DoS/Drain Guards, and Typed Rust Client SDK
- **Status:** Accepted
- **Owner:** Justin Mafie

## Context
In Milestones 1 through 11, `ClusterClient` and `NodeService` opened a fresh TCP + HTTP/2 `tonic` connection on every individual shard RPC (`append_quorum`, `replicate_wal`, `search_shard`, `get_wal_records`, `get_topology`). Under sustained multi-thousand QPS workloads, per-RPC connection setup causes ephemeral port exhaustion, kernel socket churn, and elevated tail latency. In addition, enterprise deployments require:
1. High-throughput multi-shard REST ingestion (`POST /v1/upsert_batch`) amortizing WAL `fsync` cost (`batched_fsync = true`) across logical shards.
2. Bounded memory consumption on `ShardEngine::records_by_seq` across repeated vector overwrites.
3. Ingress HTTP body size enforcement (`413 Payload Too Large`) against memory-exhaustion DoS attacks and a zero-downtime Kubernetes readiness probe (`GET /readyz`) with explicit drain signaling.
4. An ergonomic, zero-dependency typed async Rust SDK (`HttpVektorClient`) handling auth headers, tenant scoping, W3C `traceparent` propagation, and bounded transient retries.

## Decision
1. **HTTP/2 Channel Pooling (`GrpcChannelPool` in `crates/vektordist-cluster/src/client.rs` & `node.rs`):**
   - Introduced `GrpcChannelPool` backed by `Arc<RwLock<HashMap<String, tonic::transport::Channel>>>`, normalizing endpoints via `format_endpoint`, setting `connect_timeout(500ms)` + `tcp_nodelay(true)`, multiplexing concurrent HTTP/2 streams per peer, and calling `evict(endpoint)` immediately on any RPC timeout or transport error so retries establish clean connections.
   - Wired `GrpcChannelPool` into both `ClusterClient` and `NodeService` follower/catch-up replication.
2. **Batched Quorum Ingestion & In-Memory WAL History Compaction:**
   - Added `ClusterClient::upsert_batch` and `POST /v1/upsert_batch` (`TenantRole::Writer`, `"http.upsert_batch"`): validates dimensions and finite floats up front, partitions vectors across shards via `ClusterConfig::shard_for`, and dispatches concurrent `W=2` batched quorum writes (`batched_fsync = true`) with automatic topology-refresh retry on `STATUS_WRONG_EPOCH` or transport failure.
   - Added `ShardEngine::compact_history(&mut self)` to deduplicate `records_by_seq` in-place, retaining only the latest mutation (`Upsert` or `Delete` tombstone) per `VectorId` in ascending `seq_no` order.
3. **HTTP Payload DoS Protection & Kubernetes `/readyz` Graceful Drain (`crates/vektordist-cluster/src/gateway.rs`):**
   - Added `max_body_bytes` (default `16 MiB`, configurable via `with_max_body_bytes`) enforced both in the streaming TCP reader (avoiding unbounded heap allocation) and in `handle_http_request`, returning `413 Payload Too Large`.
   - Added `draining: Arc<AtomicBool>` (`set_draining`, `is_draining`) and credential-free `GET /readyz` verifying `!is_draining()`, `active_nodes > 0`, and zero `UnderQuorum` shards (`200 OK` vs `503 Service Unavailable`).
4. **Typed Async Client SDK (`crates/vektordist-cluster/src/sdk.rs`):**
   - Implemented `HttpVektorClient` providing `healthz`, `readyz`, `upsert`, `upsert_batch`, `search`, `delete`, `topology_raw`, `audit_raw`, and `send_raw`, with W3C `traceparent` injection and bounded exponential backoff on transient `429`/`503` responses.

## Alternatives Considered
- **Unpooled `connect_lazy()` Per Call:** Still allocates new HTTP/2 state machines and TCP sockets per call when the `Channel` is dropped at the end of each helper function. Explicit `HashMap<String, Channel>` caching shares a single multiplexed HTTP/2 transport per peer.
- **External `reqwest`/`hyper` Dependency for `HttpVektorClient`:** Increases compile time and dependency surface; building directly atop `gateway::send_http_request` and `tokio::net::TcpStream` preserves zero external HTTP dependencies.

## Consequences & Failure Modes
- **Stale Pooled Connections After Peer Restart:** Handled by calling `pool.evict(addr)` whenever an RPC returns a transport error or deadline timeout, ensuring the immediate retry reconnects cleanly.
- **Graceful Rolling Upgrades:** Kubernetes `preStop` hooks can toggle `set_draining(true)` so `GET /readyz` transitions to `503` and kube-proxy removes the pod from service endpoints while in-flight requests complete cleanly.
