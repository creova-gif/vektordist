//! Zero-Trust HTTP/1.1 REST, Prometheus `/metrics`, OpenAPI `3.1.0` `/openapi.json`, and
//! Multi-Collection Namespace Gateway built directly on `tokio::net::TcpListener` with bounded
//! concurrency (`Semaphore`), Slowloris read timeouts (`408`), and bounded latency histograms.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use vektordist_core::VektorError;

use crate::client::ClusterClient;
use crate::collections::CollectionRegistry;
use crate::security::{OrgId, SecurityError, SecurityGate, TenantRole};
use crate::telemetry::{
    build_http_response, build_http_response_with_headers, escape_json_str, extract_json_f32_array,
    extract_json_u64, extract_json_upsert_batch, format_hits_json, openapi_spec_json,
    parse_http_response, split_http_head_and_body,
};

pub use crate::chaos::{ChaosCampaign, ChaosCampaignReport, ChaosStep};
pub use crate::telemetry::{BoundedLatencyWindow, PrometheusMetrics, TraceSpan};

/// Default maximum HTTP request payload size (`16 MiB`) enforced before parsing JSON bodies.
pub const DEFAULT_MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Default per-connection HTTP request read timeout (`5s`) protecting against Slowloris stalls.
pub const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(5);

/// Default maximum concurrent in-flight TCP connections accepted by [`spawn_http_gateway`].
pub const DEFAULT_MAX_CONNECTIONS: usize = 512;

/// Parsed HTTP/1.1 response returned by [`send_http_request`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status_code: u16,
    pub headers: BTreeMap<String, String>,
    pub body: String,
}

/// Zero-dependency HTTP/1.1 REST and Prometheus observability gateway backed by `ClusterClient`
/// and `CollectionRegistry`, with optional zero-trust `SecurityGate` enforcement.
#[derive(Clone)]
pub struct HttpGatewayServer {
    client: ClusterClient,
    metrics: Arc<RwLock<PrometheusMetrics>>,
    latency_window: Arc<RwLock<BoundedLatencyWindow>>,
    collections: Arc<CollectionRegistry>,
    security_gate: Option<Arc<Mutex<SecurityGate>>>,
    max_body_bytes: usize,
    read_timeout: Duration,
    max_connections: usize,
    draining: Arc<AtomicBool>,
}

impl HttpGatewayServer {
    /// Create a new `HttpGatewayServer` wrapping `client` in unauthenticated mode.
    #[must_use]
    pub fn new(client: ClusterClient) -> Self {
        Self::with_metrics(client, PrometheusMetrics::default())
    }

    /// Create a new `HttpGatewayServer` with an initial `PrometheusMetrics` state.
    #[must_use]
    pub fn with_metrics(client: ClusterClient, initial: PrometheusMetrics) -> Self {
        Self {
            client,
            metrics: Arc::new(RwLock::new(initial)),
            latency_window: Arc::new(RwLock::new(BoundedLatencyWindow::default())),
            collections: Arc::new(CollectionRegistry::new()),
            security_gate: None,
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            read_timeout: DEFAULT_READ_TIMEOUT,
            max_connections: DEFAULT_MAX_CONNECTIONS,
            draining: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Attach a zero-trust `SecurityGate` to enforce bearer token authentication, tenant isolation,
    /// RBAC privilege checks, token-bucket rate limiting, and audit logging on `/v1/*` routes.
    #[must_use]
    pub fn with_security_gate(mut self, gate: Arc<Mutex<SecurityGate>>) -> Self {
        self.security_gate = Some(gate);
        self
    }

    /// Configure the maximum allowed HTTP request body size in bytes (default `16 MiB`).
    #[must_use]
    pub fn with_max_body_bytes(mut self, max_bytes: usize) -> Self {
        self.max_body_bytes = max_bytes;
        self
    }

    /// Configure the per-connection socket read timeout (default `5s`).
    #[must_use]
    pub fn with_read_timeout(mut self, timeout: Duration) -> Self {
        self.read_timeout = timeout;
        self
    }

    /// Configure the maximum concurrent TCP connections handled by `spawn_http_gateway` (default `512`).
    #[must_use]
    pub fn with_max_connections(mut self, max_conns: usize) -> Self {
        self.max_connections = max_conns.max(1);
        self
    }

    /// Share an explicit `CollectionRegistry` across gateway instances.
    #[must_use]
    pub fn with_collections(mut self, collections: Arc<CollectionRegistry>) -> Self {
        self.collections = collections;
        self
    }

    /// Borrow the multi-collection namespace registry hosted by this gateway.
    #[must_use]
    pub fn collections(&self) -> &Arc<CollectionRegistry> {
        &self.collections
    }

    /// Current number of latency samples retained in the bounded sliding window.
    #[must_use]
    pub fn latency_sample_count(&self) -> usize {
        self.latency_window.read().map_or(0, |w| w.len())
    }

    /// Toggle graceful Kubernetes drain state (`true` causes `GET /readyz` to return `503`).
    pub fn set_draining(&self, draining: bool) {
        self.draining.store(draining, Ordering::SeqCst);
    }

    /// Return whether this gateway is currently marked as draining for Kubernetes readiness probes.
    #[must_use]
    pub fn is_draining(&self) -> bool {
        self.draining.load(Ordering::SeqCst)
    }

    /// Return a point-in-time snapshot of `PrometheusMetrics`.
    #[must_use]
    pub fn metrics_snapshot(&self) -> PrometheusMetrics {
        self.metrics.read().map(|g| g.clone()).unwrap_or_default()
    }

    /// Mutate the gateway's `PrometheusMetrics` in place.
    pub fn update_metrics<F: FnOnce(&mut PrometheusMetrics)>(&self, update_fn: F) {
        if let Ok(mut guard) = self.metrics.write() {
            update_fn(&mut guard);
        }
    }

    /// Fold a completed `ChaosCampaignReport` into the gateway's Prometheus telemetry.
    pub fn apply_chaos_report(&self, report: &ChaosCampaignReport) {
        self.update_metrics(|m| {
            m.acked_writes_total = m
                .acked_writes_total
                .max(report.acked_writes_verified as u64);
            m.acked_writes_lost_total = report.acked_writes_lost as u64;
            m.recall_at_10 = report.measured_recall_at_10;
            if report.measured_p95_ms > 0.0 {
                m.search_latency_p95_ms = report.measured_p95_ms;
            }
        });
    }

    /// Refresh live cluster node counts and vector counts from the coordinator topology.
    pub async fn refresh_cluster_telemetry(&self) {
        let topo = match self.client.refresh_topology().await {
            Ok(t) => t,
            Err(_) => self.client.cached_topology(),
        };
        let configured_nodes = topo.nodes.len() as u32;
        let active_nodes = topo.nodes.iter().filter(|n| n.alive).count() as u32;
        let total_replica_vectors: u64 = topo
            .nodes
            .iter()
            .filter(|n| n.alive)
            .map(|n| n.live_vectors)
            .sum();
        let rf = u64::from(topo.replication_factor.max(1));
        let logical_vectors = total_replica_vectors / rf;

        let (p50, p95, p99) = self
            .latency_window
            .read()
            .map_or((0.0, 0.0, 0.0), |win| win.percentiles());

        self.update_metrics(|m| {
            m.active_nodes = active_nodes;
            m.configured_nodes = configured_nodes;
            if logical_vectors > 0 {
                m.indexed_vectors = logical_vectors;
            }
            if p95 > 0.0 {
                m.search_latency_p50_ms = p50;
                m.search_latency_p95_ms = p95;
                m.search_latency_p99_ms = p99;
            }
        });
    }

    /// Handle a raw HTTP/1.1 request and return a complete HTTP/1.1 response string.
    pub async fn handle_http_request(&self, raw_request: &str) -> String {
        let start = Instant::now();
        let (header_block, body) = split_http_head_and_body(raw_request);
        let mut lines = header_block.lines();
        let request_line = lines.next().unwrap_or_default().trim();
        let mut parts = request_line.split_whitespace();
        let method = parts.next().unwrap_or("GET");
        let raw_path = parts.next().unwrap_or("/");
        let path = raw_path.split('?').next().unwrap_or(raw_path);

        let mut req_headers: BTreeMap<String, String> = BTreeMap::new();
        for line in lines {
            if let Some((k, v)) = line.split_once(':') {
                req_headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
            }
        }

        let span = req_headers
            .get("traceparent")
            .and_then(|tp| TraceSpan::parse_traceparent(tp).ok())
            .map_or_else(TraceSpan::generate, |parent| parent.child_span());

        let declared_len = parse_content_length(header_block);
        if declared_len > self.max_body_bytes || body.len() > self.max_body_bytes {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            return build_http_response(
                413,
                "Payload Too Large",
                "application/json",
                &span,
                latency_ms,
                "{\"status\":\"error\",\"error\":\"payload exceeds max_body_bytes\"}",
            );
        }

        match (method, path) {
            ("GET", "/healthz") => {
                self.refresh_cluster_telemetry().await;
                let snap = self.metrics_snapshot();
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                let json = format!(
                    "{{\"status\":\"ok\",\"active_nodes\":{},\"configured_nodes\":{},\"acked_writes_lost\":{}}}",
                    snap.active_nodes, snap.configured_nodes, snap.acked_writes_lost_total
                );
                build_http_response(200, "OK", "application/json", &span, latency_ms, &json)
            }
            ("GET", "/readyz") => self.handle_readyz(&span, start).await,
            ("GET", "/metrics") => {
                self.refresh_cluster_telemetry().await;
                let text = self.metrics_snapshot().render_prometheus_text();
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                build_http_response(
                    200,
                    "OK",
                    "text/plain; version=0.0.4; charset=utf-8",
                    &span,
                    latency_ms,
                    &text,
                )
            }
            ("GET", "/openapi.json") => build_http_response(
                200,
                "OK",
                "application/json",
                &span,
                start.elapsed().as_secs_f64() * 1000.0,
                openapi_spec_json(),
            ),
            ("POST", "/v1/upsert") => match self.authorize_http_route(
                &req_headers,
                TenantRole::Writer,
                "http.upsert",
                &span,
                start,
            ) {
                Ok(_) => self.handle_upsert(body, &span, start).await,
                Err(resp) => resp,
            },
            ("POST", "/v1/upsert_batch") => match self.authorize_http_route(
                &req_headers,
                TenantRole::Writer,
                "http.upsert_batch",
                &span,
                start,
            ) {
                Ok(_) => self.handle_upsert_batch(body, &span, start).await,
                Err(resp) => resp,
            },
            ("POST", "/v1/delete") => match self.authorize_http_route(
                &req_headers,
                TenantRole::Writer,
                "http.delete",
                &span,
                start,
            ) {
                Ok(_) => self.handle_delete(body, &span, start).await,
                Err(resp) => resp,
            },
            ("POST", "/v1/search") => match self.authorize_http_route(
                &req_headers,
                TenantRole::Reader,
                "http.search",
                &span,
                start,
            ) {
                Ok(_) => self.handle_search(body, &span, start).await,
                Err(resp) => resp,
            },
            ("GET", "/v1/topology") => match self.authorize_http_route(
                &req_headers,
                TenantRole::Reader,
                "http.topology",
                &span,
                start,
            ) {
                Ok(_) => self.handle_topology(&span, start).await,
                Err(resp) => resp,
            },
            ("GET", "/v1/audit") => match self.authorize_http_route(
                &req_headers,
                TenantRole::Admin,
                "http.audit",
                &span,
                start,
            ) {
                Ok(_) => self.handle_audit(&span, start),
                Err(resp) => resp,
            },
            ("POST", "/v1/collections") => match self.authorize_http_route(
                &req_headers,
                TenantRole::Admin,
                "http.collections.create",
                &span,
                start,
            ) {
                Ok(_) => self.collections.handle_http_create(body, &span, start),
                Err(resp) => resp,
            },
            ("GET", "/v1/collections") => match self.authorize_http_route(
                &req_headers,
                TenantRole::Reader,
                "http.collections.list",
                &span,
                start,
            ) {
                Ok(_) => {
                    let resp_json = self.collections.render_list_json(&span.trace_id);
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    build_http_response(
                        200,
                        "OK",
                        "application/json",
                        &span,
                        latency_ms,
                        &resp_json,
                    )
                }
                Err(resp) => resp,
            },
            _ => {
                if let Some(subpath) = path.strip_prefix("/v1/collections/") {
                    return self.route_collection_subpath(
                        method,
                        subpath,
                        &req_headers,
                        body,
                        &span,
                        start,
                    );
                }
                not_found_response(&span, start)
            }
        }
    }

    fn route_collection_subpath(
        &self,
        method: &str,
        subpath: &str,
        req_headers: &BTreeMap<String, String>,
        body: &str,
        span: &TraceSpan,
        start: Instant,
    ) -> String {
        if subpath.is_empty() {
            return not_found_response(span, start);
        }
        if method == "DELETE" && !subpath.contains('/') {
            return match self.authorize_http_route(
                req_headers,
                TenantRole::Admin,
                "http.collections.delete",
                span,
                start,
            ) {
                Ok(_) => self.collections.handle_http_drop(subpath, span, start),
                Err(resp) => resp,
            };
        }
        if method == "POST" {
            if let Some((col_name, action)) = subpath.split_once('/') {
                if col_name.is_empty() || action.contains('/') {
                    return not_found_response(span, start);
                }
                return match action {
                    "upsert" => match self.authorize_http_route(
                        req_headers,
                        TenantRole::Writer,
                        "http.collections.upsert",
                        span,
                        start,
                    ) {
                        Ok(org) => self
                            .collections
                            .handle_http_upsert(col_name, &org, body, span, start),
                        Err(resp) => resp,
                    },
                    "search" => match self.authorize_http_route(
                        req_headers,
                        TenantRole::Reader,
                        "http.collections.search",
                        span,
                        start,
                    ) {
                        Ok(org) => {
                            let (resp, lat_opt) = self
                                .collections
                                .handle_http_search(col_name, &org, body, span, start);
                            if let Some(lat) = lat_opt {
                                self.record_search_latency(lat);
                            }
                            resp
                        }
                        Err(resp) => resp,
                    },
                    "delete" => match self.authorize_http_route(
                        req_headers,
                        TenantRole::Writer,
                        "http.collections.delete_vec",
                        span,
                        start,
                    ) {
                        Ok(org) => self
                            .collections
                            .handle_http_delete_vec(col_name, &org, body, span, start),
                        Err(resp) => resp,
                    },
                    _ => not_found_response(span, start),
                };
            }
        }
        not_found_response(span, start)
    }

    fn record_search_latency(&self, latency_ms: f64) {
        if let Ok(mut win) = self.latency_window.write() {
            win.record(latency_ms);
            let (p50, p95, p99) = win.percentiles();
            self.update_metrics(|m| {
                m.search_latency_p50_ms = p50;
                m.search_latency_p95_ms = p95;
                m.search_latency_p99_ms = p99;
            });
        }
    }

    fn authorize_http_route(
        &self,
        req_headers: &BTreeMap<String, String>,
        required_role: TenantRole,
        action_label: &str,
        span: &TraceSpan,
        start: Instant,
    ) -> Result<OrgId, String> {
        let org_id = match req_headers.get("x-org-id") {
            Some(raw_org) => match OrgId::new(raw_org.as_str()) {
                Ok(org) => org,
                Err(_) => {
                    return Err(json_error_response(
                        400,
                        "Bad Request",
                        "invalid 'x-org-id' header",
                        span,
                        start,
                    ));
                }
            },
            None => OrgId::new("default").unwrap_or_else(|_| unreachable!()),
        };

        let Some(gate_arc) = &self.security_gate else {
            return Ok(org_id);
        };

        let token = extract_bearer_token(req_headers);
        let auth_result = {
            let mut guard = gate_arc
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.authorize_request(now_unix_ms(), token, &org_id, required_role, action_label)
        };

        match auth_result {
            Ok(_) => Ok(org_id),
            Err(err) => {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                Err(map_security_error_response(&err, span, latency_ms))
            }
        }
    }

    async fn handle_readyz(&self, span: &TraceSpan, start: Instant) -> String {
        self.refresh_cluster_telemetry().await;
        let draining = self.is_draining();
        let snap = self.metrics_snapshot();
        let topo = self.client.cached_topology();
        let under_quorum_shards = topo
            .shards
            .iter()
            .filter(|s| crate::u32_to_shard_state(s.state) == crate::ShardState::UnderQuorum)
            .count();

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        if !draining && snap.active_nodes > 0 && under_quorum_shards == 0 {
            let json = format!(
                "{{\"status\":\"ready\",\"draining\":false,\"active_nodes\":{},\"under_quorum_shards\":0}}",
                snap.active_nodes
            );
            build_http_response(200, "OK", "application/json", span, latency_ms, &json)
        } else {
            let json = format!(
                "{{\"status\":\"not_ready\",\"draining\":{draining},\"active_nodes\":{},\"under_quorum_shards\":{under_quorum_shards}}}",
                snap.active_nodes
            );
            build_http_response(
                503,
                "Service Unavailable",
                "application/json",
                span,
                latency_ms,
                &json,
            )
        }
    }

    async fn handle_upsert_batch(&self, body: &str, span: &TraceSpan, start: Instant) -> String {
        let Some(records) = extract_json_upsert_batch(body) else {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            return build_http_response(
                400,
                "Bad Request",
                "application/json",
                span,
                latency_ms,
                "{\"status\":\"error\",\"error\":\"missing or invalid 'records'\"}",
            );
        };

        match self.client.upsert_batch(&records).await {
            Ok(upserted) => {
                self.update_metrics(|m| {
                    m.acked_writes_total = m.acked_writes_total.saturating_add(upserted as u64);
                    m.indexed_vectors = m.indexed_vectors.saturating_add(upserted as u64);
                });
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                let resp_json = format!(
                    "{{\"status\":\"ok\",\"upserted\":{upserted},\"trace_id\":\"{}\"}}",
                    span.trace_id
                );
                build_http_response(200, "OK", "application/json", span, latency_ms, &resp_json)
            }
            Err(
                err @ (VektorError::DimensionMismatch { .. }
                | VektorError::InvalidVector(_)
                | VektorError::InvalidConfig(_)),
            ) => json_error_response(400, "Bad Request", &err.to_string(), span, start),
            Err(err) => {
                json_error_response(503, "Service Unavailable", &err.to_string(), span, start)
            }
        }
    }

    async fn handle_upsert(&self, body: &str, span: &TraceSpan, start: Instant) -> String {
        let Some(id) = extract_json_u64(body, "id") else {
            return build_http_response(
                400,
                "Bad Request",
                "application/json",
                span,
                start.elapsed().as_secs_f64() * 1000.0,
                "{\"status\":\"error\",\"error\":\"missing or invalid 'id'\"}",
            );
        };
        let write_id = extract_json_u64(body, "write_id").unwrap_or(0);
        let Some(vector) = extract_json_f32_array(body, "vector") else {
            return build_http_response(
                400,
                "Bad Request",
                "application/json",
                span,
                start.elapsed().as_secs_f64() * 1000.0,
                "{\"status\":\"error\",\"error\":\"missing or invalid 'vector'\"}",
            );
        };

        match self.client.upsert(id, write_id, &vector).await {
            Ok((shard_id, seq_no)) => {
                self.update_metrics(|m| {
                    m.acked_writes_total = m.acked_writes_total.saturating_add(1);
                    m.indexed_vectors = m.indexed_vectors.saturating_add(1);
                });
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                let resp_json = format!(
                    "{{\"status\":\"ok\",\"shard_id\":{shard_id},\"seq_no\":{seq_no},\"trace_id\":\"{}\"}}",
                    span.trace_id
                );
                build_http_response(200, "OK", "application/json", span, latency_ms, &resp_json)
            }
            Err(err) => {
                json_error_response(503, "Service Unavailable", &err.to_string(), span, start)
            }
        }
    }

    async fn handle_search(&self, body: &str, span: &TraceSpan, start: Instant) -> String {
        let Some(query) = extract_json_f32_array(body, "query") else {
            return build_http_response(
                400,
                "Bad Request",
                "application/json",
                span,
                start.elapsed().as_secs_f64() * 1000.0,
                "{\"status\":\"error\",\"error\":\"missing or invalid 'query'\"}",
            );
        };
        let top_k = extract_json_u64(body, "top_k").unwrap_or(10) as usize;
        let ef_search = extract_json_u64(body, "ef_search").unwrap_or(64) as usize;

        match self.client.search(&query, top_k, ef_search).await {
            Ok(hits) => {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                self.record_search_latency(latency_ms);
                let hits_json = format_hits_json(&hits);
                let resp_json = format!(
                    "{{\"status\":\"ok\",\"hits\":{hits_json},\"trace_id\":\"{}\"}}",
                    span.trace_id
                );
                build_http_response(200, "OK", "application/json", span, latency_ms, &resp_json)
            }
            Err(err) => {
                json_error_response(503, "Service Unavailable", &err.to_string(), span, start)
            }
        }
    }

    async fn handle_delete(&self, body: &str, span: &TraceSpan, start: Instant) -> String {
        let Some(id) = extract_json_u64(body, "id") else {
            return build_http_response(
                400,
                "Bad Request",
                "application/json",
                span,
                start.elapsed().as_secs_f64() * 1000.0,
                "{\"status\":\"error\",\"error\":\"missing or invalid 'id'\"}",
            );
        };
        let write_id = extract_json_u64(body, "write_id").unwrap_or(0);

        match self.client.delete(id, write_id).await {
            Ok((shard_id, seq_no)) => {
                self.update_metrics(|m| {
                    m.indexed_vectors = m.indexed_vectors.saturating_sub(1);
                    m.acked_writes_total = m.acked_writes_total.saturating_add(1);
                });
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                let resp_json = format!(
                    "{{\"status\":\"ok\",\"shard_id\":{shard_id},\"seq_no\":{seq_no},\"trace_id\":\"{}\"}}",
                    span.trace_id
                );
                build_http_response(200, "OK", "application/json", span, latency_ms, &resp_json)
            }
            Err(err) => {
                json_error_response(503, "Service Unavailable", &err.to_string(), span, start)
            }
        }
    }

    async fn handle_topology(&self, span: &TraceSpan, start: Instant) -> String {
        let topo = match self.client.refresh_topology().await {
            Ok(t) => t,
            Err(_) => self.client.cached_topology(),
        };
        let epoch = topo.shards.iter().map(|s| s.epoch).max().unwrap_or(0);
        let configured_nodes = topo.nodes.len() as u32;
        let active_nodes = topo.nodes.iter().filter(|n| n.alive).count() as u32;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let resp_json = format!(
            "{{\"status\":\"ok\",\"epoch\":{epoch},\"num_shards\":{},\"replication_factor\":{},\"write_quorum\":{},\"dimensions\":{},\"configured_nodes\":{configured_nodes},\"active_nodes\":{active_nodes}}}",
            topo.num_shards, topo.replication_factor, topo.write_quorum, topo.dimensions
        );
        build_http_response(200, "OK", "application/json", span, latency_ms, &resp_json)
    }

    fn handle_audit(&self, span: &TraceSpan, start: Instant) -> String {
        let Some(gate_arc) = &self.security_gate else {
            return build_http_response(
                200,
                "OK",
                "application/json",
                span,
                start.elapsed().as_secs_f64() * 1000.0,
                "{\"status\":\"ok\",\"security_enabled\":false,\"entries_count\":0,\"head_hash\":0,\"chain_valid\":true}",
            );
        };

        let (verify_res, head_hash) = {
            let guard = gate_arc
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (
                guard.audit_log.verify_integrity(),
                guard.audit_log.head_hash(),
            )
        };

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        match verify_res {
            Ok(entries_count) => {
                let resp_json = format!(
                    "{{\"status\":\"ok\",\"security_enabled\":true,\"entries_count\":{entries_count},\"head_hash\":{head_hash},\"chain_valid\":true}}"
                );
                build_http_response(200, "OK", "application/json", span, latency_ms, &resp_json)
            }
            Err(corrupted_seq_no) => map_security_error_response(
                &SecurityError::AuditIntegrityViolation { corrupted_seq_no },
                span,
                latency_ms,
            ),
        }
    }
}

/// Bind a `tokio::net::TcpListener` on `bind_addr` and spawn the hardened HTTP/1.1 gateway loop
/// enforcing `gateway.max_connections` via `Semaphore` and `gateway.read_timeout` via `tokio::time::timeout`.
pub async fn spawn_http_gateway(
    bind_addr: &str,
    gateway: HttpGatewayServer,
) -> Result<(String, JoinHandle<()>), VektorError> {
    let listener = TcpListener::bind(bind_addr)
        .await
        .map_err(|e| VektorError::WalIo(format!("failed to bind HTTP gateway: {e}")))?;
    let local_addr = listener
        .local_addr()
        .map_err(|e| VektorError::WalIo(format!("failed to read local_addr: {e}")))?
        .to_string();

    let conn_semaphore = Arc::new(Semaphore::new(gateway.max_connections.max(1)));
    let handle = tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            let Ok(permit) = Arc::clone(&conn_semaphore).acquire_owned().await else {
                break;
            };
            let gw = gateway.clone();
            tokio::spawn(async move {
                let _permit = permit;
                let read_fut = read_full_http_request(&mut stream, gw.max_body_bytes);
                match tokio::time::timeout(gw.read_timeout, read_fut).await {
                    Ok(Ok(HttpReadOutcome::Complete(req_str))) => {
                        let resp_str = gw.handle_http_request(&req_str).await;
                        let _ = stream.write_all(resp_str.as_bytes()).await;
                        let _ = stream.shutdown().await;
                    }
                    Ok(Ok(HttpReadOutcome::PayloadTooLarge { traceparent })) => {
                        let span = traceparent
                            .as_deref()
                            .and_then(|tp| TraceSpan::parse_traceparent(tp).ok())
                            .map_or_else(TraceSpan::generate, |p| p.child_span());
                        let resp_str = build_http_response(
                            413,
                            "Payload Too Large",
                            "application/json",
                            &span,
                            0.0,
                            "{\"status\":\"error\",\"error\":\"payload exceeds max_body_bytes\"}",
                        );
                        let _ = stream.write_all(resp_str.as_bytes()).await;
                        let _ = stream.shutdown().await;
                    }
                    Ok(Err(_)) => {}
                    Err(_elapsed) => {
                        let span = TraceSpan::generate();
                        let resp_str = build_http_response(
                            408,
                            "Request Timeout",
                            "application/json",
                            &span,
                            gw.read_timeout.as_secs_f64() * 1000.0,
                            "{\"status\":\"error\",\"error\":\"request read timeout\"}",
                        );
                        let _ = stream.write_all(resp_str.as_bytes()).await;
                        let _ = stream.shutdown().await;
                    }
                }
            });
        }
    });

    Ok((local_addr, handle))
}

/// Send an HTTP/1.1 request over a `tokio::net::TcpStream` and parse the `HttpResponse`.
pub async fn send_http_request(
    addr: &str,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &str,
) -> Result<HttpResponse, VektorError> {
    let mut stream = TcpStream::connect(addr)
        .await
        .map_err(|e| VektorError::WalIo(format!("http connect error: {e}")))?;

    let mut raw_req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Length: {}\r\n",
        body.len()
    );
    for &(k, v) in headers {
        let _ = write!(raw_req, "{k}: {v}\r\n");
    }
    raw_req.push_str("\r\n");
    raw_req.push_str(body);

    let write_res = stream.write_all(raw_req.as_bytes()).await;
    let mut resp_bytes = Vec::new();
    let read_res = stream.read_to_end(&mut resp_bytes).await;
    if !resp_bytes.is_empty() {
        let raw_resp = String::from_utf8_lossy(&resp_bytes);
        if let Ok(parsed) = parse_http_response(&raw_resp) {
            return Ok(parsed);
        }
    }

    write_res.map_err(|e| VektorError::WalIo(format!("http write error: {e}")))?;
    read_res.map_err(|e| VektorError::WalIo(format!("http read error: {e}")))?;
    parse_http_response(&String::from_utf8_lossy(&resp_bytes))
}

enum HttpReadOutcome {
    Complete(String),
    PayloadTooLarge { traceparent: Option<String> },
}

async fn read_full_http_request(
    stream: &mut TcpStream,
    max_body_bytes: usize,
) -> Result<HttpReadOutcome, std::io::Error> {
    const MAX_HEADER_BYTES: usize = 32 * 1024;
    const DRAIN_CAP_BYTES: usize = 1024 * 1024;

    let mut buf = Vec::with_capacity(2048);
    let mut chunk = [0_u8; 4096];
    let mut header_end_opt: Option<usize> = None;
    let mut declared_content_len: usize = 0;

    loop {
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);

        if header_end_opt.is_none() {
            if let Some(header_end) = find_header_end(&buf) {
                header_end_opt = Some(header_end);
                let header_str = String::from_utf8_lossy(&buf[..header_end]);
                declared_content_len = parse_content_length(&header_str);
                let traceparent = extract_header_value(&header_str, "traceparent");
                if declared_content_len > max_body_bytes {
                    let already_read_body = buf.len().saturating_sub(header_end);
                    let remaining = declared_content_len.saturating_sub(already_read_body);
                    drain_bounded_bytes(stream, remaining.min(DRAIN_CAP_BYTES)).await;
                    return Ok(HttpReadOutcome::PayloadTooLarge { traceparent });
                }
            } else if buf.len() > MAX_HEADER_BYTES {
                return Ok(HttpReadOutcome::PayloadTooLarge { traceparent: None });
            }
        }

        if let Some(header_end) = header_end_opt {
            let body_read = buf.len().saturating_sub(header_end);
            if body_read > max_body_bytes {
                let header_str = String::from_utf8_lossy(&buf[..header_end]);
                let traceparent = extract_header_value(&header_str, "traceparent");
                return Ok(HttpReadOutcome::PayloadTooLarge { traceparent });
            }
            if buf.len() >= header_end.saturating_add(declared_content_len) {
                break;
            }
        }
    }

    Ok(HttpReadOutcome::Complete(
        String::from_utf8_lossy(&buf).into_owned(),
    ))
}

async fn drain_bounded_bytes(stream: &mut TcpStream, mut remaining: usize) {
    let mut scratch = [0_u8; 4096];
    while remaining > 0 {
        let to_read = remaining.min(scratch.len());
        match stream.read(&mut scratch[..to_read]).await {
            Ok(0) | Err(_) => break,
            Ok(n) => remaining = remaining.saturating_sub(n),
        }
    }
}

fn not_found_response(span: &TraceSpan, start: Instant) -> String {
    build_http_response(
        404,
        "Not Found",
        "application/json",
        span,
        start.elapsed().as_secs_f64() * 1000.0,
        "{\"status\":\"error\",\"error\":\"route not found\"}",
    )
}

fn json_error_response(
    status_code: u16,
    reason: &str,
    msg: &str,
    span: &TraceSpan,
    start: Instant,
) -> String {
    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    let body = format!(
        "{{\"status\":\"error\",\"error\":\"{}\",\"trace_id\":\"{}\"}}",
        escape_json_str(msg),
        span.trace_id
    );
    build_http_response(
        status_code,
        reason,
        "application/json",
        span,
        latency_ms,
        &body,
    )
}

fn extract_header_value(header_str: &str, name: &str) -> Option<String> {
    for line in header_str.lines() {
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case(name) {
                return Some(v.trim().to_string());
            }
        }
    }
    None
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
        return Some(pos + 4);
    }
    buf.windows(2).position(|w| w == b"\n\n").map(|pos| pos + 2)
}

fn parse_content_length(header_str: &str) -> usize {
    for line in header_str.lines() {
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                if let Ok(len) = v.trim().parse::<usize>() {
                    return len;
                }
            }
        }
    }
    0
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn extract_bearer_token(headers: &BTreeMap<String, String>) -> &str {
    let Some(raw) = headers.get("authorization") else {
        return "";
    };
    let trimmed = raw.trim();
    if trimmed
        .get(..7)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("bearer "))
    {
        trimmed.get(7..).map_or("", str::trim)
    } else {
        ""
    }
}

fn map_security_error_response(err: &SecurityError, span: &TraceSpan, latency_ms: f64) -> String {
    let escaped = escape_json_str(&err.to_string());
    match err {
        SecurityError::Unauthorized => {
            let body = format!(
                "{{\"status\":\"error\",\"error\":\"{escaped}\",\"trace_id\":\"{}\"}}",
                span.trace_id
            );
            build_http_response(
                401,
                "Unauthorized",
                "application/json",
                span,
                latency_ms,
                &body,
            )
        }
        SecurityError::CrossTenantDenied { .. } | SecurityError::InsufficientRole { .. } => {
            let body = format!(
                "{{\"status\":\"error\",\"error\":\"{escaped}\",\"trace_id\":\"{}\"}}",
                span.trace_id
            );
            build_http_response(
                403,
                "Forbidden",
                "application/json",
                span,
                latency_ms,
                &body,
            )
        }
        SecurityError::RateLimited { retry_after_ms, .. } => {
            let retry_secs = retry_after_ms.div_ceil(1000).max(1).to_string();
            let body = format!(
                "{{\"status\":\"error\",\"error\":\"{escaped}\",\"retry_after_ms\":{retry_after_ms},\"trace_id\":\"{}\"}}",
                span.trace_id
            );
            build_http_response_with_headers(
                429,
                "Too Many Requests",
                "application/json",
                span,
                latency_ms,
                &[("Retry-After", &retry_secs)],
                &body,
            )
        }
        SecurityError::StorageQuotaExceeded { .. } => {
            let body = format!(
                "{{\"status\":\"error\",\"error\":\"{escaped}\",\"trace_id\":\"{}\"}}",
                span.trace_id
            );
            build_http_response(
                507,
                "Insufficient Storage",
                "application/json",
                span,
                latency_ms,
                &body,
            )
        }
        SecurityError::AuditIntegrityViolation { .. } => {
            let body = format!(
                "{{\"status\":\"error\",\"error\":\"{escaped}\",\"trace_id\":\"{}\"}}",
                span.trace_id
            );
            build_http_response(
                500,
                "Internal Server Error",
                "application/json",
                span,
                latency_ms,
                &body,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_extraction_and_security_error_http_mappings_cover_all_variants() {
        let mut headers = BTreeMap::new();
        assert_eq!(extract_bearer_token(&headers), "");
        headers.insert("authorization".to_string(), "BEARER vdk.k1.abc".to_string());
        assert_eq!(extract_bearer_token(&headers), "vdk.k1.abc");
        headers.insert("authorization".to_string(), "Basic xyz".to_string());
        assert_eq!(extract_bearer_token(&headers), "");

        let span = TraceSpan::new_root(42);
        let org_a = OrgId::new("org-a").unwrap();
        let org_b = OrgId::new("org-b").unwrap();

        let r401 = parse_http_response(&map_security_error_response(
            &SecurityError::Unauthorized,
            &span,
            1.0,
        ))
        .unwrap();
        assert_eq!(r401.status_code, 401);

        let r403_tenant = parse_http_response(&map_security_error_response(
            &SecurityError::CrossTenantDenied {
                key_org: org_a.clone(),
                target_org: org_b,
            },
            &span,
            1.0,
        ))
        .unwrap();
        assert_eq!(r403_tenant.status_code, 403);

        let r403_role = parse_http_response(&map_security_error_response(
            &SecurityError::InsufficientRole {
                actual: TenantRole::Reader,
                required: TenantRole::Admin,
            },
            &span,
            1.0,
        ))
        .unwrap();
        assert_eq!(r403_role.status_code, 403);

        let r429 = parse_http_response(&map_security_error_response(
            &SecurityError::RateLimited {
                org_id: org_a.clone(),
                retry_after_ms: 250,
            },
            &span,
            1.0,
        ))
        .unwrap();
        assert_eq!(r429.status_code, 429);
        assert_eq!(
            r429.headers.get("retry-after").map(String::as_str),
            Some("1")
        );

        let r507 = parse_http_response(&map_security_error_response(
            &SecurityError::StorageQuotaExceeded {
                org_id: org_a,
                current_vectors: 11,
                max_vectors: 10,
                current_bytes: 1024,
                max_bytes: 512,
            },
            &span,
            1.0,
        ))
        .unwrap();
        assert_eq!(r507.status_code, 507);

        let r500 = parse_http_response(&map_security_error_response(
            &SecurityError::AuditIntegrityViolation {
                corrupted_seq_no: 7,
            },
            &span,
            1.0,
        ))
        .unwrap();
        assert_eq!(r500.status_code, 500);
    }
}
