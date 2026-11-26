//! Milestone 10: HTTP/1.1 REST & Prometheus `/metrics` Gateway, W3C `traceparent` Distributed
//! Trace Spans, and Automated Chaos Campaign Runner.
//!
//! Built directly on `tokio::net::TcpListener` without external HTTP framework dependencies.

use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;
use tonic::transport::Server;
use vektordist_core::exact::{ExactIndex, SeededRng};
use vektordist_core::hnsw::HnswParams;
use vektordist_core::{ClusterConfig, NodeId, ShardId, VectorId, VektorError, WriteId};
use vektordist_proto::v1::shard_replica_rpc_server::ShardReplicaRpcServer;
use vektordist_proto::v1::topology_rpc_server::TopologyRpcServer;

use crate::client::ClusterClient;
use crate::coord::CoordinatorService;
use crate::node::NodeService;

static TRACE_COUNTER: AtomicU64 = AtomicU64::new(1);
static CHAOS_DIR_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Live cluster & gateway telemetry exported in Prometheus text exposition format (`0.0.4`).
#[derive(Debug, Clone, PartialEq)]
pub struct PrometheusMetrics {
    pub active_nodes: u32,
    pub configured_nodes: u32,
    pub indexed_vectors: u64,
    pub search_latency_p50_ms: f64,
    pub search_latency_p95_ms: f64,
    pub search_latency_p99_ms: f64,
    pub recall_at_10: f64,
    pub acked_writes_total: u64,
    pub acked_writes_lost_total: u64,
}

impl Default for PrometheusMetrics {
    fn default() -> Self {
        Self {
            active_nodes: 0,
            configured_nodes: 0,
            indexed_vectors: 0,
            search_latency_p50_ms: 0.0,
            search_latency_p95_ms: 0.0,
            search_latency_p99_ms: 0.0,
            recall_at_10: 1.0,
            acked_writes_total: 0,
            acked_writes_lost_total: 0,
        }
    }
}

impl PrometheusMetrics {
    /// Format all cluster metrics into standard Prometheus `text/plain; version=0.0.4` format.
    #[must_use]
    pub fn render_prometheus_text(&self) -> String {
        let mut out = String::with_capacity(1024);
        let _ = writeln!(
            out,
            "# HELP vektordist_active_nodes Number of currently alive physical cluster nodes."
        );
        let _ = writeln!(out, "# TYPE vektordist_active_nodes gauge");
        let _ = writeln!(out, "vektordist_active_nodes {}", self.active_nodes);

        let _ = writeln!(
            out,
            "# HELP vektordist_configured_nodes Total number of registered physical cluster nodes."
        );
        let _ = writeln!(out, "# TYPE vektordist_configured_nodes gauge");
        let _ = writeln!(out, "vektordist_configured_nodes {}", self.configured_nodes);

        let _ = writeln!(
            out,
            "# HELP vektordist_indexed_vectors Total live vectors indexed across active replicas."
        );
        let _ = writeln!(out, "# TYPE vektordist_indexed_vectors gauge");
        let _ = writeln!(out, "vektordist_indexed_vectors {}", self.indexed_vectors);

        let _ = writeln!(
            out,
            "# HELP vektordist_search_latency_p50_ms 50th percentile scatter-gather search latency in milliseconds."
        );
        let _ = writeln!(out, "# TYPE vektordist_search_latency_p50_ms gauge");
        let _ = writeln!(
            out,
            "vektordist_search_latency_p50_ms {:.4}",
            self.search_latency_p50_ms
        );

        let _ = writeln!(
            out,
            "# HELP vektordist_search_latency_p95_ms 95th percentile scatter-gather search latency in milliseconds."
        );
        let _ = writeln!(out, "# TYPE vektordist_search_latency_p95_ms gauge");
        let _ = writeln!(
            out,
            "vektordist_search_latency_p95_ms {:.4}",
            self.search_latency_p95_ms
        );

        let _ = writeln!(
            out,
            "# HELP vektordist_search_latency_p99_ms 99th percentile scatter-gather search latency in milliseconds."
        );
        let _ = writeln!(out, "# TYPE vektordist_search_latency_p99_ms gauge");
        let _ = writeln!(
            out,
            "vektordist_search_latency_p99_ms {:.4}",
            self.search_latency_p99_ms
        );

        let _ = writeln!(
            out,
            "# HELP vektordist_recall_at_10 Measured top-10 recall against exact brute-force ground truth."
        );
        let _ = writeln!(out, "# TYPE vektordist_recall_at_10 gauge");
        let _ = writeln!(out, "vektordist_recall_at_10 {:.4}", self.recall_at_10);

        let _ = writeln!(
            out,
            "# HELP vektordist_acked_writes_total Total W=2 quorum-acknowledged vector writes."
        );
        let _ = writeln!(out, "# TYPE vektordist_acked_writes_total counter");
        let _ = writeln!(
            out,
            "vektordist_acked_writes_total {}",
            self.acked_writes_total
        );

        let _ = writeln!(
            out,
            "# HELP vektordist_acked_writes_lost_total Total quorum-acknowledged writes lost across failovers."
        );
        let _ = writeln!(out, "# TYPE vektordist_acked_writes_lost_total counter");
        let _ = writeln!(
            out,
            "vektordist_acked_writes_lost_total {}",
            self.acked_writes_lost_total
        );
        out
    }
}

/// W3C Trace Context (`traceparent` version `00`) span representation:
/// `00-<32-hex trace_id>-<16-hex span_id>-<2-hex trace_flags>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceSpan {
    pub version: u8,
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub sampled: bool,
}

impl TraceSpan {
    /// Generate a fresh root `TraceSpan` with deterministic-friendly SplitMix64 entropy.
    #[must_use]
    pub fn generate() -> Self {
        let seq = TRACE_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self::new_root(seq)
    }

    /// Create a root `TraceSpan` seeded by `seed`.
    #[must_use]
    pub fn new_root(seed: u64) -> Self {
        let mut rng = SeededRng::new(seed ^ 0x517c_c1b7_2722_0a95);
        let hi = rng.next_u64().max(1);
        let lo = rng.next_u64().max(1);
        let span = rng.next_u64().max(1);
        Self {
            version: 0,
            trace_id: format!("{hi:016x}{lo:016x}"),
            span_id: format!("{span:016x}"),
            parent_span_id: None,
            sampled: true,
        }
    }

    /// Parse a W3C `traceparent` header (`00-<32-hex-trace-id>-<16-hex-span-id>-<2-hex-flags>`).
    pub fn parse_traceparent(header: &str) -> Result<Self, VektorError> {
        let trimmed = header.trim();
        let parts: Vec<&str> = trimmed.split('-').collect();
        if parts.len() != 4 {
            return Err(VektorError::InvalidConfig(
                "traceparent must have 4 dash-separated fields",
            ));
        }
        if parts[0] != "00" {
            return Err(VektorError::InvalidConfig(
                "unsupported traceparent version",
            ));
        }
        let trace_id = parts[1].to_ascii_lowercase();
        if trace_id.len() != 32
            || !trace_id.bytes().all(|b| b.is_ascii_hexdigit())
            || trace_id.bytes().all(|b| b == b'0')
        {
            return Err(VektorError::InvalidConfig("invalid 32-hex trace_id"));
        }
        let span_id = parts[2].to_ascii_lowercase();
        if span_id.len() != 16
            || !span_id.bytes().all(|b| b.is_ascii_hexdigit())
            || span_id.bytes().all(|b| b == b'0')
        {
            return Err(VektorError::InvalidConfig("invalid 16-hex span_id"));
        }
        let flags_str = parts[3];
        if flags_str.len() != 2 || !flags_str.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(VektorError::InvalidConfig("invalid 2-hex trace_flags"));
        }
        let flags = u8::from_str_radix(flags_str, 16)
            .map_err(|_| VektorError::InvalidConfig("invalid trace_flags hex"))?;
        Ok(Self {
            version: 0,
            trace_id,
            span_id,
            parent_span_id: None,
            sampled: (flags & 0x01) != 0,
        })
    }

    /// Create a child span preserving `trace_id` and recording `self.span_id` as `parent_span_id`.
    #[must_use]
    pub fn child_span(&self) -> Self {
        let seq = TRACE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut rng = SeededRng::new(seq ^ 0x9e37_79b9_7f4a_7c15);
        let next_span = rng.next_u64().max(1);
        Self {
            version: self.version,
            trace_id: self.trace_id.clone(),
            span_id: format!("{next_span:016x}"),
            parent_span_id: Some(self.span_id.clone()),
            sampled: self.sampled,
        }
    }

    /// Format as a W3C `traceparent` header string (`00-<trace_id>-<span_id>-01`).
    #[must_use]
    pub fn to_traceparent(&self) -> String {
        let flags = if self.sampled { "01" } else { "00" };
        format!(
            "{:02x}-{}-{}-{}",
            self.version, self.trace_id, self.span_id, flags
        )
    }
}

/// Parsed HTTP/1.1 response returned by `send_http_request`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status_code: u16,
    pub headers: BTreeMap<String, String>,
    pub body: String,
}

/// Zero-dependency HTTP/1.1 REST and Prometheus observability gateway backed by `ClusterClient`.
#[derive(Clone)]
pub struct HttpGatewayServer {
    client: ClusterClient,
    metrics: Arc<RwLock<PrometheusMetrics>>,
    latency_samples_ms: Arc<RwLock<Vec<f64>>>,
}

impl HttpGatewayServer {
    /// Create a new `HttpGatewayServer` wrapping `client`.
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
            latency_samples_ms: Arc::new(RwLock::new(Vec::new())),
        }
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

        let (p50, p95, p99) = {
            let guard = self.latency_samples_ms.read();
            if let Ok(samples) = guard {
                compute_percentiles(&samples)
            } else {
                (0.0, 0.0, 0.0)
            }
        };

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
        let path = parts.next().unwrap_or("/");

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
            ("GET", "/metrics") => {
                self.refresh_cluster_telemetry().await;
                let snap = self.metrics_snapshot();
                let text = snap.render_prometheus_text();
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
            ("POST", "/v1/upsert") => self.handle_upsert(body, &span, start).await,
            ("POST", "/v1/search") => self.handle_search(body, &span, start).await,
            _ => {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                build_http_response(
                    404,
                    "Not Found",
                    "application/json",
                    &span,
                    latency_ms,
                    "{\"status\":\"error\",\"error\":\"route not found\"}",
                )
            }
        }
    }

    async fn handle_upsert(&self, body: &str, span: &TraceSpan, start: Instant) -> String {
        let Some(id) = extract_json_u64(body, "id") else {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            return build_http_response(
                400,
                "Bad Request",
                "application/json",
                span,
                latency_ms,
                "{\"status\":\"error\",\"error\":\"missing or invalid 'id'\"}",
            );
        };
        let write_id = extract_json_u64(body, "write_id").unwrap_or(id);
        let Some(vector) = extract_json_f32_array(body, "vector") else {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            return build_http_response(
                400,
                "Bad Request",
                "application/json",
                span,
                latency_ms,
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
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                let resp_json = format!(
                    "{{\"status\":\"error\",\"error\":\"{}\",\"trace_id\":\"{}\"}}",
                    escape_json_str(&err.to_string()),
                    span.trace_id
                );
                build_http_response(
                    503,
                    "Service Unavailable",
                    "application/json",
                    span,
                    latency_ms,
                    &resp_json,
                )
            }
        }
    }

    async fn handle_search(&self, body: &str, span: &TraceSpan, start: Instant) -> String {
        let Some(query) = extract_json_f32_array(body, "query") else {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            return build_http_response(
                400,
                "Bad Request",
                "application/json",
                span,
                latency_ms,
                "{\"status\":\"error\",\"error\":\"missing or invalid 'query'\"}",
            );
        };
        let top_k = extract_json_u64(body, "top_k").unwrap_or(10) as usize;
        let ef_search = extract_json_u64(body, "ef_search").unwrap_or(64) as usize;

        match self.client.search(&query, top_k, ef_search).await {
            Ok(hits) => {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                if let Ok(mut samples) = self.latency_samples_ms.write() {
                    samples.push(latency_ms);
                    let (p50, p95, p99) = compute_percentiles(&samples);
                    self.update_metrics(|m| {
                        m.search_latency_p50_ms = p50;
                        m.search_latency_p95_ms = p95;
                        m.search_latency_p99_ms = p99;
                    });
                }
                let mut hits_json = String::from("[");
                for (idx, h) in hits.iter().enumerate() {
                    if idx > 0 {
                        hits_json.push(',');
                    }
                    let _ = write!(hits_json, "{{\"id\":{},\"score\":{:.6}}}", h.id, h.score);
                }
                hits_json.push(']');
                let resp_json = format!(
                    "{{\"status\":\"ok\",\"hits\":{hits_json},\"trace_id\":\"{}\"}}",
                    span.trace_id
                );
                build_http_response(200, "OK", "application/json", span, latency_ms, &resp_json)
            }
            Err(err) => {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                let resp_json = format!(
                    "{{\"status\":\"error\",\"error\":\"{}\",\"trace_id\":\"{}\"}}",
                    escape_json_str(&err.to_string()),
                    span.trace_id
                );
                build_http_response(
                    503,
                    "Service Unavailable",
                    "application/json",
                    span,
                    latency_ms,
                    &resp_json,
                )
            }
        }
    }
}

/// Bind a `tokio::net::TcpListener` on `bind_addr` and spawn the HTTP/1.1 gateway server task.
/// Returns `(bound_socket_addr, join_handle)`.
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

    let handle = tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            let gw = gateway.clone();
            tokio::spawn(async move {
                if let Ok(req_str) = read_full_http_request(&mut stream).await {
                    let resp_str = gw.handle_http_request(&req_str).await;
                    let _ = stream.write_all(resp_str.as_bytes()).await;
                    let _ = stream.shutdown().await;
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

    stream
        .write_all(raw_req.as_bytes())
        .await
        .map_err(|e| VektorError::WalIo(format!("http write error: {e}")))?;

    let mut resp_bytes = Vec::new();
    stream
        .read_to_end(&mut resp_bytes)
        .await
        .map_err(|e| VektorError::WalIo(format!("http read error: {e}")))?;

    let raw_resp = String::from_utf8_lossy(&resp_bytes);
    parse_http_response(&raw_resp)
}

async fn read_full_http_request(stream: &mut TcpStream) -> Result<String, std::io::Error> {
    let mut buf = Vec::with_capacity(2048);
    let mut chunk = [0_u8; 1024];
    let mut expected_total_len: Option<usize> = None;

    loop {
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);

        if expected_total_len.is_none() {
            if let Some(header_end) = find_header_end(&buf) {
                let header_str = String::from_utf8_lossy(&buf[..header_end]);
                let content_len = parse_content_length(&header_str);
                expected_total_len = Some(header_end + content_len);
            }
        }
        if let Some(total) = expected_total_len {
            if buf.len() >= total {
                break;
            }
        }
    }

    Ok(String::from_utf8_lossy(&buf).into_owned())
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

fn split_http_head_and_body(raw: &str) -> (&str, &str) {
    if let Some((head, body)) = raw.split_once("\r\n\r\n") {
        (head, body)
    } else if let Some((head, body)) = raw.split_once("\n\n") {
        (head, body)
    } else {
        (raw, "")
    }
}

fn build_http_response(
    status_code: u16,
    reason: &str,
    content_type: &str,
    span: &TraceSpan,
    latency_ms: f64,
    body: &str,
) -> String {
    format!(
        "HTTP/1.1 {status_code} {reason}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nx-trace-id: {}\r\ntraceparent: {}\r\nx-latency-ms: {latency_ms:.3}\r\nconnection: close\r\n\r\n{body}",
        body.len(),
        span.trace_id,
        span.to_traceparent(),
    )
}

fn parse_http_response(raw: &str) -> Result<HttpResponse, VektorError> {
    let (head, body) = split_http_head_and_body(raw);
    let mut lines = head.lines();
    let status_line = lines
        .next()
        .ok_or(VektorError::InvalidConfig("empty HTTP response"))?;
    let mut parts = status_line.split_whitespace();
    let _http_ver = parts.next();
    let status_code = parts
        .next()
        .and_then(|s| s.parse::<u16>().ok())
        .ok_or(VektorError::InvalidConfig("invalid HTTP status code"))?;

    let mut headers = BTreeMap::new();
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }

    Ok(HttpResponse {
        status_code,
        headers,
        body: body.to_string(),
    })
}

fn extract_json_u64(json: &str, key: &str) -> Option<u64> {
    let pattern = format!("\"{key}\"");
    let start_idx = json.find(&pattern)? + pattern.len();
    let rest = json.get(start_idx..)?.trim_start();
    let after_colon = rest.strip_prefix(':')?.trim_start();
    let num_end = after_colon
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after_colon.len());
    after_colon.get(..num_end)?.parse::<u64>().ok()
}

fn extract_json_f32_array(json: &str, key: &str) -> Option<Vec<f32>> {
    let pattern = format!("\"{key}\"");
    let start_idx = json.find(&pattern)? + pattern.len();
    let rest = json.get(start_idx..)?.trim_start();
    let after_colon = rest.strip_prefix(':')?.trim_start();
    let after_open = after_colon.strip_prefix('[')?;
    let close_idx = after_open.find(']')?;
    let inner = after_open.get(..close_idx)?.trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    let mut out = Vec::new();
    for item in inner.split(',') {
        let val = item.trim().parse::<f32>().ok()?;
        out.push(val);
    }
    Some(out)
}

fn escape_json_str(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn compute_percentiles(samples: &[f64]) -> (f64, f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let p50 = percentile_sorted(&sorted, 0.50);
    let p95 = percentile_sorted(&sorted, 0.95);
    let p99 = percentile_sorted(&sorted, 0.99);
    (p50, p95, p99)
}

fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len().saturating_sub(1) as f64) * q.clamp(0.0, 1.0)).round() as usize;
    sorted.get(idx).copied().unwrap_or(0.0)
}

/// A single programmable step in an automated chaos campaign.
#[derive(Debug, Clone)]
pub enum ChaosStep {
    /// Seed `count` deterministic unit vectors across all shards using batched WAL fsync.
    SeedVectors {
        count: usize,
        batch_size: usize,
        seed: u64,
    },
    /// Execute concurrent `W=2` quorum writes and scatter-gather reads.
    ConcurrentWorkload {
        writes: usize,
        reads: usize,
        top_k: usize,
        ef_search: usize,
        seed: u64,
    },
    /// Abruptly terminate `node_id` (simulating `SIGKILL`) and trigger coordinator failover + union WAL reconciliation.
    KillNode { node_id: NodeId },
    /// Audit all surviving shard WALs against the external acknowledged-write ledger (`acked_writes_lost == 0`).
    VerifyZeroWriteLoss,
    /// Spawn a replacement node `node_id`, stream historical + live WAL catch-up (`RECOVERING` -> `HEALTHY`).
    AddReplacementNode { node_id: NodeId },
    /// Measure distributed `recall@10` against an exact brute-force oracle and verify `p95` latency SLA.
    AssertRecallAndLatencySla {
        queries: usize,
        top_k: usize,
        ef_search: usize,
        min_recall_at_10: f64,
        max_p95_ms: f64,
        seed: u64,
    },
}

/// Falsifiable report produced by `ChaosCampaign::run`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChaosCampaignReport {
    pub steps_executed: usize,
    pub acked_writes_verified: usize,
    pub acked_writes_lost: usize,
    pub measured_recall_at_10: f64,
    pub measured_p95_ms: f64,
    pub sla_passed: bool,
    pub step_logs: Vec<String>,
}

/// Automated multi-step fault-injection and SLA verification runner against a live `CoordinatorService` + `NodeService` cluster.
#[derive(Debug, Clone)]
pub struct ChaosCampaign {
    pub config: ClusterConfig,
    pub hnsw_params: HnswParams,
    pub initial_nodes: Vec<NodeId>,
    pub steps: Vec<ChaosStep>,
}

impl ChaosCampaign {
    /// Construct a new `ChaosCampaign` with default `HnswParams`.
    #[must_use]
    pub fn new(config: ClusterConfig, initial_nodes: Vec<NodeId>, steps: Vec<ChaosStep>) -> Self {
        Self {
            config,
            hnsw_params: HnswParams::default(),
            initial_nodes,
            steps,
        }
    }

    /// Override `HnswParams` used by cluster nodes during the chaos campaign.
    #[must_use]
    pub fn with_hnsw_params(mut self, params: HnswParams) -> Self {
        self.hnsw_params = params;
        self
    }

    /// Execute the entire chaos schedule against real localhost gRPC node & coordinator servers.
    pub async fn run(&self) -> Result<ChaosCampaignReport, VektorError> {
        self.config.validate()?;
        if self.initial_nodes.is_empty() {
            return Err(VektorError::InvalidConfig(
                "chaos campaign requires at least 1 initial node",
            ));
        }

        let mut node_handles: BTreeMap<NodeId, (String, PathBuf, JoinHandle<()>)> = BTreeMap::new();
        let mut initial_descriptors = Vec::with_capacity(self.initial_nodes.len());

        for &node_id in &self.initial_nodes {
            let (addr, dir, handle) =
                spawn_chaos_node(node_id, self.config.clone(), self.hnsw_params).await?;
            initial_descriptors.push((node_id, addr.clone()));
            node_handles.insert(node_id, (addr, dir, handle));
        }

        let coord = CoordinatorService::new(self.config.clone(), initial_descriptors)?;
        coord.push_all_shard_configs().await?;
        let (coord_addr, coord_handle) = spawn_chaos_coord(coord.clone()).await?;
        let client = ClusterClient::connect(&coord_addr).await?;

        let mut exact_oracle = ExactIndex::new(self.config.dimensions, self.config.metric)?;
        let mut acked_ledger: Vec<(WriteId, VectorId, ShardId)> = Vec::new();
        let mut next_vec_id: VectorId = 1;
        let mut next_write_id: WriteId = 10_000;

        let mut steps_executed = 0_usize;
        let mut acked_writes_verified = 0_usize;
        let mut acked_writes_lost = 0_usize;
        let mut measured_recall_at_10 = 1.0_f64;
        let mut measured_p95_ms = 0.0_f64;
        let mut sla_passed = true;
        let mut step_logs = Vec::with_capacity(self.steps.len());

        for step in &self.steps {
            match step {
                ChaosStep::SeedVectors {
                    count,
                    batch_size,
                    seed,
                } => {
                    let mut rng = SeededRng::new(*seed);
                    let effective_batch = (*batch_size).max(1);
                    let mut by_shard: BTreeMap<ShardId, Vec<(VectorId, WriteId, Vec<f32>)>> =
                        BTreeMap::new();

                    for _ in 0..*count {
                        let mut vec_buf = vec![0.0_f32; self.config.dimensions];
                        rng.fill_unit_vector(&mut vec_buf);
                        let id = next_vec_id;
                        let write_id = next_write_id;
                        next_vec_id += 1;
                        next_write_id += 1;
                        let shard_id = self.config.shard_for(id);
                        by_shard
                            .entry(shard_id)
                            .or_default()
                            .push((id, write_id, vec_buf));
                    }

                    for (shard_id, records) in by_shard {
                        for chunk in records.chunks(effective_batch) {
                            let assigned = client.seed_shard_batch(shard_id, chunk).await?;
                            if assigned.len() == chunk.len() {
                                for (id, write_id, vec_buf) in chunk {
                                    exact_oracle.upsert(*id, vec_buf)?;
                                    acked_ledger.push((*write_id, *id, shard_id));
                                }
                            }
                        }
                    }
                    steps_executed += 1;
                    step_logs.push(format!("SeedVectors(count={count})"));
                }
                ChaosStep::ConcurrentWorkload {
                    writes,
                    reads,
                    top_k,
                    ef_search,
                    seed,
                } => {
                    let base_id = next_vec_id;
                    let base_write_id = next_write_id;
                    next_vec_id += *writes as u64;
                    next_write_id += *writes as u64;

                    let dims = self.config.dimensions;
                    let write_client = client.clone();
                    let write_count = *writes;
                    let write_seed = *seed;

                    let writer_task = tokio::spawn(async move {
                        let mut rng = SeededRng::new(write_seed);
                        let mut acked = Vec::with_capacity(write_count);
                        for offset in 0..write_count {
                            let mut v = vec![0.0_f32; dims];
                            rng.fill_unit_vector(&mut v);
                            let id = base_id + offset as u64;
                            let wid = base_write_id + offset as u64;
                            if let Ok((shard_id, _seq)) = write_client.upsert(id, wid, &v).await {
                                acked.push((wid, id, shard_id, v));
                            }
                        }
                        acked
                    });

                    let read_client = client.clone();
                    let read_count = *reads;
                    let k = (*top_k).max(1);
                    let ef = (*ef_search).max(k);
                    let read_seed = seed.wrapping_add(7919);

                    let reader_task = tokio::spawn(async move {
                        let mut rng = SeededRng::new(read_seed);
                        let mut completed_reads = 0_usize;
                        for _ in 0..read_count {
                            let mut q = vec![0.0_f32; dims];
                            rng.fill_unit_vector(&mut q);
                            if let Ok(hits) = read_client.search(&q, k, ef).await {
                                if !hits.is_empty() {
                                    completed_reads += 1;
                                }
                            }
                        }
                        completed_reads
                    });

                    let acked_batch = writer_task
                        .await
                        .map_err(|e| VektorError::WalIo(format!("writer task join error: {e}")))?;
                    let completed_reads = reader_task
                        .await
                        .map_err(|e| VektorError::WalIo(format!("reader task join error: {e}")))?;

                    for (wid, id, shard_id, v) in acked_batch {
                        exact_oracle.upsert(id, &v)?;
                        acked_ledger.push((wid, id, shard_id));
                    }

                    steps_executed += 1;
                    step_logs.push(format!(
                        "ConcurrentWorkload(writes={writes}, reads_ok={completed_reads}/{reads})"
                    ));
                }
                ChaosStep::KillNode { node_id } => {
                    if let Some((_addr, _dir, handle)) = node_handles.get(node_id) {
                        handle.abort();
                    }
                    coord.handle_node_failure(*node_id).await?;
                    let _ = client.refresh_topology().await?;
                    steps_executed += 1;
                    step_logs.push(format!("KillNode(node_id={node_id})"));
                }
                ChaosStep::VerifyZeroWriteLoss => {
                    let mut durable_write_ids: HashSet<WriteId> =
                        HashSet::with_capacity(acked_ledger.len());
                    for shard_id in 0..self.config.num_shards {
                        let recs = client.verify_shard_wal_write_ids(shard_id).await?;
                        for rec in recs {
                            durable_write_ids.insert(rec.write_id);
                        }
                    }

                    let mut verified = 0_usize;
                    let mut lost = 0_usize;
                    for &(wid, _vec_id, _shard_id) in &acked_ledger {
                        if durable_write_ids.contains(&wid) {
                            verified += 1;
                        } else {
                            lost += 1;
                        }
                    }
                    acked_writes_verified = verified;
                    acked_writes_lost += lost;
                    if lost > 0 {
                        sla_passed = false;
                    }
                    steps_executed += 1;
                    step_logs.push(format!(
                        "VerifyZeroWriteLoss(verified={verified}, lost={lost})"
                    ));
                }
                ChaosStep::AddReplacementNode { node_id } => {
                    let (addr, dir, handle) =
                        spawn_chaos_node(*node_id, self.config.clone(), self.hnsw_params).await?;
                    node_handles.insert(*node_id, (addr.clone(), dir, handle));
                    let recovered_shards = coord.add_replacement_node(*node_id, addr).await?;
                    let _ = client.refresh_topology().await?;
                    steps_executed += 1;
                    step_logs.push(format!(
                        "AddReplacementNode(node_id={node_id}, recovered_shards={})",
                        recovered_shards.len()
                    ));
                }
                ChaosStep::AssertRecallAndLatencySla {
                    queries,
                    top_k,
                    ef_search,
                    min_recall_at_10,
                    max_p95_ms,
                    seed,
                } => {
                    let k = (*top_k).max(1);
                    let ef = (*ef_search).max(k);
                    let q_count = (*queries).max(1);
                    let mut rng = SeededRng::new(*seed);
                    let mut latencies_ms = Vec::with_capacity(q_count);
                    let mut recall_sum = 0.0_f64;

                    for _ in 0..q_count {
                        let mut q = vec![0.0_f32; self.config.dimensions];
                        rng.fill_unit_vector(&mut q);
                        let exact_hits = exact_oracle.search(&q, k)?;
                        let expected_ids: HashSet<VectorId> =
                            exact_hits.iter().map(|h| h.id).collect();

                        let t0 = Instant::now();
                        let approx_hits = client.search(&q, k, ef).await?;
                        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                        latencies_ms.push(elapsed_ms);

                        if expected_ids.is_empty() {
                            recall_sum += 1.0;
                        } else {
                            let matched = approx_hits
                                .iter()
                                .filter(|h| expected_ids.contains(&h.id))
                                .count();
                            recall_sum += (matched as f64) / (expected_ids.len() as f64);
                        }
                    }

                    measured_recall_at_10 = recall_sum / (q_count as f64);
                    let (_p50, p95, _p99) = compute_percentiles(&latencies_ms);
                    measured_p95_ms = p95;

                    let step_ok = acked_writes_lost == 0
                        && measured_recall_at_10 >= *min_recall_at_10
                        && measured_p95_ms <= *max_p95_ms;
                    if !step_ok {
                        sla_passed = false;
                    }
                    steps_executed += 1;
                    step_logs.push(format!(
                        "AssertRecallAndLatencySla(recall={measured_recall_at_10:.4}, p95_ms={measured_p95_ms:.3}, passed={step_ok})"
                    ));
                }
            }
        }

        coord_handle.abort();
        for (_id, (_addr, dir, handle)) in node_handles {
            handle.abort();
            let _ = std::fs::remove_dir_all(&dir);
        }

        Ok(ChaosCampaignReport {
            steps_executed,
            acked_writes_verified,
            acked_writes_lost,
            measured_recall_at_10,
            measured_p95_ms,
            sla_passed,
            step_logs,
        })
    }
}

async fn spawn_chaos_node(
    node_id: NodeId,
    config: ClusterConfig,
    params: HnswParams,
) -> Result<(String, PathBuf, JoinHandle<()>), VektorError> {
    let seq = CHAOS_DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("vektordist-m10-chaos-{pid}-{seq}-n{node_id}"));
    let _ = std::fs::remove_dir_all(&dir);
    let svc = NodeService::open(node_id, &dir, config, params)?;

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| VektorError::WalIo(e.to_string()))?;
    let addr = listener
        .local_addr()
        .map_err(|e| VektorError::WalIo(e.to_string()))?
        .to_string();
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);

    let handle = tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(ShardReplicaRpcServer::new(svc))
            .serve_with_incoming(incoming)
            .await;
    });

    Ok((addr, dir, handle))
}

async fn spawn_chaos_coord(
    coord: CoordinatorService,
) -> Result<(String, JoinHandle<()>), VektorError> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| VektorError::WalIo(e.to_string()))?;
    let addr = listener
        .local_addr()
        .map_err(|e| VektorError::WalIo(e.to_string()))?
        .to_string();
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);

    let handle = tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(TopologyRpcServer::new(coord))
            .serve_with_incoming(incoming)
            .await;
    });

    Ok((addr, handle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traceparent_roundtrip_and_child_span_propagation() {
        let header = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        let span = TraceSpan::parse_traceparent(header).unwrap();
        assert_eq!(span.version, 0);
        assert_eq!(span.trace_id, "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_eq!(span.span_id, "00f067aa0ba902b7");
        assert!(span.sampled);
        assert_eq!(span.to_traceparent(), header);

        let child = span.child_span();
        assert_eq!(child.trace_id, span.trace_id);
        assert_ne!(child.span_id, span.span_id);
        assert_eq!(child.parent_span_id.as_deref(), Some("00f067aa0ba902b7"));
    }

    #[test]
    fn invalid_traceparent_headers_are_rejected() {
        assert!(TraceSpan::parse_traceparent("invalid").is_err());
        assert!(TraceSpan::parse_traceparent(
            "00-00000000000000000000000000000000-00f067aa0ba902b7-01"
        )
        .is_err());
        assert!(TraceSpan::parse_traceparent(
            "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01"
        )
        .is_err());
    }

    #[test]
    fn prometheus_exposition_renders_all_required_metrics() {
        let m = PrometheusMetrics {
            active_nodes: 3,
            configured_nodes: 3,
            indexed_vectors: 1200,
            search_latency_p50_ms: 1.25,
            search_latency_p95_ms: 3.75,
            search_latency_p99_ms: 5.50,
            recall_at_10: 0.985,
            acked_writes_total: 1200,
            acked_writes_lost_total: 0,
        };
        let text = m.render_prometheus_text();
        assert!(text.contains("# TYPE vektordist_active_nodes gauge"));
        assert!(text.contains("vektordist_active_nodes 3"));
        assert!(text.contains("vektordist_indexed_vectors 1200"));
        assert!(text.contains("vektordist_recall_at_10 0.9850"));
        assert!(text.contains("# TYPE vektordist_acked_writes_lost_total counter"));
        assert!(text.contains("vektordist_acked_writes_lost_total 0"));
    }
}
