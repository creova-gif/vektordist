//! Gateway telemetry (`PrometheusMetrics`, W3C `TraceSpan`, `BoundedLatencyWindow`), OpenAPI
//! `3.1.0` specification, and shared zero-dependency HTTP/JSON wire helpers.

use std::collections::{BTreeMap, VecDeque};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};

use vektordist_core::exact::{ScoredId, SeededRng};
use vektordist_core::{VectorId, VektorError, WriteId};

use crate::gateway::HttpResponse;

static TRACE_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Default capacity of [`BoundedLatencyWindow`] (`1,024` recent search latency samples).
pub const DEFAULT_LATENCY_WINDOW_CAPACITY: usize = 1_024;

/// Fixed-capacity sliding-window ring buffer of latency observations (in milliseconds),
/// bounding telemetry memory to `O(capacity)` and percentile computation to `O(capacity log capacity)`.
#[derive(Debug, Clone)]
pub struct BoundedLatencyWindow {
    samples: VecDeque<f64>,
    capacity: usize,
}

impl Default for BoundedLatencyWindow {
    fn default() -> Self {
        Self::new(DEFAULT_LATENCY_WINDOW_CAPACITY)
    }
}

impl BoundedLatencyWindow {
    /// Construct a new `BoundedLatencyWindow` retaining at most `capacity.max(1)` samples.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            samples: VecDeque::with_capacity(cap),
            capacity: cap,
        }
    }

    /// Record a new finite non-negative latency observation in milliseconds, evicting the oldest
    /// sample in FIFO order if the ring buffer is at capacity.
    pub fn record(&mut self, ms: f64) {
        if !ms.is_finite() || ms < 0.0 {
            return;
        }
        while self.samples.len() >= self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(ms);
    }

    /// Compute `(p50_ms, p95_ms, p99_ms)` over the currently retained window.
    #[must_use]
    pub fn percentiles(&self) -> (f64, f64, f64) {
        if self.samples.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let mut sorted: Vec<f64> = self.samples.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        (
            percentile_sorted(&sorted, 0.50),
            percentile_sorted(&sorted, 0.95),
            percentile_sorted(&sorted, 0.99),
        )
    }

    /// Number of latency samples currently stored in the ring buffer (`<= capacity()`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Return `true` if no latency samples have been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Maximum number of samples retained by this window.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Compute `(p50, p95, p99)` over an arbitrary slice of latency samples.
#[must_use]
pub(crate) fn compute_percentiles(samples: &[f64]) -> (f64, f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    (
        percentile_sorted(&sorted, 0.50),
        percentile_sorted(&sorted, 0.95),
        percentile_sorted(&sorted, 0.99),
    )
}

fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len().saturating_sub(1) as f64) * q.clamp(0.0, 1.0)).round() as usize;
    sorted.get(idx).copied().unwrap_or(0.0)
}

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
        let _ = writeln!(out, "# HELP vektordist_search_latency_p50_ms 50th percentile scatter-gather search latency in milliseconds.");
        let _ = writeln!(out, "# TYPE vektordist_search_latency_p50_ms gauge");
        let _ = writeln!(
            out,
            "vektordist_search_latency_p50_ms {:.4}",
            self.search_latency_p50_ms
        );
        let _ = writeln!(out, "# HELP vektordist_search_latency_p95_ms 95th percentile scatter-gather search latency in milliseconds.");
        let _ = writeln!(out, "# TYPE vektordist_search_latency_p95_ms gauge");
        let _ = writeln!(
            out,
            "vektordist_search_latency_p95_ms {:.4}",
            self.search_latency_p95_ms
        );
        let _ = writeln!(out, "# HELP vektordist_search_latency_p99_ms 99th percentile scatter-gather search latency in milliseconds.");
        let _ = writeln!(out, "# TYPE vektordist_search_latency_p99_ms gauge");
        let _ = writeln!(
            out,
            "vektordist_search_latency_p99_ms {:.4}",
            self.search_latency_p99_ms
        );
        let _ = writeln!(out, "# HELP vektordist_recall_at_10 Measured top-10 recall against exact brute-force ground truth.");
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
        let _ = writeln!(out, "# HELP vektordist_acked_writes_lost_total Total quorum-acknowledged writes lost across failovers.");
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
        let parts: Vec<&str> = header.trim().split('-').collect();
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

/// Static OpenAPI `3.1.0` JSON document describing all public and zero-trust gateway endpoints.
#[must_use]
pub const fn openapi_spec_json() -> &'static str {
    concat!(
        "{\"openapi\":\"3.1.0\",",
        "\"info\":{\"title\":\"VektorDist Zero-Trust Gateway API\",\"version\":\"0.4.0\",",
        "\"description\":\"Distributed quorum-durable vector database and multi-collection namespace gateway.\"},",
        "\"components\":{\"securitySchemes\":{\"BearerAuth\":{\"type\":\"http\",\"scheme\":\"bearer\"}}},",
        "\"paths\":{",
        "\"/\":{\"get\":{\"summary\":\"Interactive Web Control Console\",\"responses\":{\"200\":{\"description\":\"Interactive Web Control Console\"}}}},",
        "\"/console\":{\"get\":{\"summary\":\"Interactive Web Control Console\",\"responses\":{\"200\":{\"description\":\"Interactive Web Control Console\"}}}},",
        "\"/healthz\":{\"get\":{\"summary\":\"Kubernetes liveness probe\",\"responses\":{\"200\":{\"description\":\"Alive\"}}}},",
        "\"/readyz\":{\"get\":{\"summary\":\"Kubernetes quorum and drain readiness probe\",\"responses\":{\"200\":{\"description\":\"Ready\"},\"503\":{\"description\":\"Draining or under quorum\"}}}},",
        "\"/metrics\":{\"get\":{\"summary\":\"Prometheus 0.0.4 exposition endpoint\",\"responses\":{\"200\":{\"description\":\"Prometheus text metrics\"}}}},",
        "\"/openapi.json\":{\"get\":{\"summary\":\"OpenAPI 3.1.0 specification\",\"responses\":{\"200\":{\"description\":\"OpenAPI JSON document\"}}}},",
        "\"/v1/upsert\":{\"post\":{\"summary\":\"Quorum-durable single vector upsert\",\"security\":[{\"BearerAuth\":[]}],\"responses\":{\"200\":{\"description\":\"Upserted\"},\"400\":{\"description\":\"Invalid vector or id\"},\"503\":{\"description\":\"Quorum unavailable\"}}}},",
        "\"/v1/upsert_batch\":{\"post\":{\"summary\":\"Batched quorum-durable vector upsert\",\"security\":[{\"BearerAuth\":[]}],\"responses\":{\"200\":{\"description\":\"Batch upserted\"},\"400\":{\"description\":\"Invalid batch\"},\"413\":{\"description\":\"Payload too large\"}}}},",
        "\"/v1/search\":{\"post\":{\"summary\":\"Distributed scatter-gather top-k vector search\",\"security\":[{\"BearerAuth\":[]}],\"responses\":{\"200\":{\"description\":\"Search hits\"}}}},",
        "\"/v1/delete\":{\"post\":{\"summary\":\"Quorum-durable vector delete\",\"security\":[{\"BearerAuth\":[]}],\"responses\":{\"200\":{\"description\":\"Deleted\"}}}},",
        "\"/v1/topology\":{\"get\":{\"summary\":\"Cluster shard topology snapshot\",\"security\":[{\"BearerAuth\":[]}],\"responses\":{\"200\":{\"description\":\"Topology snapshot\"}}}},",
        "\"/v1/audit\":{\"get\":{\"summary\":\"Tamper-evident hash-chained audit verification\",\"security\":[{\"BearerAuth\":[]}],\"responses\":{\"200\":{\"description\":\"Audit chain verified\"}}}},",
        "\"/v1/collections\":{",
        "\"get\":{\"summary\":\"List multi-collection namespaces\",\"security\":[{\"BearerAuth\":[]}],\"responses\":{\"200\":{\"description\":\"Collection list\"}}},",
        "\"post\":{\"summary\":\"Create isolated vector collection\",\"security\":[{\"BearerAuth\":[]}],\"responses\":{\"201\":{\"description\":\"Created\"},\"400\":{\"description\":\"Invalid spec\"},\"409\":{\"description\":\"Duplicate collection\"}}}",
        "},",
        "\"/v1/collections/{name}\":{\"delete\":{\"summary\":\"Drop vector collection\",\"security\":[{\"BearerAuth\":[]}],\"responses\":{\"200\":{\"description\":\"Dropped\"},\"404\":{\"description\":\"Collection not found\"}}}},",
        "\"/v1/collections/{name}/upsert\":{\"post\":{\"summary\":\"Tenant-isolated collection vector upsert\",\"security\":[{\"BearerAuth\":[]}],\"responses\":{\"200\":{\"description\":\"Upserted\"},\"400\":{\"description\":\"Dimension mismatch\"},\"404\":{\"description\":\"Collection not found\"}}}},",
        "\"/v1/collections/{name}/search\":{\"post\":{\"summary\":\"Tenant-isolated collection top-k search\",\"security\":[{\"BearerAuth\":[]}],\"responses\":{\"200\":{\"description\":\"Search hits\"},\"404\":{\"description\":\"Collection not found\"}}}},",
        "\"/v1/collections/{name}/delete\":{\"post\":{\"summary\":\"Tenant-isolated collection vector delete\",\"security\":[{\"BearerAuth\":[]}],\"responses\":{\"200\":{\"description\":\"Deleted\"},\"404\":{\"description\":\"Collection not found\"}}}}",
        "}}"
    )
}

pub(crate) fn split_http_head_and_body(raw: &str) -> (&str, &str) {
    if let Some((head, body)) = raw.split_once("\r\n\r\n") {
        (head, body)
    } else if let Some((head, body)) = raw.split_once("\n\n") {
        (head, body)
    } else {
        (raw, "")
    }
}

pub(crate) fn build_http_response(
    status_code: u16,
    reason: &str,
    content_type: &str,
    span: &TraceSpan,
    latency_ms: f64,
    body: &str,
) -> String {
    build_http_response_with_headers(
        status_code,
        reason,
        content_type,
        span,
        latency_ms,
        &[],
        body,
    )
}

pub(crate) fn build_http_response_with_headers(
    status_code: u16,
    reason: &str,
    content_type: &str,
    span: &TraceSpan,
    latency_ms: f64,
    extra_headers: &[(&str, &str)],
    body: &str,
) -> String {
    const DEFAULT_SECURITY_HEADERS: [(&str, &str); 4] = [
        ("x-content-type-options", "nosniff"),
        ("x-frame-options", "DENY"),
        ("referrer-policy", "no-referrer"),
        ("cache-control", "no-store"),
    ];

    let mut out = format!(
        "HTTP/1.1 {status_code} {reason}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nx-trace-id: {}\r\ntraceparent: {}\r\nx-latency-ms: {latency_ms:.3}\r\n",
        body.len(),
        span.trace_id,
        span.to_traceparent(),
    );
    for &(sec_k, sec_v) in &DEFAULT_SECURITY_HEADERS {
        let already_set = extra_headers
            .iter()
            .any(|&(k, _)| k.eq_ignore_ascii_case(sec_k));
        if !already_set {
            let _ = write!(out, "{sec_k}: {sec_v}\r\n");
        }
    }
    for &(k, v) in extra_headers {
        let _ = write!(out, "{k}: {v}\r\n");
    }
    out.push_str("connection: close\r\n\r\n");
    out.push_str(body);
    out
}

pub(crate) fn parse_http_response(raw: &str) -> Result<HttpResponse, VektorError> {
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

pub(crate) fn format_hits_json(hits: &[ScoredId]) -> String {
    let mut hits_json = String::from("[");
    for (idx, h) in hits.iter().enumerate() {
        if idx > 0 {
            hits_json.push(',');
        }
        let _ = write!(hits_json, "{{\"id\":{},\"score\":{:.6}}}", h.id, h.score);
    }
    hits_json.push(']');
    hits_json
}

pub(crate) fn escape_json_str(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Locate the JSON value slice immediately following `"<key>":` while scanning strictly outside
/// JSON string values so attacker-controlled string payloads (e.g. `{"note": "\"id\": 999", "id": 42}`)
/// can never shadow real object keys.
fn find_json_key_value<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let bytes = json.as_bytes();
    let mut idx = 0_usize;

    while idx < bytes.len() {
        if bytes[idx] != b'"' {
            idx += 1;
            continue;
        }

        // Parse JSON string literal starting at `idx`, decoding escapes if needed to compare with `key`.
        let str_start = idx + 1;
        let mut cursor = str_start;
        let mut escaped = false;
        let mut has_escapes = false;
        let mut str_end_opt: Option<usize> = None;

        while cursor < bytes.len() {
            let b = bytes[cursor];
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
                has_escapes = true;
            } else if b == b'"' {
                str_end_opt = Some(cursor);
                break;
            }
            cursor += 1;
        }

        let str_end = str_end_opt?;
        let after_quote = json.get(str_end + 1..)?;
        let trimmed_after = after_quote.trim_start();

        // A JSON string is an object key iff the next non-whitespace token is `:`.
        if let Some(after_colon) = trimmed_after.strip_prefix(':') {
            let raw_key_slice = json.get(str_start..str_end)?;
            let matches_key = if !has_escapes {
                raw_key_slice == key
            } else {
                unescape_json_simple(raw_key_slice).as_deref() == Some(key)
            };
            if matches_key {
                return Some(after_colon.trim_start());
            }
        }

        // Advance past the closing quote of this string (whether it was a non-matching key or a string value!).
        idx = str_end + 1;
    }

    None
}

fn unescape_json_simple(raw: &str) -> Option<String> {
    let mut out = String::with_capacity(raw.len());
    let mut escaped = false;
    for ch in raw.chars() {
        if escaped {
            out.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else {
            out.push(ch);
        }
    }
    if escaped {
        None
    } else {
        Some(out)
    }
}

pub(crate) fn extract_json_str(json: &str, key: &str) -> Option<String> {
    let val = find_json_key_value(json, key)?;
    let after_quote = val.strip_prefix('"')?;
    let mut out = String::new();
    let mut escaped = false;
    for ch in after_quote.chars() {
        if escaped {
            out.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(out);
        } else {
            out.push(ch);
        }
    }
    None
}

pub(crate) fn extract_json_u64(json: &str, key: &str) -> Option<u64> {
    let after_colon = find_json_key_value(json, key)?;
    let num_end = after_colon
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after_colon.len());
    after_colon.get(..num_end)?.parse::<u64>().ok()
}

pub(crate) fn extract_json_f32_array(json: &str, key: &str) -> Option<Vec<f32>> {
    let after_colon = find_json_key_value(json, key)?;
    let after_open = after_colon.strip_prefix('[')?;
    let close_idx = find_matching_delimiter(after_open, b'[', b']')?;
    let inner = after_open.get(..close_idx)?.trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    let mut out = Vec::new();
    for item in inner.split(',') {
        let token = item.trim();
        if token.is_empty() {
            return None;
        }
        // Reject explicit non-JSON float tokens like NaN/inf/Infinity as well as overflow (e.g. 1e999 -> inf).
        let val = token.parse::<f32>().ok()?;
        if !val.is_finite() {
            return None;
        }
        out.push(val);
    }
    Some(out)
}

pub(crate) fn extract_json_upsert_batch(json: &str) -> Option<Vec<(VectorId, WriteId, Vec<f32>)>> {
    let after_colon = find_json_key_value(json, "records")?;
    let after_open = after_colon.strip_prefix('[')?;
    let array_end = find_matching_delimiter(after_open, b'[', b']')?;
    let inner = after_open.get(..array_end)?.trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }

    let mut records = Vec::new();
    let mut cursor = inner;
    while !cursor.is_empty() {
        cursor = cursor.trim_start();
        if let Some(stripped) = cursor.strip_prefix(',') {
            cursor = stripped.trim_start();
        }
        if cursor.is_empty() {
            break;
        }
        let after_brace = cursor.strip_prefix('{')?;
        let obj_end = find_matching_delimiter(after_brace, b'{', b'}')?;
        let obj_body = after_brace.get(..obj_end)?;

        let id = extract_json_u64(obj_body, "id")?;
        let write_id = extract_json_u64(obj_body, "write_id").unwrap_or(id);
        let vector = extract_json_f32_array(obj_body, "vector")?;
        records.push((id, write_id, vector));
        cursor = after_brace.get(obj_end + 1..)?.trim_start();
    }
    Some(records)
}

fn find_matching_delimiter(input: &str, open: u8, close: u8) -> Option<usize> {
    let mut depth = 1_usize;
    let mut in_string = false;
    let mut escaped = false;
    for (idx, &b) in input.as_bytes().iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        if b == b'"' {
            in_string = true;
        } else if b == open {
            depth = depth.saturating_add(1);
        } else if b == close {
            depth -= 1;
            if depth == 0 {
                return Some(idx);
            }
        }
    }
    None
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

    #[test]
    fn bounded_latency_window_evicts_fifo_and_computes_percentiles() {
        let mut win = BoundedLatencyWindow::default();
        assert_eq!(win.capacity(), 1024);
        assert!(win.is_empty());
        assert_eq!(win.percentiles(), (0.0, 0.0, 0.0));

        for i in 1..=3000 {
            win.record(f64::from(i));
        }
        assert_eq!(win.len(), 1024);
        let (p50, p95, p99) = win.percentiles();
        assert!((2480.0..=2500.0).contains(&p50));
        assert!((2940.0..=2960.0).contains(&p95));
        assert!((2985.0..=3000.0).contains(&p99));
    }
}
