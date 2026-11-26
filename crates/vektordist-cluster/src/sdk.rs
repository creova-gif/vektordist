//! Ergonomic typed async Rust Client SDK (`HttpVektorClient`) over the HTTP/1.1 zero-trust
//! gateway with automatic `Authorization`, `X-Org-Id`, and W3C `traceparent` header injection,
//! bounded exponential retry on transient `429` / `503` responses, OpenAPI `3.1.0` discovery,
//! and multi-collection namespace management.

use std::fmt::Write as _;
use std::time::Duration;

use vektordist_core::exact::ScoredId;
use vektordist_core::{DistanceMetric, SeqNo, ShardId, VectorId, VektorError, WriteId};

use crate::collections::{metric_as_str, CollectionQuantization};
use crate::gateway::{send_http_request, HttpResponse, TraceSpan};

const DEFAULT_MAX_RETRIES: usize = 3;
const DEFAULT_BASE_BACKOFF: Duration = Duration::from_millis(25);
const MAX_BACKOFF_CAP: Duration = Duration::from_millis(500);

/// Structured response returned by [`HttpVektorClient::healthz`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthStatus {
    pub status_code: u16,
    pub ok: bool,
    pub active_nodes: u32,
    pub configured_nodes: u32,
    pub acked_writes_lost: u64,
    pub body: String,
}

/// Structured response returned by [`HttpVektorClient::readyz`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadinessStatus {
    pub status_code: u16,
    pub ready: bool,
    pub draining: bool,
    pub active_nodes: u32,
    pub under_quorum_shards: u32,
    pub body: String,
}

/// Ergonomic typed async HTTP/1.1 client for `HttpGatewayServer`.
#[derive(Debug, Clone)]
pub struct HttpVektorClient {
    gateway_addr: String,
    bearer_token: Option<String>,
    org_id: Option<String>,
    max_retries: usize,
    base_backoff: Duration,
}

impl HttpVektorClient {
    /// Construct a new `HttpVektorClient` targeting `gateway_addr` (`host:port` or `http://host:port`).
    #[must_use]
    pub fn new(gateway_addr: impl Into<String>) -> Self {
        let raw = gateway_addr.into();
        let trimmed = raw.trim();
        let without_scheme = trimmed
            .strip_prefix("http://")
            .or_else(|| trimmed.strip_prefix("https://"))
            .unwrap_or(trimmed);
        let clean = without_scheme.trim_end_matches('/').to_string();
        Self {
            gateway_addr: clean,
            bearer_token: None,
            org_id: None,
            max_retries: DEFAULT_MAX_RETRIES,
            base_backoff: DEFAULT_BASE_BACKOFF,
        }
    }

    /// Attach a scoped API bearer token (`vdk.<key_id>.<hex>`) injected as `Authorization: Bearer ...`.
    #[must_use]
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    /// Attach a tenant identifier injected as `X-Org-Id: ...`.
    #[must_use]
    pub fn with_org_id(mut self, org_id: impl Into<String>) -> Self {
        self.org_id = Some(org_id.into());
        self
    }

    /// Set the maximum retry count for transient `429 Too Many Requests` and `503 Service Unavailable` responses.
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Override the initial backoff delay used between transient retries.
    #[must_use]
    pub fn with_base_backoff(mut self, base_backoff: Duration) -> Self {
        self.base_backoff = base_backoff;
        self
    }

    /// Query `GET /healthz` without auth requirements.
    pub async fn healthz(&self) -> Result<HealthStatus, VektorError> {
        let resp = self.send_once("GET", "/healthz", "").await?;
        let active_nodes = parse_json_u64(&resp.body, "active_nodes").unwrap_or(0) as u32;
        let configured_nodes = parse_json_u64(&resp.body, "configured_nodes").unwrap_or(0) as u32;
        let acked_writes_lost = parse_json_u64(&resp.body, "acked_writes_lost").unwrap_or(0);
        Ok(HealthStatus {
            ok: resp.status_code == 200,
            status_code: resp.status_code,
            active_nodes,
            configured_nodes,
            acked_writes_lost,
            body: resp.body,
        })
    }

    /// Query `GET /readyz` (Kubernetes readiness probe; returns both `200` and `503` states cleanly
    /// without retrying `503` so callers can inspect live drain/quorum readiness immediately).
    pub async fn readyz(&self) -> Result<ReadinessStatus, VektorError> {
        let resp = self.send_once("GET", "/readyz", "").await?;
        let draining = parse_json_bool(&resp.body, "draining").unwrap_or(false);
        let active_nodes = parse_json_u64(&resp.body, "active_nodes").unwrap_or(0) as u32;
        let under_quorum_shards =
            parse_json_u64(&resp.body, "under_quorum_shards").unwrap_or(0) as u32;
        Ok(ReadinessStatus {
            ready: resp.status_code == 200,
            status_code: resp.status_code,
            draining,
            active_nodes,
            under_quorum_shards,
            body: resp.body,
        })
    }

    /// Fetch the public OpenAPI `3.1.0` JSON specification from `GET /openapi.json`.
    pub async fn openapi_raw(&self) -> Result<String, VektorError> {
        let resp = self.send_once("GET", "/openapi.json", "").await?;
        ensure_http_ok(&resp)?;
        Ok(resp.body)
    }

    /// Perform a single quorum-durable vector upsert via `POST /v1/upsert`.
    pub async fn upsert(
        &self,
        id: VectorId,
        write_id: WriteId,
        vector: &[f32],
    ) -> Result<(ShardId, SeqNo), VektorError> {
        let mut body = format!("{{\"id\":{id},\"write_id\":{write_id},\"vector\":[");
        append_f32_slice(&mut body, vector);
        body.push_str("]}");

        let resp = self.send_with_retry("POST", "/v1/upsert", &body).await?;
        ensure_http_ok(&resp)?;
        let shard_id = parse_json_u64(&resp.body, "shard_id").unwrap_or(0) as ShardId;
        let seq_no = parse_json_u64(&resp.body, "seq_no").unwrap_or(0);
        Ok((shard_id, seq_no))
    }

    /// Perform a high-throughput batched quorum upsert via `POST /v1/upsert_batch`.
    /// Returns the total count of durably acknowledged vectors.
    pub async fn upsert_batch(
        &self,
        records: &[(VectorId, WriteId, Vec<f32>)],
    ) -> Result<usize, VektorError> {
        let mut body = String::with_capacity(32 + records.len() * 64);
        body.push_str("{\"records\":[");
        for (idx, (id, write_id, vector)) in records.iter().enumerate() {
            if idx > 0 {
                body.push(',');
            }
            let _ = write!(body, "{{\"id\":{id},\"write_id\":{write_id},\"vector\":[");
            append_f32_slice(&mut body, vector);
            body.push_str("]}");
        }
        body.push_str("]}");

        let resp = self
            .send_with_retry("POST", "/v1/upsert_batch", &body)
            .await?;
        ensure_http_ok(&resp)?;
        let upserted = parse_json_u64(&resp.body, "upserted").ok_or(VektorError::InvalidConfig(
            "missing 'upserted' in upsert_batch response",
        ))? as usize;
        Ok(upserted)
    }

    /// Execute a distributed top-`k` scatter-gather search via `POST /v1/search`.
    pub async fn search(
        &self,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
    ) -> Result<Vec<ScoredId>, VektorError> {
        let mut body = format!("{{\"top_k\":{top_k},\"ef_search\":{ef_search},\"query\":[");
        append_f32_slice(&mut body, query);
        body.push_str("]}");

        let resp = self.send_with_retry("POST", "/v1/search", &body).await?;
        ensure_http_ok(&resp)?;
        parse_search_hits(&resp.body).ok_or(VektorError::InvalidConfig(
            "invalid 'hits' payload in search response",
        ))
    }

    /// Perform a quorum-durable vector delete via `POST /v1/delete`.
    pub async fn delete(
        &self,
        id: VectorId,
        write_id: WriteId,
    ) -> Result<(ShardId, SeqNo), VektorError> {
        let body = format!("{{\"id\":{id},\"write_id\":{write_id}}}");
        let resp = self.send_with_retry("POST", "/v1/delete", &body).await?;
        ensure_http_ok(&resp)?;
        let shard_id = parse_json_u64(&resp.body, "shard_id").unwrap_or(0) as ShardId;
        let seq_no = parse_json_u64(&resp.body, "seq_no").unwrap_or(0);
        Ok((shard_id, seq_no))
    }

    /// Fetch the raw JSON topology snapshot from `GET /v1/topology`.
    pub async fn topology_raw(&self) -> Result<String, VektorError> {
        let resp = self.send_with_retry("GET", "/v1/topology", "").await?;
        ensure_http_ok(&resp)?;
        Ok(resp.body)
    }

    /// Fetch the raw JSON cryptographic audit verification report from `GET /v1/audit`.
    pub async fn audit_raw(&self) -> Result<String, VektorError> {
        let resp = self.send_with_retry("GET", "/v1/audit", "").await?;
        ensure_http_ok(&resp)?;
        Ok(resp.body)
    }

    /// Create a new multi-collection vector namespace via `POST /v1/collections` (`TenantRole::Admin`).
    pub async fn create_collection(
        &self,
        name: &str,
        dims: usize,
        metric: DistanceMetric,
        quantization: CollectionQuantization,
    ) -> Result<String, VektorError> {
        let body = format!(
            "{{\"name\":\"{name}\",\"dimensions\":{dims},\"metric\":\"{}\",\"quantization\":\"{}\"}}",
            metric_as_str(metric),
            quantization.as_str()
        );
        let resp = self
            .send_with_retry("POST", "/v1/collections", &body)
            .await?;
        ensure_http_status(&resp, &[200, 201])?;
        Ok(resp.body)
    }

    /// List all collections in raw JSON format via `GET /v1/collections` (`TenantRole::Reader`).
    pub async fn list_collections_raw(&self) -> Result<String, VektorError> {
        let resp = self.send_with_retry("GET", "/v1/collections", "").await?;
        ensure_http_ok(&resp)?;
        Ok(resp.body)
    }

    /// Drop an existing collection via `DELETE /v1/collections/<name>` (`TenantRole::Admin`).
    pub async fn drop_collection(&self, name: &str) -> Result<(), VektorError> {
        let path = format!("/v1/collections/{name}");
        let resp = self.send_with_retry("DELETE", &path, "").await?;
        ensure_http_ok(&resp)?;
        Ok(())
    }

    /// Upsert `(id, vector)` into `collection` scoped to caller's `X-Org-Id` via `POST /v1/collections/<name>/upsert`.
    pub async fn collection_upsert(
        &self,
        collection: &str,
        id: VectorId,
        vector: &[f32],
    ) -> Result<(), VektorError> {
        let path = format!("/v1/collections/{collection}/upsert");
        let mut body = format!("{{\"id\":{id},\"vector\":[");
        append_f32_slice(&mut body, vector);
        body.push_str("]}");

        let resp = self.send_with_retry("POST", &path, &body).await?;
        ensure_http_ok(&resp)?;
        Ok(())
    }

    /// Execute a tenant-isolated top-`k` search inside `collection` via `POST /v1/collections/<name>/search`.
    pub async fn collection_search(
        &self,
        collection: &str,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
    ) -> Result<Vec<ScoredId>, VektorError> {
        let path = format!("/v1/collections/{collection}/search");
        let mut body = format!("{{\"top_k\":{top_k},\"ef_search\":{ef_search},\"query\":[");
        append_f32_slice(&mut body, query);
        body.push_str("]}");

        let resp = self.send_with_retry("POST", &path, &body).await?;
        ensure_http_ok(&resp)?;
        parse_search_hits(&resp.body).ok_or(VektorError::InvalidConfig(
            "invalid 'hits' payload in collection search response",
        ))
    }

    /// Delete `id` inside `collection` scoped to caller's `X-Org-Id` via `POST /v1/collections/<name>/delete`.
    pub async fn collection_delete(
        &self,
        collection: &str,
        id: VectorId,
    ) -> Result<bool, VektorError> {
        let path = format!("/v1/collections/{collection}/delete");
        let body = format!("{{\"id\":{id}}}");
        let resp = self.send_with_retry("POST", &path, &body).await?;
        ensure_http_ok(&resp)?;
        Ok(parse_json_bool(&resp.body, "deleted").unwrap_or(false))
    }

    /// Send an arbitrary HTTP request with automatic `Authorization`, `X-Org-Id`, and `traceparent`
    /// headers, returning the raw [`HttpResponse`] after bounded transient retries.
    pub async fn send_raw(
        &self,
        method: &str,
        path: &str,
        body: &str,
    ) -> Result<HttpResponse, VektorError> {
        self.send_with_retry(method, path, body).await
    }

    async fn send_with_retry(
        &self,
        method: &str,
        path: &str,
        body: &str,
    ) -> Result<HttpResponse, VektorError> {
        let mut attempt = 0_usize;
        loop {
            let resp = self.send_once(method, path, body).await?;
            let is_transient = resp.status_code == 429 || resp.status_code == 503;
            if !is_transient || attempt >= self.max_retries {
                return Ok(resp);
            }
            let shift = (attempt as u32).min(10);
            let factor = 1_u32 << shift;
            let delay = self
                .base_backoff
                .saturating_mul(factor)
                .min(MAX_BACKOFF_CAP);
            tokio::time::sleep(delay).await;
            attempt += 1;
        }
    }

    async fn send_once(
        &self,
        method: &str,
        path: &str,
        body: &str,
    ) -> Result<HttpResponse, VektorError> {
        let span = TraceSpan::generate();
        let traceparent = span.to_traceparent();
        let auth_header = self.bearer_token.as_ref().map(|tok| {
            let trimmed = tok.trim();
            if trimmed
                .get(..7)
                .is_some_and(|p| p.eq_ignore_ascii_case("bearer "))
            {
                trimmed.to_string()
            } else {
                format!("Bearer {trimmed}")
            }
        });

        let mut headers: Vec<(&str, &str)> = Vec::with_capacity(3);
        headers.push(("traceparent", &traceparent));
        if let Some(auth) = &auth_header {
            headers.push(("Authorization", auth.as_str()));
        }
        if let Some(org) = &self.org_id {
            headers.push(("X-Org-Id", org.as_str()));
        }

        send_http_request(&self.gateway_addr, method, path, &headers, body).await
    }
}

fn ensure_http_ok(resp: &HttpResponse) -> Result<(), VektorError> {
    ensure_http_status(resp, &[200])
}

fn ensure_http_status(resp: &HttpResponse, allowed: &[u16]) -> Result<(), VektorError> {
    if allowed.contains(&resp.status_code) {
        Ok(())
    } else {
        Err(VektorError::WalIo(format!(
            "HTTP {}: {}",
            resp.status_code, resp.body
        )))
    }
}

fn append_f32_slice(out: &mut String, values: &[f32]) {
    for (idx, val) in values.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        let _ = write!(out, "{val}");
    }
}

fn parse_json_u64(json: &str, key: &str) -> Option<u64> {
    let pattern = format!("\"{key}\"");
    let start_idx = json.find(&pattern)? + pattern.len();
    let rest = json.get(start_idx..)?.trim_start();
    let after_colon = rest.strip_prefix(':')?.trim_start();
    let num_end = after_colon
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after_colon.len());
    after_colon.get(..num_end)?.parse::<u64>().ok()
}

fn parse_json_bool(json: &str, key: &str) -> Option<bool> {
    let pattern = format!("\"{key}\"");
    let start_idx = json.find(&pattern)? + pattern.len();
    let rest = json.get(start_idx..)?.trim_start();
    let after_colon = rest.strip_prefix(':')?.trim_start();
    if after_colon.starts_with("true") {
        Some(true)
    } else if after_colon.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn parse_json_f32(json: &str, key: &str) -> Option<f32> {
    let pattern = format!("\"{key}\"");
    let start_idx = json.find(&pattern)? + pattern.len();
    let rest = json.get(start_idx..)?.trim_start();
    let after_colon = rest.strip_prefix(':')?.trim_start();
    let num_end = after_colon
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')))
        .unwrap_or(after_colon.len());
    after_colon.get(..num_end)?.parse::<f32>().ok()
}

fn parse_search_hits(json: &str) -> Option<Vec<ScoredId>> {
    let pattern = "\"hits\"";
    let start_idx = json.find(pattern)? + pattern.len();
    let rest = json.get(start_idx..)?.trim_start();
    let after_colon = rest.strip_prefix(':')?.trim_start();
    let after_open = after_colon.strip_prefix('[')?;
    let close_idx = after_open.find(']')?;
    let inner = after_open.get(..close_idx)?.trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }

    let mut hits = Vec::new();
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
        let end_brace = after_brace.find('}')?;
        let obj_str = after_brace.get(..end_brace)?;
        let id = parse_json_u64(obj_str, "id")?;
        let score = parse_json_f32(obj_str, "score")?;
        hits.push(ScoredId { id, score });
        cursor = after_brace.get(end_brace + 1..)?.trim_start();
    }
    Some(hits)
}
