//! Milestone 18: Multi-collection namespace management (`CollectionRegistry`) with independent
//! dimensionality, distance metrics (`Cosine`, `L2`), quantization profiles (`F32`, `Sq8`, `Bq`),
//! and strict per-vector `OrgId` tenant isolation.

use std::collections::BTreeMap;
use std::fmt::{self, Write as _};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use vektordist_core::arena::{validate_vector, VectorArena};
use vektordist_core::exact::ScoredId;
use vektordist_core::hnsw::{HnswIndex, HnswParams};
use vektordist_core::pq_bq::BqIndex;
use vektordist_core::quant::{Sq8HnswIndex, Sq8Quantizer};
use vektordist_core::{DistanceMetric, VectorId, VektorError};

use crate::security::OrgId;
use crate::telemetry::{
    build_http_response, escape_json_str, extract_json_f32_array, extract_json_str,
    extract_json_u64, format_hits_json, TraceSpan,
};

/// Maximum permitted byte length of a collection identifier.
pub const MAX_COLLECTION_NAME_LEN: usize = 128;

/// Maximum permitted vector dimensionality for a single collection.
pub const MAX_COLLECTION_DIMENSIONS: usize = 65_536;

/// Storage and search quantization profile configured per collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CollectionQuantization {
    /// Full-precision 32-bit IEEE 754 floating-point HNSW index (`HnswIndex`).
    #[default]
    F32,
    /// 8-bit uniform scalar quantized HNSW index (`Sq8HnswIndex`, 4x payload compression).
    Sq8,
    /// 1-bit binary quantized `XOR + POPCNT` index with exact reranking (`BqIndex`, 32x code compression).
    Bq,
}

impl CollectionQuantization {
    /// Canonical lowercase JSON string representation (`"f32"`, `"sq8"`, `"bq"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Sq8 => "sq8",
            Self::Bq => "bq",
        }
    }

    /// Parse a case-insensitive quantization identifier (`"f32"`, `"sq8"`, `"bq"`).
    pub fn parse(raw: &str) -> Result<Self, CollectionError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "f32" => Ok(Self::F32),
            "sq8" => Ok(Self::Sq8),
            "bq" => Ok(Self::Bq),
            _ => Err(CollectionError::InvalidSpec(format!(
                "unsupported quantization '{raw}': expected 'f32', 'sq8', or 'bq'"
            ))),
        }
    }
}

impl fmt::Display for CollectionQuantization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Format a [`DistanceMetric`] into its canonical lowercase JSON label (`"cosine"` or `"l2"`).
#[must_use]
pub const fn metric_as_str(metric: DistanceMetric) -> &'static str {
    match metric {
        DistanceMetric::Cosine => "cosine",
        DistanceMetric::L2 => "l2",
    }
}

/// Parse a case-insensitive distance metric string (`"cosine"` or `"l2"`).
pub fn parse_distance_metric(raw: &str) -> Result<DistanceMetric, CollectionError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "cosine" => Ok(DistanceMetric::Cosine),
        "l2" | "euclidean" => Ok(DistanceMetric::L2),
        _ => Err(CollectionError::InvalidSpec(format!(
            "unsupported metric '{raw}': expected 'cosine' or 'l2'"
        ))),
    }
}

/// Errors produced by [`CollectionRegistry`] lifecycle and data-plane operations.
#[derive(Debug, PartialEq, Eq)]
pub enum CollectionError {
    /// A collection with this name already exists (`409 Conflict`).
    AlreadyExists(String),
    /// The target collection does not exist (`404 Not Found`).
    NotFound(String),
    /// Invalid collection name, dimensionality, metric, or quantization (`400 Bad Request`).
    InvalidSpec(String),
    /// Underlying vector validation or index error (`400 Bad Request`).
    Core(VektorError),
}

impl fmt::Display for CollectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyExists(name) => write!(f, "collection '{name}' already exists"),
            Self::NotFound(name) => write!(f, "collection '{name}' not found"),
            Self::InvalidSpec(msg) => write!(f, "invalid collection spec: {msg}"),
            Self::Core(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for CollectionError {}

impl From<VektorError> for CollectionError {
    fn from(value: VektorError) -> Self {
        Self::Core(value)
    }
}

/// Schema specification for creating a new isolated vector collection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionSpec {
    pub name: String,
    pub dimensions: usize,
    pub metric: DistanceMetric,
    pub quantization: CollectionQuantization,
}

impl CollectionSpec {
    /// Construct a new `CollectionSpec`.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        dimensions: usize,
        metric: DistanceMetric,
        quantization: CollectionQuantization,
    ) -> Self {
        Self {
            name: name.into(),
            dimensions,
            metric,
            quantization,
        }
    }

    /// Validate collection identifier characters and dimension bounds.
    pub fn validate(&self) -> Result<(), CollectionError> {
        validate_collection_name(&self.name)?;
        if self.dimensions == 0 || self.dimensions > MAX_COLLECTION_DIMENSIONS {
            return Err(CollectionError::InvalidSpec(format!(
                "dimensions must be in 1..={MAX_COLLECTION_DIMENSIONS}, got {}",
                self.dimensions
            )));
        }
        Ok(())
    }
}

/// Point-in-time metadata and resource usage summary for a registered collection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionInfo {
    pub name: String,
    pub dimensions: usize,
    pub metric: DistanceMetric,
    pub quantization: CollectionQuantization,
    pub live_vectors: usize,
    pub memory_bytes: usize,
}

impl CollectionInfo {
    /// Render this `CollectionInfo` as a compact JSON object.
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            "{{\"name\":\"{}\",\"dimensions\":{},\"metric\":\"{}\",\"quantization\":\"{}\",\"live_vectors\":{},\"memory_bytes\":{}}}",
            self.name,
            self.dimensions,
            metric_as_str(self.metric),
            self.quantization.as_str(),
            self.live_vectors,
            self.memory_bytes
        )
    }
}

enum TenantCollectionIndex {
    F32(HnswIndex),
    Sq8(Sq8HnswIndex),
    Bq { bq: BqIndex, rerank: VectorArena },
}

impl TenantCollectionIndex {
    fn new(spec: &CollectionSpec) -> Result<Self, CollectionError> {
        let params = HnswParams::default();
        match spec.quantization {
            CollectionQuantization::F32 => Ok(Self::F32(HnswIndex::new(
                spec.dimensions,
                spec.metric,
                params,
            )?)),
            CollectionQuantization::Sq8 => {
                let quantizer = match spec.metric {
                    DistanceMetric::Cosine => Sq8Quantizer::for_unit_vectors(spec.dimensions)?,
                    DistanceMetric::L2 => Sq8Quantizer::new(spec.dimensions, -16.0, 16.0)?,
                };
                Ok(Self::Sq8(Sq8HnswIndex::new(
                    quantizer,
                    spec.metric,
                    params,
                )?))
            }
            CollectionQuantization::Bq => Ok(Self::Bq {
                bq: BqIndex::new(spec.dimensions, spec.metric)?,
                rerank: VectorArena::new(spec.dimensions)?,
            }),
        }
    }

    fn upsert(&mut self, id: VectorId, vector: &[f32]) -> Result<(), CollectionError> {
        match self {
            Self::F32(idx) => {
                idx.upsert(id, vector)?;
            }
            Self::Sq8(idx) => {
                idx.upsert(id, vector)?;
            }
            Self::Bq { bq, rerank } => {
                bq.upsert(id, vector)?;
                rerank.upsert(id, vector)?;
            }
        }
        Ok(())
    }

    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
    ) -> Result<Vec<ScoredId>, CollectionError> {
        let k = top_k.max(1);
        let ef = ef_search.max(k);
        match self {
            Self::F32(idx) => Ok(idx.search_with_ef(query, k, ef)?),
            Self::Sq8(idx) => Ok(idx.search_with_ef(query, k, ef)?),
            Self::Bq { bq, rerank } => {
                let oversample = (ef / k).max(4);
                Ok(bq.search_reranked(query, k, oversample, rerank)?)
            }
        }
    }

    fn delete(&mut self, id: VectorId) -> bool {
        match self {
            Self::F32(idx) => idx.delete(id),
            Self::Sq8(idx) => idx.delete(id),
            Self::Bq { bq, rerank } => {
                let deleted_bq = bq.delete(id);
                let deleted_rerank = rerank.delete(id);
                deleted_bq || deleted_rerank
            }
        }
    }

    fn live_vectors(&self) -> usize {
        match self {
            Self::F32(idx) => idx.len(),
            Self::Sq8(idx) => idx.len(),
            Self::Bq { bq, .. } => bq.len(),
        }
    }

    fn memory_bytes(&self) -> usize {
        match self {
            Self::F32(idx) => idx.memory_bytes(),
            Self::Sq8(idx) => idx.memory_bytes(),
            Self::Bq { bq, rerank } => bq.arena().memory_bytes() + rerank.memory_bytes(),
        }
    }
}

struct CollectionInstance {
    spec: CollectionSpec,
    tenants: BTreeMap<OrgId, TenantCollectionIndex>,
}

impl CollectionInstance {
    fn new(spec: CollectionSpec) -> Self {
        Self {
            spec,
            tenants: BTreeMap::new(),
        }
    }

    fn info(&self) -> CollectionInfo {
        let mut live_vectors = 0_usize;
        let mut memory_bytes = 0_usize;
        for tenant_idx in self.tenants.values() {
            live_vectors = live_vectors.saturating_add(tenant_idx.live_vectors());
            memory_bytes = memory_bytes.saturating_add(tenant_idx.memory_bytes());
        }
        CollectionInfo {
            name: self.spec.name.clone(),
            dimensions: self.spec.dimensions,
            metric: self.spec.metric,
            quantization: self.spec.quantization,
            live_vectors,
            memory_bytes,
        }
    }
}

/// Thread-safe multi-collection namespace registry enforcing independent per-collection vector
/// schemas (`dimensions`, `metric`, `quantization`) and strict per-tenant `OrgId` isolation.
#[derive(Default)]
pub struct CollectionRegistry {
    collections: RwLock<BTreeMap<String, Arc<RwLock<CollectionInstance>>>>,
}

impl CollectionRegistry {
    /// Construct an empty `CollectionRegistry`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create and register a new collection from `spec`.
    pub fn create_collection(
        &self,
        spec: CollectionSpec,
    ) -> Result<CollectionInfo, CollectionError> {
        spec.validate()?;
        let mut guard = self
            .collections
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.contains_key(&spec.name) {
            return Err(CollectionError::AlreadyExists(spec.name));
        }
        let instance = CollectionInstance::new(spec.clone());
        let info = instance.info();
        guard.insert(spec.name, Arc::new(RwLock::new(instance)));
        Ok(info)
    }

    /// List metadata and live memory/vector counters for all collections sorted by `name`.
    #[must_use]
    pub fn list_collections(&self) -> Vec<CollectionInfo> {
        let guard = self
            .collections
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut out = Vec::with_capacity(guard.len());
        for col_arc in guard.values() {
            let col = col_arc
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            out.push(col.info());
        }
        out
    }

    /// Drop `name` from the registry and reclaim all associated index memory.
    pub fn drop_collection(&self, name: &str) -> Result<CollectionInfo, CollectionError> {
        let removed = {
            let mut guard = self
                .collections
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.remove(name)
        };
        let Some(col_arc) = removed else {
            return Err(CollectionError::NotFound(name.to_string()));
        };
        let col = col_arc
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Ok(col.info())
    }

    /// Upsert `(id, vector)` into `collection` strictly scoped to `org_id`.
    pub fn upsert(
        &self,
        collection: &str,
        org_id: &OrgId,
        id: VectorId,
        vector: &[f32],
    ) -> Result<(), CollectionError> {
        let col_arc = self.lookup(collection)?;
        let mut col = col_arc
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        validate_vector(vector, col.spec.dimensions)?;
        if !col.tenants.contains_key(org_id) {
            let created = TenantCollectionIndex::new(&col.spec)?;
            col.tenants.insert(org_id.clone(), created);
        }
        if let Some(tenant_idx) = col.tenants.get_mut(org_id) {
            tenant_idx.upsert(id, vector)?;
        }
        Ok(())
    }

    /// Search `collection` for top-`top_k` neighbors strictly scoped to `org_id`'s isolated index.
    pub fn search(
        &self,
        collection: &str,
        org_id: &OrgId,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
    ) -> Result<Vec<ScoredId>, CollectionError> {
        let col_arc = self.lookup(collection)?;
        let col = col_arc
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        validate_vector(query, col.spec.dimensions)?;
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let Some(tenant_idx) = col.tenants.get(org_id) else {
            return Ok(Vec::new());
        };
        tenant_idx.search(query, top_k, ef_search)
    }

    /// Delete `id` from `collection` strictly within `org_id`'s isolated namespace.
    pub fn delete(
        &self,
        collection: &str,
        org_id: &OrgId,
        id: VectorId,
    ) -> Result<bool, CollectionError> {
        let col_arc = self.lookup(collection)?;
        let mut col = col_arc
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(tenant_idx) = col.tenants.get_mut(org_id) else {
            return Ok(false);
        };
        Ok(tenant_idx.delete(id))
    }

    /// Render the full `GET /v1/collections` JSON payload.
    #[must_use]
    pub fn render_list_json(&self, trace_id: &str) -> String {
        let list = self.list_collections();
        let mut arr = String::from("[");
        for (idx, item) in list.iter().enumerate() {
            if idx > 0 {
                arr.push(',');
            }
            let _ = write!(arr, "{}", item.to_json());
        }
        arr.push(']');
        format!("{{\"status\":\"ok\",\"collections\":{arr},\"trace_id\":\"{trace_id}\"}}")
    }

    pub(crate) fn handle_http_create(
        &self,
        body: &str,
        span: &TraceSpan,
        start: Instant,
    ) -> String {
        let Some(name) = extract_json_str(body, "name") else {
            return col_err_http(400, "Bad Request", "missing or invalid 'name'", span, start);
        };
        let Some(dims_u64) = extract_json_u64(body, "dimensions") else {
            return col_err_http(
                400,
                "Bad Request",
                "missing or invalid 'dimensions'",
                span,
                start,
            );
        };
        let metric = match extract_json_str(body, "metric") {
            Some(raw) => match parse_distance_metric(&raw) {
                Ok(m) => m,
                Err(e) => return col_err_http(400, "Bad Request", &e.to_string(), span, start),
            },
            None => DistanceMetric::Cosine,
        };
        let quantization = match extract_json_str(body, "quantization") {
            Some(raw) => match CollectionQuantization::parse(&raw) {
                Ok(q) => q,
                Err(e) => return col_err_http(400, "Bad Request", &e.to_string(), span, start),
            },
            None => CollectionQuantization::F32,
        };
        let spec = CollectionSpec::new(name, dims_u64 as usize, metric, quantization);
        match self.create_collection(spec) {
            Ok(info) => {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                let resp_json = format!(
                    "{{\"status\":\"created\",\"collection\":{},\"trace_id\":\"{}\"}}",
                    info.to_json(),
                    span.trace_id
                );
                build_http_response(
                    201,
                    "Created",
                    "application/json",
                    span,
                    latency_ms,
                    &resp_json,
                )
            }
            Err(err) => map_col_err(&err, span, start),
        }
    }

    pub(crate) fn handle_http_drop(&self, name: &str, span: &TraceSpan, start: Instant) -> String {
        match self.drop_collection(name) {
            Ok(info) => {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                let resp_json = format!(
                    "{{\"status\":\"ok\",\"deleted\":\"{}\",\"trace_id\":\"{}\"}}",
                    info.name, span.trace_id
                );
                build_http_response(200, "OK", "application/json", span, latency_ms, &resp_json)
            }
            Err(err) => map_col_err(&err, span, start),
        }
    }

    pub(crate) fn handle_http_upsert(
        &self,
        name: &str,
        org_id: &OrgId,
        body: &str,
        span: &TraceSpan,
        start: Instant,
    ) -> String {
        let Some(id) = extract_json_u64(body, "id") else {
            return col_err_http(400, "Bad Request", "missing or invalid 'id'", span, start);
        };
        let Some(vector) = extract_json_f32_array(body, "vector") else {
            return col_err_http(
                400,
                "Bad Request",
                "missing or invalid 'vector'",
                span,
                start,
            );
        };
        match self.upsert(name, org_id, id, &vector) {
            Ok(()) => {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                let resp_json = format!(
                    "{{\"status\":\"ok\",\"collection\":\"{name}\",\"id\":{id},\"trace_id\":\"{}\"}}",
                    span.trace_id
                );
                build_http_response(200, "OK", "application/json", span, latency_ms, &resp_json)
            }
            Err(err) => map_col_err(&err, span, start),
        }
    }

    pub(crate) fn handle_http_search(
        &self,
        name: &str,
        org_id: &OrgId,
        body: &str,
        span: &TraceSpan,
        start: Instant,
    ) -> (String, Option<f64>) {
        let Some(query) = extract_json_f32_array(body, "query") else {
            return (
                col_err_http(
                    400,
                    "Bad Request",
                    "missing or invalid 'query'",
                    span,
                    start,
                ),
                None,
            );
        };
        let top_k = extract_json_u64(body, "top_k").unwrap_or(10) as usize;
        let ef_search = extract_json_u64(body, "ef_search").unwrap_or(64) as usize;
        match self.search(name, org_id, &query, top_k, ef_search) {
            Ok(hits) => {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                let hits_json = format_hits_json(&hits);
                let resp_json = format!(
                    "{{\"status\":\"ok\",\"collection\":\"{name}\",\"hits\":{hits_json},\"trace_id\":\"{}\"}}",
                    span.trace_id
                );
                (
                    build_http_response(
                        200,
                        "OK",
                        "application/json",
                        span,
                        latency_ms,
                        &resp_json,
                    ),
                    Some(latency_ms),
                )
            }
            Err(err) => (map_col_err(&err, span, start), None),
        }
    }

    pub(crate) fn handle_http_delete_vec(
        &self,
        name: &str,
        org_id: &OrgId,
        body: &str,
        span: &TraceSpan,
        start: Instant,
    ) -> String {
        let Some(id) = extract_json_u64(body, "id") else {
            return col_err_http(400, "Bad Request", "missing or invalid 'id'", span, start);
        };
        match self.delete(name, org_id, id) {
            Ok(deleted) => {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                let resp_json = format!(
                    "{{\"status\":\"ok\",\"collection\":\"{name}\",\"id\":{id},\"deleted\":{deleted},\"trace_id\":\"{}\"}}",
                    span.trace_id
                );
                build_http_response(200, "OK", "application/json", span, latency_ms, &resp_json)
            }
            Err(err) => map_col_err(&err, span, start),
        }
    }

    fn lookup(&self, name: &str) -> Result<Arc<RwLock<CollectionInstance>>, CollectionError> {
        let guard = self
            .collections
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard
            .get(name)
            .cloned()
            .ok_or_else(|| CollectionError::NotFound(name.to_string()))
    }
}

fn col_err_http(code: u16, reason: &str, msg: &str, span: &TraceSpan, start: Instant) -> String {
    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    let body = format!(
        "{{\"status\":\"error\",\"error\":\"{}\",\"trace_id\":\"{}\"}}",
        escape_json_str(msg),
        span.trace_id
    );
    build_http_response(code, reason, "application/json", span, latency_ms, &body)
}

fn map_col_err(err: &CollectionError, span: &TraceSpan, start: Instant) -> String {
    match err {
        CollectionError::AlreadyExists(_) => {
            col_err_http(409, "Conflict", &err.to_string(), span, start)
        }
        CollectionError::NotFound(_) => {
            col_err_http(404, "Not Found", &err.to_string(), span, start)
        }
        CollectionError::InvalidSpec(_) | CollectionError::Core(_) => {
            col_err_http(400, "Bad Request", &err.to_string(), span, start)
        }
    }
}

fn validate_collection_name(name: &str) -> Result<(), CollectionError> {
    if name.is_empty() || name.len() > MAX_COLLECTION_NAME_LEN {
        return Err(CollectionError::InvalidSpec(
            "collection name must be 1..=128 characters".to_string(),
        ));
    }
    if !name
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return Err(CollectionError::InvalidSpec(
            "collection name must contain only ASCII alphanumeric, '_', or '-' characters"
                .to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collection_registry_enforces_spec_validation_quantization_and_org_isolation() {
        let reg = CollectionRegistry::new();
        assert!(reg
            .create_collection(CollectionSpec::new(
                "bad/name",
                16,
                DistanceMetric::Cosine,
                CollectionQuantization::F32
            ))
            .is_err());
        assert!(reg
            .create_collection(CollectionSpec::new(
                "zero_dims",
                0,
                DistanceMetric::Cosine,
                CollectionQuantization::F32
            ))
            .is_err());

        for (name, quant) in [
            ("col_f32", CollectionQuantization::F32),
            ("col_sq8", CollectionQuantization::Sq8),
            ("col_bq", CollectionQuantization::Bq),
        ] {
            reg.create_collection(CollectionSpec::new(name, 8, DistanceMetric::Cosine, quant))
                .unwrap();
        }
        assert!(matches!(
            reg.create_collection(CollectionSpec::new(
                "col_f32",
                8,
                DistanceMetric::Cosine,
                CollectionQuantization::F32
            )),
            Err(CollectionError::AlreadyExists(_))
        ));

        let org_a = OrgId::new("org-a").unwrap();
        let org_b = OrgId::new("org-b").unwrap();
        let v1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        for col in ["col_f32", "col_sq8", "col_bq"] {
            reg.upsert(col, &org_a, 10, &v1).unwrap();
            reg.upsert(col, &org_b, 20, &v2).unwrap();

            let hits_a = reg.search(col, &org_a, &v1, 5, 32).unwrap();
            assert_eq!(hits_a.len(), 1);
            assert_eq!(hits_a[0].id, 10);

            let hits_b = reg.search(col, &org_b, &v1, 5, 32).unwrap();
            assert_eq!(hits_b.len(), 1);
            assert_eq!(hits_b[0].id, 20);

            assert!(!reg.delete(col, &org_b, 10).unwrap());
            assert!(reg.delete(col, &org_a, 10).unwrap());
            assert!(reg.search(col, &org_a, &v1, 5, 32).unwrap().is_empty());
        }
    }
}
