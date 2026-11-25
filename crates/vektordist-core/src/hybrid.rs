use std::cmp::Ordering;
use std::collections::HashMap;

use crate::exact::ScoredId;
use crate::hnsw::{HnswIndex, HnswParams};
use crate::{DistanceMetric, VectorId, VektorError};

/// Default Reciprocal Rank Fusion (RRF) smoothing constant (`k = 60.0`).
pub const DEFAULT_RRF_K: f32 = 60.0;

/// Validated non-empty multi-tenant organization identifier (`org_id`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OrgId(String);

impl OrgId {
    /// Construct a validated `OrgId`, rejecting empty or whitespace-only tenant identifiers.
    pub fn new(raw: impl Into<String>) -> Result<Self, VektorError> {
        let s = raw.into();
        if s.trim().is_empty() {
            return Err(VektorError::InvalidConfig("org_id must not be empty"));
        }
        Ok(Self(s))
    }

    /// Borrow the underlying tenant identifier string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<&str> for OrgId {
    type Error = VektorError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<String> for OrgId {
    type Error = VektorError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

/// Typed metadata attribute value attached to a vector record.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl MetadataValue {
    #[must_use]
    fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Int(v) => Some(*v as f64),
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }
}

/// Key-value metadata map attached to a vector record.
pub type Metadata = HashMap<String, MetadataValue>;

/// Declarative metadata filter evaluated one-stage during HNSW candidate traversal
/// and BM25 lexical scoring.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataFilter {
    /// Match every record belonging to the authenticated caller tenant.
    All,
    /// Require `metadata[key] == value`.
    Eq { key: String, value: MetadataValue },
    /// Require `metadata[key]` to be a member of `values`.
    In {
        key: String,
        values: Vec<MetadataValue>,
    },
    /// Require numeric `metadata[key]` (`Int` or `Float`) to lie within `[min, max]` inclusive.
    Range {
        key: String,
        min: Option<f64>,
        max: Option<f64>,
    },
    /// Conjunction: all child filters must match.
    And(Vec<MetadataFilter>),
    /// Disjunction: at least one child filter must match.
    Or(Vec<MetadataFilter>),
}

impl MetadataFilter {
    /// Evaluate whether `metadata` satisfies this filter predicate.
    #[must_use]
    pub fn matches(&self, metadata: &Metadata) -> bool {
        match self {
            Self::All => true,
            Self::Eq { key, value } => metadata.get(key).is_some_and(|v| v == value),
            Self::In { key, values } => metadata
                .get(key)
                .is_some_and(|v| values.iter().any(|candidate| candidate == v)),
            Self::Range { key, min, max } => {
                let Some(val) = metadata.get(key).and_then(MetadataValue::as_f64) else {
                    return false;
                };
                if !val.is_finite() {
                    return false;
                }
                if let Some(lo) = min {
                    if val < *lo {
                        return false;
                    }
                }
                if let Some(hi) = max {
                    if val > *hi {
                        return false;
                    }
                }
                true
            }
            Self::And(children) => children.iter().all(|c| c.matches(metadata)),
            Self::Or(children) => children.iter().any(|c| c.matches(metadata)),
        }
    }
}

/// Case-folded alphanumeric tokenizer for BM25 indexing and querying.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' || ch == '-' {
            for lower in ch.to_lowercase() {
                current.push(lower);
            }
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

#[derive(Debug, Clone, Default)]
struct TenantBm25Index {
    /// `postings[term]` maps `VectorId -> term_frequency`.
    postings: HashMap<String, HashMap<VectorId, u32>>,
    /// Document token count per `VectorId`.
    doc_lengths: HashMap<VectorId, u32>,
    /// Sum of token counts across all live documents in this tenant.
    total_tokens: u64,
}

impl TenantBm25Index {
    fn upsert_doc(&mut self, id: VectorId, text: &str) {
        self.remove_doc(id);
        let tokens = tokenize(text);
        let doc_len = tokens.len() as u32;
        self.doc_lengths.insert(id, doc_len);
        self.total_tokens = self.total_tokens.saturating_add(u64::from(doc_len));

        let mut tf_map: HashMap<String, u32> = HashMap::new();
        for token in tokens {
            *tf_map.entry(token).or_insert(0) += 1;
        }
        for (term, tf) in tf_map {
            self.postings.entry(term).or_default().insert(id, tf);
        }
    }

    fn remove_doc(&mut self, id: VectorId) -> bool {
        let Some(prev_len) = self.doc_lengths.remove(&id) else {
            return false;
        };
        self.total_tokens = self.total_tokens.saturating_sub(u64::from(prev_len));
        self.postings.retain(|_, doc_map| {
            doc_map.remove(&id);
            !doc_map.is_empty()
        });
        true
    }

    fn search<F>(
        &self,
        query_terms: &[String],
        k: usize,
        k1: f32,
        b: f32,
        mut filter_fn: F,
    ) -> Vec<ScoredId>
    where
        F: FnMut(VectorId) -> bool,
    {
        if k == 0 || query_terms.is_empty() || self.doc_lengths.is_empty() {
            return Vec::new();
        }
        let n_docs = self.doc_lengths.len() as f32;
        let avgdl = (self.total_tokens as f32 / n_docs).max(1.0);

        let mut scores: HashMap<VectorId, f32> = HashMap::new();
        for term in query_terms {
            let Some(posting) = self.postings.get(term) else {
                continue;
            };
            let df = posting.len() as f32;
            // Lucene/Robertson non-negative BM25 IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
            let idf = (1.0_f32 + (n_docs - df + 0.5) / (df + 0.5)).ln();
            for (&doc_id, &tf_u32) in posting {
                if !filter_fn(doc_id) {
                    continue;
                }
                let doc_len = self.doc_lengths.get(&doc_id).copied().unwrap_or(1) as f32;
                let tf = tf_u32 as f32;
                let norm = 1.0 - b + b * (doc_len / avgdl);
                let term_score = idf * ((tf * (k1 + 1.0)) / (tf + k1 * norm));
                *scores.entry(doc_id).or_insert(0.0) += term_score;
            }
        }

        let mut hits: Vec<ScoredId> = scores
            .into_iter()
            .filter(|&(_, score)| score > 0.0)
            .map(|(id, score)| ScoredId { id, score })
            .collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        hits.truncate(k);
        hits
    }
}

/// Multi-tenant BM25 inverted index with per-tenant isolated posting lists (`k1 = 1.2`, `b = 0.75`).
#[derive(Debug, Clone)]
pub struct Bm25Index {
    k1: f32,
    b: f32,
    tenants: HashMap<OrgId, TenantBm25Index>,
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new(1.2, 0.75)
    }
}

impl Bm25Index {
    /// Construct a new BM25 inverted index with custom `k1` and `b` hyperparameters.
    #[must_use]
    pub fn new(k1: f32, b: f32) -> Self {
        Self {
            k1,
            b,
            tenants: HashMap::new(),
        }
    }

    /// Upsert a document under `org_id` and `id`.
    pub fn upsert(&mut self, org_id: &OrgId, id: VectorId, text: &str) {
        self.tenants
            .entry(org_id.clone())
            .or_default()
            .upsert_doc(id, text);
    }

    /// Remove `id` from `org_id`'s posting lists.
    pub fn delete(&mut self, org_id: &OrgId, id: VectorId) -> bool {
        self.tenants
            .get_mut(org_id)
            .is_some_and(|idx| idx.remove_doc(id))
    }

    /// Search `caller_org_id`'s isolated BM25 posting lists for `query_text`.
    #[must_use]
    pub fn search(&self, caller_org_id: &OrgId, query_text: &str, k: usize) -> Vec<ScoredId> {
        self.search_filtered(caller_org_id, query_text, k, |_| true)
    }

    /// Search `caller_org_id`'s isolated BM25 posting lists with an inline filter over `VectorId`.
    #[must_use]
    pub fn search_filtered<F>(
        &self,
        caller_org_id: &OrgId,
        query_text: &str,
        k: usize,
        filter_fn: F,
    ) -> Vec<ScoredId>
    where
        F: FnMut(VectorId) -> bool,
    {
        let Some(tenant_idx) = self.tenants.get(caller_org_id) else {
            return Vec::new();
        };
        let query_terms = tokenize(query_text);
        tenant_idx.search(&query_terms, k, self.k1, self.b, filter_fn)
    }
}

/// Result entry returned by `hybrid_search_rrf`.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridHit {
    pub id: VectorId,
    /// Reciprocal Rank Fusion score (`sum(1.0 / (rrf_k + rank_1_based))`).
    pub rrf_score: f32,
    /// 1-based rank in the dense HNSW results if present.
    pub dense_rank: Option<usize>,
    /// 1-based rank in the lexical BM25 results if present.
    pub lexical_rank: Option<usize>,
}

/// Combine dense and lexical ranked lists using Reciprocal Rank Fusion (`score = sum(1.0 / (rrf_k + rank))`).
#[must_use]
pub fn fuse_rankings_rrf(
    dense_hits: &[ScoredId],
    lexical_hits: &[ScoredId],
    top_k: usize,
    rrf_k: f32,
) -> Vec<HybridHit> {
    if top_k == 0 {
        return Vec::new();
    }
    let effective_rrf_k = if rrf_k > 0.0 && rrf_k.is_finite() {
        rrf_k
    } else {
        DEFAULT_RRF_K
    };

    let mut fused: HashMap<VectorId, HybridHit> =
        HashMap::with_capacity(dense_hits.len() + lexical_hits.len());

    for (idx, hit) in dense_hits.iter().enumerate() {
        let rank = idx + 1;
        let contrib = 1.0_f32 / (effective_rrf_k + rank as f32);
        let entry = fused.entry(hit.id).or_insert(HybridHit {
            id: hit.id,
            rrf_score: 0.0,
            dense_rank: None,
            lexical_rank: None,
        });
        entry.rrf_score += contrib;
        entry.dense_rank = Some(rank);
    }

    for (idx, hit) in lexical_hits.iter().enumerate() {
        let rank = idx + 1;
        let contrib = 1.0_f32 / (effective_rrf_k + rank as f32);
        let entry = fused.entry(hit.id).or_insert(HybridHit {
            id: hit.id,
            rrf_score: 0.0,
            dense_rank: None,
            lexical_rank: None,
        });
        entry.rrf_score += contrib;
        entry.lexical_rank = Some(rank);
    }

    let mut out: Vec<HybridHit> = fused.into_values().collect();
    out.sort_by(|a, b| {
        match b.rrf_score.total_cmp(&a.rrf_score) {
            Ordering::Equal => {
                // Prefer items appearing in both rankings, then smaller best rank, then smaller VectorId.
                let a_both = a.dense_rank.is_some() && a.lexical_rank.is_some();
                let b_both = b.dense_rank.is_some() && b.lexical_rank.is_some();
                b_both.cmp(&a_both).then_with(|| a.id.cmp(&b.id))
            }
            ord => ord,
        }
    });
    out.truncate(top_k);
    out
}

#[derive(Debug, Clone)]
struct RecordMeta {
    org_id: OrgId,
    metadata: Metadata,
}

/// Multi-tenant hybrid retrieval engine combining:
/// 1. Mandatory `OrgId` tenant isolation evaluated one-stage during HNSW traversal.
/// 2. One-stage metadata predicate filtering (`Eq`, `In`, `Range`, `And`, `Or`).
/// 3. Per-tenant BM25 lexical inverted index + Reciprocal Rank Fusion (`hybrid_search_rrf`).
#[derive(Debug, Clone)]
pub struct TenantHybridEngine {
    hnsw: HnswIndex,
    bm25: Bm25Index,
    records: HashMap<VectorId, RecordMeta>,
}

impl TenantHybridEngine {
    /// Create a new multi-tenant hybrid retrieval engine.
    pub fn new(
        dims: usize,
        metric: DistanceMetric,
        params: HnswParams,
    ) -> Result<Self, VektorError> {
        Ok(Self {
            hnsw: HnswIndex::new(dims, metric, params)?,
            bm25: Bm25Index::default(),
            records: HashMap::new(),
        })
    }

    /// Create a new multi-tenant hybrid retrieval engine with pre-allocated vector capacity.
    pub fn with_capacity(
        dims: usize,
        capacity: usize,
        metric: DistanceMetric,
        params: HnswParams,
    ) -> Result<Self, VektorError> {
        Ok(Self {
            hnsw: HnswIndex::with_capacity(dims, capacity, metric, params)?,
            bm25: Bm25Index::default(),
            records: HashMap::with_capacity(capacity),
        })
    }

    /// Total number of live records across all tenants.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the engine contains zero live records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Look up the owning `OrgId` of `id` (used for security verification).
    #[must_use]
    pub fn owner_of(&self, id: VectorId) -> Option<&OrgId> {
        self.records.get(&id).map(|r| &r.org_id)
    }

    /// Look up the metadata attached to `id`, scoped strictly to `caller_org_id`.
    #[must_use]
    pub fn get_metadata(&self, caller_org_id: &OrgId, id: VectorId) -> Option<&Metadata> {
        let rec = self.records.get(&id)?;
        if &rec.org_id == caller_org_id {
            Some(&rec.metadata)
        } else {
            None
        }
    }

    /// Upsert a multi-tenant record with dense `vector`, `metadata`, and lexical `text`.
    ///
    /// If `id` already existed under a previous tenant, its previous BM25 posting entries
    /// are cleaned up before re-indexing.
    pub fn upsert(
        &mut self,
        org_id: OrgId,
        id: VectorId,
        vector: &[f32],
        metadata: Metadata,
        text: &str,
    ) -> Result<usize, VektorError> {
        let slot = self.hnsw.upsert(id, vector)?;
        if let Some(prev) = self.records.insert(
            id,
            RecordMeta {
                org_id: org_id.clone(),
                metadata,
            },
        ) {
            if prev.org_id != org_id {
                self.bm25.delete(&prev.org_id, id);
            }
        }
        self.bm25.upsert(&org_id, id, text);
        Ok(slot)
    }

    /// Delete `id` only if it belongs to `caller_org_id`.
    ///
    /// Prevents cross-tenant deletion attacks: a tenant cannot delete another tenant's `VectorId`.
    pub fn delete(&mut self, caller_org_id: &OrgId, id: VectorId) -> bool {
        let Some(rec) = self.records.get(&id) else {
            return false;
        };
        if &rec.org_id != caller_org_id {
            return false;
        }
        self.records.remove(&id);
        self.bm25.delete(caller_org_id, id);
        self.hnsw.delete(id)
    }

    /// One-stage tenant-isolated and metadata-filtered dense HNSW search.
    pub fn search_dense_filtered(
        &self,
        caller_org_id: &OrgId,
        query_vector: &[f32],
        filter: &MetadataFilter,
        top_k: usize,
    ) -> Result<Vec<ScoredId>, VektorError> {
        self.hnsw.search_filtered_with_ef(
            query_vector,
            top_k,
            self.hnsw.params().ef_search,
            |_slot, id| {
                self.records.get(&id).is_some_and(|rec| {
                    &rec.org_id == caller_org_id && filter.matches(&rec.metadata)
                })
            },
        )
    }

    /// Tenant-isolated and metadata-filtered BM25 lexical search.
    #[must_use]
    pub fn search_lexical_filtered(
        &self,
        caller_org_id: &OrgId,
        query_text: &str,
        filter: &MetadataFilter,
        top_k: usize,
    ) -> Vec<ScoredId> {
        self.bm25
            .search_filtered(caller_org_id, query_text, top_k, |id| {
                self.records.get(&id).is_some_and(|rec| {
                    &rec.org_id == caller_org_id && filter.matches(&rec.metadata)
                })
            })
    }

    /// Execute hybrid dense (HNSW) + lexical (BM25) search fused via Reciprocal Rank Fusion (`RRF`),
    /// strictly scoped to `caller_org_id` and matching `filter`.
    pub fn hybrid_search_rrf(
        &self,
        caller_org_id: &OrgId,
        query_vector: &[f32],
        query_text: &str,
        filter: &MetadataFilter,
        top_k: usize,
        rrf_k: f32,
    ) -> Result<Vec<HybridHit>, VektorError> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let candidate_depth = top_k.saturating_mul(4).max(32);
        let dense_hits =
            self.search_dense_filtered(caller_org_id, query_vector, filter, candidate_depth)?;
        let lexical_hits =
            self.search_lexical_filtered(caller_org_id, query_text, filter, candidate_depth);
        Ok(fuse_rankings_rrf(&dense_hits, &lexical_hits, top_k, rrf_k))
    }
}
