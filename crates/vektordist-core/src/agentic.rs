//! Milestone 9 (M9): Programmable Agentic RAG Memory, Late-Interaction Multi-Vector
//! `ColBERT MaxSim`, Time-Decay Episodic Scoring, and Tenant-Isolated Semantic Query Cache.
//!
//! # Architecture Overview
//! 1. **Late-Interaction Multi-Vector Retrieval (`ColbertDocument`, `ColbertIndex`, `maxsim_score`):**
//!    Stores L2-normalized token-level embeddings (`Vec<Vec<f32>>`) per document and evaluates
//!    exact late-interaction `MaxSim`:
//!    $$\text{MaxSim}(Q, D) = \sum_{q \in Q} \max_{d \in D} (q \cdot d)$$
//!    using hardware-accelerated SIMD dot products (`crate::arena::dot_product`).
//! 2. **Episodic Agent Memory Store (`MemoryEntry`, `MemoryScoringWeights`, `AgentMemoryStore`):**
//!    Stores tenant-scoped (`OrgId`) episodic memories and ranks them via a programmable
//!    composite objective combining semantic similarity, exponential time-decay recency,
//!    and memory importance:
//!    $$\text{score}(m) = w_{\text{sem}} \cdot \cos(q, m) + w_{\text{rec}} \cdot e^{-\lambda \cdot \Delta t} + w_{\text{imp}} \cdot \text{importance}(m)$$
//!    Greedily packs top-scoring memories into an LLM context window subject to `max_token_budget`
//!    while updating `access_count` on retrieved memories.
//! 3. **Tenant-Isolated Semantic Query Cache (`SemanticCache`, `CacheEntry`):**
//!    Caches `(OrgId, filter_hash, query_vector) -> Vec<ScoredId>` with configurable cosine
//!    similarity threshold (e.g. `>= 0.96`), strict tenant + filter isolation, TTL expiration,
//!    and LRU/FIFO bounded capacity eviction.

use std::collections::{HashMap, VecDeque};
use std::ops::Deref;

use crate::arena::{dot_product, normalize_l2, validate_vector};
use crate::exact::ScoredId;
use crate::hybrid::{tokenize, OrgId};
use crate::{VectorId, VektorError};

/// Compute the exact ColBERT late-interaction `MaxSim` score between L2-normalized
/// query token vectors and document token vectors:
///
/// `sum_{q in query_tokens} max_{d in doc_tokens} dot_product(q, d)`
///
/// Returns `0.0` if either token slice is empty.
#[must_use]
pub fn maxsim_score(query_tokens: &[Vec<f32>], doc_tokens: &[Vec<f32>]) -> f32 {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }
    let mut total = 0.0_f32;
    for q in query_tokens {
        let mut max_sim = f32::NEG_INFINITY;
        for d in doc_tokens {
            let sim = dot_product(q, d);
            if sim > max_sim {
                max_sim = sim;
            }
        }
        if max_sim.is_finite() {
            total += max_sim;
        }
    }
    total
}

/// Multi-vector document representation for ColBERT late-interaction scoring.
///
/// Each token vector in `token_vectors` is validated to match the index dimensionality
/// and L2-normalized upon insertion into `ColbertIndex`.
#[derive(Debug, Clone, PartialEq)]
pub struct ColbertDocument {
    pub id: VectorId,
    pub token_vectors: Vec<Vec<f32>>,
}

/// Multi-vector late-interaction index supporting exact ColBERT `MaxSim` search and
/// candidate re-ranking.
#[derive(Debug, Clone)]
pub struct ColbertIndex {
    dims: usize,
    docs: HashMap<VectorId, ColbertDocument>,
}

impl ColbertIndex {
    /// Construct an empty `ColbertIndex` for token embeddings of dimension `dims`.
    pub fn new(dims: usize) -> Result<Self, VektorError> {
        if dims == 0 {
            return Err(VektorError::InvalidConfig(
                "colbert token dimensions must be > 0",
            ));
        }
        Ok(Self {
            dims,
            docs: HashMap::new(),
        })
    }

    /// Fixed dimensionality of each token embedding in this index.
    #[must_use]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Number of indexed multi-vector documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Whether the index contains zero documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Look up a stored `ColbertDocument` by `id`.
    #[must_use]
    pub fn get(&self, id: VectorId) -> Option<&ColbertDocument> {
        self.docs.get(&id)
    }

    /// Upsert a multi-vector document by `id`.
    ///
    /// Every token vector is validated for dimensionality and finiteness, then L2-normalized
    /// so subsequent `MaxSim` evaluations execute pure SIMD dot products.
    pub fn upsert(
        &mut self,
        id: VectorId,
        token_vectors: Vec<Vec<f32>>,
    ) -> Result<(), VektorError> {
        if token_vectors.is_empty() {
            return Err(VektorError::InvalidVector(
                "colbert document must contain at least one token vector",
            ));
        }
        let mut normalized_tokens = Vec::with_capacity(token_vectors.len());
        for mut vec in token_vectors {
            validate_vector(&vec, self.dims)?;
            if !normalize_l2(&mut vec) {
                return Err(VektorError::InvalidVector(
                    "colbert token vector must have non-zero L2 norm",
                ));
            }
            normalized_tokens.push(vec);
        }
        self.docs.insert(
            id,
            ColbertDocument {
                id,
                token_vectors: normalized_tokens,
            },
        );
        Ok(())
    }

    /// Remove a document from the index by `id`.
    pub fn delete(&mut self, id: VectorId) -> bool {
        self.docs.remove(&id).is_some()
    }

    /// Validate and L2-normalize a slice of query token vectors.
    fn normalize_query_tokens(
        &self,
        query_tokens: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, VektorError> {
        if query_tokens.is_empty() {
            return Err(VektorError::InvalidVector(
                "colbert query must contain at least one token vector",
            ));
        }
        let mut out = Vec::with_capacity(query_tokens.len());
        for q in query_tokens {
            validate_vector(q, self.dims)?;
            let mut copy = q.clone();
            if !normalize_l2(&mut copy) {
                return Err(VektorError::InvalidVector(
                    "colbert query token vector must have non-zero L2 norm",
                ));
            }
            out.push(copy);
        }
        Ok(out)
    }

    /// Rank all indexed documents by exact ColBERT `MaxSim` against `query_tokens`,
    /// returning the top `top_k` documents in descending score order (ties broken by ascending `id`).
    pub fn search(
        &self,
        query_tokens: &[Vec<f32>],
        top_k: usize,
    ) -> Result<Vec<ScoredId>, VektorError> {
        let normalized_query = self.normalize_query_tokens(query_tokens)?;
        if top_k == 0 || self.docs.is_empty() {
            return Ok(Vec::new());
        }

        let mut hits: Vec<ScoredId> = self
            .docs
            .values()
            .map(|doc| ScoredId {
                id: doc.id,
                score: maxsim_score(&normalized_query, &doc.token_vectors),
            })
            .collect();

        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        hits.truncate(top_k);
        Ok(hits)
    }

    /// Re-rank a specific set of first-stage `candidate_ids` using ColBERT `MaxSim`.
    pub fn rerank(
        &self,
        query_tokens: &[Vec<f32>],
        candidate_ids: &[VectorId],
        top_k: usize,
    ) -> Result<Vec<ScoredId>, VektorError> {
        let normalized_query = self.normalize_query_tokens(query_tokens)?;
        if top_k == 0 || candidate_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut hits: Vec<ScoredId> = candidate_ids
            .iter()
            .filter_map(|id| {
                self.docs.get(id).map(|doc| ScoredId {
                    id: doc.id,
                    score: maxsim_score(&normalized_query, &doc.token_vectors),
                })
            })
            .collect();

        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        hits.truncate(top_k);
        Ok(hits)
    }
}

/// Programmable scoring weights and exponential decay rate for `AgentMemoryStore`.
///
/// Composite score:
/// `total_score = w_semantic * cosine_sim + w_recency * (-decay_lambda * age_secs).exp() + w_importance * importance`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryScoringWeights {
    pub w_semantic: f32,
    pub w_recency: f32,
    pub w_importance: f32,
    /// Exponential decay constant $\lambda$ (in $\text{s}^{-1}$) applied to memory age in seconds.
    pub decay_lambda: f32,
}

impl Default for MemoryScoringWeights {
    fn default() -> Self {
        Self {
            w_semantic: 0.60,
            w_recency: 0.25,
            w_importance: 0.15,
            // ~1 hour half-life by default: ln(2) / 3600 ≈ 0.0001925
            decay_lambda: 0.000_192_5,
        }
    }
}

impl MemoryScoringWeights {
    /// Validate that all scoring weights and decay parameters are finite and non-negative.
    pub fn validate(&self) -> Result<(), VektorError> {
        if !self.w_semantic.is_finite()
            || !self.w_recency.is_finite()
            || !self.w_importance.is_finite()
            || !self.decay_lambda.is_finite()
        {
            return Err(VektorError::InvalidConfig(
                "memory scoring weights and decay_lambda must be finite",
            ));
        }
        if self.w_semantic < 0.0
            || self.w_recency < 0.0
            || self.w_importance < 0.0
            || self.decay_lambda < 0.0
        {
            return Err(VektorError::InvalidConfig(
                "memory scoring weights and decay_lambda must be >= 0.0",
            ));
        }
        Ok(())
    }

    /// Evaluate the composite memory score given `cosine_sim`, `age_secs`, and `importance`.
    #[inline]
    #[must_use]
    pub fn score(&self, cosine_sim: f32, age_secs: u64, importance: f32) -> f32 {
        let recency = (-self.decay_lambda * (age_secs as f32)).exp();
        let clamped_importance = importance.clamp(0.0, 1.0);
        self.w_semantic * cosine_sim
            + self.w_recency * recency
            + self.w_importance * clamped_importance
    }
}

/// Tenant-scoped episodic memory record stored in `AgentMemoryStore`.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryEntry {
    pub id: VectorId,
    pub org_id: OrgId,
    /// L2-normalized dense semantic embedding of the memory.
    pub embedding: Vec<f32>,
    /// Raw text payload of the episodic memory.
    pub text: String,
    /// Unix timestamp (seconds) when the memory was recorded.
    pub created_at_secs: u64,
    /// Intrinsic importance score in `0.0..=1.0`.
    pub importance: f32,
    /// Number of times this memory has been retrieved into an agent context window.
    pub access_count: u32,
    /// Token count consumed when packing this memory into an LLM context window.
    pub token_count: usize,
}

/// A ranked memory returned by `AgentMemoryStore::retrieve_context_window`.
///
/// Implements `Deref<Target = MemoryEntry>` so callers can access `MemoryEntry` fields
/// (`id`, `text`, `token_count`, `access_count`, etc.) directly as well as `total_score`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredMemory {
    pub entry: MemoryEntry,
    pub total_score: f32,
    pub semantic_score: f32,
    pub recency_score: f32,
}

impl Deref for ScoredMemory {
    type Target = MemoryEntry;

    fn deref(&self) -> &Self::Target {
        &self.entry
    }
}

/// Multi-tenant episodic memory store for LLM agents with time-decay recency,
/// importance weighting, and token-budget-aware context window packing.
#[derive(Debug, Clone)]
pub struct AgentMemoryStore {
    dims: usize,
    weights: MemoryScoringWeights,
    tenants: HashMap<OrgId, HashMap<VectorId, MemoryEntry>>,
}

impl AgentMemoryStore {
    /// Create a new `AgentMemoryStore` for embeddings of dimension `dims` and programmable `weights`.
    pub fn new(dims: usize, weights: MemoryScoringWeights) -> Result<Self, VektorError> {
        if dims == 0 {
            return Err(VektorError::InvalidConfig(
                "memory embedding dimensions must be > 0",
            ));
        }
        weights.validate()?;
        Ok(Self {
            dims,
            weights,
            tenants: HashMap::new(),
        })
    }

    /// Return the configured embedding dimensionality.
    #[must_use]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Return the active `MemoryScoringWeights`.
    #[must_use]
    pub fn weights(&self) -> MemoryScoringWeights {
        self.weights
    }

    /// Update the programmable scoring weights.
    pub fn set_weights(&mut self, weights: MemoryScoringWeights) -> Result<(), VektorError> {
        weights.validate()?;
        self.weights = weights;
        Ok(())
    }

    /// Total number of memories stored for `org_id`.
    #[must_use]
    pub fn tenant_len(&self, org_id: &OrgId) -> usize {
        self.tenants.get(org_id).map_or(0, HashMap::len)
    }

    /// Look up a memory entry strictly within `caller_org`.
    #[must_use]
    pub fn get(&self, caller_org: &OrgId, id: VectorId) -> Option<&MemoryEntry> {
        self.tenants.get(caller_org)?.get(&id)
    }

    /// Upsert an episodic memory under `org_id`.
    ///
    /// If `token_count` is `0`, it is automatically estimated from `text` via `tokenize(text).len().max(1)`.
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_memory(
        &mut self,
        org_id: OrgId,
        id: VectorId,
        embedding: &[f32],
        text: impl Into<String>,
        created_at_secs: u64,
        importance: f32,
        token_count: usize,
    ) -> Result<(), VektorError> {
        validate_vector(embedding, self.dims)?;
        if !importance.is_finite() || !(0.0..=1.0).contains(&importance) {
            return Err(VektorError::InvalidConfig(
                "memory importance must be in 0.0..=1.0",
            ));
        }
        let mut normalized = embedding.to_vec();
        if !normalize_l2(&mut normalized) {
            return Err(VektorError::InvalidVector(
                "memory embedding must have non-zero L2 norm",
            ));
        }
        let text_str = text.into();
        let effective_tokens = if token_count == 0 {
            tokenize(&text_str).len().max(1)
        } else {
            token_count
        };

        let tenant_map = self.tenants.entry(org_id.clone()).or_default();
        let prev_access_count = tenant_map.get(&id).map_or(0, |m| m.access_count);
        tenant_map.insert(
            id,
            MemoryEntry {
                id,
                org_id,
                embedding: normalized,
                text: text_str,
                created_at_secs,
                importance,
                access_count: prev_access_count,
                token_count: effective_tokens,
            },
        );
        Ok(())
    }

    /// Delete a memory entry strictly within `caller_org`.
    pub fn delete(&mut self, caller_org: &OrgId, id: VectorId) -> bool {
        self.tenants
            .get_mut(caller_org)
            .is_some_and(|map| map.remove(&id).is_some())
    }

    /// Retrieve top memories for `caller_org` ranked by composite semantic + time-decay + importance
    /// score, greedily packing memories that fit within `max_token_budget` tokens (up to `top_k`
    /// memories) and incrementing `access_count` on each selected memory.
    pub fn retrieve_context_window(
        &mut self,
        caller_org: &OrgId,
        query: &[f32],
        now_secs: u64,
        top_k: usize,
        max_token_budget: usize,
    ) -> Result<Vec<ScoredMemory>, VektorError> {
        validate_vector(query, self.dims)?;
        let mut normalized_query = query.to_vec();
        if !normalize_l2(&mut normalized_query) {
            return Err(VektorError::InvalidVector(
                "query embedding must have non-zero L2 norm",
            ));
        }
        if top_k == 0 || max_token_budget == 0 {
            return Ok(Vec::new());
        }
        let Some(tenant_map) = self.tenants.get_mut(caller_org) else {
            return Ok(Vec::new());
        };
        if tenant_map.is_empty() {
            return Ok(Vec::new());
        }

        // Score all candidate memories belonging to `caller_org`.
        let mut candidates: Vec<(VectorId, f32, f32, f32)> = Vec::with_capacity(tenant_map.len());
        for entry in tenant_map.values() {
            let semantic_sim = dot_product(&normalized_query, &entry.embedding);
            let age_secs = now_secs.saturating_sub(entry.created_at_secs);
            let recency = (-self.weights.decay_lambda * (age_secs as f32)).exp();
            let total_score = self.weights.score(semantic_sim, age_secs, entry.importance);
            candidates.push((entry.id, total_score, semantic_sim, recency));
        }

        candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // Greedily pack candidate memories that fit within `max_token_budget` up to `top_k`.
        let mut remaining_tokens = max_token_budget;
        let mut packed = Vec::with_capacity(top_k.min(candidates.len()));

        for (id, total_score, semantic_score, recency_score) in candidates {
            if packed.len() >= top_k || remaining_tokens == 0 {
                break;
            }
            let Some(entry) = tenant_map.get_mut(&id) else {
                continue;
            };
            if entry.token_count <= remaining_tokens {
                remaining_tokens -= entry.token_count;
                entry.access_count = entry.access_count.saturating_add(1);
                packed.push(ScoredMemory {
                    entry: entry.clone(),
                    total_score,
                    semantic_score,
                    recency_score,
                });
            }
        }

        Ok(packed)
    }
}

/// Cached query entry stored inside a tenant- and filter-partitioned `SemanticCache` bucket.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheEntry {
    pub org_id: OrgId,
    pub filter_hash: u64,
    /// L2-normalized query embedding.
    pub query_vector: Vec<f32>,
    /// Cached ranked search results.
    pub results: Vec<ScoredId>,
    /// Unix timestamp (seconds) when the entry was inserted.
    pub inserted_at_secs: u64,
    /// Number of cache hits served by this entry.
    pub hit_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CachePartitionKey {
    org_id: OrgId,
    filter_hash: u64,
}

/// Tenant-isolated semantic query cache (`(OrgId, filter_hash, query_vector) -> Vec<ScoredId>`).
///
/// Serves sub-50-microsecond cache hits when a new query within the same `(OrgId, filter_hash)`
/// partition achieves cosine similarity `>= similarity_threshold` against a non-expired cached
/// query vector. Enforces strict tenant/filter isolation, TTL expiration, and bounded capacity
/// eviction.
#[derive(Debug, Clone)]
pub struct SemanticCache {
    dims: usize,
    similarity_threshold: f32,
    ttl_secs: u64,
    max_entries: usize,
    total_entries: usize,
    partitions: HashMap<CachePartitionKey, VecDeque<CacheEntry>>,
    eviction_queue: VecDeque<CachePartitionKey>,
}

impl SemanticCache {
    /// Construct a new `SemanticCache`.
    ///
    /// - `dims`: Query embedding dimensionality (`> 0`).
    /// - `similarity_threshold`: Minimum cosine similarity for a semantic hit (e.g. `0.96`, in `(0.0, 1.0]`).
    /// - `ttl_secs`: Time-to-live in seconds (`> 0`).
    /// - `max_entries`: Maximum total cached queries across all partitions (`> 0`).
    pub fn new(
        dims: usize,
        similarity_threshold: f32,
        ttl_secs: u64,
        max_entries: usize,
    ) -> Result<Self, VektorError> {
        if dims == 0 {
            return Err(VektorError::InvalidConfig(
                "semantic cache dimensions must be > 0",
            ));
        }
        if !similarity_threshold.is_finite()
            || similarity_threshold <= 0.0
            || similarity_threshold > 1.0
        {
            return Err(VektorError::InvalidConfig(
                "similarity_threshold must be in (0.0, 1.0]",
            ));
        }
        if ttl_secs == 0 {
            return Err(VektorError::InvalidConfig("ttl_secs must be > 0"));
        }
        if max_entries == 0 {
            return Err(VektorError::InvalidConfig("max_entries must be > 0"));
        }
        Ok(Self {
            dims,
            similarity_threshold,
            ttl_secs,
            max_entries,
            total_entries: 0,
            partitions: HashMap::new(),
            eviction_queue: VecDeque::new(),
        })
    }

    /// Configured query vector dimensionality.
    #[must_use]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Configured cosine similarity threshold for cache hits.
    #[must_use]
    pub fn similarity_threshold(&self) -> f32 {
        self.similarity_threshold
    }

    /// Configured entry TTL in seconds.
    #[must_use]
    pub fn ttl_secs(&self) -> u64 {
        self.ttl_secs
    }

    /// Maximum number of cached entries retained across all tenants.
    #[must_use]
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Current number of cached entries across all partitions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.total_entries
    }

    /// Whether the cache currently holds zero entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total_entries == 0
    }

    /// Purge all expired entries as of `now_secs` across all partitions.
    pub fn evict_expired(&mut self, now_secs: u64) -> usize {
        let ttl = self.ttl_secs;
        let mut removed = 0_usize;
        self.partitions.retain(|_, bucket| {
            let before = bucket.len();
            bucket.retain(|entry| now_secs.saturating_sub(entry.inserted_at_secs) < ttl);
            removed += before - bucket.len();
            !bucket.is_empty()
        });
        if removed > 0 {
            self.total_entries = self.total_entries.saturating_sub(removed);
            self.eviction_queue
                .retain(|k| self.partitions.contains_key(k));
        }
        removed
    }

    /// Look up a semantic cache hit for `(caller_org, filter_hash, query_vector)` at `now_secs`.
    ///
    /// Lazily evicts expired entries in the target partition and returns the cached `Vec<ScoredId>`
    /// from the highest-similarity entry whose cosine similarity is `>= similarity_threshold`.
    pub fn get(
        &mut self,
        caller_org: &OrgId,
        filter_hash: u64,
        query_vector: &[f32],
        now_secs: u64,
    ) -> Result<Option<Vec<ScoredId>>, VektorError> {
        validate_vector(query_vector, self.dims)?;
        let mut normalized_query = query_vector.to_vec();
        if !normalize_l2(&mut normalized_query) {
            return Err(VektorError::InvalidVector(
                "semantic cache query vector must have non-zero L2 norm",
            ));
        }

        let key = CachePartitionKey {
            org_id: caller_org.clone(),
            filter_hash,
        };
        let Some(bucket) = self.partitions.get_mut(&key) else {
            return Ok(None);
        };

        // Evict expired entries in this partition first.
        let ttl = self.ttl_secs;
        let before = bucket.len();
        bucket.retain(|entry| now_secs.saturating_sub(entry.inserted_at_secs) < ttl);
        let expired_count = before - bucket.len();
        if expired_count > 0 {
            self.total_entries = self.total_entries.saturating_sub(expired_count);
        }
        if bucket.is_empty() {
            self.partitions.remove(&key);
            return Ok(None);
        }

        // Find the cached entry with maximum cosine similarity >= similarity_threshold.
        let mut best_idx: Option<usize> = None;
        let mut best_sim = self.similarity_threshold;
        for (idx, entry) in bucket.iter().enumerate() {
            let sim = dot_product(&normalized_query, &entry.query_vector);
            if sim >= self.similarity_threshold && (best_idx.is_none() || sim > best_sim) {
                best_sim = sim;
                best_idx = Some(idx);
            }
        }

        if let Some(idx) = best_idx {
            if let Some(entry) = bucket.get_mut(idx) {
                entry.hit_count = entry.hit_count.saturating_add(1);
                return Ok(Some(entry.results.clone()));
            }
        }

        Ok(None)
    }

    /// Insert or refresh a semantic cache entry under `(org_id, filter_hash)`.
    ///
    /// Enforces `max_entries` capacity by evicting expired entries first and then the oldest
    /// live entry if at capacity.
    pub fn put(
        &mut self,
        org_id: OrgId,
        filter_hash: u64,
        query_vector: &[f32],
        results: Vec<ScoredId>,
        now_secs: u64,
    ) -> Result<(), VektorError> {
        validate_vector(query_vector, self.dims)?;
        let mut normalized_query = query_vector.to_vec();
        if !normalize_l2(&mut normalized_query) {
            return Err(VektorError::InvalidVector(
                "semantic cache query vector must have non-zero L2 norm",
            ));
        }

        if self.total_entries >= self.max_entries {
            self.evict_expired(now_secs);
        }
        while self.total_entries >= self.max_entries {
            let Some(oldest_key) = self.eviction_queue.pop_front() else {
                break;
            };
            if let Some(bucket) = self.partitions.get_mut(&oldest_key) {
                if bucket.pop_front().is_some() {
                    self.total_entries = self.total_entries.saturating_sub(1);
                }
                if bucket.is_empty() {
                    self.partitions.remove(&oldest_key);
                }
            }
        }

        let key = CachePartitionKey {
            org_id: org_id.clone(),
            filter_hash,
        };
        let entry = CacheEntry {
            org_id,
            filter_hash,
            query_vector: normalized_query,
            results,
            inserted_at_secs: now_secs,
            hit_count: 0,
        };
        self.partitions
            .entry(key.clone())
            .or_default()
            .push_back(entry);
        self.eviction_queue.push_back(key);
        self.total_entries += 1;
        Ok(())
    }

    /// Invalidate all cached queries belonging to `org_id` (e.g., on tenant write/delete).
    pub fn invalidate_tenant(&mut self, org_id: &OrgId) -> usize {
        let mut removed = 0_usize;
        self.partitions.retain(|k, bucket| {
            if &k.org_id == org_id {
                removed += bucket.len();
                false
            } else {
                true
            }
        });
        if removed > 0 {
            self.total_entries = self.total_entries.saturating_sub(removed);
            self.eviction_queue.retain(|k| &k.org_id != org_id);
        }
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maxsim_exact_alignment_math() {
        let q = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let d_both = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let d_single = vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];

        assert!((maxsim_score(&q, &d_both) - 2.0).abs() < 1e-5);
        assert!((maxsim_score(&q, &d_single) - 1.0).abs() < 1e-5);
    }
}
