# ADR-009: Programmable Agentic RAG Memory, Late-Interaction ColBERT MaxSim, and Tenant-Isolated Semantic Cache
- **Status:** Accepted
- **Owner:** Justin Mafie

## Context
Modern agentic RAG workloads expose three retrieval bottlenecks that single-vector ANN search alone cannot solve:
1. **Single-Vector Information Bottleneck on Multi-Concept Queries:** Compressing an entire document and a complex multi-clause agent query into single pooled vectors loses fine-grained token alignment (e.g., matching *all* required technical entities vs. repeating a single dominant keyword).
2. **Episodic Memory Drift and Context Window Overflow:** Long-running autonomous agents accumulate thousands of episodic memories per tenant. Ranking solely by cosine similarity surfaces stale, superseded memories and ignores intrinsic importance or downstream LLM token budget limits (`max_token_budget`).
3. **Redundant Agent Sub-Query Latency:** Multi-step reasoning loops frequently issue repeated or slightly paraphrased vector queries (`cosine_sim >= 0.96`). Re-traversing HNSW graphs for near-identical queries wastes CPU, while naive global query caching risks cross-tenant data leakage or stale post-TTL results.

## Decision
We introduce `vektordist_core::agentic` with three cohesive components:
1. **Multi-Vector Late Interaction (`ColbertDocument`, `ColbertIndex`, `maxsim_score`):**
   - Stores L2-normalized token-level embeddings (`Vec<Vec<f32>>`) per document ID and computes exact ColBERT late-interaction scoring:
     $$\text{MaxSim}(Q, D) = \sum_{q \in Q} \max_{d \in D} (q \cdot d)$$
     using hardware-accelerated SIMD inner products (`dot_product_simd`).
   - Supports both standalone multi-vector search (`ColbertIndex::search`) and two-stage candidate re-ranking (`ColbertIndex::rerank`).
2. **Episodic Agent Memory Store (`MemoryEntry`, `MemoryScoringWeights`, `AgentMemoryStore`):**
   - Partitions episodic memories strictly by `OrgId` and ranks them with a programmable composite score:
     $$\text{score}(m) = w_{\text{semantic}} \cdot \cos(q, m) + w_{\text{recency}} \cdot e^{-\lambda \cdot \Delta t} + w_{\text{importance}} \cdot \text{importance}(m)$$
   - `retrieve_context_window` evaluates the composite score at `now_secs`, greedily packs the highest-ranked memories that fit within `max_token_budget` (up to `top_k`), and increments `access_count` on every packed memory.
3. **Tenant-Isolated Semantic Query Cache (`SemanticCache`, `CacheEntry`):**
   - Partitions cached query vectors by `(OrgId, filter_hash)` so a cache lookup never inspects or returns entries belonging to another tenant or metadata filter predicate.
   - Within the caller's `(OrgId, filter_hash)` partition, lazily evicts entries older than `ttl_secs`, scans remaining normalized query embeddings via SIMD dot product, and returns a cache hit when the best match satisfies `cosine_sim >= similarity_threshold` (e.g. `0.96`).
   - Enforces a hard `max_entries` bound via expired-first + FIFO eviction and supports instant tenant invalidation (`invalidate_tenant`).

## Alternatives Considered
- **Cross-Encoder Reranking Inside `vektordist-core`:** Running full transformer cross-attention over every candidate pair requires GPU inference inside the storage engine (`10-100ms`). ColBERT `MaxSim` pre-computes document token representations at ingest time and reduces late interaction to cache-friendly SIMD dot products (`< 0.1ms`).
- **Hard Token Truncation Instead of Greedy Budget Packing:** Truncating the ranked list as soon as the next candidate exceeds `remaining_tokens` wastes remaining context capacity when a slightly lower-ranked concise memory fits cleanly inside `remaining_tokens`.
- **Global Unpartitioned Semantic Cache with Post-Filter Check:** Storing all tenants in a single vector cache and checking `org_id` after similarity search risks timing side channels and cross-tenant cache eviction pressure. Keying buckets by `(OrgId, filter_hash)` guarantees hard structural isolation.

## Consequences & Failure Modes
- **Multi-Vector Memory Footprint:** Storing $T$ token vectors per document increases memory usage by $T\times$ relative to single-vector indices. In large deployments, `ColbertIndex::rerank` should be paired with first-stage HNSW candidate generation (`top_k = 100..200`).
- **Cache Staleness Within TTL Window:** Between a write and `ttl_secs` expiration, `SemanticCache` may serve slightly stale hits unless the write path calls `SemanticCache::invalidate_tenant(&org_id)`.
