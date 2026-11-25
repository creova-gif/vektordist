# ADR-004: Multi-Tenant `OrgId` Isolation, One-Stage In-Graph Filtering, and Hybrid RRF Search
- **Status:** Accepted
- **Owner:** Justin Mafie

## Context
Enterprise vector workloads require strict multi-tenant isolation (`org_id`) so no query or deletion from one tenant can ever observe or mutate another tenant's records (`cross_tenant_leakage == 0`), alongside selective attribute filtering (`Eq`, `In`, `Range`, `And`, `Or`) and hybrid semantic + keyword retrieval for exact identifiers (error codes, SKUs, symbols).

## Decision
1. **Validated `OrgId` Scoping:** Introduce `OrgId` (non-empty, non-whitespace string wrapper) required on every `TenantHybridEngine` upsert, metadata lookup, deletion, and search operation.
2. **One-Stage In-Graph Predicate Traversal (`HnswIndex::search_filtered_with_ef`):** Evaluate tenant ownership (`rec.org_id == caller_org_id`) and `MetadataFilter` inline during HNSW layer-0 beam expansion. Non-matching and tombstoned nodes remain traversable as routing bridges in an expanded routing window (`routing_ef = 4 * ef`), while only live matching nodes enter `matched_results`, preventing post-filtering starvation under low selectivity (e.g., 10% matching predicates).
3. **Per-Tenant BM25 Inverted Index (`Bm25Index`) + RRF (`hybrid_search_rrf`):** Maintain physically partitioned per-`OrgId` posting lists (`k1 = 1.2`, `b = 0.75`) with case-folded alphanumeric tokenization (`tokenize`) and fuse dense HNSW rankings with lexical BM25 rankings using Reciprocal Rank Fusion (`score = sum(1.0 / (rrf_k + rank))`, default `rrf_k = 60.0`).

## Alternatives Considered
- **Two-Stage Post-Filtering:** Retrieve top-`k` globally from HNSW and filter afterward. Rejected because selective filters (`<= 10%`) or dominant co-tenants starve the top-`k` candidate set and leak cross-tenant timing/scoring signals unless heavily over-fetched.
- **Separate HNSW Graph per Tenant:** Eliminates shared graph routing but incurs high per-tenant memory overhead for thousands of small tenants.

## Consequences & Failure Modes
- **Zero Cross-Tenant Leakage:** Verified across 1,000 interleaved multi-tenant vectors and adversarial identical-vector/keyword queries (`cross_tenant_leakage == 0`).
- **Selectivity Backfill:** If extreme predicate selectivity leaves fewer than `k` matches inside the HNSW beam, `search_filtered_with_ef` backfills from live matching slots so valid tenant records are never dropped.
