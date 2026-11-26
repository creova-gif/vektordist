# ADR-018: Multi-Collection Namespace Engine with Per-Collection Quantization and `OrgId` Isolation
- **Status:** Accepted
- **Owner:** Justin Mafie

## Context
Prior to Milestone 18, `HttpGatewayServer` exposed a single cluster-wide vector space (`/v1/upsert`, `/v1/search`, `/v1/delete`) bound to one fixed dimensionality (`ClusterConfig::dimensions`) and distance metric. Production AI workloads routinely host heterogeneous embedding models side-by-side within a single deployment — for example, a `64`- or `768`-dimensional full-precision `f32` document retrieval index alongside a `32`- or `1024`-dimensional binary-quantized (`bq`) semantic cache or `sq8` code index — while requiring strict zero-trust multi-tenant (`OrgId`) isolation inside every collection.

## Decision
1. **`CollectionRegistry` (`crates/vektordist-cluster/src/collections.rs`):**
   - Defined `CollectionQuantization` (`F32`, `Sq8`, `Bq`), `CollectionSpec` (`name`, `dimensions`, `metric`, `quantization`), `CollectionInfo` (`name`, `dimensions`, `metric`, `quantization`, `live_vectors`, `memory_bytes`), and `CollectionError`.
   - Implemented thread-safe `CollectionRegistry` backed by `RwLock<BTreeMap<String, Arc<RwLock<CollectionInstance>>>>` so operations on distinct collections execute without cross-collection lock contention.
   - Inside each `CollectionInstance`, partitioned vectors into per-tenant `BTreeMap<OrgId, TenantCollectionIndex>` sub-indices (`HnswIndex` for `F32`, `Sq8HnswIndex` for `Sq8`, and two-stage `BqIndex` + `VectorArena` reranking for `Bq`), structurally guaranteeing that `search` and `delete` for `org-b` can never read, score, or delete vectors owned by `org-a`.
2. **Zero-Trust REST Routes (`gateway.rs`) & Typed Client SDK (`sdk.rs`):**
   - Wired `collections: Arc<CollectionRegistry>` into `HttpGatewayServer` with RBAC-enforced routes:
     - `POST /v1/collections` (`TenantRole::Admin`, `"http.collections.create"` -> `201 Created`, `409 Conflict`, or `400 Bad Request`)
     - `GET /v1/collections` (`TenantRole::Reader`, `"http.collections.list"` -> `200 OK`)
     - `DELETE /v1/collections/<name>` (`TenantRole::Admin`, `"http.collections.delete"` -> `200 OK` or `404 Not Found`)
     - `POST /v1/collections/<name>/upsert` (`TenantRole::Writer`, `"http.collections.upsert"`, scoped to `X-Org-Id`)
     - `POST /v1/collections/<name>/search` (`TenantRole::Reader`, `"http.collections.search"`, scoped to `X-Org-Id`)
     - `POST /v1/collections/<name>/delete` (`TenantRole::Writer`, `"http.collections.delete_vec"`, scoped to `X-Org-Id`)
   - Extended `HttpVektorClient` with `create_collection`, `list_collections_raw`, `drop_collection`, `collection_upsert`, `collection_search`, and `collection_delete`.

## Alternatives Considered
- **Shared Cross-Tenant HNSW Graph with Post-Filtering:** Mixing multiple tenants' vectors inside one HNSW graph risks disconnected subgraphs for small tenants when a dominant tenant owns 99% of neighbors, and allows VectorId collisions across tenants unless composite keys are minted. Maintaining isolated per-`OrgId` indices per collection guarantees both zero cross-tenant information leakage and high recall regardless of tenant size skew.
- **Single-Stage Hamming-Only Search for `Bq` Collections:** Pure 1-bit sign quantization maps all non-negative coordinates (`>= 0.0`) to `1`, causing sparse non-negative basis vectors to tie in Hamming distance. Pairing `BqIndex` shortlisting with exact `VectorArena` reranking preserves `XOR + POPCNT` candidate filtering while maintaining high precision across arbitrary vector distributions.

## Consequences & Failure Modes
- **Gateway-Scoped In-Memory Collection State:** `CollectionRegistry` provides fast multi-model namespace isolation at the gateway tier (`Arc<CollectionRegistry>` shareable across embedded handlers via `with_collections`), separate from the single-schema gRPC shard WAL protocol; dropping a collection immediately frees 100% of its tenant arenas and HNSW adjacency lists.
- **Strict Collection Name Validation:** Collection names are restricted to `1..=128` ASCII `[A-Za-z0-9_-]` characters, preventing URL path traversal or routing ambiguity on `/v1/collections/<name>/*`.
