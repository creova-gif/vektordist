use vektordist_core::agentic::{
    maxsim_score, AgentMemoryStore, ColbertIndex, MemoryScoringWeights, SemanticCache,
};
use vektordist_core::exact::ScoredId;
use vektordist_core::hybrid::OrgId;
use vektordist_core::VektorError;

#[test]
fn colbert_maxsim_surfaces_multi_concept_documents_over_single_concept_distractors() {
    let mut index = ColbertIndex::new(4).unwrap();

    // Query has 3 distinct sub-concepts along basis axes e0, e1, e2:
    // e.g. ["rust", "raft_consensus", "simd_neon"]
    let query_tokens = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];

    // Doc 10: Single-concept distractor repeating only concept e0 three times.
    // A naive sum/average embedding might score reasonably on e0, but MaxSim gives:
    // q0 -> 1.0, q1 -> 0.0, q2 -> 0.0 => total = 1.0.
    index
        .upsert(
            10,
            vec![
                vec![2.0, 0.0, 0.0, 0.0], // auto-normalized on upsert
                vec![5.0, 0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0, 0.0],
            ],
        )
        .unwrap();

    // Doc 20: Two-concept partial match covering e0 and e1 plus unrelated e3.
    // MaxSim: q0 -> 1.0, q1 -> 1.0, q2 -> 0.0 => total = 2.0.
    index
        .upsert(
            20,
            vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 3.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
            ],
        )
        .unwrap();

    // Doc 30: Full multi-concept document covering all three query concepts (e0, e1, e2)
    // alongside an extra context token on e3.
    // MaxSim: q0 -> 1.0, q1 -> 1.0, q2 -> 1.0 => total = 3.0.
    index
        .upsert(
            30,
            vec![
                vec![0.0, 0.0, 0.0, 1.0],
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
            ],
        )
        .unwrap();

    let hits = index.search(&query_tokens, 3).unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].id, 30);
    assert!((hits[0].score - 3.0).abs() < 1e-5);
    assert_eq!(hits[1].id, 20);
    assert!((hits[1].score - 2.0).abs() < 1e-5);
    assert_eq!(hits[2].id, 10);
    assert!((hits[2].score - 1.0).abs() < 1e-5);

    // Verify candidate re-ranking on a subset of IDs.
    let reranked = index.rerank(&query_tokens, &[10, 20], 2).unwrap();
    assert_eq!(reranked.len(), 2);
    assert_eq!(reranked[0].id, 20);
    assert_eq!(reranked[1].id, 10);

    // Direct maxsim_score helper check.
    let doc30 = index.get(30).unwrap();
    assert!((maxsim_score(&query_tokens, &doc30.token_vectors) - 3.0).abs() < 1e-5);
}

#[test]
fn agent_memory_store_balances_semantic_recency_importance_and_packs_token_budget_with_tenant_isolation(
) {
    // Programmable weights: semantic = 0.5, recency = 0.3, importance = 0.2, decay_lambda = 0.01/sec
    let weights = MemoryScoringWeights {
        w_semantic: 0.50,
        w_recency: 0.30,
        w_importance: 0.20,
        decay_lambda: 0.01,
    };
    let mut store = AgentMemoryStore::new(3, weights).unwrap();

    let org_alpha = OrgId::new("org_alpha").unwrap();
    let org_beta = OrgId::new("org_beta").unwrap();

    // Now is t = 1_000s.
    let now_secs = 1_000;

    // Memory 1 (org_alpha): Old memory (created at t = 0 => age = 1000s, recency ≈ 0),
    // exact semantic match (sim = 1.0), low importance (0.1), 40 tokens.
    // Score = 0.50 * 1.0 + 0.30 * exp(-10) + 0.20 * 0.1 ≈ 0.520
    store
        .upsert_memory(
            org_alpha.clone(),
            1,
            &[1.0, 0.0, 0.0],
            "stale low-importance fact",
            0,
            0.1,
            40,
        )
        .unwrap();

    // Memory 2 (org_alpha): Fresh critical memory (created at t = 1_000 => age = 0s, recency = 1.0),
    // strong semantic match (sim = 0.96), high importance (1.0), 60 tokens.
    // Score = 0.50 * 0.96 + 0.30 * 1.0 + 0.20 * 1.0 = 0.48 + 0.30 + 0.20 = 0.980
    store
        .upsert_memory(
            org_alpha.clone(),
            2,
            &[0.96, 0.28, 0.0],
            "fresh critical user preference: always use strict clippy lints",
            1_000,
            1.0,
            60,
        )
        .unwrap();

    // Memory 3 (org_alpha): Oversized memory (90 tokens) with medium score:
    // age = 100s (recency = exp(-1) ≈ 0.3679), sim = 0.80, importance = 0.80.
    // Score = 0.50 * 0.80 + 0.30 * 0.3679 + 0.20 * 0.80 ≈ 0.40 + 0.1104 + 0.16 = 0.6704
    store
        .upsert_memory(
            org_alpha.clone(),
            3,
            &[0.80, 0.60, 0.0],
            "large architectural transcript that exceeds remaining token budget after memory 2",
            900,
            0.8,
            90,
        )
        .unwrap();

    // Memory 99 (org_beta): Another tenant's secret memory with maximal score.
    // Must NEVER leak into org_alpha's context window.
    store
        .upsert_memory(
            org_beta.clone(),
            99,
            &[1.0, 0.0, 0.0],
            "org_beta confidential API key memory",
            1_000,
            1.0,
            10,
        )
        .unwrap();

    // Query with budget = 110 tokens:
    // - Memory 2 ranks #1 (score ~0.98, 60 tokens) -> packed! (50 tokens remaining)
    // - Memory 3 ranks #2 (score ~0.67, 90 tokens) -> skipped because 90 > 50 remaining tokens!
    // - Memory 1 ranks #3 (score ~0.52, 40 tokens) -> fits in remaining 50 tokens -> packed!
    let window = store
        .retrieve_context_window(&org_alpha, &[1.0, 0.0, 0.0], now_secs, 5, 110)
        .unwrap();

    assert_eq!(window.len(), 2);
    assert_eq!(window[0].id, 2);
    assert_eq!(window[1].id, 1);
    assert!(window[0].total_score > window[1].total_score);

    // Total packed tokens must respect max_token_budget (60 + 40 = 100 <= 110).
    let packed_tokens: usize = window.iter().map(|m| m.token_count).sum();
    assert_eq!(packed_tokens, 100);

    // Access count must be incremented only for packed memories in org_alpha.
    assert_eq!(store.get(&org_alpha, 2).unwrap().access_count, 1);
    assert_eq!(store.get(&org_alpha, 1).unwrap().access_count, 1);
    assert_eq!(store.get(&org_alpha, 3).unwrap().access_count, 0);
    assert_eq!(store.get(&org_beta, 99).unwrap().access_count, 0);
    assert!(store.get(&org_alpha, 99).is_none());
}

#[test]
fn semantic_cache_hits_near_identical_queries_isolates_tenants_and_filters_and_evicts_ttl() {
    // Threshold = 0.96, TTL = 60s, max_entries = 2
    let mut cache = SemanticCache::new(3, 0.96, 60, 2).unwrap();

    let tenant_a = OrgId::new("tenant_a").unwrap();
    let tenant_b = OrgId::new("tenant_b").unwrap();
    let filter_1 = 0x1111_u64;
    let filter_2 = 0x2222_u64;

    let canonical_query = [1.0_f32, 0.0, 0.0];
    let expected_hits = vec![
        ScoredId {
            id: 42,
            score: 0.99,
        },
        ScoredId {
            id: 77,
            score: 0.88,
        },
    ];

    cache
        .put(
            tenant_a.clone(),
            filter_1,
            &canonical_query,
            expected_hits.clone(),
            100,
        )
        .unwrap();

    // 1. Paraphrased / slightly perturbed query with cosine similarity = 0.98 >= 0.96 hits!
    let paraphrased_query = [0.98_f32, 0.198_997_5, 0.0];
    let hit = cache
        .get(&tenant_a, filter_1, &paraphrased_query, 120)
        .unwrap();
    assert_eq!(hit, Some(expected_hits.clone()));

    // 2. Semantically different query with cosine similarity = 0.80 < 0.96 misses!
    let distant_query = [0.80_f32, 0.60, 0.0];
    assert_eq!(
        cache.get(&tenant_a, filter_1, &distant_query, 120).unwrap(),
        None
    );

    // 3. Cross-tenant query with identical vector and filter_hash MUST miss (zero cross-tenant pollution).
    assert_eq!(
        cache
            .get(&tenant_b, filter_1, &canonical_query, 120)
            .unwrap(),
        None
    );

    // 4. Different filter_hash within the same tenant MUST miss.
    assert_eq!(
        cache
            .get(&tenant_a, filter_2, &canonical_query, 120)
            .unwrap(),
        None
    );

    // 5. After TTL expiration (t = 160 => age = 60s >= ttl_secs = 60s), entry is evicted and misses.
    assert_eq!(
        cache
            .get(&tenant_a, filter_1, &canonical_query, 160)
            .unwrap(),
        None
    );
    assert_eq!(cache.len(), 0);

    // 6. Verify bounded capacity eviction (max_entries = 2).
    cache
        .put(
            tenant_a.clone(),
            filter_1,
            &[1.0, 0.0, 0.0],
            vec![ScoredId { id: 1, score: 1.0 }],
            200,
        )
        .unwrap();
    cache
        .put(
            tenant_a.clone(),
            filter_1,
            &[0.0, 1.0, 0.0],
            vec![ScoredId { id: 2, score: 1.0 }],
            201,
        )
        .unwrap();
    assert_eq!(cache.len(), 2);

    // Inserting a 3rd live entry evicts the oldest entry ([1.0, 0.0, 0.0]).
    cache
        .put(
            tenant_a.clone(),
            filter_1,
            &[0.0, 0.0, 1.0],
            vec![ScoredId { id: 3, score: 1.0 }],
            202,
        )
        .unwrap();
    assert_eq!(cache.len(), 2);
    assert_eq!(
        cache
            .get(&tenant_a, filter_1, &[1.0, 0.0, 0.0], 205)
            .unwrap(),
        None
    );
    assert_eq!(
        cache
            .get(&tenant_a, filter_1, &[0.0, 1.0, 0.0], 205)
            .unwrap(),
        Some(vec![ScoredId { id: 2, score: 1.0 }])
    );
    assert_eq!(
        cache
            .get(&tenant_a, filter_1, &[0.0, 0.0, 1.0], 205)
            .unwrap(),
        Some(vec![ScoredId { id: 3, score: 1.0 }])
    );

    // Validation errors on invalid config.
    assert!(matches!(
        SemanticCache::new(3, 1.05, 60, 10),
        Err(VektorError::InvalidConfig(_))
    ));
}
