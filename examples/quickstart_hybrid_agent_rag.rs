//! End-to-end runnable showcase of VektorDist's enterprise retrieval & AI agent memory stack:
//! 1. Multi-tenant `TenantHybridEngine` (`OrgId` isolation, `MetadataFilter`, dense + BM25 `RRF`).
//! 2. Multi-vector `ColbertIndex` late-interaction `MaxSim` candidate re-ranking.
//! 3. `AgentMemoryStore` composite semantic + time-decay + importance scoring with token-budget packing.
//! 4. `SemanticCache` sub-50us cosine-thresholded query hit verification.

#![deny(clippy::unwrap_used, clippy::expect_used)]

use vektordist_core::agentic::{
    AgentMemoryStore, ColbertIndex, MemoryScoringWeights, SemanticCache,
};
use vektordist_core::exact::{ScoredId, SeededRng};
use vektordist_core::hnsw::HnswParams;
use vektordist_core::hybrid::{
    Metadata, MetadataFilter, MetadataValue, OrgId, TenantHybridEngine, DEFAULT_RRF_K,
};
use vektordist_core::DistanceMetric;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dims = 64;
    let mut rng = SeededRng::new(2026);
    let org_acme = OrgId::new("acme-prod")?;
    let org_globex = OrgId::new("globex-corp")?;

    // 1. Multi-tenant Hybrid Dense + BM25 RRF Search with One-Stage Metadata Filtering
    let mut hybrid = TenantHybridEngine::new(dims, DistanceMetric::Cosine, HnswParams::default())?;
    let mut query_vec = vec![0.0_f32; dims];
    rng.fill_unit_vector(&mut query_vec);

    let passages = [
        (
            1_u64,
            "tier-1",
            "quorum wal replication prevents acknowledged write loss during sigkill",
        ),
        (
            2_u64,
            "tier-1",
            "arm neon simd kernels accelerate cosine and sq8 distance scoring",
        ),
        (
            3_u64,
            "tier-2",
            "raft topology consensus fences stale shard primaries across epochs",
        ),
    ];

    for (id, tier, text) in passages {
        let mut v = query_vec.clone();
        if id != 1 {
            rng.fill_unit_vector(&mut v);
        }
        let mut meta = Metadata::new();
        meta.insert("tier".to_string(), MetadataValue::String(tier.to_string()));
        hybrid.upsert(org_acme.clone(), id, &v, meta, text)?;
    }

    // Adversarial twin record under a different tenant (`globex-corp`) — must never leak to `acme-prod`
    hybrid.upsert(
        org_globex,
        999,
        &query_vec,
        Metadata::new(),
        "quorum wal replication prevents acknowledged write loss",
    )?;

    let tier1_filter = MetadataFilter::Eq {
        key: "tier".to_string(),
        value: MetadataValue::String("tier-1".to_string()),
    };
    let rrf_hits = hybrid.hybrid_search_rrf(
        &org_acme,
        &query_vec,
        "quorum wal sigkill",
        &tier1_filter,
        3,
        DEFAULT_RRF_K,
    )?;

    println!("=== 1. Multi-Tenant Hybrid RRF Hits (org=acme-prod, filter=tier-1) ===");
    for hit in &rrf_hits {
        println!(
            "  doc_id={} rrf_score={:.5} dense_rank={:?} lexical_rank={:?}",
            hit.id, hit.rrf_score, hit.dense_rank, hit.lexical_rank
        );
    }

    // 2. ColBERT Late-Interaction Multi-Vector MaxSim Re-Ranking
    let mut colbert = ColbertIndex::new(dims)?;
    let mut tok_b = vec![0.0_f32; dims];
    rng.fill_unit_vector(&mut tok_b);
    colbert.upsert(1, vec![query_vec.clone(), tok_b.clone()])?;
    colbert.upsert(2, vec![tok_b.clone()])?;

    let candidate_ids: Vec<u64> = rrf_hits.iter().map(|h| h.id).collect();
    let reranked = colbert.rerank(&[query_vec.clone(), tok_b], &candidate_ids, 2)?;
    println!("=== 2. ColBERT MaxSim Reranked Top Candidates ===");
    for hit in &reranked {
        println!("  doc_id={} maxsim_score={:.4}", hit.id, hit.score);
    }

    // 3. Tenant-Scoped Semantic Query Cache
    let mut cache = SemanticCache::new(dims, 0.96, 300, 128)?;
    let filter_hash = 0xCAFE_BABE_u64;
    let scored_for_cache: Vec<ScoredId> = reranked.clone();
    cache.put(
        org_acme.clone(),
        filter_hash,
        &query_vec,
        scored_for_cache,
        1_000,
    )?;
    let cached_hit = cache.get(&org_acme, filter_hash, &query_vec, 1_005)?;
    println!(
        "=== 3. Semantic Query Cache ===\n  cache_hit={} returned_ids={:?}",
        cached_hit.is_some(),
        cached_hit
            .unwrap_or_default()
            .iter()
            .map(|s| s.id)
            .collect::<Vec<_>>()
    );

    // 4. Programmable Agentic Episodic Memory with Token-Budget Context Packing
    let mut memory_store = AgentMemoryStore::new(dims, MemoryScoringWeights::default())?;
    memory_store.upsert_memory(
        org_acme.clone(),
        101,
        &query_vec,
        "Operator ran chaos failover drill; node-5 recovered in 42ms with 0 writes lost.",
        9_980,
        0.95,
        16,
    )?;
    memory_store.upsert_memory(
        org_acme.clone(),
        102,
        &query_vec,
        "Archived maintenance note from last week.",
        1_000,
        0.30,
        12,
    )?;

    let packed = memory_store.retrieve_context_window(&org_acme, &query_vec, 10_000, 5, 20)?;
    println!(
        "=== 4. AgentMemoryStore Context Window (budget=20 tokens) ===\n  packed_memories={} top_id={} score={:.4} tokens={}",
        packed.len(),
        packed.first().map_or(0, |m| m.id),
        packed.first().map_or(0.0, |m| m.total_score),
        packed.iter().map(|m| m.token_count).sum::<usize>()
    );

    Ok(())
}
