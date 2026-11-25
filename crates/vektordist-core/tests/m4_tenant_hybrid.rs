use std::collections::HashMap;

use vektordist_core::exact::SeededRng;
use vektordist_core::hnsw::HnswParams;
use vektordist_core::hybrid::{
    Metadata, MetadataFilter, MetadataValue, OrgId, TenantHybridEngine, DEFAULT_RRF_K,
};
use vektordist_core::{DistanceMetric, VektorError};

fn meta_from(pairs: &[(&str, MetadataValue)]) -> Metadata {
    let mut m = HashMap::with_capacity(pairs.len());
    for (k, v) in pairs {
        m.insert((*k).to_string(), v.clone());
    }
    m
}

#[test]
fn org_id_rejects_empty_or_whitespace_identifiers() {
    assert_eq!(
        OrgId::new(""),
        Err(VektorError::InvalidConfig("org_id must not be empty"))
    );
    assert_eq!(
        OrgId::new("   \t\n "),
        Err(VektorError::InvalidConfig("org_id must not be empty"))
    );
    let valid = OrgId::new("org_acme_prod").unwrap();
    assert_eq!(valid.as_str(), "org_acme_prod");
}

#[test]
fn zero_cross_tenant_leakage_across_1000_interleaved_vectors_and_adversarial_twins() {
    let dims = 32;
    let total_vectors = 1_000;
    let top_k = 20;

    let mut rng = SeededRng::new(2026);
    let mut engine = TenantHybridEngine::with_capacity(
        dims,
        total_vectors + 10,
        DistanceMetric::Cosine,
        HnswParams::default(),
    )
    .unwrap();

    let tenants = [
        OrgId::new("org_alpha").unwrap(),
        OrgId::new("org_bravo").unwrap(),
        OrgId::new("org_charlie").unwrap(),
        OrgId::new("org_delta").unwrap(),
    ];

    let mut vec_buf = vec![0.0_f32; dims];
    for id in 0..total_vectors {
        rng.fill_unit_vector(&mut vec_buf);
        let tenant = tenants[id % tenants.len()].clone();
        let meta = meta_from(&[
            ("env", MetadataValue::String("prod".to_string())),
            ("seq", MetadataValue::Int(id as i64)),
        ]);
        let text = format!(
            "document_{id} service_mesh auth_gateway cluster_node_{}",
            id % 17
        );
        engine
            .upsert(tenant, id as u64, &vec_buf, meta, &text)
            .unwrap();
    }

    // Insert an exact adversarial twin under `org_bravo` with the exact same vector and keywords
    // that `org_alpha` will query for.
    let mut secret_vec = vec![0.0_f32; dims];
    rng.fill_unit_vector(&mut secret_vec);
    let secret_text = "classified_token_994812_xkz zero_trust_master_key";
    let victim_id = 9_999_u64;
    engine
        .upsert(
            tenants[1].clone(), // org_bravo owns this secret vector
            victim_id,
            &secret_vec,
            meta_from(&[("env", MetadataValue::String("prod".to_string()))]),
            secret_text,
        )
        .unwrap();

    // Verify `org_alpha` cannot delete or inspect `org_bravo`'s secret record.
    assert!(!engine.delete(&tenants[0], victim_id));
    assert!(engine.get_metadata(&tenants[0], victim_id).is_none());

    let mut cross_tenant_leakage = 0_usize;

    // 1. Adversarial query by `org_alpha` using `org_bravo`'s exact vector & secret keywords.
    let adv_dense = engine
        .search_dense_filtered(&tenants[0], &secret_vec, &MetadataFilter::All, top_k)
        .unwrap();
    let adv_lex =
        engine.search_lexical_filtered(&tenants[0], secret_text, &MetadataFilter::All, top_k);
    let adv_hybrid = engine
        .hybrid_search_rrf(
            &tenants[0],
            &secret_vec,
            secret_text,
            &MetadataFilter::All,
            top_k,
            DEFAULT_RRF_K,
        )
        .unwrap();

    for hit in &adv_dense {
        if engine.owner_of(hit.id) != Some(&tenants[0]) {
            cross_tenant_leakage += 1;
        }
    }
    for hit in &adv_lex {
        if engine.owner_of(hit.id) != Some(&tenants[0]) {
            cross_tenant_leakage += 1;
        }
    }
    for hit in &adv_hybrid {
        if engine.owner_of(hit.id) != Some(&tenants[0]) {
            cross_tenant_leakage += 1;
        }
    }

    // 2. Sweep 40 randomized queries across all 4 tenants.
    for q in 0..40 {
        rng.fill_unit_vector(&mut vec_buf);
        let caller = &tenants[q % tenants.len()];
        let dense_hits = engine
            .search_dense_filtered(caller, &vec_buf, &MetadataFilter::All, top_k)
            .unwrap();
        assert_eq!(dense_hits.len(), top_k);
        for h in dense_hits {
            if engine.owner_of(h.id) != Some(caller) {
                cross_tenant_leakage += 1;
            }
        }

        let hybrid_hits = engine
            .hybrid_search_rrf(
                caller,
                &vec_buf,
                "service_mesh auth_gateway cluster_node_3",
                &MetadataFilter::All,
                top_k,
                DEFAULT_RRF_K,
            )
            .unwrap();
        assert_eq!(hybrid_hits.len(), top_k);
        for h in hybrid_hits {
            if engine.owner_of(h.id) != Some(caller) {
                cross_tenant_leakage += 1;
            }
        }
    }

    assert_eq!(
        cross_tenant_leakage, 0,
        "expected strictly 0 cross-tenant leakage across all dense, lexical, and hybrid queries"
    );
}

#[test]
fn one_stage_metadata_filtering_maintains_full_top_k_under_10_percent_selectivity() {
    let dims = 32;
    let total_vectors = 1_000;
    let top_k = 15;

    let mut rng = SeededRng::new(777);
    let mut engine = TenantHybridEngine::with_capacity(
        dims,
        total_vectors,
        DistanceMetric::Cosine,
        HnswParams::default(),
    )
    .unwrap();
    let org = OrgId::new("org_enterprise").unwrap();

    let mut vec_buf = vec![0.0_f32; dims];
    for id in 0..total_vectors {
        rng.fill_unit_vector(&mut vec_buf);
        // Exactly 10% of vectors (id % 10 == 0) match:
        // tier == "gold" AND latency_ms in [5.0, 25.0] AND region in ["us-east", "eu-west"].
        let is_target = id % 10 == 0;
        let tier = if is_target { "gold" } else { "bronze" };
        let latency = if is_target { 12.5 } else { 250.0 };
        let region = if id % 20 == 0 {
            "us-east"
        } else if is_target {
            "eu-west"
        } else {
            "ap-south"
        };

        let meta = meta_from(&[
            ("tier", MetadataValue::String(tier.to_string())),
            ("latency_ms", MetadataValue::Float(latency)),
            ("region", MetadataValue::String(region.to_string())),
            ("active", MetadataValue::Bool(true)),
        ]);
        engine
            .upsert(
                org.clone(),
                id as u64,
                &vec_buf,
                meta,
                "distributed consensus raft wal",
            )
            .unwrap();
    }

    let filter = MetadataFilter::And(vec![
        MetadataFilter::Eq {
            key: "tier".to_string(),
            value: MetadataValue::String("gold".to_string()),
        },
        MetadataFilter::Range {
            key: "latency_ms".to_string(),
            min: Some(5.0),
            max: Some(25.0),
        },
        MetadataFilter::In {
            key: "region".to_string(),
            values: vec![
                MetadataValue::String("us-east".to_string()),
                MetadataValue::String("eu-west".to_string()),
            ],
        },
    ]);

    for _ in 0..10 {
        rng.fill_unit_vector(&mut vec_buf);
        let hits = engine
            .search_dense_filtered(&org, &vec_buf, &filter, top_k)
            .unwrap();
        assert_eq!(
            hits.len(),
            top_k,
            "one-stage filtered HNSW search must return full top_k={} under 10% predicate selectivity",
            top_k
        );
        for hit in hits {
            assert_eq!(
                hit.id % 10,
                0,
                "returned vector id={} did not satisfy the 10% metadata filter",
                hit.id
            );
            let meta = engine.get_metadata(&org, hit.id).unwrap();
            assert!(filter.matches(meta));
        }
    }
}

#[test]
fn hybrid_rrf_outperforms_pure_dense_and_pure_lexical_on_combined_queries() {
    let dims = 16;
    let org = OrgId::new("org_search_eval").unwrap();
    let mut engine =
        TenantHybridEngine::new(dims, DistanceMetric::Cosine, HnswParams::default()).unwrap();

    // Base semantic direction for "payment settlement failure".
    let mut target_semantic = vec![0.0_f32; dims];
    target_semantic[0] = 0.92;
    target_semantic[1] = 0.39;

    let mut orthogonal_semantic = vec![0.0_f32; dims];
    orthogonal_semantic[2] = 0.95;
    orthogonal_semantic[3] = 0.31;

    // 1. Distractor group A (ids 1..=2): highest dense similarity to query vector (ranks #1, #2 in dense),
    //    but paraphrased text with wrong error code (`ERR_TIMEOUT_504`) so lexical BM25 does not match at all.
    for i in 1..=2_u64 {
        let mut v = target_semantic.clone();
        v[1] += (i as f32) * 0.005;
        engine
            .upsert(
                org.clone(),
                i,
                &v,
                Metadata::new(),
                "wire transfer clearing delay gateway timeout ERR_TIMEOUT_504 retry queue",
            )
            .unwrap();
    }

    // 2. Distractor group B (ids 101..=102): contains exact identifier `ERR_LEDGER_9021` repeated 3x
    //    so pure lexical BM25 ranks them #1, #2, but semantically opposite vector (last in dense).
    let mut opposite_semantic = vec![0.0_f32; dims];
    opposite_semantic[0] = -0.92;
    opposite_semantic[1] = -0.39;
    for i in 101..=102_u64 {
        let mut v = opposite_semantic.clone();
        v[3] += ((i - 100) as f32) * 0.01;
        engine
            .upsert(
                org.clone(),
                i,
                &v,
                Metadata::new(),
                "ERR_LEDGER_9021 ERR_LEDGER_9021 ERR_LEDGER_9021 schema migration alter table",
            )
            .unwrap();
    }

    // 3. Background semantic records (ids 201..=220): moderate semantic alignment so opposite_semantic
    //    records rank at the bottom of the dense list (> rank 20).
    for i in 201..=220_u64 {
        let mut v = orthogonal_semantic.clone();
        v[0] = 0.25;
        v[2] += ((i - 200) as f32) * 0.01;
        engine
            .upsert(
                org.clone(),
                i,
                &v,
                Metadata::new(),
                "general ledger reconciliation batch job",
            )
            .unwrap();
    }

    // 4. True target record (id = 777): strong semantic alignment (rank #3 in dense, behind distractors 1..=2)
    //    AND contains the exact error identifier `ERR_LEDGER_9021` (rank #3 in lexical, behind distractors 101..=102).
    let true_target_id = 777_u64;
    let mut target_vec = target_semantic.clone();
    target_vec[0] = 0.85;
    target_vec[1] = 0.52;
    engine
        .upsert(
            org.clone(),
            true_target_id,
            &target_vec,
            Metadata::new(),
            "incident report quorum commit abort ERR_LEDGER_9021",
        )
        .unwrap();

    let query_text = "ERR_LEDGER_9021";

    let dense_top5 = engine
        .search_dense_filtered(&org, &target_semantic, &MetadataFilter::All, 5)
        .unwrap();
    let lexical_top5 = engine.search_lexical_filtered(&org, query_text, &MetadataFilter::All, 5);
    let hybrid_hits = engine
        .hybrid_search_rrf(
            &org,
            &target_semantic,
            query_text,
            &MetadataFilter::All,
            5,
            DEFAULT_RRF_K,
        )
        .unwrap();

    // Neither pure dense top-1 nor pure lexical top-1 finds `true_target_id` (777):
    // - Pure dense top-1 is a semantic distractor (`1..=5`) with the wrong error code.
    // - Pure lexical top-1 is a keyword distractor (`101..=105`) with an orthogonal semantic vector.
    assert_ne!(dense_top5[0].id, true_target_id);
    assert_ne!(lexical_top5[0].id, true_target_id);

    // Hybrid RRF promotes `true_target_id` (777) to rank #1 because it is the ONLY record
    // with high rank in BOTH dense HNSW similarity and BM25 lexical matching.
    assert_eq!(
        hybrid_hits[0].id, true_target_id,
        "expected hybrid RRF to rank true target 777 at #1; got {:?} (dense top: {:?}, lexical top: {:?})",
        hybrid_hits[0], dense_top5[0], lexical_top5[0]
    );
    assert!(hybrid_hits[0].dense_rank.is_some());
    assert!(hybrid_hits[0].lexical_rank.is_some());
}
