use vektordist_core::arena::VectorArena;
use vektordist_core::exact::{generate_seeded_unit_vectors, ExactIndex};
use vektordist_core::hnsw::recall_at_k;
use vektordist_core::pq_bq::{
    hamming_distance_u64, BqIndex, BqQuantizer, PqCodebook, PqIndex, TieredCompactor,
};
use vektordist_core::{DistanceMetric, VectorId};

#[test]
fn bq_index_achieves_32x_compression_and_high_recall_with_reranking() {
    let dims = 768;
    let num_vectors = 600;
    let num_queries = 40;
    let top_k = 10;
    let oversample = 8;

    // Generate 60 semantic topic centroids in 768-d space (~10 vectors per topic cluster)
    // plus per-vector Gaussian-like perturbations to model realistic 768-d embedding workloads.
    let centers = generate_seeded_unit_vectors(60, dims, 202_607);
    let perturbations = generate_seeded_unit_vectors(num_vectors, dims, 505_101);
    let mut dataset = Vec::with_capacity(num_vectors);
    for (i, pert) in perturbations.iter().enumerate() {
        let c1 = &centers[i % centers.len()];
        let c2 = &centers[(i * 7 + 3) % centers.len()];
        let mut v = vec![0.0_f32; dims];
        for d in 0..dims {
            v[d] = 0.55 * c1[d] + 0.25 * c2[d] + 0.20 * pert[d];
        }
        vektordist_core::arena::normalize_l2(&mut v);
        dataset.push(v);
    }

    let query_noise = generate_seeded_unit_vectors(num_queries, dims, 900_107);
    let mut queries = Vec::with_capacity(num_queries);
    for (i, qn) in query_noise.iter().enumerate() {
        let base = &dataset[(i * 13) % dataset.len()];
        let mut q = vec![0.0_f32; dims];
        for d in 0..dims {
            q[d] = 0.70 * base[d] + 0.30 * qn[d];
        }
        vektordist_core::arena::normalize_l2(&mut q);
        queries.push(q);
    }

    let mut exact = ExactIndex::with_capacity(dims, num_vectors, DistanceMetric::Cosine).unwrap();
    let mut bq = BqIndex::with_capacity(dims, num_vectors, DistanceMetric::Cosine).unwrap();

    for (i, vec) in dataset.iter().enumerate() {
        let id = (i + 1) as VectorId;
        exact.upsert(id, vec).unwrap();
        bq.upsert(id, vec).unwrap();
    }

    // Prove exact 32x payload reduction on 768-d vectors (3,072 bytes -> 96 bytes per vector).
    let exact_bytes = exact.arena().vector_bytes();
    let bq_bytes = bq.arena().vector_bytes();
    assert_eq!(exact_bytes, num_vectors * 3072);
    assert_eq!(bq_bytes, num_vectors * 96);
    assert_eq!(exact_bytes / bq_bytes, 32);
    assert!((bq.quantizer().compression_ratio() - 32.0).abs() < 1e-6);

    // Prove deterministic Hamming distance symmetry and triangle inequality on quantized codes.
    let q_quant = BqQuantizer::new(dims).unwrap();
    let c0 = q_quant.encode(&dataset[0]).unwrap();
    let c1 = q_quant.encode(&dataset[1]).unwrap();
    let c2 = q_quant.encode(&dataset[2]).unwrap();
    let d01 = hamming_distance_u64(&c0, &c1);
    let d10 = hamming_distance_u64(&c1, &c0);
    let d12 = hamming_distance_u64(&c1, &c2);
    let d02 = hamming_distance_u64(&c0, &c2);
    assert_eq!(d01, d10);
    assert!(d02 <= d01 + d12);

    // Evaluate recall@10 across all queries with two-stage oversampled reranking.
    let mut total_recall = 0.0_f64;
    for q in &queries {
        let truth = exact.search(q, top_k).unwrap();
        let approx = bq
            .search_reranked(q, top_k, oversample, exact.arena())
            .unwrap();
        assert_eq!(approx.len(), top_k);
        total_recall += recall_at_k(&truth, &approx, top_k);
    }
    let mean_recall = total_recall / (num_queries as f64);
    assert!(
        mean_recall >= 0.95,
        "expected BQ reranked recall@10 >= 0.95, got {mean_recall:.4}"
    );
}

#[test]
fn pq_index_deterministic_codebook_and_adc_high_recall() {
    let dims = 64;
    let m_subspaces = 16; // 4 dimensions per subspace -> 16x payload compression (256B -> 16B)
    let k_centroids = 64;
    let num_vectors = 320;
    let num_queries = 30;
    let top_k = 10;

    // Generate structured topic clusters (~10 vectors per semantic cluster) to model real embedding manifolds.
    let centers = generate_seeded_unit_vectors(32, dims, 88_123);
    let noise = generate_seeded_unit_vectors(num_vectors, dims, 99_456);
    let mut dataset = Vec::with_capacity(num_vectors);
    for (i, n) in noise.iter().enumerate() {
        let c = &centers[i % centers.len()];
        let mut v = vec![0.0_f32; dims];
        for d in 0..dims {
            v[d] = 0.82 * c[d] + 0.18 * n[d];
        }
        vektordist_core::arena::normalize_l2(&mut v);
        dataset.push(v);
    }

    // Verify deterministic training given the same seed.
    let cb1 = PqCodebook::train(dims, m_subspaces, k_centroids, &dataset, 15, 4242).unwrap();
    let cb2 = PqCodebook::train(dims, m_subspaces, k_centroids, &dataset, 15, 4242).unwrap();
    for sample in dataset.iter().take(16) {
        assert_eq!(cb1.encode(sample).unwrap(), cb2.encode(sample).unwrap());
    }

    let mut exact = ExactIndex::with_capacity(dims, num_vectors, DistanceMetric::Cosine).unwrap();
    let mut pq = PqIndex::with_capacity(cb1, num_vectors, DistanceMetric::Cosine);

    for (i, vec) in dataset.iter().enumerate() {
        let id = (i + 1) as VectorId;
        exact.upsert(id, vec).unwrap();
        pq.upsert(id, vec).unwrap();
    }

    // 64 floats (256 bytes) -> 16 bytes per vector = 16x compression.
    assert_eq!(exact.arena().vector_bytes(), num_vectors * 256);
    assert_eq!(pq.vector_bytes(), num_vectors * 16);
    assert!((pq.codebook().compression_ratio() - 16.0).abs() < 1e-6);

    let mut adc_only_recall = 0.0_f64;
    let mut adc_reranked_recall = 0.0_f64;

    for q in dataset.iter().take(num_queries) {
        let truth = exact.search(q, top_k).unwrap();
        let adc_hits = pq.search(q, top_k).unwrap();
        let reranked_hits = pq.search_reranked(q, top_k, 4, exact.arena()).unwrap();

        adc_only_recall += recall_at_k(&truth, &adc_hits, top_k);
        adc_reranked_recall += recall_at_k(&truth, &reranked_hits, top_k);
    }

    let mean_adc_recall = adc_only_recall / (num_queries as f64);
    let mean_reranked_recall = adc_reranked_recall / (num_queries as f64);
    assert!(
        mean_adc_recall >= 0.85,
        "expected pure ADC recall@10 >= 0.85, got {mean_adc_recall:.4}"
    );
    assert!(
        mean_reranked_recall >= 0.98,
        "expected PQ ADC + rerank recall@10 >= 0.98, got {mean_reranked_recall:.4}"
    );
}

#[test]
fn tiered_compactor_reclaims_all_tombstones_and_preserves_exact_search() {
    let dims = 64;
    let base_vecs = generate_seeded_unit_vectors(120, dims, 5050);
    let updated_vecs = generate_seeded_unit_vectors(40, dims, 6060);

    // Ground-truth single arena that receives the exact same sequence of operations.
    let mut reference_index = ExactIndex::new(dims, DistanceMetric::Cosine).unwrap();

    // Segment 0 (oldest): IDs 1..=60, plus internal overwrite on IDs 1..=10.
    let mut seg0 = VectorArena::new(dims).unwrap();
    for (i, vec) in base_vecs.iter().enumerate().take(60) {
        let id = (i + 1) as VectorId;
        seg0.upsert(id, vec).unwrap();
        reference_index.upsert(id, vec).unwrap();
    }
    for i in 0..10 {
        let id = (i + 1) as VectorId;
        seg0.upsert(id, &base_vecs[60 + i]).unwrap();
        reference_index.upsert(id, &base_vecs[60 + i]).unwrap();
    }

    // Segment 1 (middle): IDs 61..=100, overwrites IDs 11..=30, and deletes IDs 31..=40 in-segment.
    let mut seg1 = VectorArena::new(dims).unwrap();
    for (i, vec) in base_vecs.iter().enumerate().take(100).skip(60) {
        let id = (i + 1) as VectorId;
        seg1.upsert(id, vec).unwrap();
        reference_index.upsert(id, vec).unwrap();
    }
    for (i, vec) in updated_vecs.iter().enumerate().take(20) {
        let id = (11 + i) as VectorId;
        seg1.upsert(id, vec).unwrap();
        reference_index.upsert(id, vec).unwrap();
    }
    for id in 31..=40 {
        // Upsert then delete in seg1 to record an in-segment deletion tombstone.
        seg1.upsert(id, &updated_vecs[(id - 31) as usize]).unwrap();
        seg1.delete(id);
        reference_index.delete(id);
    }

    // Segment 2 (newest): Overwrites IDs 15..=25, resurrects ID 35, and we also pass explicit
    // cross-segment deletes for IDs 41..=50.
    let mut seg2 = VectorArena::new(dims).unwrap();
    for i in 0..11 {
        let id = (15 + i) as VectorId;
        seg2.upsert(id, &updated_vecs[20 + i]).unwrap();
        reference_index.upsert(id, &updated_vecs[20 + i]).unwrap();
    }
    seg2.upsert(35, &updated_vecs[35]).unwrap();
    reference_index.upsert(35, &updated_vecs[35]).unwrap();

    let explicit_deletes: Vec<VectorId> = (41..=50).collect();
    for &del_id in &explicit_deletes {
        reference_index.delete(del_id);
    }

    let segments = [&seg0, &seg1, &seg2];
    let (compacted, stats) =
        TieredCompactor::compact_with_deletes(&segments, &explicit_deletes).unwrap();

    // Verify 100% tombstone reclamation (`slot_count == live_len`).
    assert_eq!(compacted.slot_count(), compacted.live_len());
    assert_eq!(compacted.live_len(), reference_index.len());
    assert_eq!(stats.input_segments, 3);
    assert_eq!(
        stats.input_slots,
        seg0.slot_count() + seg1.slot_count() + seg2.slot_count()
    );
    assert_eq!(stats.output_live_vectors, compacted.live_len());
    assert_eq!(
        stats.reclaimed_tombstones,
        stats.input_slots - stats.output_live_vectors
    );
    assert!(stats.reclaimed_tombstones > 40);
    assert_eq!(
        stats.reclaimed_bytes,
        stats.reclaimed_tombstones * dims * std::mem::size_of::<f32>()
    );
    for slot in 0..compacted.slot_count() {
        assert!(!compacted.is_tombstoned(slot));
    }

    // Build an ExactIndex from the compacted arena and verify identical top-10 search hits.
    let mut compacted_index =
        ExactIndex::with_capacity(dims, compacted.live_len(), DistanceMetric::Cosine).unwrap();
    for slot in 0..compacted.slot_count() {
        let id = compacted.id_at_slot(slot).unwrap();
        let vec = compacted.vector_at_slot(slot).unwrap();
        compacted_index.upsert(id, vec).unwrap();
    }

    let queries = generate_seeded_unit_vectors(20, dims, 7070);
    for q in &queries {
        let expected_hits = reference_index.search(q, 10).unwrap();
        let compacted_hits = compacted_index.search(q, 10).unwrap();
        assert_eq!(expected_hits, compacted_hits);
    }
}
