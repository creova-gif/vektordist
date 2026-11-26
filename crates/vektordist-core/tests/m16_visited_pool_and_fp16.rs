use std::thread;

use vektordist_core::exact::{ExactIndex, SeededRng};
use vektordist_core::fp16::{
    dot_product_f16_f16, dot_product_f32_f16, f16_bits_to_f32, f32_to_f16_bits, l2_squared_f16_f16,
    l2_squared_f32_f16, Fp16HnswIndex,
};
use vektordist_core::hnsw::{recall_at_k, HnswIndex, HnswParams};
use vektordist_core::DistanceMetric;

#[test]
fn hnsw_visited_pool_reuses_capacity_across_1000_queries_with_exact_parity() {
    fn assert_traits<T: std::fmt::Debug + Clone + Send + Sync>() {}
    assert_traits::<HnswIndex>();
    assert_traits::<Fp16HnswIndex>();

    let dims = 64;
    let count = 500;
    let total_queries = 1_000;
    let top_k = 10;

    let mut rng = SeededRng::new(2026);
    let mut exact = ExactIndex::with_capacity(dims, count, DistanceMetric::Cosine).unwrap();
    let params = HnswParams {
        m: 16,
        m_max0: 32,
        ef_construction: 128,
        ef_search: 64,
        ..HnswParams::default()
    };
    let mut hnsw = HnswIndex::with_capacity(dims, count, DistanceMetric::Cosine, params).unwrap();

    let num_clusters = 16;
    let mut centroids = vec![vec![0.0_f32; dims]; num_clusters];
    for c in &mut centroids {
        rng.fill_unit_vector(c);
    }

    let mut buf = vec![0.0_f32; dims];
    for id in 0..count {
        let centroid = &centroids[id % num_clusters];
        for (d, &c_val) in buf.iter_mut().zip(centroid.iter()) {
            *d = c_val + 0.30 * rng.next_f32_signed();
        }
        exact.upsert(id as u64, &buf).unwrap();
        hnsw.upsert(id as u64, &buf).unwrap();
    }

    // Generate 1,000 deterministic queries up front.
    let mut queries = Vec::with_capacity(total_queries);
    for q_idx in 0..total_queries {
        let mut q = vec![0.0_f32; dims];
        let centroid = &centroids[q_idx % num_clusters];
        for (d, &c_val) in q.iter_mut().zip(centroid.iter()) {
            *d = c_val + 0.30 * rng.next_f32_signed();
        }
        queries.push(q);
    }

    // 1. Single-threaded 1,000 queries: pool never exceeds 1 table and retains >= 500 slots.
    let mut single_thread_results = Vec::with_capacity(total_queries);
    let mut total_recall = 0.0_f64;
    for q in &queries {
        let exact_hits = exact.search(q, top_k).unwrap();
        let approx_hits = hnsw.search(q, top_k).unwrap();
        assert_eq!(approx_hits.len(), top_k);
        total_recall += recall_at_k(&exact_hits, &approx_hits, top_k);
        single_thread_results.push(approx_hits);
    }
    let mean_recall = total_recall / (total_queries as f64);
    assert_eq!(
        hnsw.visited_pool_len(),
        1,
        "single-threaded queries must reuse a single VisitedTable"
    );
    assert!(
        hnsw.visited_pool_capacity_slots() >= count,
        "expected visited_pool_capacity_slots() >= {count}, got {}",
        hnsw.visited_pool_capacity_slots()
    );
    assert!(
        mean_recall >= 0.98,
        "expected recall@10 >= 0.98, got {mean_recall:.4}"
    );

    // Filtered search also reuses the exact same VisitedTable without growing pool count.
    let filtered = hnsw
        .search_filtered_with_ef(&queries[0], top_k, 64, |_slot, id| id % 2 == 0)
        .unwrap();
    assert_eq!(filtered.len(), top_k);
    assert!(filtered.iter().all(|hit| hit.id % 2 == 0));
    assert_eq!(hnsw.visited_pool_len(), 1);

    // Clone starts with an empty independent pool while producing bit-identical hits.
    let cloned = hnsw.clone();
    assert_eq!(cloned.visited_pool_len(), 0);
    assert_eq!(
        cloned.search(&queries[0], top_k).unwrap(),
        single_thread_results[0]
    );
    assert_eq!(cloned.visited_pool_len(), 1);

    // 2. Concurrent 4-thread 1,000 queries via std::thread::scope: exact result parity & bounded pool.
    let num_threads = 4_usize;
    let chunk_size = total_queries / num_threads;
    let concurrent_chunks: Vec<Vec<Vec<_>>> = thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);
        for chunk in queries.chunks(chunk_size) {
            let hnsw_ref = &hnsw;
            handles.push(s.spawn(move || {
                chunk
                    .iter()
                    .map(|q| hnsw_ref.search(q, top_k).unwrap())
                    .collect::<Vec<_>>()
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let concurrent_results: Vec<_> = concurrent_chunks.into_iter().flatten().collect();
    assert_eq!(concurrent_results.len(), total_queries);
    assert_eq!(
        concurrent_results, single_thread_results,
        "concurrent VisitedPool queries must match single-threaded results bit-for-bit"
    );
    assert!(
        (1..=num_threads).contains(&hnsw.visited_pool_len()),
        "expected 1..={num_threads} pooled VisitedTables after 4-thread run, got {}",
        hnsw.visited_pool_len()
    );
    assert!(hnsw.visited_pool_capacity_slots() >= count);
}

#[test]
fn fp16_ieee754_conversions_and_2x_hnsw_index_achieve_99_percent_recall() {
    // 1. Verify IEEE 754 binary16 conversions across [-1.0, 1.0], +-0.0, subnormals, and large values.
    assert_eq!(f32_to_f16_bits(0.0_f32), 0x0000);
    assert_eq!(f32_to_f16_bits(-0.0_f32), 0x8000);
    assert!(f16_bits_to_f32(0x8000).is_sign_negative() && f16_bits_to_f32(0x8000) == 0.0);

    for step in -1_000..=1_000 {
        let x = (step as f32) / 1_000.0;
        let roundtrip = f16_bits_to_f32(f32_to_f16_bits(x));
        assert!(
            (x - roundtrip).abs() <= 1e-3,
            "unexpected FP16 error for {x}: got {roundtrip}"
        );
    }

    // Subnormal values (2^-24 .. 2^-14)
    for exp in -24..=-15 {
        let sub = 2.0_f32.powi(exp);
        let bits = f32_to_f16_bits(sub);
        assert_eq!((bits >> 10) & 0x1f, 0, "expected subnormal exponent 0");
        assert_ne!(bits & 0x03ff, 0);
        assert_eq!(f16_bits_to_f32(bits), sub);
    }

    // Large finite values & overflow
    let large = 32_768.0_f32;
    assert_eq!(f16_bits_to_f32(f32_to_f16_bits(large)), large);
    assert_eq!(f16_bits_to_f32(f32_to_f16_bits(65_504.0)), 65_504.0);
    assert!(f16_bits_to_f32(f32_to_f16_bits(100_000.0)).is_infinite());

    // Verify 4-lane unrolled asymmetric & symmetric distance kernels (including non-multiple-of-4 tail).
    let q_f32 = [0.25_f32, -0.5, 0.75, -0.125, 0.5, -0.25, 0.125];
    let q_f16: Vec<u16> = q_f32.iter().copied().map(f32_to_f16_bits).collect();
    let expected_norm_sq: f32 = q_f32.iter().map(|v| v * v).sum();
    assert!((dot_product_f32_f16(&q_f32, &q_f16) - expected_norm_sq).abs() < 1e-6);
    assert!((dot_product_f16_f16(&q_f16, &q_f16) - expected_norm_sq).abs() < 1e-6);
    assert!(l2_squared_f32_f16(&q_f32, &q_f16).abs() < 1e-6);
    assert!(l2_squared_f16_f16(&q_f16, &q_f16).abs() < 1e-6);

    // 2. Build ExactIndex (f32) vs Fp16HnswIndex (600 x 768d) and verify 2.0x RAM & >= 0.99 recall@10.
    let dims = 768;
    let count = 600;
    let num_queries = 100;
    let top_k = 10;

    let mut rng = SeededRng::new(16_016);
    let mut exact = ExactIndex::with_capacity(dims, count, DistanceMetric::Cosine).unwrap();
    let params = HnswParams {
        m: 16,
        m_max0: 32,
        ef_construction: 128,
        ef_search: 64,
        ..HnswParams::default()
    };
    let mut fp16 =
        Fp16HnswIndex::with_capacity(dims, count, DistanceMetric::Cosine, params).unwrap();

    let num_clusters = 15;
    let mut centroids = vec![vec![0.0_f32; dims]; num_clusters];
    for c in &mut centroids {
        rng.fill_unit_vector(c);
    }

    let mut buf = vec![0.0_f32; dims];
    for id in 0..count {
        let centroid = &centroids[id % num_clusters];
        for (d, &c_val) in buf.iter_mut().zip(centroid.iter()) {
            *d = c_val + 0.30 * rng.next_f32_signed();
        }
        exact.upsert(id as u64, &buf).unwrap();
        fp16.upsert(id as u64, &buf).unwrap();
    }

    assert_eq!(exact.arena().vector_bytes(), 1_843_200);
    assert_eq!(fp16.arena().vector_bytes(), 921_600);
    assert_eq!(
        exact.arena().vector_bytes() / fp16.arena().vector_bytes(),
        2
    );

    let mut total_recall = 0.0_f64;
    for q_idx in 0..num_queries {
        let centroid = &centroids[q_idx % num_clusters];
        for (d, &c_val) in buf.iter_mut().zip(centroid.iter()) {
            *d = c_val + 0.30 * rng.next_f32_signed();
        }
        let exact_hits = exact.search(&buf, top_k).unwrap();
        let fp16_hits = fp16.search(&buf, top_k).unwrap();
        assert_eq!(fp16_hits.len(), top_k);
        total_recall += recall_at_k(&exact_hits, &fp16_hits, top_k);
    }

    let mean_recall = total_recall / (num_queries as f64);
    assert_eq!(fp16.visited_pool_len(), 1);
    assert!(fp16.visited_pool_capacity_slots() >= count);
    assert!(
        mean_recall >= 0.99,
        "expected Fp16HnswIndex recall@10 >= 0.99 without external f32 rerank, got {mean_recall:.4}"
    );
}

#[test]
fn fp16_hnsw_l2_metric_tombstones_overwrites_and_slot_decode() {
    let dims = 32;
    let mut fp16 = Fp16HnswIndex::new(dims, DistanceMetric::L2, HnswParams::default()).unwrap();
    assert!(fp16.is_empty());

    let mut v1 = vec![0.0_f32; dims];
    let mut v2 = vec![0.0_f32; dims];
    let mut v3 = vec![0.0_f32; dims];
    v1[0] = 1.0;
    v2[1] = 2.0;
    v3[2] = 3.0;

    let s0 = fp16.upsert(10, &v1).unwrap();
    fp16.upsert(20, &v2).unwrap();
    fp16.upsert(30, &v3).unwrap();
    assert_eq!(fp16.len(), 3);
    assert_eq!(fp16.arena().decode_at_slot(s0).unwrap(), v1);

    // Overwrite id=10 from v1 to v2; old slot is tombstoned and id=10 appears once.
    fp16.upsert(10, &v2).unwrap();
    assert_eq!(fp16.len(), 3);
    let hits = fp16.search(&v2, 3).unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits.iter().filter(|h| h.id == 10).count(), 1);
    assert!(hits[0].score < 1e-5);

    // Tombstone id=20; only ids 10 and 30 remain.
    assert!(fp16.delete(20));
    assert!(!fp16.delete(20));
    assert_eq!(fp16.len(), 2);

    let hits_after = fp16.search(&v2, 5).unwrap();
    assert_eq!(hits_after.len(), 2);
    assert_eq!(hits_after[0].id, 10);
    assert_eq!(hits_after[1].id, 30);
    assert!(fp16.memory_bytes() >= fp16.arena().vector_bytes());
}
