use vektordist_core::exact::{ExactIndex, SeededRng};
use vektordist_core::hnsw::{recall_at_k, HnswIndex, HnswParams};
use vektordist_core::DistanceMetric;

#[test]
fn hnsw_recall_at_10_exceeds_95_percent_on_seeded_dataset() {
    let dims = 64;
    let count = 2_000;
    let queries = 50;
    let top_k = 10;

    let mut rng = SeededRng::new(42);
    let mut exact = ExactIndex::with_capacity(dims, count, DistanceMetric::Cosine).unwrap();
    let params = HnswParams {
        m: 16,
        m_max0: 32,
        ef_construction: 128,
        ef_search: 64,
        ..HnswParams::default()
    };
    let mut hnsw = HnswIndex::with_capacity(dims, count, DistanceMetric::Cosine, params).unwrap();

    // Generate clustered + uniform high-dimensional unit vectors so nearest-neighbor structure is realistic.
    let num_clusters = 20;
    let mut centroids = vec![vec![0.0_f32; dims]; num_clusters];
    for c in &mut centroids {
        rng.fill_unit_vector(c);
    }

    let mut buf = vec![0.0_f32; dims];
    for id in 0..count {
        let centroid = &centroids[id % num_clusters];
        for (d, &c_val) in buf.iter_mut().zip(centroid.iter()) {
            *d = c_val + 0.35 * rng.next_f32_signed();
        }
        exact.upsert(id as u64, &buf).unwrap();
        hnsw.upsert(id as u64, &buf).unwrap();
    }

    let mut total_recall = 0.0_f64;
    for q_idx in 0..queries {
        let centroid = &centroids[q_idx % num_clusters];
        for (d, &c_val) in buf.iter_mut().zip(centroid.iter()) {
            *d = c_val + 0.35 * rng.next_f32_signed();
        }
        let exact_hits = exact.search(&buf, top_k).unwrap();
        let approx_hits = hnsw.search(&buf, top_k).unwrap();
        assert_eq!(approx_hits.len(), top_k);
        total_recall += recall_at_k(&exact_hits, &approx_hits, top_k);
    }

    let mean_recall = total_recall / (queries as f64);
    println!(
        "HNSW recall@10 across {queries} queries on {count}x{dims}d: {:.4}",
        mean_recall
    );
    assert!(
        mean_recall >= 0.95,
        "expected recall@10 >= 0.95, got {mean_recall:.4}"
    );
}

#[test]
fn hnsw_tombstone_delete_and_overwrite_exclude_stale_entries() {
    let dims = 8;
    let mut hnsw = HnswIndex::new(dims, DistanceMetric::Cosine, HnswParams::default()).unwrap();

    let v_x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let v_y = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let v_z = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    hnsw.upsert(1, &v_x).unwrap();
    hnsw.upsert(2, &v_y).unwrap();
    hnsw.upsert(3, &v_z).unwrap();

    // Overwrite id=1 from x-axis to y-axis; duplicate id=1 must never appear.
    hnsw.upsert(1, &v_y).unwrap();
    let hits = hnsw.search(&v_x, 3).unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits.iter().filter(|h| h.id == 1).count(), 1);

    // Delete id=2; only ids 1 and 3 remain.
    assert!(hnsw.delete(2));
    assert!(!hnsw.delete(2));
    assert_eq!(hnsw.len(), 2);

    let hits_after = hnsw.search(&v_y, 5).unwrap();
    assert_eq!(hits_after.len(), 2);
    assert!(hits_after.iter().all(|h| h.id != 2));
    assert_eq!(hits_after[0].id, 1);
}
