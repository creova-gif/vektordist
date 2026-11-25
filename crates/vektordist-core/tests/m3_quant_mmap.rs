use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::time::Instant;

use vektordist_core::arena::VectorArena;
use vektordist_core::exact::{ExactIndex, SeededRng};
use vektordist_core::hnsw::{recall_at_k, HnswIndex, HnswParams};
use vektordist_core::mmap::PersistentVectorSegment;
use vektordist_core::quant::{Sq8HnswIndex, Sq8Quantizer};
use vektordist_core::simd::{
    dot_product_scalar, dot_product_simd, dot_product_u8_asymmetric,
    dot_product_u8_asymmetric_scalar, l2_squared_scalar, l2_squared_simd, l2_squared_u8_asymmetric,
    l2_squared_u8_asymmetric_scalar,
};
use vektordist_core::{DistanceMetric, VektorError};

#[test]
fn simd_and_scalar_kernels_match_across_odd_and_768_dimensions() {
    let mut rng = SeededRng::new(2026);
    for dims in [1, 5, 13, 19, 33, 64, 128, 384, 768, 769, 1536] {
        let mut a = vec![0.0_f32; dims];
        let mut b = vec![0.0_f32; dims];
        rng.fill_unit_vector(&mut a);
        rng.fill_unit_vector(&mut b);

        let dot_s = dot_product_scalar(&a, &b);
        let dot_v = dot_product_simd(&a, &b);
        assert!(
            (dot_s - dot_v).abs() < 1e-5,
            "dot_product parity failed for dims={dims}: scalar={dot_s}, simd={dot_v}"
        );

        let l2_s = l2_squared_scalar(&a, &b);
        let l2_v = l2_squared_simd(&a, &b);
        assert!(
            (l2_s - l2_v).abs() < 1e-5,
            "l2_squared parity failed for dims={dims}: scalar={l2_s}, simd={l2_v}"
        );

        let quant = Sq8Quantizer::for_unit_vectors(dims).unwrap();
        let code = quant.encode(&b).unwrap();
        let asym_dot_s =
            dot_product_u8_asymmetric_scalar(&a, &code, quant.scale(), quant.min_val());
        let asym_dot_v = dot_product_u8_asymmetric(&a, &code, quant.scale(), quant.min_val());
        assert!(
            (asym_dot_s - asym_dot_v).abs() < 1e-4,
            "asymmetric dot parity failed for dims={dims}: scalar={asym_dot_s}, simd={asym_dot_v}"
        );

        let asym_l2_s = l2_squared_u8_asymmetric_scalar(&a, &code, quant.scale(), quant.min_val());
        let asym_l2_v = l2_squared_u8_asymmetric(&a, &code, quant.scale(), quant.min_val());
        assert!(
            (asym_l2_s - asym_l2_v).abs() < 1e-4,
            "asymmetric l2 parity failed for dims={dims}: scalar={asym_l2_s}, simd={asym_l2_v}"
        );
    }
}

#[test]
fn sq8_quantizer_achieves_4x_compression_and_95_percent_recall_at_10() {
    let dims = 64;
    let count = 2_000;
    let queries = 50;
    let top_k = 10;

    let mut rng = SeededRng::new(42);
    let num_clusters = 20;
    let mut centroids = vec![vec![0.0_f32; dims]; num_clusters];
    for c in &mut centroids {
        rng.fill_unit_vector(c);
    }

    let mut dataset = Vec::with_capacity(count);
    for id in 0..count {
        let centroid = &centroids[id % num_clusters];
        let mut v = vec![0.0_f32; dims];
        for (d, &c_val) in v.iter_mut().zip(centroid.iter()) {
            *d = c_val + 0.35 * rng.next_f32_signed();
        }
        dataset.push(v);
    }

    let quantizer = Sq8Quantizer::train(dims, &dataset[..256], true).unwrap();
    assert_eq!(quantizer.compression_ratio(), 4.0);

    let params = HnswParams {
        m: 16,
        m_max0: 32,
        ef_construction: 128,
        ef_search: 64,
        ..HnswParams::default()
    };

    let t_exact_build = Instant::now();
    let mut exact = ExactIndex::with_capacity(dims, count, DistanceMetric::Cosine).unwrap();
    for (id, vec) in dataset.iter().enumerate() {
        exact.upsert(id as u64, vec).unwrap();
    }
    let exact_build_ms = t_exact_build.elapsed().as_secs_f64() * 1000.0;

    let t_hnsw_build = Instant::now();
    let mut hnsw_f32 =
        HnswIndex::with_capacity(dims, count, DistanceMetric::Cosine, params).unwrap();
    for (id, vec) in dataset.iter().enumerate() {
        hnsw_f32.upsert(id as u64, vec).unwrap();
    }
    let hnsw_build_ms = t_hnsw_build.elapsed().as_secs_f64() * 1000.0;

    let t_sq8_build = Instant::now();
    let mut hnsw_sq8 =
        Sq8HnswIndex::with_capacity(quantizer, count, DistanceMetric::Cosine, params).unwrap();
    for (id, vec) in dataset.iter().enumerate() {
        hnsw_sq8.upsert(id as u64, vec).unwrap();
    }
    let sq8_build_ms = t_sq8_build.elapsed().as_secs_f64() * 1000.0;

    // Verify exact 4x vector payload reduction (4 bytes/dim -> 1 byte/dim).
    let f32_vec_bytes = hnsw_f32.arena().vector_bytes();
    let sq8_vec_bytes = hnsw_sq8.arena().vector_bytes();
    assert_eq!(f32_vec_bytes, count * dims * 4);
    assert_eq!(sq8_vec_bytes, count * dims);
    assert_eq!(f32_vec_bytes / sq8_vec_bytes, 4);

    let mut query_vecs = Vec::with_capacity(queries);
    for q_idx in 0..queries {
        let centroid = &centroids[q_idx % num_clusters];
        let mut q = vec![0.0_f32; dims];
        for (d, &c_val) in q.iter_mut().zip(centroid.iter()) {
            *d = c_val + 0.35 * rng.next_f32_signed();
        }
        query_vecs.push(q);
    }

    let mut f32_recall_sum = 0.0_f64;
    let mut sq8_asym_recall_sum = 0.0_f64;
    let mut sq8_rerank_recall_sum = 0.0_f64;

    let mut exact_us_total = 0.0_f64;
    let mut f32_us_total = 0.0_f64;
    let mut sq8_asym_us_total = 0.0_f64;
    let mut sq8_rerank_us_total = 0.0_f64;

    for q in &query_vecs {
        let t0 = Instant::now();
        let exact_hits = exact.search(q, top_k).unwrap();
        exact_us_total += t0.elapsed().as_secs_f64() * 1_000_000.0;

        let t1 = Instant::now();
        let f32_hits = hnsw_f32.search(q, top_k).unwrap();
        f32_us_total += t1.elapsed().as_secs_f64() * 1_000_000.0;

        let t2 = Instant::now();
        let sq8_asym_hits = hnsw_sq8.search(q, top_k).unwrap();
        sq8_asym_us_total += t2.elapsed().as_secs_f64() * 1_000_000.0;

        let t3 = Instant::now();
        let sq8_rerank_hits = hnsw_sq8
            .search_reranked_with_arena(q, top_k, 64, 3, Some(exact.arena()))
            .unwrap();
        sq8_rerank_us_total += t3.elapsed().as_secs_f64() * 1_000_000.0;

        f32_recall_sum += recall_at_k(&exact_hits, &f32_hits, top_k);
        sq8_asym_recall_sum += recall_at_k(&exact_hits, &sq8_asym_hits, top_k);
        sq8_rerank_recall_sum += recall_at_k(&exact_hits, &sq8_rerank_hits, top_k);
    }

    let f32_recall = f32_recall_sum / (queries as f64);
    let sq8_asym_recall = sq8_asym_recall_sum / (queries as f64);
    let sq8_rerank_recall = sq8_rerank_recall_sum / (queries as f64);

    println!(
        "M3 Benchmark ({count}x{dims}d, {queries} queries):
  ExactIndex: build={exact_build_ms:.2}ms, vector_bytes={f32_vec_bytes}, query_avg={:.2}us
  HnswIndex(f32): build={hnsw_build_ms:.2}ms, vector_bytes={f32_vec_bytes}, total_bytes={}, recall@10={f32_recall:.4}, query_avg={:.2}us
  Sq8HnswIndex(asym): build={sq8_build_ms:.2}ms, vector_bytes={sq8_vec_bytes}, total_bytes={}, recall@10={sq8_asym_recall:.4}, query_avg={:.2}us
  Sq8HnswIndex(rerank x3): recall@10={sq8_rerank_recall:.4}, query_avg={:.2}us",
        exact_us_total / (queries as f64),
        hnsw_f32.memory_bytes(),
        f32_us_total / (queries as f64),
        hnsw_sq8.memory_bytes(),
        sq8_asym_us_total / (queries as f64),
        sq8_rerank_us_total / (queries as f64),
    );

    assert!(
        sq8_asym_recall >= 0.95,
        "expected SQ8 asymmetric recall@10 >= 0.95, got {sq8_asym_recall:.4}"
    );
    assert!(
        sq8_rerank_recall >= 0.95,
        "expected SQ8 reranked recall@10 >= 0.95, got {sq8_rerank_recall:.4}"
    );
}

#[test]
fn persistent_vector_segment_round_trip_slice_reads_and_crc32_corruption_detection() {
    let tmp_dir = std::env::temp_dir().join(format!("vektordist_m3_seg_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let seg_path = tmp_dir.join("shard_0.vdsg");

    let dims = 768;
    let mut rng = SeededRng::new(99);
    let mut arena = VectorArena::with_capacity(dims, 16).unwrap();

    let mut v1 = vec![0.0_f32; dims];
    let mut v2 = vec![0.0_f32; dims];
    let mut v3 = vec![0.0_f32; dims];
    rng.fill_unit_vector(&mut v1);
    rng.fill_unit_vector(&mut v2);
    rng.fill_unit_vector(&mut v3);

    arena.upsert(100, &v1).unwrap();
    arena.upsert(200, &v2).unwrap();
    arena.upsert(300, &v3).unwrap();
    assert!(arena.delete(200));

    let seg = PersistentVectorSegment::write_from_arena(&seg_path, &arena).unwrap();
    assert_eq!(seg.dims(), dims);
    assert_eq!(seg.slot_count(), 3);
    assert_eq!(seg.live_len(), 2);
    assert_eq!(seg.get(100), Some(&v1[..]));
    assert_eq!(seg.get(200), None);
    assert_eq!(seg.get(300), Some(&v3[..]));
    assert!(seg.is_tombstoned(1));

    // Reopen from disk and verify zero-copy slices + exact search parity.
    let reopened = PersistentVectorSegment::open(&seg_path).unwrap();
    assert_eq!(reopened.get(100), Some(&v1[..]));
    assert_eq!(reopened.get(200), None);
    assert_eq!(reopened.get(300), Some(&v3[..]));

    let hits = reopened
        .search_exact(&v1, 2, DistanceMetric::Cosine)
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].id, 100);
    assert!((hits[0].score - 1.0).abs() < 1e-5);

    let restored_arena = reopened.to_arena().unwrap();
    assert_eq!(restored_arena.live_len(), 2);
    assert_eq!(restored_arena.get(100), Some(&v1[..]));
    assert_eq!(restored_arena.get(300), Some(&v3[..]));

    // Flip a byte inside the vector payload and confirm CRC32 corruption detection.
    {
        let mut f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&seg_path)
            .unwrap();
        f.seek(SeekFrom::Start(64)).unwrap();
        f.write_all(&[0xFF]).unwrap();
        f.sync_all().unwrap();
    }

    let corrupted = PersistentVectorSegment::open(&seg_path);
    assert!(
        matches!(corrupted, Err(VektorError::WalCorruption { .. })),
        "expected WalCorruption on bit flip, got {corrupted:?}"
    );

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
