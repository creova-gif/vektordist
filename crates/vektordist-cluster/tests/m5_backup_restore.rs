use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use vektordist_cluster::backup::{
    restore_node_from_snapshot, shard_wal_filename, verify_snapshot, ClusterSnapshotManifest,
    SnapshotManager, MANIFEST_FILENAME, SNAPSHOT_MANIFEST_MAGIC,
};
use vektordist_cluster::node::NodeService;
use vektordist_core::exact::SeededRng;
use vektordist_core::hnsw::HnswParams;
use vektordist_core::wal::{WalOp, WalRecord};
use vektordist_core::{ClusterConfig, DistanceMetric, VektorError};

static SNAPSHOT_TEST_ID: AtomicU64 = AtomicU64::new(1);

fn temp_test_dir(tag: &str) -> PathBuf {
    let seq = SNAPSHOT_TEST_ID.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("vektordist-m5-test-{tag}-{pid}-{seq}"))
}

#[tokio::test]
async fn point_in_time_snapshot_verify_and_full_restore_preserves_all_vectors_and_search() {
    let config = ClusterConfig {
        num_shards: 4,
        replication_factor: 3,
        write_quorum: 2,
        dimensions: 16,
        metric: DistanceMetric::Cosine,
    };
    let params = HnswParams::default();

    let src_dir = temp_test_dir("src-node");
    let snap_dir = temp_test_dir("snapshot");
    let restore_dir = temp_test_dir("restored-node");
    let _ = std::fs::remove_dir_all(&src_dir);
    let _ = std::fs::remove_dir_all(&snap_dir);
    let _ = std::fs::remove_dir_all(&restore_dir);

    let node = NodeService::open(1, &src_dir, config.clone(), params).unwrap();
    let mut rng = SeededRng::new(2026);
    let mut vec_buf = vec![0.0_f32; config.dimensions];

    // Track per-shard sequence numbers and expected write_ids.
    let mut shard_seq = vec![0_u64; config.num_shards as usize];
    let mut expected_write_ids = HashSet::new();
    let total_vectors = 80_u64;

    for vec_id in 1..=total_vectors {
        rng.fill_unit_vector(&mut vec_buf);
        let shard_id = config.shard_for(vec_id);
        shard_seq[shard_id as usize] += 1;
        let seq_no = shard_seq[shard_id as usize];
        let write_id = 50_000 + vec_id;
        expected_write_ids.insert(write_id);

        let shard_lock = node.shards.get(&shard_id).unwrap();
        let mut engine = shard_lock.write().await;
        engine.epoch = 3;
        engine
            .apply_records(
                &[WalRecord {
                    seq_no,
                    write_id,
                    op: WalOp::Upsert {
                        id: vec_id,
                        vector: vec_buf.clone(),
                    },
                }],
                false,
            )
            .unwrap();
    }

    // Capture baseline top-K search results across every shard before taking snapshot.
    let mut query_vec = vec![0.0_f32; config.dimensions];
    rng.fill_unit_vector(&mut query_vec);
    let mut baseline_hits_per_shard = Vec::new();
    for shard_id in 0..config.num_shards {
        let engine = node.shards.get(&shard_id).unwrap().read().await;
        let hits = engine.hnsw.search_with_ef(&query_vec, 5, 32).unwrap();
        baseline_hits_per_shard.push((shard_id, engine.hnsw.last_seq_no(), hits));
    }

    // 1. Create point-in-time snapshot across all 4 shards.
    let manifest = SnapshotManager::create_snapshot(&node, &snap_dir)
        .await
        .unwrap();
    assert_eq!(manifest.node_id, 1);
    assert_eq!(manifest.shards.len(), config.num_shards as usize);

    let raw_manifest = std::fs::read(snap_dir.join(MANIFEST_FILENAME)).unwrap();
    assert_eq!(&raw_manifest[0..4], &SNAPSHOT_MANIFEST_MAGIC);
    let decoded_manifest = ClusterSnapshotManifest::from_bytes(&raw_manifest).unwrap();
    assert_eq!(decoded_manifest, manifest);

    // 2. Verify snapshot integrity report.
    let report = verify_snapshot(&snap_dir).unwrap();
    assert_eq!(report.verified_shards, config.num_shards);
    assert_eq!(report.total_vectors, total_vectors);
    assert!(report.total_bytes > 0);
    assert!(report.corrupt_shards.is_empty());

    // 3. Restore into a clean target directory and verify 100% vector, write_id, epoch, and search preservation.
    let restored =
        restore_node_from_snapshot(&snap_dir, &restore_dir, config.clone(), params).unwrap();
    assert_eq!(restored.node_id, 1);

    let mut restored_write_ids = HashSet::new();
    let mut restored_total_vectors = 0_u64;
    for (shard_id, expected_last_seq, baseline_hits) in &baseline_hits_per_shard {
        let engine = restored.shards.get(shard_id).unwrap().read().await;
        assert_eq!(engine.epoch, 3);
        assert_eq!(engine.hnsw.last_seq_no(), *expected_last_seq);
        restored_total_vectors += engine.hnsw.len() as u64;

        for rec in &engine.records_by_seq {
            restored_write_ids.insert(rec.write_id);
        }

        let restored_hits = engine.hnsw.search_with_ef(&query_vec, 5, 32).unwrap();
        assert_eq!(
            &restored_hits, baseline_hits,
            "restored HNSW top-K hits diverged on shard {shard_id}"
        );
    }

    assert_eq!(restored_total_vectors, total_vectors);
    assert_eq!(restored_write_ids, expected_write_ids);

    let _ = std::fs::remove_dir_all(&src_dir);
    let _ = std::fs::remove_dir_all(&snap_dir);
    let _ = std::fs::remove_dir_all(&restore_dir);
}

#[tokio::test]
async fn tamper_and_bit_rot_in_shard_snapshot_rejected_with_wal_corruption() {
    let config = ClusterConfig {
        num_shards: 4,
        replication_factor: 3,
        write_quorum: 2,
        dimensions: 8,
        metric: DistanceMetric::Cosine,
    };
    let params = HnswParams::default();

    let src_dir = temp_test_dir("tamper-src");
    let snap_dir = temp_test_dir("tamper-snap");
    let restore_dir = temp_test_dir("tamper-restore");
    let _ = std::fs::remove_dir_all(&src_dir);
    let _ = std::fs::remove_dir_all(&snap_dir);
    let _ = std::fs::remove_dir_all(&restore_dir);

    let node = NodeService::open(7, &src_dir, config.clone(), params).unwrap();
    let mut rng = SeededRng::new(99);
    let mut vec_buf = vec![0.0_f32; config.dimensions];

    for shard_id in 0..config.num_shards {
        rng.fill_unit_vector(&mut vec_buf);
        let shard_lock = node.shards.get(&shard_id).unwrap();
        let mut engine = shard_lock.write().await;
        engine
            .apply_records(
                &[WalRecord {
                    seq_no: 1,
                    write_id: 900 + u64::from(shard_id),
                    op: WalOp::Upsert {
                        id: u64::from(shard_id) + 1,
                        vector: vec_buf.clone(),
                    },
                }],
                false,
            )
            .unwrap();
    }

    SnapshotManager::create_snapshot(&node, &snap_dir)
        .await
        .unwrap();

    // Flip 1 byte in shard-0002.wal to simulate bit-rot / tampering.
    let shard2_path = snap_dir.join(shard_wal_filename(2));
    let mut shard2_bytes = std::fs::read(&shard2_path).unwrap();
    assert!(!shard2_bytes.is_empty());
    let mid = shard2_bytes.len() / 2;
    shard2_bytes[mid] ^= 0xFF;
    std::fs::write(&shard2_path, &shard2_bytes).unwrap();

    // Audit reports shard 2 in corrupt_shards.
    let audit = SnapshotManager::audit_snapshot(&snap_dir).unwrap();
    assert_eq!(audit.corrupt_shards, vec![2]);
    assert_eq!(audit.verified_shards, 3);

    // verify_snapshot and restore_node_from_snapshot both reject with VektorError::WalCorruption.
    let verify_err = verify_snapshot(&snap_dir).unwrap_err();
    assert!(matches!(verify_err, VektorError::WalCorruption { .. }));

    let restore_res = restore_node_from_snapshot(&snap_dir, &restore_dir, config, params);
    assert!(matches!(
        restore_res,
        Err(VektorError::WalCorruption { .. })
    ));

    let _ = std::fs::remove_dir_all(&src_dir);
    let _ = std::fs::remove_dir_all(&snap_dir);
    let _ = std::fs::remove_dir_all(&restore_dir);
}
