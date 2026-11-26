use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use vektordist_core::checkpoint::ShardCheckpointManager;
use vektordist_core::exact::generate_seeded_unit_vectors;
use vektordist_core::hnsw::{HnswIndex, HnswParams};
use vektordist_core::wal::{replay_into_hnsw, WalLog, WalOp, WalRecord};
use vektordist_core::{DistanceMetric, ShardId, VektorError};

static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(1);

fn create_temp_shard_dir(tag: &str) -> PathBuf {
    let n = TEST_DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("vektordist-m12-{tag}-{pid}-{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn checkpoint_compacts_wal_over_90_percent_reclaims_tombstones_and_recovers_exact_state() {
    let data_dir = create_temp_shard_dir("compaction-and-recovery");
    let shard_id: ShardId = 7;
    let dims = 32;
    let metric = DistanceMetric::Cosine;
    let params = HnswParams {
        m: 16,
        m_max0: 32,
        ef_construction: 200,
        ef_search: 600,
        ..HnswParams::default()
    };

    let vectors = generate_seeded_unit_vectors(600, dims, 202_612);
    let mut ref_hnsw = HnswIndex::with_capacity(dims, 550, metric, params).unwrap();
    let mut shard_hnsw = HnswIndex::with_capacity(dims, 550, metric, params).unwrap();

    let wal_path = ShardCheckpointManager::wal_path(&data_dir, shard_id);
    let (mut wal, initial_replay) = WalLog::open(&wal_path, dims).unwrap();
    assert!(initial_replay.records.is_empty());

    let mut pre_ckpt_records = Vec::with_capacity(650);

    // 1..=300: 300 initial upserts (VectorIds 1..=300)
    for seq in 1..=300_u64 {
        pre_ckpt_records.push(WalRecord {
            seq_no: seq,
            write_id: 10_000 + seq,
            op: WalOp::Upsert {
                id: seq,
                vector: vectors[(seq - 1) as usize].clone(),
            },
        });
    }

    // 301..=500: 200 overwrites on VectorIds 1..=200 (creates 200 overwritten tombstones)
    for i in 0..200_u64 {
        let seq = 301 + i;
        pre_ckpt_records.push(WalRecord {
            seq_no: seq,
            write_id: 10_000 + seq,
            op: WalOp::Upsert {
                id: i + 1,
                vector: vectors[(300 + i) as usize].clone(),
            },
        });
    }

    // 501..=650: 150 deletes on VectorIds 1..=150 (creates 150 delete tombstones; 350 total tombstones)
    for i in 0..150_u64 {
        let seq = 501 + i;
        pre_ckpt_records.push(WalRecord {
            seq_no: seq,
            write_id: 10_000 + seq,
            op: WalOp::Delete { id: i + 1 },
        });
    }

    wal.append_batch_and_sync(&pre_ckpt_records).unwrap();
    assert_eq!(
        replay_into_hnsw(&pre_ckpt_records, &mut ref_hnsw).unwrap(),
        650
    );
    assert_eq!(
        replay_into_hnsw(&pre_ckpt_records, &mut shard_hnsw).unwrap(),
        650
    );
    assert_eq!(shard_hnsw.len(), 150);
    assert_eq!(shard_hnsw.arena().slot_count(), 500);
    assert_eq!(shard_hnsw.last_seq_no(), 650);

    // Create atomic checkpoint with empty retained tail -> 100% (> 99%) WAL reclamation!
    let manifest =
        ShardCheckpointManager::create_checkpoint(&data_dir, shard_id, &shard_hnsw, &mut wal, &[])
            .unwrap();

    assert_eq!(manifest.shard_id, shard_id);
    assert_eq!(manifest.checkpoint_seq_no, 650);
    assert_eq!(manifest.dims, dims);
    assert_eq!(manifest.live_vectors, 150);
    assert_eq!(manifest.reclaimed_tombstones, 350);
    assert!(manifest.wal_bytes_before > 70_000);
    assert_eq!(manifest.wal_bytes_after, 0);
    assert_eq!(wal.file_size_bytes().unwrap(), 0);

    let reduction_ratio =
        1.0 - (manifest.wal_bytes_after as f64) / (manifest.wal_bytes_before as f64);
    assert!(reduction_ratio > 0.99);

    // 651..=700: 50 post-checkpoint delta operations (35 upserts + 15 deletes)
    let mut delta_records = Vec::with_capacity(50);
    for i in 0..35_u64 {
        let seq = 651 + i;
        // Re-insert 15 previously deleted IDs (1..=15) and insert 20 brand-new IDs (301..=320)
        let target_id = if i < 15 { i + 1 } else { 301 + (i - 15) };
        delta_records.push(WalRecord {
            seq_no: seq,
            write_id: 20_000 + seq,
            op: WalOp::Upsert {
                id: target_id,
                vector: vectors[(500 + i) as usize].clone(),
            },
        });
    }
    for i in 0..15_u64 {
        let seq = 686 + i;
        delta_records.push(WalRecord {
            seq_no: seq,
            write_id: 20_000 + seq,
            op: WalOp::Delete { id: 151 + i },
        });
    }

    wal.append_batch_and_sync(&delta_records).unwrap();
    assert_eq!(replay_into_hnsw(&delta_records, &mut ref_hnsw).unwrap(), 50);
    drop(shard_hnsw);
    drop(wal);

    // Simulate cold restart: recover from .ckpt + .vdsg + post-checkpoint delta WAL
    let (recovered_hnsw, _recovered_wal, replay, recovered_manifest) =
        ShardCheckpointManager::recover_shard(&data_dir, shard_id, dims, metric, params).unwrap();

    let recovered_manifest = recovered_manifest.unwrap();
    assert_eq!(recovered_manifest.checkpoint_seq_no, 650);
    assert_eq!(recovered_manifest.live_vectors, 150);
    assert_eq!(recovered_manifest.segment_crc32, manifest.segment_crc32);
    assert_eq!(replay.records.len(), 50);
    assert_eq!(replay.last_seq_no, 700);

    assert_eq!(recovered_hnsw.len(), ref_hnsw.len());
    assert_eq!(recovered_hnsw.len(), 170);
    assert_eq!(recovered_hnsw.last_seq_no(), 700);
    // Pre-checkpoint tombstones remain stripped in the recovered arena (185 slots vs 535 slots).
    assert_eq!(recovered_hnsw.arena().slot_count(), 185);
    assert_eq!(ref_hnsw.arena().slot_count(), 535);

    let queries = generate_seeded_unit_vectors(20, dims, 999_123);
    for (q_idx, q) in queries.iter().enumerate() {
        let expected_top10 = ref_hnsw.search(q, 10).unwrap();
        let recovered_top10 = recovered_hnsw.search(q, 10).unwrap();
        assert_eq!(
            recovered_top10, expected_top10,
            "top-10 mismatch on query #{q_idx}"
        );
    }

    let _ = std::fs::remove_dir_all(&data_dir);
}

#[test]
fn corrupted_checkpoint_manifest_or_vdsg_segment_rejected_with_wal_corruption() {
    let data_dir = create_temp_shard_dir("corruption-detection");
    let shard_id: ShardId = 3;
    let dims = 16;
    let metric = DistanceMetric::L2;
    let params = HnswParams::default();

    let vectors = generate_seeded_unit_vectors(32, dims, 77);
    let mut hnsw = HnswIndex::new(dims, metric, params).unwrap();
    let wal_path = ShardCheckpointManager::wal_path(&data_dir, shard_id);
    let (mut wal, _) = WalLog::open(&wal_path, dims).unwrap();

    for (idx, v) in vectors.iter().enumerate() {
        let seq = (idx + 1) as u64;
        let rec = WalRecord {
            seq_no: seq,
            write_id: 500 + seq,
            op: WalOp::Upsert {
                id: seq * 10,
                vector: v.clone(),
            },
        };
        wal.append_and_sync(&rec).unwrap();
        replay_into_hnsw(&[rec], &mut hnsw).unwrap();
    }

    ShardCheckpointManager::create_checkpoint(&data_dir, shard_id, &hnsw, &mut wal, &[]).unwrap();
    drop(wal);

    let ckpt_path = ShardCheckpointManager::checkpoint_path(&data_dir, shard_id);
    let vdsg_path = ShardCheckpointManager::segment_path(&data_dir, shard_id);

    let original_ckpt = std::fs::read(&ckpt_path).unwrap();
    assert_eq!(original_ckpt.len(), 40);
    let original_vdsg = std::fs::read(&vdsg_path).unwrap();

    // 1. Flip 1 byte in `checkpoint_seq_no` inside `.ckpt` -> must fail with `WalCorruption`.
    let mut corrupted_ckpt = original_ckpt.clone();
    corrupted_ckpt[18] ^= 0x01;
    std::fs::write(&ckpt_path, &corrupted_ckpt).unwrap();

    let ckpt_err = ShardCheckpointManager::recover_shard(&data_dir, shard_id, dims, metric, params)
        .unwrap_err();
    assert!(matches!(ckpt_err, VektorError::WalCorruption { .. }));

    // 2. Restore `.ckpt`, flip 1 byte inside `.vdsg` vector payload -> must fail with `WalCorruption`.
    std::fs::write(&ckpt_path, &original_ckpt).unwrap();
    let mut corrupted_vdsg = original_vdsg.clone();
    let mid_offset = corrupted_vdsg.len() / 2;
    corrupted_vdsg[mid_offset] ^= 0x80;
    std::fs::write(&vdsg_path, &corrupted_vdsg).unwrap();

    let vdsg_err = ShardCheckpointManager::recover_shard(&data_dir, shard_id, dims, metric, params)
        .unwrap_err();
    assert!(matches!(vdsg_err, VektorError::WalCorruption { .. }));

    // 3. Restore `.vdsg` -> clean recovery succeeds again.
    std::fs::write(&vdsg_path, &original_vdsg).unwrap();
    let (recovered, _, _, manifest) =
        ShardCheckpointManager::recover_shard(&data_dir, shard_id, dims, metric, params).unwrap();
    assert_eq!(recovered.len(), 32);
    assert_eq!(manifest.unwrap().checkpoint_seq_no, 32);

    let _ = std::fs::remove_dir_all(&data_dir);
}

#[test]
fn retained_wal_tail_compaction_and_uncheckpointed_recovery_are_idempotent() {
    let data_dir = create_temp_shard_dir("retained-tail");
    let shard_id: ShardId = 11;
    let dims = 8;
    let metric = DistanceMetric::L2;
    let params = HnswParams::default();

    let vectors = generate_seeded_unit_vectors(40, dims, 4242);
    let wal_path = ShardCheckpointManager::wal_path(&data_dir, shard_id);
    let (mut wal, _) = WalLog::open(&wal_path, dims).unwrap();

    let mut all_records = Vec::with_capacity(40);
    for (idx, v) in vectors.iter().enumerate() {
        let seq = (idx + 1) as u64;
        all_records.push(WalRecord {
            seq_no: seq,
            write_id: 900 + seq,
            op: WalOp::Upsert {
                id: seq,
                vector: v.clone(),
            },
        });
    }
    wal.append_batch_and_sync(&all_records).unwrap();
    drop(wal);

    // Cold recovery when no `.ckpt` exists yet replays all 40 WAL records cleanly.
    let (hnsw_before_ckpt, mut wal, replay0, ckpt0) =
        ShardCheckpointManager::recover_shard(&data_dir, shard_id, dims, metric, params).unwrap();
    assert!(ckpt0.is_none());
    assert_eq!(replay0.records.len(), 40);
    assert_eq!(hnsw_before_ckpt.len(), 40);
    assert_eq!(hnsw_before_ckpt.last_seq_no(), 40);

    // Checkpoint at seq_no = 40 while retaining the last 2 WAL records (`39..=40`) for follower catch-up.
    let retained_tail = &all_records[38..40];
    let manifest = ShardCheckpointManager::create_checkpoint(
        &data_dir,
        shard_id,
        &hnsw_before_ckpt,
        &mut wal,
        retained_tail,
    )
    .unwrap();
    assert!(manifest.wal_bytes_after > 0);
    assert!(manifest.wal_bytes_after * 10 < manifest.wal_bytes_before);

    // Recovering with overlapping retained frames (`seq_no <= 40`) skips them idempotently without duplicating slots.
    drop(wal);
    let (recovered, _, replay1, ckpt1) =
        ShardCheckpointManager::recover_shard(&data_dir, shard_id, dims, metric, params).unwrap();
    assert!(ckpt1.is_some());
    assert_eq!(replay1.records.len(), 2);
    assert_eq!(recovered.len(), 40);
    assert_eq!(recovered.arena().slot_count(), 40);
    assert_eq!(recovered.last_seq_no(), 40);

    let _ = std::fs::remove_dir_all(&data_dir);
}
