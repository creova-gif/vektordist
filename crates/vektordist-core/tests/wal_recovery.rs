use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use vektordist_core::hnsw::{HnswIndex, HnswParams};
use vektordist_core::wal::{encode_frame, replay_into_hnsw, WalLog, WalOp, WalRecord};
use vektordist_core::{DistanceMetric, VektorError};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(1);

fn temp_wal_path(tag: &str) -> PathBuf {
    let n = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("vektordist-wal-test-{tag}-{pid}-{n}.wal"))
}

#[test]
fn wal_round_trip_and_idempotent_hnsw_replay() {
    let path = temp_wal_path("roundtrip");
    let _ = std::fs::remove_file(&path);
    let dims = 4;

    {
        let (mut wal, replay) = WalLog::open(&path, dims).unwrap();
        assert!(replay.records.is_empty());
        assert_eq!(replay.truncated_tail_bytes, 0);

        let batch = vec![
            WalRecord {
                seq_no: 1,
                write_id: 101,
                op: WalOp::Upsert {
                    id: 10,
                    vector: vec![1.0, 0.0, 0.0, 0.0],
                },
            },
            WalRecord {
                seq_no: 2,
                write_id: 102,
                op: WalOp::Upsert {
                    id: 20,
                    vector: vec![0.0, 1.0, 0.0, 0.0],
                },
            },
        ];
        wal.append_batch_and_sync(&batch).unwrap();
        wal.append_and_sync(&WalRecord {
            seq_no: 3,
            write_id: 103,
            op: WalOp::Delete { id: 10 },
        })
        .unwrap();
    }

    let (_, replay) = WalLog::open(&path, dims).unwrap();
    assert_eq!(replay.records.len(), 3);
    assert_eq!(replay.last_seq_no, 3);
    assert_eq!(replay.truncated_tail_bytes, 0);

    let mut hnsw = HnswIndex::new(dims, DistanceMetric::Cosine, HnswParams::default()).unwrap();
    let applied = replay_into_hnsw(&replay.records, &mut hnsw).unwrap();
    assert_eq!(applied, 3);
    assert_eq!(hnsw.len(), 1);
    assert_eq!(hnsw.last_seq_no(), 3);

    // Replaying the same records again is a no-op (idempotent seq_no guard).
    let applied_again = replay_into_hnsw(&replay.records, &mut hnsw).unwrap();
    assert_eq!(applied_again, 0);
    assert_eq!(hnsw.len(), 1);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn wal_truncates_partial_tail_frame_after_simulated_sigkill() {
    let path = temp_wal_path("partial-tail");
    let _ = std::fs::remove_file(&path);
    let dims = 4;

    {
        let (mut wal, _) = WalLog::open(&path, dims).unwrap();
        wal.append_and_sync(&WalRecord {
            seq_no: 1,
            write_id: 201,
            op: WalOp::Upsert {
                id: 1,
                vector: vec![1.0, 0.0, 0.0, 0.0],
            },
        })
        .unwrap();
        wal.append_and_sync(&WalRecord {
            seq_no: 2,
            write_id: 202,
            op: WalOp::Upsert {
                id: 2,
                vector: vec![0.0, 1.0, 0.0, 0.0],
            },
        })
        .unwrap();
    }

    // Append a torn 3rd frame (cut off mid-payload as if SIGKILL interrupted write()).
    let full_third = encode_frame(
        &WalRecord {
            seq_no: 3,
            write_id: 203,
            op: WalOp::Upsert {
                id: 3,
                vector: vec![0.0, 0.0, 1.0, 0.0],
            },
        },
        dims,
    )
    .unwrap();
    let torn_len = full_third.len() - 7;
    {
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(&full_third[..torn_len]).unwrap();
        f.sync_all().unwrap();
    }

    let (mut wal_after, replay) = WalLog::open(&path, dims).unwrap();
    assert_eq!(replay.records.len(), 2);
    assert_eq!(replay.last_seq_no, 2);
    assert_eq!(replay.truncated_tail_bytes, torn_len as u64);

    // Subsequent appends succeed cleanly after tail truncation.
    wal_after
        .append_and_sync(&WalRecord {
            seq_no: 3,
            write_id: 203,
            op: WalOp::Upsert {
                id: 3,
                vector: vec![0.0, 0.0, 1.0, 0.0],
            },
        })
        .unwrap();

    let (_, replay2) = WalLog::open(&path, dims).unwrap();
    assert_eq!(replay2.records.len(), 3);
    assert_eq!(replay2.truncated_tail_bytes, 0);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn wal_detects_mid_file_crc32_corruption_and_hard_fails() {
    let path = temp_wal_path("mid-corruption");
    let _ = std::fs::remove_file(&path);
    let dims = 4;

    {
        let (mut wal, _) = WalLog::open(&path, dims).unwrap();
        wal.append_and_sync(&WalRecord {
            seq_no: 1,
            write_id: 301,
            op: WalOp::Upsert {
                id: 1,
                vector: vec![1.0, 0.0, 0.0, 0.0],
            },
        })
        .unwrap();
        wal.append_and_sync(&WalRecord {
            seq_no: 2,
            write_id: 302,
            op: WalOp::Upsert {
                id: 2,
                vector: vec![0.0, 1.0, 0.0, 0.0],
            },
        })
        .unwrap();
    }

    // Flip one bit inside the first frame's payload (byte offset 20).
    {
        let mut f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        f.seek(SeekFrom::Start(20)).unwrap();
        f.write_all(&[0xFF]).unwrap();
        f.sync_all().unwrap();
    }

    let err = WalLog::open(&path, dims).unwrap_err();
    assert_eq!(err, VektorError::WalCorruption { offset: 0 });

    let _ = std::fs::remove_file(&path);
}
