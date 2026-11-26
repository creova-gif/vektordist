//! Milestone 12: Atomic CRC32-verified shard checkpointing (`b"VDCK"` + zero-tombstone `b"VDSG"`)
//! and crash-safe WAL log truncation.
//!
//! Checkpointing eliminates unbounded WAL disk growth and `O(|WAL|)` cold-restart replay:
//! 1. [`TieredCompactor::compact`] strips 100% of tombstoned and overwritten vector slots from
//!    the active [`HnswIndex`]'s [`crate::arena::VectorArena`].
//! 2. [`PersistentVectorSegment::write_from_arena`] writes the zero-tombstone arena to a sibling
//!    `.vdsg.tmp` file with `fsync` + trailing IEEE CRC32, then atomically `rename`s it to
//!    `shard-{shard_id:04}.vdsg`.
//! 3. [`ShardCheckpointManager`] writes a 40-byte `b"VDCK"` manifest (`shard-{shard_id:04}.ckpt`)
//!    via `.ckpt.tmp` -> `sync_all()` -> atomic `rename`, recording `checkpoint_seq_no` and
//!    `segment_crc32` protected by its own trailing IEEE CRC32.
//! 4. [`WalLog::compact_to`] atomically replaces the shard WAL with only `retained_tail` frames
//!    (typically `&[]`), reclaiming `> 99%` of historical WAL bytes without risking data loss on
//!    mid-compaction `SIGKILL`.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::hnsw::{HnswIndex, HnswParams};
use crate::mmap::PersistentVectorSegment;
use crate::pq_bq::TieredCompactor;
use crate::wal::{replay_into_hnsw, WalLog, WalRecord, WalReplayResult};
use crate::{DistanceMetric, SeqNo, ShardId, VektorError};

const CKPT_MAGIC: [u8; 4] = *b"VDCK";
const CKPT_VERSION: u32 = 1;
const CKPT_PAYLOAD_LEN: usize = 36;
const CKPT_TOTAL_LEN: usize = 40;

/// Durable metadata and telemetry describing a completed or recovered shard checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointManifest {
    pub shard_id: ShardId,
    pub checkpoint_seq_no: SeqNo,
    pub dims: usize,
    pub live_vectors: usize,
    pub reclaimed_tombstones: usize,
    pub segment_crc32: u32,
    pub wal_bytes_before: u64,
    pub wal_bytes_after: u64,
}

/// Coordinates atomic shard checkpoint creation (`b"VDCK"` + `b"VDSG"` + WAL truncation)
/// and fast CRC32-verified cold shard recovery.
#[derive(Debug, Default, Clone, Copy)]
pub struct ShardCheckpointManager;

impl ShardCheckpointManager {
    /// Canonical binary checkpoint manifest path (`shard-{shard_id:04}.ckpt`).
    #[must_use]
    pub fn checkpoint_path(data_dir: &Path, shard_id: ShardId) -> PathBuf {
        data_dir.join(format!("shard-{shard_id:04}.ckpt"))
    }

    /// Canonical compacted vector segment path (`shard-{shard_id:04}.vdsg`).
    #[must_use]
    pub fn segment_path(data_dir: &Path, shard_id: ShardId) -> PathBuf {
        data_dir.join(format!("shard-{shard_id:04}.vdsg"))
    }

    /// Canonical shard WAL path (`shard-{shard_id:04}.wal`).
    #[must_use]
    pub fn wal_path(data_dir: &Path, shard_id: ShardId) -> PathBuf {
        data_dir.join(format!("shard-{shard_id:04}.wal"))
    }

    /// Atomically snapshot `hnsw` into a zero-tombstone `b"VDSG"` segment + `b"VDCK"` checkpoint
    /// manifest and compact `wal` down to `retained_tail`.
    pub fn create_checkpoint(
        data_dir: &Path,
        shard_id: ShardId,
        hnsw: &HnswIndex,
        wal: &mut WalLog,
        retained_tail: &[WalRecord],
    ) -> Result<CheckpointManifest, VektorError> {
        std::fs::create_dir_all(data_dir).map_err(|e| VektorError::WalIo(e.to_string()))?;

        let wal_bytes_before = wal.file_size_bytes()?;
        let (compacted_arena, stats) = TieredCompactor::compact(&[hnsw.arena()])?;

        let vdsg_path = Self::segment_path(data_dir, shard_id);
        let vdsg_tmp_path = append_tmp_suffix(&vdsg_path);
        let seg = PersistentVectorSegment::write_from_arena(&vdsg_tmp_path, &compacted_arena)?;
        let segment_crc32 = seg.crc32();
        drop(seg);

        std::fs::rename(&vdsg_tmp_path, &vdsg_path)
            .map_err(|e| VektorError::WalIo(e.to_string()))?;

        let checkpoint_seq_no = hnsw.last_seq_no();
        let dims = compacted_arena.dims();
        let live_vectors = compacted_arena.live_len();

        let ckpt_path = Self::checkpoint_path(data_dir, shard_id);
        let ckpt_tmp_path = append_tmp_suffix(&ckpt_path);
        let ckpt_bytes = encode_ckpt_bytes(
            shard_id,
            dims,
            checkpoint_seq_no,
            live_vectors,
            segment_crc32,
        )?;

        {
            let mut ckpt_file = OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(true)
                .open(&ckpt_tmp_path)
                .map_err(|e| VektorError::WalIo(e.to_string()))?;
            ckpt_file
                .write_all(&ckpt_bytes)
                .map_err(|e| VektorError::WalIo(e.to_string()))?;
            ckpt_file
                .sync_all()
                .map_err(|e| VektorError::WalIo(e.to_string()))?;
        }

        std::fs::rename(&ckpt_tmp_path, &ckpt_path)
            .map_err(|e| VektorError::WalIo(e.to_string()))?;
        sync_dir_best_effort(data_dir);

        let wal_bytes_after = wal.compact_to(retained_tail)?;

        Ok(CheckpointManifest {
            shard_id,
            checkpoint_seq_no,
            dims,
            live_vectors,
            reclaimed_tombstones: stats.reclaimed_tombstones,
            segment_crc32,
            wal_bytes_before,
            wal_bytes_after,
        })
    }

    /// Recover a shard from its latest `b"VDCK"` + `b"VDSG"` checkpoint (if present) followed by
    /// idempotent replay of post-checkpoint WAL delta frames (`seq_no > checkpoint_seq_no`).
    pub fn recover_shard(
        data_dir: &Path,
        shard_id: ShardId,
        dims: usize,
        metric: DistanceMetric,
        params: HnswParams,
    ) -> Result<
        (
            HnswIndex,
            WalLog,
            WalReplayResult,
            Option<CheckpointManifest>,
        ),
        VektorError,
    > {
        if dims == 0 {
            return Err(VektorError::InvalidConfig("dimensions must be > 0"));
        }
        std::fs::create_dir_all(data_dir).map_err(|e| VektorError::WalIo(e.to_string()))?;

        let ckpt_path = Self::checkpoint_path(data_dir, shard_id);
        let vdsg_path = Self::segment_path(data_dir, shard_id);
        let wal_path = Self::wal_path(data_dir, shard_id);

        if ckpt_path.exists() {
            let ckpt_raw =
                std::fs::read(&ckpt_path).map_err(|e| VektorError::WalIo(e.to_string()))?;
            let decoded = decode_and_verify_ckpt(&ckpt_raw, shard_id, dims)?;

            let seg = PersistentVectorSegment::open(&vdsg_path)?;
            if seg.crc32() != decoded.segment_crc32
                || seg.dims() != dims
                || seg.live_len() != decoded.live_vectors
            {
                return Err(VektorError::WalCorruption { offset: 0 });
            }

            let mut hnsw = HnswIndex::with_capacity(dims, seg.live_len(), metric, params)?;
            for slot in 0..seg.slot_count() {
                if seg.is_tombstoned(slot) {
                    continue;
                }
                let (Some(id), Some(vec)) = (seg.id_at_slot(slot), seg.vector_at_slot(slot)) else {
                    return Err(VektorError::WalCorruption { offset: 0 });
                };
                hnsw.upsert_checkpoint_vector(id, vec)?;
            }
            hnsw.set_last_seq_no(decoded.checkpoint_seq_no);

            let (wal, replay) = WalLog::open(&wal_path, dims)?;
            let wal_bytes_after = wal.file_size_bytes()?;
            let delta_records: Vec<WalRecord> = replay
                .records
                .iter()
                .filter(|rec| rec.seq_no > hnsw.last_seq_no())
                .cloned()
                .collect();
            replay_into_hnsw(&delta_records, &mut hnsw)?;

            let manifest = CheckpointManifest {
                shard_id,
                checkpoint_seq_no: decoded.checkpoint_seq_no,
                dims,
                live_vectors: decoded.live_vectors,
                reclaimed_tombstones: 0,
                segment_crc32: decoded.segment_crc32,
                wal_bytes_before: 0,
                wal_bytes_after,
            };
            Ok((hnsw, wal, replay, Some(manifest)))
        } else {
            let (wal, replay) = WalLog::open(&wal_path, dims)?;
            let mut hnsw = HnswIndex::with_capacity(dims, replay.records.len(), metric, params)?;
            replay_into_hnsw(&replay.records, &mut hnsw)?;
            Ok((hnsw, wal, replay, None))
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct DecodedCkptHeader {
    checkpoint_seq_no: SeqNo,
    live_vectors: usize,
    segment_crc32: u32,
}

fn encode_ckpt_bytes(
    shard_id: ShardId,
    dims: usize,
    checkpoint_seq_no: SeqNo,
    live_vectors: usize,
    segment_crc32: u32,
) -> Result<[u8; CKPT_TOTAL_LEN], VektorError> {
    let dims_u32 = u32::try_from(dims)
        .map_err(|_| VektorError::InvalidConfig("dimensions overflow u32 checkpoint header"))?;
    let live_u64 = live_vectors as u64;

    let mut buf = [0_u8; CKPT_TOTAL_LEN];
    buf[0..4].copy_from_slice(&CKPT_MAGIC);
    buf[4..8].copy_from_slice(&CKPT_VERSION.to_le_bytes());
    buf[8..12].copy_from_slice(&shard_id.to_le_bytes());
    buf[12..16].copy_from_slice(&dims_u32.to_le_bytes());
    buf[16..24].copy_from_slice(&checkpoint_seq_no.to_le_bytes());
    buf[24..32].copy_from_slice(&live_u64.to_le_bytes());
    buf[32..36].copy_from_slice(&segment_crc32.to_le_bytes());

    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&buf[..CKPT_PAYLOAD_LEN]);
    let ckpt_crc32 = hasher.finalize();
    buf[36..40].copy_from_slice(&ckpt_crc32.to_le_bytes());
    Ok(buf)
}

fn decode_and_verify_ckpt(
    bytes: &[u8],
    expected_shard_id: ShardId,
    expected_dims: usize,
) -> Result<DecodedCkptHeader, VektorError> {
    if bytes.len() != CKPT_TOTAL_LEN {
        return Err(VektorError::WalCorruption { offset: 0 });
    }

    let expected_crc = u32::from_le_bytes([bytes[36], bytes[37], bytes[38], bytes[39]]);
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&bytes[..CKPT_PAYLOAD_LEN]);
    let actual_crc = hasher.finalize();
    if actual_crc != expected_crc {
        return Err(VektorError::WalCorruption { offset: 0 });
    }

    if bytes[0..4] != CKPT_MAGIC {
        return Err(VektorError::WalCorruption { offset: 0 });
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != CKPT_VERSION {
        return Err(VektorError::WalCorruption { offset: 0 });
    }
    let shard_id = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if shard_id != expected_shard_id {
        return Err(VektorError::WalCorruption { offset: 0 });
    }
    let dims = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    if dims != expected_dims {
        return Err(VektorError::WalCorruption { offset: 0 });
    }

    let checkpoint_seq_no = u64::from_le_bytes([
        bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
    ]);
    let live_u64 = u64::from_le_bytes([
        bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
    ]);
    let live_vectors =
        usize::try_from(live_u64).map_err(|_| VektorError::WalCorruption { offset: 0 })?;
    let segment_crc32 = u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);

    Ok(DecodedCkptHeader {
        checkpoint_seq_no,
        live_vectors,
        segment_crc32,
    })
}

fn append_tmp_suffix(path: &Path) -> PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(".tmp");
    PathBuf::from(os)
}

fn sync_dir_best_effort(dir: &Path) {
    if !dir.as_os_str().is_empty() {
        if let Ok(handle) = File::open(dir) {
            let _ = handle.sync_all();
        }
    }
}
