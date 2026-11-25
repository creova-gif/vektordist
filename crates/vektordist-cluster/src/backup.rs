//! Point-in-time cluster snapshots (`b"VDSN"`), CRC32 integrity verification, and deterministic node restore.
//!
//! `SnapshotManager` captures a consistent point-in-time snapshot across all shards hosted by a
//! `NodeService` into a portable directory containing:
//! - `manifest.bin`: binary header (`magic: b"VDSN"`, version, node ID, cluster config, per-shard
//!   `ShardSnapshotEntry` metadata, and trailing whole-manifest IEEE CRC32 checksum).
//! - `shard-XXXX.wal`: byte-exact copy of each shard's durable WAL file verified by IEEE CRC32.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use vektordist_core::hnsw::HnswParams;
use vektordist_core::{ClusterConfig, DistanceMetric, Epoch, NodeId, SeqNo, ShardId, VektorError};

use crate::node::NodeService;

/// 4-byte magic header identifying a VektorDist snapshot manifest (`b"VDSN"`).
pub const SNAPSHOT_MANIFEST_MAGIC: [u8; 4] = *b"VDSN";

/// Binary format version for `manifest.bin`.
pub const SNAPSHOT_MANIFEST_VERSION: u32 = 1;

/// Standard filename for the snapshot manifest inside a snapshot directory.
pub const MANIFEST_FILENAME: &str = "manifest.bin";

const MANIFEST_HEADER_LEN: usize = 28;
const SHARD_ENTRY_LEN: usize = 40;
const MANIFEST_CRC_LEN: usize = 4;

/// Per-shard metadata and WAL checksum recorded in a [`ClusterSnapshotManifest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardSnapshotEntry {
    pub shard_id: ShardId,
    pub epoch: Epoch,
    pub last_seq_no: SeqNo,
    pub vector_count: u64,
    pub wal_file_bytes: u64,
    pub wal_crc32: u32,
}

/// Portable point-in-time cluster snapshot manifest (`manifest.bin`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterSnapshotManifest {
    pub node_id: NodeId,
    pub config: ClusterConfig,
    pub shards: Vec<ShardSnapshotEntry>,
    pub manifest_crc32: u32,
}

impl ClusterSnapshotManifest {
    /// Serialize this manifest into its canonical little-endian binary representation with a
    /// trailing IEEE CRC32 checksum.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let total_payload_len = MANIFEST_HEADER_LEN + self.shards.len() * SHARD_ENTRY_LEN;
        let mut buf = Vec::with_capacity(total_payload_len + MANIFEST_CRC_LEN);

        buf.extend_from_slice(&SNAPSHOT_MANIFEST_MAGIC);
        buf.extend_from_slice(&SNAPSHOT_MANIFEST_VERSION.to_le_bytes());
        buf.extend_from_slice(&self.node_id.to_le_bytes());
        buf.extend_from_slice(&self.config.num_shards.to_le_bytes());
        buf.push(self.config.replication_factor);
        buf.push(self.config.write_quorum);
        buf.push(metric_to_u8(self.config.metric));
        buf.push(0_u8);
        buf.extend_from_slice(&(self.config.dimensions as u32).to_le_bytes());
        buf.extend_from_slice(&(self.shards.len() as u32).to_le_bytes());
        debug_assert_eq!(buf.len(), MANIFEST_HEADER_LEN);

        for entry in &self.shards {
            buf.extend_from_slice(&entry.shard_id.to_le_bytes());
            buf.extend_from_slice(&entry.epoch.to_le_bytes());
            buf.extend_from_slice(&entry.last_seq_no.to_le_bytes());
            buf.extend_from_slice(&entry.vector_count.to_le_bytes());
            buf.extend_from_slice(&entry.wal_file_bytes.to_le_bytes());
            buf.extend_from_slice(&entry.wal_crc32.to_le_bytes());
        }

        let crc = compute_crc32(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Parse and validate a `ClusterSnapshotManifest` from raw bytes, checking the `b"VDSN"` magic,
    /// format version, entry count, and trailing IEEE CRC32 checksum.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VektorError> {
        if bytes.len() < MANIFEST_HEADER_LEN + MANIFEST_CRC_LEN {
            return Err(VektorError::WalCorruption { offset: 0 });
        }
        if bytes[0..4] != SNAPSHOT_MANIFEST_MAGIC {
            return Err(VektorError::WalCorruption { offset: 0 });
        }
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if version != SNAPSHOT_MANIFEST_VERSION {
            return Err(VektorError::WalCorruption { offset: 4 });
        }

        let node_id = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let num_shards = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let replication_factor = bytes[16];
        let write_quorum = bytes[17];
        let metric = u8_to_metric(bytes[18])?;
        let dimensions = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]) as usize;
        let shard_count = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]) as usize;

        let crc_offset =
            MANIFEST_HEADER_LEN.saturating_add(shard_count.saturating_mul(SHARD_ENTRY_LEN));
        let expected_len = crc_offset.saturating_add(MANIFEST_CRC_LEN);
        if bytes.len() != expected_len {
            return Err(VektorError::WalCorruption {
                offset: bytes.len() as u64,
            });
        }

        let expected_crc = u32::from_le_bytes([
            bytes[crc_offset],
            bytes[crc_offset + 1],
            bytes[crc_offset + 2],
            bytes[crc_offset + 3],
        ]);
        let actual_crc = compute_crc32(&bytes[..crc_offset]);
        if actual_crc != expected_crc {
            return Err(VektorError::WalCorruption {
                offset: crc_offset as u64,
            });
        }

        let config = ClusterConfig {
            num_shards,
            replication_factor,
            write_quorum,
            dimensions,
            metric,
        };
        config.validate()?;

        let mut shards = Vec::with_capacity(shard_count);
        for idx in 0..shard_count {
            let off = MANIFEST_HEADER_LEN + idx * SHARD_ENTRY_LEN;
            let shard_id =
                u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
            let epoch = u64::from_le_bytes([
                bytes[off + 4],
                bytes[off + 5],
                bytes[off + 6],
                bytes[off + 7],
                bytes[off + 8],
                bytes[off + 9],
                bytes[off + 10],
                bytes[off + 11],
            ]);
            let last_seq_no = u64::from_le_bytes([
                bytes[off + 12],
                bytes[off + 13],
                bytes[off + 14],
                bytes[off + 15],
                bytes[off + 16],
                bytes[off + 17],
                bytes[off + 18],
                bytes[off + 19],
            ]);
            let vector_count = u64::from_le_bytes([
                bytes[off + 20],
                bytes[off + 21],
                bytes[off + 22],
                bytes[off + 23],
                bytes[off + 24],
                bytes[off + 25],
                bytes[off + 26],
                bytes[off + 27],
            ]);
            let wal_file_bytes = u64::from_le_bytes([
                bytes[off + 28],
                bytes[off + 29],
                bytes[off + 30],
                bytes[off + 31],
                bytes[off + 32],
                bytes[off + 33],
                bytes[off + 34],
                bytes[off + 35],
            ]);
            let wal_crc32 = u32::from_le_bytes([
                bytes[off + 36],
                bytes[off + 37],
                bytes[off + 38],
                bytes[off + 39],
            ]);
            shards.push(ShardSnapshotEntry {
                shard_id,
                epoch,
                last_seq_no,
                vector_count,
                wal_file_bytes,
                wal_crc32,
            });
        }

        Ok(Self {
            node_id,
            config,
            shards,
            manifest_crc32: expected_crc,
        })
    }

    /// Persist `manifest.bin` to `path` and call `sync_all()`.
    pub fn write_to_file(&self, path: &Path) -> Result<(), VektorError> {
        let bytes = self.to_bytes();
        write_and_sync_file(path, &bytes)
    }

    /// Read and validate `manifest.bin` from `path`.
    pub fn read_from_file(path: &Path) -> Result<Self, VektorError> {
        let bytes = read_file_bytes(path)?;
        Self::from_bytes(&bytes)
    }
}

/// Structured integrity report returned by [`SnapshotManager::verify_snapshot`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotVerificationReport {
    pub verified_shards: u32,
    pub total_vectors: u64,
    pub total_bytes: u64,
    pub corrupt_shards: Vec<ShardId>,
}

/// Coordinator for creating, verifying, and restoring point-in-time node/cluster snapshots.
pub struct SnapshotManager;

impl SnapshotManager {
    /// Capture a consistent point-in-time snapshot across all shards in `node` into `snapshot_dir`.
    ///
    /// Acquires read locks across all shards in deterministic `shard_id` order (`0..num_shards`)
    /// so no concurrent mutation can interleave across shards while WAL files and metadata are
    /// snapshotted.
    pub async fn create_snapshot(
        node: &NodeService,
        snapshot_dir: &Path,
    ) -> Result<ClusterSnapshotManifest, VektorError> {
        std::fs::create_dir_all(snapshot_dir).map_err(|e| VektorError::WalIo(e.to_string()))?;

        let mut shard_ids: Vec<ShardId> = node.shards.keys().copied().collect();
        shard_ids.sort_unstable();

        // Hold read guards across all shards simultaneously in deterministic order for a
        // consistent multi-shard point-in-time cut.
        let mut guards = Vec::with_capacity(shard_ids.len());
        for &shard_id in &shard_ids {
            let lock = node
                .shards
                .get(&shard_id)
                .ok_or(VektorError::InvalidConfig("missing shard in node service"))?;
            guards.push(lock.read().await);
        }

        let mut entries = Vec::with_capacity(guards.len());
        for engine in &guards {
            let wal_bytes = read_file_bytes(engine.wal.path())?;
            let wal_crc32 = compute_crc32(&wal_bytes);
            let wal_file_bytes = wal_bytes.len() as u64;

            let snap_wal_path = snapshot_dir.join(shard_wal_filename(engine.shard_id));
            write_and_sync_file(&snap_wal_path, &wal_bytes)?;

            entries.push(ShardSnapshotEntry {
                shard_id: engine.shard_id,
                epoch: engine.epoch,
                last_seq_no: engine.hnsw.last_seq_no(),
                vector_count: engine.hnsw.len() as u64,
                wal_file_bytes,
                wal_crc32,
            });
        }

        let mut manifest = ClusterSnapshotManifest {
            node_id: node.node_id,
            config: node.config.clone(),
            shards: entries,
            manifest_crc32: 0,
        };
        let raw_bytes = manifest.to_bytes();
        let crc_off = raw_bytes.len() - MANIFEST_CRC_LEN;
        manifest.manifest_crc32 = u32::from_le_bytes([
            raw_bytes[crc_off],
            raw_bytes[crc_off + 1],
            raw_bytes[crc_off + 2],
            raw_bytes[crc_off + 3],
        ]);

        let manifest_path = snapshot_dir.join(MANIFEST_FILENAME);
        write_and_sync_file(&manifest_path, &raw_bytes)?;

        Ok(manifest)
    }

    /// Audit a snapshot directory and return a [`SnapshotVerificationReport`] recording any
    /// shards whose WAL file length or CRC32 checksum mismatches the manifest.
    pub fn audit_snapshot(snapshot_dir: &Path) -> Result<SnapshotVerificationReport, VektorError> {
        let manifest_path = snapshot_dir.join(MANIFEST_FILENAME);
        let manifest = ClusterSnapshotManifest::read_from_file(&manifest_path)?;

        let mut verified_shards = 0_u32;
        let mut total_vectors = 0_u64;
        let mut total_bytes = 0_u64;
        let mut corrupt_shards = Vec::new();

        for entry in &manifest.shards {
            let wal_path = snapshot_dir.join(shard_wal_filename(entry.shard_id));
            let Ok(wal_bytes) = read_file_bytes(&wal_path) else {
                corrupt_shards.push(entry.shard_id);
                continue;
            };
            let actual_len = wal_bytes.len() as u64;
            let actual_crc = compute_crc32(&wal_bytes);
            if actual_len != entry.wal_file_bytes || actual_crc != entry.wal_crc32 {
                corrupt_shards.push(entry.shard_id);
            } else {
                verified_shards = verified_shards.saturating_add(1);
                total_vectors = total_vectors.saturating_add(entry.vector_count);
                total_bytes = total_bytes.saturating_add(actual_len);
            }
        }

        Ok(SnapshotVerificationReport {
            verified_shards,
            total_vectors,
            total_bytes,
            corrupt_shards,
        })
    }

    /// Validate `manifest.bin` magic/CRC32 and every shard's `.wal` length and IEEE CRC32 checksum.
    ///
    /// Rejects any tampered or bit-rotted manifest or shard `.wal` file with
    /// [`VektorError::WalCorruption`].
    pub fn verify_snapshot(snapshot_dir: &Path) -> Result<SnapshotVerificationReport, VektorError> {
        let report = Self::audit_snapshot(snapshot_dir)?;
        if let Some(&first_corrupt_shard) = report.corrupt_shards.first() {
            return Err(VektorError::WalCorruption {
                offset: u64::from(first_corrupt_shard),
            });
        }
        Ok(report)
    }

    /// Restore all shard WALs from `snapshot_dir` into `target_node_dir` and deterministically
    /// rebuild the `NodeService` / `ShardEngine` HNSW indexes, verifying that every shard's
    /// restored `last_seq_no` and live vector count match the snapshot manifest.
    pub fn restore_node_from_snapshot(
        snapshot_dir: &Path,
        target_node_dir: &Path,
        config: ClusterConfig,
        params: HnswParams,
    ) -> Result<NodeService, VektorError> {
        Self::verify_snapshot(snapshot_dir)?;
        let manifest_path = snapshot_dir.join(MANIFEST_FILENAME);
        let manifest = ClusterSnapshotManifest::read_from_file(&manifest_path)?;

        if config.num_shards != manifest.config.num_shards
            || config.dimensions != manifest.config.dimensions
            || config.metric != manifest.config.metric
        {
            return Err(VektorError::InvalidConfig(
                "restore ClusterConfig does not match snapshot manifest config",
            ));
        }

        std::fs::create_dir_all(target_node_dir).map_err(|e| VektorError::WalIo(e.to_string()))?;

        for entry in &manifest.shards {
            let src_wal = snapshot_dir.join(shard_wal_filename(entry.shard_id));
            let dst_wal = target_node_dir.join(shard_wal_filename(entry.shard_id));
            let wal_bytes = read_file_bytes(&src_wal)?;
            write_and_sync_file(&dst_wal, &wal_bytes)?;
        }

        let service = NodeService::open(manifest.node_id, target_node_dir, config, params)?;

        for entry in &manifest.shards {
            let shard_lock = service
                .shards
                .get(&entry.shard_id)
                .ok_or(VektorError::InvalidConfig("restored node missing shard"))?;
            let mut engine = shard_lock
                .try_write()
                .map_err(|_| VektorError::InvalidConfig("restored shard lock contended"))?;
            engine.epoch = entry.epoch;

            if engine.hnsw.last_seq_no() != entry.last_seq_no {
                return Err(VektorError::WalCorruption {
                    offset: entry.last_seq_no,
                });
            }
            if engine.hnsw.len() as u64 != entry.vector_count {
                return Err(VektorError::WalCorruption {
                    offset: entry.vector_count,
                });
            }
        }

        Ok(service)
    }
}

/// Convenience wrapper for [`SnapshotManager::verify_snapshot`].
pub fn verify_snapshot(snapshot_dir: &Path) -> Result<SnapshotVerificationReport, VektorError> {
    SnapshotManager::verify_snapshot(snapshot_dir)
}

/// Convenience wrapper for [`SnapshotManager::restore_node_from_snapshot`].
pub fn restore_node_from_snapshot(
    snapshot_dir: &Path,
    target_node_dir: &Path,
    config: ClusterConfig,
    params: HnswParams,
) -> Result<NodeService, VektorError> {
    SnapshotManager::restore_node_from_snapshot(snapshot_dir, target_node_dir, config, params)
}

#[must_use]
pub fn shard_wal_filename(shard_id: ShardId) -> String {
    format!("shard-{shard_id:04}.wal")
}

fn compute_crc32(bytes: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

fn metric_to_u8(metric: DistanceMetric) -> u8 {
    match metric {
        DistanceMetric::Cosine => 0,
        DistanceMetric::L2 => 1,
    }
}

fn u8_to_metric(code: u8) -> Result<DistanceMetric, VektorError> {
    match code {
        0 => Ok(DistanceMetric::Cosine),
        1 => Ok(DistanceMetric::L2),
        _ => Err(VektorError::WalCorruption { offset: 18 }),
    }
}

fn read_file_bytes(path: &Path) -> Result<Vec<u8>, VektorError> {
    let mut file = File::open(path).map_err(|e| VektorError::WalIo(e.to_string()))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| VektorError::WalIo(e.to_string()))?;
    Ok(bytes)
}

fn write_and_sync_file(path: &Path, bytes: &[u8]) -> Result<(), VektorError> {
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|e| VektorError::WalIo(e.to_string()))?;
    file.write_all(bytes)
        .map_err(|e| VektorError::WalIo(e.to_string()))?;
    file.sync_all()
        .map_err(|e| VektorError::WalIo(e.to_string()))?;
    Ok(())
}
