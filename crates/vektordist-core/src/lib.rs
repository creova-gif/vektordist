#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod arena;
pub mod exact;
pub mod hnsw;
pub mod hybrid;
pub mod mmap;
pub mod quant;
pub mod simd;
pub mod wal;

use thiserror::Error;

/// Unique identifier for a stored vector.
pub type VectorId = u64;

/// Logical shard identifier (`0..num_shards`).
pub type ShardId = u32;

/// Physical node identifier in the cluster.
pub type NodeId = u32;

/// Monotonic shard leadership epoch used for stale-primary fencing.
pub type Epoch = u64;

/// Monotonic gap-free per-shard WAL sequence number.
pub type SeqNo = u64;

/// Caller-assigned unique write identifier tracked by the external verification ledger.
pub type WriteId = u64;

/// Supported vector distance/similarity metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Cosine similarity over L2-normalized vectors (higher is closer).
    Cosine,
    /// Squared Euclidean distance (lower is closer).
    L2,
}

/// Authoritative cluster configuration distributed by the coordinator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterConfig {
    pub num_shards: u32,
    pub replication_factor: u8,
    pub write_quorum: u8,
    pub dimensions: usize,
    pub metric: DistanceMetric,
}

impl ClusterConfig {
    /// Validate invariants required for M1 quorum durability.
    pub fn validate(&self) -> Result<(), VektorError> {
        if self.num_shards == 0 {
            return Err(VektorError::InvalidConfig("num_shards must be > 0"));
        }
        if self.dimensions == 0 {
            return Err(VektorError::InvalidConfig("dimensions must be > 0"));
        }
        if self.replication_factor == 0 {
            return Err(VektorError::InvalidConfig("replication_factor must be > 0"));
        }
        if self.write_quorum == 0 || self.write_quorum > self.replication_factor {
            return Err(VektorError::InvalidConfig(
                "write_quorum must be in 1..=replication_factor",
            ));
        }
        Ok(())
    }

    /// Deterministically map a `VectorId` to a logical `ShardId`.
    #[must_use]
    pub fn shard_for(&self, id: VectorId) -> ShardId {
        // SplitMix64 finalizer for uniform deterministic shard distribution.
        let mut z = id.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z % u64::from(self.num_shards)) as ShardId
    }
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            num_shards: 16,
            replication_factor: 3,
            write_quorum: 2,
            dimensions: 768,
            metric: DistanceMetric::Cosine,
        }
    }
}

/// Core error type for VektorDist indexing, WAL, and cluster boundaries.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum VektorError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),

    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("invalid vector: {0}")]
    InvalidVector(&'static str),

    #[error("stale shard epoch: expected >= {current}, got {received}")]
    WrongEpoch { current: Epoch, received: Epoch },

    #[error("write quorum unavailable for shard {shard_id}: required {required}, alive {alive}")]
    QuorumUnavailable {
        shard_id: ShardId,
        required: u8,
        alive: u8,
    },

    #[error("WAL corruption detected at byte offset {offset}")]
    WalCorruption { offset: u64 },

    #[error("WAL I/O error: {0}")]
    WalIo(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let cfg = ClusterConfig::default();
        assert_eq!(cfg.validate(), Ok(()));
        assert_eq!(cfg.num_shards, 16);
        assert_eq!(cfg.replication_factor, 3);
        assert_eq!(cfg.write_quorum, 2);
        assert_eq!(cfg.dimensions, 768);
    }

    #[test]
    fn invalid_quorum_rejected() {
        let cfg = ClusterConfig {
            write_quorum: 4,
            ..ClusterConfig::default()
        };
        assert!(matches!(cfg.validate(), Err(VektorError::InvalidConfig(_))));
    }

    #[test]
    fn shard_mapping_is_deterministic_and_in_bounds() {
        let cfg = ClusterConfig::default();
        for id in 0..10_000 {
            let s1 = cfg.shard_for(id);
            let s2 = cfg.shard_for(id);
            assert_eq!(s1, s2);
            assert!(s1 < cfg.num_shards);
        }
    }
}
