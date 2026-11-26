#![deny(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::result_large_err)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

//! Distributed cluster control plane (`TopologyStore`) and data plane (`ShardEngine`, `ClusterClient`).

pub mod backup;
pub mod client;
pub mod coord;
pub mod gateway;
pub mod node;
pub mod placement;
pub mod raft;
pub mod sdk;
pub mod security;

pub use vektordist_core::{ClusterConfig, Epoch, NodeId, ShardId, VektorError};

/// Lifecycle state of a logical shard across failover and recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardState {
    Healthy,
    Reconciling,
    DegradedOk,
    Recovering,
    UnderQuorum,
}

#[must_use]
pub fn shard_state_to_u32(state: ShardState) -> u32 {
    match state {
        ShardState::Healthy => 0,
        ShardState::Reconciling => 1,
        ShardState::DegradedOk => 2,
        ShardState::Recovering => 3,
        ShardState::UnderQuorum => 4,
    }
}

#[must_use]
pub fn u32_to_shard_state(code: u32) -> ShardState {
    match code {
        0 => ShardState::Healthy,
        1 => ShardState::Reconciling,
        2 => ShardState::DegradedOk,
        3 => ShardState::Recovering,
        _ => ShardState::UnderQuorum,
    }
}

/// Authoritative replica placement and epoch for a single logical shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardAssignment {
    pub shard_id: ShardId,
    pub epoch: Epoch,
    pub primary: NodeId,
    pub followers: Vec<NodeId>,
    pub state: ShardState,
}

/// Point-in-time cluster topology cached by `ClusterClient` for direct-to-primary routing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologySnapshot {
    pub config: ClusterConfig,
    pub shards: Vec<ShardAssignment>,
}

/// Control-plane abstraction implemented by the M1 single authoritative coordinator
/// and replaceable by Raft consensus in M2 without changing node or client callers.
pub trait TopologyStore: Send + Sync {
    /// Return the current cluster topology snapshot.
    fn snapshot(&self) -> TopologySnapshot;

    /// Advance a shard's epoch and promote a new primary after union WAL reconciliation.
    fn promote_primary(
        &mut self,
        shard_id: ShardId,
        new_primary: NodeId,
    ) -> Result<Epoch, VektorError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubStore {
        snapshot: TopologySnapshot,
    }

    impl TopologyStore for StubStore {
        fn snapshot(&self) -> TopologySnapshot {
            self.snapshot.clone()
        }

        fn promote_primary(
            &mut self,
            shard_id: ShardId,
            new_primary: NodeId,
        ) -> Result<Epoch, VektorError> {
            let shard = self
                .snapshot
                .shards
                .iter_mut()
                .find(|s| s.shard_id == shard_id)
                .ok_or(VektorError::InvalidConfig("unknown shard_id"))?;
            shard.epoch += 1;
            shard.primary = new_primary;
            Ok(shard.epoch)
        }
    }

    #[test]
    fn promote_primary_increments_epoch() {
        let mut store = StubStore {
            snapshot: TopologySnapshot {
                config: ClusterConfig::default(),
                shards: vec![ShardAssignment {
                    shard_id: 0,
                    epoch: 1,
                    primary: 1,
                    followers: vec![2, 3],
                    state: ShardState::Healthy,
                }],
            },
        };
        let next_epoch = store.promote_primary(0, 2).unwrap();
        assert_eq!(next_epoch, 2);
        assert_eq!(store.snapshot().shards[0].primary, 2);
    }
}
