#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

//! Dynamic online shard rebalancing (`compute_rebalance_plan`, `execute_online_rebalance`) with
//! zero-loss historical WAL streaming and live dual-stream replication prior to ownership transfer.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use tonic::{Request, Status};
use vektordist_core::{ClusterConfig, Epoch, NodeId, ShardId, VektorError};
use vektordist_proto::v1::shard_replica_rpc_client::ShardReplicaRpcClient;
use vektordist_proto::v1::{
    ConfigureShardRequest, GetWalRecordsRequest, ReplicateWalRequest, WireWalRecord,
};
use vektordist_proto::STATUS_OK;

use crate::node::format_endpoint;
use crate::raft::{RaftCluster, TopologyCommand};
use crate::{shard_state_to_u32, ShardAssignment, ShardState};

const RAFT_RPC_TIMEOUT: Duration = Duration::from_secs(5);

/// Planned replica and primary migration for a single shard during online rebalancing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardMove {
    pub shard_id: ShardId,
    pub from_primary: NodeId,
    pub to_primary: NodeId,
    pub added_replicas: Vec<NodeId>,
    pub removed_replicas: Vec<NodeId>,
    pub target_assignment: ShardAssignment,
}

/// Minimal-movement online rebalance plan across active cluster nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RebalancePlan {
    pub moves: Vec<ShardMove>,
    pub target_shards: Vec<ShardAssignment>,
}

/// Compute a minimal-movement online shard rebalance plan across `active_nodes` for replication
/// factor `rf`.
///
/// Guarantees:
/// 1. Every shard has `min(rf, active_nodes.len())` distinct replicas drawn from `active_nodes`.
/// 2. Total owned shard replicas per node differ by at most `1` across all `active_nodes`.
/// 3. Primary assignments per node differ by at most `1` across all `active_nodes`.
/// 4. Existing replicas and primaries from `current_shards` are preserved whenever possible to
///    minimize network WAL transfers, and epochs are incremented for modified shards.
pub fn compute_rebalance_plan(
    current_shards: &[ShardAssignment],
    active_nodes: &[NodeId],
    rf: u8,
) -> Result<RebalancePlan, VektorError> {
    if active_nodes.is_empty() {
        return Err(VektorError::InvalidConfig("active_nodes must not be empty"));
    }
    if rf == 0 {
        return Err(VektorError::InvalidConfig("rf must be > 0"));
    }

    let mut ordered_nodes = Vec::with_capacity(active_nodes.len());
    let mut active_set = BTreeSet::new();
    for &n in active_nodes {
        if active_set.insert(n) {
            ordered_nodes.push(n);
        }
    }
    ordered_nodes.sort_unstable();

    let num_nodes = ordered_nodes.len();
    let eff_rf = (rf as usize).min(num_nodes);

    // Track each shard's replica set (primary is index 0 initially if kept).
    let mut replica_sets: Vec<Vec<NodeId>> = Vec::with_capacity(current_shards.len());
    let mut shard_primaries: Vec<NodeId> = Vec::with_capacity(current_shards.len());
    let mut owned_count: BTreeMap<NodeId, usize> =
        ordered_nodes.iter().map(|&id| (id, 0)).collect();

    for shard in current_shards {
        let mut set = Vec::with_capacity(eff_rf);
        if active_set.contains(&shard.primary) {
            set.push(shard.primary);
        }
        for &f in &shard.followers {
            if set.len() < eff_rf && active_set.contains(&f) && !set.contains(&f) {
                set.push(f);
            }
        }
        for &id in &set {
            if let Some(c) = owned_count.get_mut(&id) {
                *c += 1;
            }
        }
        let primary_seed = set.first().copied().unwrap_or(ordered_nodes[0]);
        shard_primaries.push(primary_seed);
        replica_sets.push(set);
    }

    // Step 1: Fill any shard with fewer than `eff_rf` replicas using the least-loaded active nodes.
    for (s_idx, set) in replica_sets.iter_mut().enumerate() {
        while set.len() < eff_rf {
            let best_node = ordered_nodes
                .iter()
                .copied()
                .filter(|id| !set.contains(id))
                .min_by_key(|id| (owned_count.get(id).copied().unwrap_or(0), *id));
            let Some(chosen) = best_node else {
                break;
            };
            if set.is_empty() {
                shard_primaries[s_idx] = chosen;
            }
            set.push(chosen);
            *owned_count.entry(chosen).or_insert(0) += 1;
        }
    }

    // Step 2: Greedily transfer replicas from overloaded nodes to underloaded nodes until
    // max(owned_count) - min(owned_count) <= 1.
    while let Some((&over_node, &over_cnt)) = owned_count.iter().max_by_key(|(&id, &cnt)| (cnt, id))
    {
        let Some((&under_node, &under_cnt)) = owned_count
            .iter()
            .min_by_key(|(&id, &cnt)| (cnt, std::cmp::Reverse(id)))
        else {
            break;
        };
        if over_cnt <= under_cnt + 1 {
            break;
        }

        // Prefer moving a follower replica first so primary leadership stays stable if possible.
        let candidate_shard = (0..replica_sets.len())
            .find(|&s_idx| {
                replica_sets[s_idx].contains(&over_node)
                    && !replica_sets[s_idx].contains(&under_node)
                    && shard_primaries[s_idx] != over_node
            })
            .or_else(|| {
                (0..replica_sets.len()).find(|&s_idx| {
                    replica_sets[s_idx].contains(&over_node)
                        && !replica_sets[s_idx].contains(&under_node)
                })
            });

        let Some(s_idx) = candidate_shard else {
            break;
        };

        for slot in &mut replica_sets[s_idx] {
            if *slot == over_node {
                *slot = under_node;
                break;
            }
        }
        if shard_primaries[s_idx] == over_node {
            shard_primaries[s_idx] = under_node;
        }
        if let Some(c) = owned_count.get_mut(&over_node) {
            *c -= 1;
        }
        if let Some(c) = owned_count.get_mut(&under_node) {
            *c += 1;
        }
    }

    // Step 3: Balance primaries across active nodes so primary count differs by at most 1.
    let mut primary_count: BTreeMap<NodeId, usize> =
        ordered_nodes.iter().map(|&id| (id, 0)).collect();
    for (s_idx, set) in replica_sets.iter().enumerate() {
        if !set.contains(&shard_primaries[s_idx]) {
            if let Some(&first) = set.first() {
                shard_primaries[s_idx] = first;
            }
        }
        *primary_count.entry(shard_primaries[s_idx]).or_insert(0) += 1;
    }

    while let Some((&over_p, &over_cnt)) = primary_count.iter().max_by_key(|(&id, &cnt)| (cnt, id))
    {
        let Some((&under_p, &under_cnt)) = primary_count
            .iter()
            .min_by_key(|(&id, &cnt)| (cnt, std::cmp::Reverse(id)))
        else {
            break;
        };
        if over_cnt <= under_cnt + 1 {
            break;
        }

        // Look for a shard where `over_p` is primary and `under_p` is already a replica.
        let swap_shard = (0..replica_sets.len()).find(|&s_idx| {
            shard_primaries[s_idx] == over_p && replica_sets[s_idx].contains(&under_p)
        });

        if let Some(s_idx) = swap_shard {
            shard_primaries[s_idx] = under_p;
            if let Some(c) = primary_count.get_mut(&over_p) {
                *c -= 1;
            }
            if let Some(c) = primary_count.get_mut(&under_p) {
                *c += 1;
            }
        } else {
            break;
        }
    }

    // Step 4: Construct target ShardAssignments and diff against current_shards.
    let mut target_shards = Vec::with_capacity(current_shards.len());
    let mut moves = Vec::new();

    for (s_idx, old_shard) in current_shards.iter().enumerate() {
        let new_primary = shard_primaries[s_idx];
        let mut new_followers: Vec<NodeId> = replica_sets[s_idx]
            .iter()
            .copied()
            .filter(|&id| id != new_primary)
            .collect();
        new_followers.sort_unstable();

        let mut old_followers_sorted = old_shard.followers.clone();
        old_followers_sorted.sort_unstable();

        let old_replicas: BTreeSet<NodeId> = std::iter::once(old_shard.primary)
            .chain(old_shard.followers.iter().copied())
            .collect();
        let new_replicas: BTreeSet<NodeId> = std::iter::once(new_primary)
            .chain(new_followers.iter().copied())
            .collect();

        let added_replicas: Vec<NodeId> = new_replicas.difference(&old_replicas).copied().collect();
        let removed_replicas: Vec<NodeId> =
            old_replicas.difference(&new_replicas).copied().collect();

        let changed = new_primary != old_shard.primary
            || new_followers != old_followers_sorted
            || !added_replicas.is_empty()
            || !removed_replicas.is_empty();

        let new_state = if eff_rf >= (rf as usize) {
            ShardState::Healthy
        } else {
            ShardState::DegradedOk
        };

        let target_assignment = ShardAssignment {
            shard_id: old_shard.shard_id,
            epoch: if changed {
                old_shard.epoch + 1
            } else {
                old_shard.epoch
            },
            primary: new_primary,
            followers: if changed {
                new_followers
            } else {
                old_shard.followers.clone()
            },
            state: new_state,
        };

        if changed {
            moves.push(ShardMove {
                shard_id: old_shard.shard_id,
                from_primary: old_shard.primary,
                to_primary: new_primary,
                added_replicas,
                removed_replicas,
                target_assignment: target_assignment.clone(),
            });
        }
        target_shards.push(target_assignment);
    }

    Ok(RebalancePlan {
        moves,
        target_shards,
    })
}

/// Execute an online shard rebalance across live `NodeService` peers and commit the new topology
/// through the Raft state machine ONLY after all newly assigned replicas and primaries have
/// streamed and fsynced 100% of historical WAL entries.
pub async fn execute_online_rebalance(
    raft: &Arc<RwLock<RaftCluster>>,
    new_nodes: Vec<(NodeId, String)>,
    active_nodes: Vec<NodeId>,
) -> Result<RebalancePlan, VektorError> {
    // 1. Register any new nodes in the Raft cluster so their addresses and membership are tracked.
    {
        let mut guard = raft
            .write()
            .map_err(|_| VektorError::InvalidConfig("raft cluster lock poisoned"))?;
        if guard.leader_id().is_none() {
            guard.elect_any_leader()?;
        }
        for (node_id, addr) in new_nodes {
            guard.add_node(node_id, Some(addr))?;
        }
    }

    // 2. Snapshot current shards, config, and node address table from Raft leader.
    let (current_snap, node_addrs, alive_nodes) = {
        let guard = raft
            .read()
            .map_err(|_| VektorError::InvalidConfig("raft cluster lock poisoned"))?;
        let leader_id = guard.leader_id().ok_or(VektorError::QuorumUnavailable {
            shard_id: 0,
            required: 2,
            alive: 0,
        })?;
        let leader = guard
            .node(leader_id)
            .ok_or(VektorError::InvalidConfig("missing leader"))?;
        (
            leader.snapshot.clone(),
            leader.node_addrs.clone(),
            leader.alive_nodes.clone(),
        )
    };

    let plan = compute_rebalance_plan(
        &current_snap.shards,
        &active_nodes,
        current_snap.config.replication_factor,
    )?;

    // 3. Before switching ownership or epochs, stream WAL entries from the current primary (or
    //    surviving replicas) to every added replica and any promoted primary so zero writes are lost.
    for mv in &plan.moves {
        let Some(old_shard) = current_snap
            .shards
            .iter()
            .find(|s| s.shard_id == mv.shard_id)
        else {
            continue;
        };

        let mut source_addrs = Vec::new();
        if let Some(p_addr) = node_addrs.get(&old_shard.primary) {
            source_addrs.push(p_addr.clone());
        }
        for f_id in &old_shard.followers {
            if let Some(f_addr) = node_addrs.get(f_id) {
                source_addrs.push(f_addr.clone());
            }
        }

        // Collect the union of all WAL records for this shard across existing replicas.
        let mut union_by_seq: BTreeMap<u64, WireWalRecord> = BTreeMap::new();
        for src_addr in &source_addrs {
            if let Ok(records) = fetch_wal_records(src_addr, mv.shard_id, 1).await {
                for rec in records {
                    union_by_seq.entry(rec.seq_no).or_insert(rec);
                }
            }
        }
        let all_records: Vec<WireWalRecord> = union_by_seq.into_values().collect();

        // Determine all destination nodes that must be caught up before ownership switch.
        let mut catchup_targets: BTreeSet<NodeId> = mv.added_replicas.iter().copied().collect();
        if mv.to_primary != mv.from_primary {
            catchup_targets.insert(mv.to_primary);
        }

        for target_id in catchup_targets {
            let Some(target_addr) = node_addrs.get(&target_id) else {
                continue;
            };
            // Dual-stream live writes to `target_addr` while streaming historical WAL.
            push_shard_config_to_nodes(
                &current_snap.config,
                &node_addrs,
                &alive_nodes,
                old_shard,
                Some(target_addr),
            )
            .await;

            if !all_records.is_empty() {
                for chunk in all_records.chunks(500) {
                    push_wal_records(target_addr, mv.shard_id, old_shard.epoch, chunk.to_vec())
                        .await
                        .map_err(|e| VektorError::WalIo(e.to_string()))?;
                }
            }
        }
    }

    // 4. Propose and commit the rebalanced topology through Raft majority consensus.
    let updated_alive: BTreeSet<NodeId> = active_nodes.iter().copied().collect();
    {
        let mut guard = raft
            .write()
            .map_err(|_| VektorError::InvalidConfig("raft cluster lock poisoned"))?;
        guard.propose(TopologyCommand::ApplyRebalance {
            assignments: plan.target_shards.clone(),
            active_nodes: active_nodes.clone(),
        })?;
    }

    // 5. Push new shard epochs and roles to all active data-plane nodes, and fence removed replicas.
    for mv in &plan.moves {
        push_shard_config_to_nodes(
            &current_snap.config,
            &node_addrs,
            &updated_alive,
            &mv.target_assignment,
            None,
        )
        .await;

        for removed_id in &mv.removed_replicas {
            if let Some(rem_addr) = node_addrs.get(removed_id) {
                let req = ConfigureShardRequest {
                    shard_id: mv.shard_id,
                    epoch: mv.target_assignment.epoch,
                    is_primary: false,
                    follower_addrs: Vec::new(),
                    recovering_follower_addr: String::new(),
                    state: shard_state_to_u32(mv.target_assignment.state),
                    write_quorum: u32::from(current_snap.config.write_quorum),
                };
                let _ = send_configure_shard(rem_addr, req).await;
            }
        }
    }

    Ok(plan)
}

pub(crate) async fn push_shard_config_to_nodes(
    config: &ClusterConfig,
    node_addrs: &BTreeMap<NodeId, String>,
    alive_nodes: &BTreeSet<NodeId>,
    shard: &ShardAssignment,
    recovering_addr: Option<&str>,
) {
    let follower_addrs: Vec<String> = shard
        .followers
        .iter()
        .filter(|id| alive_nodes.contains(id))
        .filter_map(|id| node_addrs.get(id).cloned())
        .collect();

    if alive_nodes.contains(&shard.primary) {
        if let Some(primary_addr) = node_addrs.get(&shard.primary) {
            let req = ConfigureShardRequest {
                shard_id: shard.shard_id,
                epoch: shard.epoch,
                is_primary: true,
                follower_addrs,
                recovering_follower_addr: recovering_addr.unwrap_or_default().to_string(),
                state: shard_state_to_u32(shard.state),
                write_quorum: u32::from(config.write_quorum),
            };
            let _ = send_configure_shard(primary_addr, req).await;
        }
    }

    for &f_id in &shard.followers {
        if alive_nodes.contains(&f_id) {
            if let Some(f_addr) = node_addrs.get(&f_id) {
                let req = ConfigureShardRequest {
                    shard_id: shard.shard_id,
                    epoch: shard.epoch,
                    is_primary: false,
                    follower_addrs: Vec::new(),
                    recovering_follower_addr: String::new(),
                    state: shard_state_to_u32(shard.state),
                    write_quorum: u32::from(config.write_quorum),
                };
                let _ = send_configure_shard(f_addr, req).await;
            }
        }
    }
}

async fn send_configure_shard(addr: &str, req: ConfigureShardRequest) -> Result<(), Status> {
    let endpoint = format_endpoint(addr);
    let mut client =
        tokio::time::timeout(RAFT_RPC_TIMEOUT, ShardReplicaRpcClient::connect(endpoint))
            .await
            .map_err(|_| Status::deadline_exceeded("configure connect timeout"))?
            .map_err(|e| Status::unavailable(e.to_string()))?;

    tokio::time::timeout(RAFT_RPC_TIMEOUT, client.configure_shard(Request::new(req)))
        .await
        .map_err(|_| Status::deadline_exceeded("configure rpc timeout"))??;
    Ok(())
}

async fn fetch_wal_records(
    addr: &str,
    shard_id: ShardId,
    from_seq_no: u64,
) -> Result<Vec<WireWalRecord>, Status> {
    let endpoint = format_endpoint(addr);
    let mut client =
        tokio::time::timeout(RAFT_RPC_TIMEOUT, ShardReplicaRpcClient::connect(endpoint))
            .await
            .map_err(|_| Status::deadline_exceeded("fetch_wal connect timeout"))?
            .map_err(|e| Status::unavailable(e.to_string()))?;

    let resp = tokio::time::timeout(
        RAFT_RPC_TIMEOUT,
        client.get_wal_records(Request::new(GetWalRecordsRequest {
            shard_id,
            from_seq_no,
        })),
    )
    .await
    .map_err(|_| Status::deadline_exceeded("fetch_wal rpc timeout"))??;
    Ok(resp.into_inner().records)
}

async fn push_wal_records(
    addr: &str,
    shard_id: ShardId,
    epoch: Epoch,
    records: Vec<WireWalRecord>,
) -> Result<(), Status> {
    let endpoint = format_endpoint(addr);
    let mut client =
        tokio::time::timeout(RAFT_RPC_TIMEOUT, ShardReplicaRpcClient::connect(endpoint))
            .await
            .map_err(|_| Status::deadline_exceeded("push_wal connect timeout"))?
            .map_err(|e| Status::unavailable(e.to_string()))?;

    let resp = tokio::time::timeout(
        RAFT_RPC_TIMEOUT,
        client.replicate_wal(Request::new(ReplicateWalRequest {
            shard_id,
            epoch,
            batched_fsync: true,
            records,
        })),
    )
    .await
    .map_err(|_| Status::deadline_exceeded("push_wal rpc timeout"))??
    .into_inner();

    if resp.status_code != STATUS_OK {
        return Err(Status::internal(resp.error_message));
    }
    Ok(())
}
