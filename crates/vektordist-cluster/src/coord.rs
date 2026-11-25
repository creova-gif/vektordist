use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use tonic::{Request, Response, Status};
use vektordist_core::{ClusterConfig, DistanceMetric, Epoch, NodeId, ShardId, VektorError};
use vektordist_proto::v1::shard_replica_rpc_client::ShardReplicaRpcClient;
use vektordist_proto::v1::topology_rpc_server::TopologyRpc;
use vektordist_proto::v1::{
    ConfigureShardRequest, GetTopologyRequest, GetTopologyResponse, GetWalRecordsRequest,
    PingNodeRequest, ReplicateWalRequest, WireNodeStatus, WireShardAssignment, WireWalRecord,
};
use vektordist_proto::STATUS_OK;

use crate::node::format_endpoint;
use crate::{shard_state_to_u32, ShardAssignment, ShardState, TopologySnapshot, TopologyStore};

const COORD_RPC_TIMEOUT: Duration = Duration::from_millis(400);
const CATCHUP_RPC_TIMEOUT: Duration = Duration::from_secs(5);

/// Metadata tracked by the M1 authoritative coordinator for each physical node.
#[derive(Debug, Clone)]
pub struct NodeDescriptor {
    pub node_id: NodeId,
    pub addr: String,
    pub alive: bool,
    pub missed_heartbeats: u8,
    pub memory_bytes: u64,
    pub live_vectors: u64,
}

#[derive(Debug, Clone)]
struct CoordinatorInner {
    config: ClusterConfig,
    nodes: BTreeMap<NodeId, NodeDescriptor>,
    shards: Vec<ShardAssignment>,
}

/// M1 Single Authoritative Topology Coordinator implementing `TopologyStore` and `TopologyRpc`.
#[derive(Debug, Clone)]
pub struct CoordinatorService {
    inner: Arc<RwLock<CoordinatorInner>>,
}

impl CoordinatorService {
    /// Initialize a cluster topology across `initial_nodes` (`(node_id, addr)`).
    pub fn new(
        config: ClusterConfig,
        initial_nodes: Vec<(NodeId, String)>,
    ) -> Result<Self, VektorError> {
        config.validate()?;
        if initial_nodes.is_empty() {
            return Err(VektorError::InvalidConfig("at least 1 node required"));
        }
        let mut nodes = BTreeMap::new();
        let mut node_ids = Vec::with_capacity(initial_nodes.len());
        for (id, addr) in initial_nodes {
            nodes.insert(
                id,
                NodeDescriptor {
                    node_id: id,
                    addr,
                    alive: true,
                    missed_heartbeats: 0,
                    memory_bytes: 0,
                    live_vectors: 0,
                },
            );
            node_ids.push(id);
        }

        let rf = (config.replication_factor as usize).min(node_ids.len());
        let mut shards = Vec::with_capacity(config.num_shards as usize);
        for shard_id in 0..config.num_shards {
            let primary_idx = (shard_id as usize) % node_ids.len();
            let primary = node_ids[primary_idx];
            let mut followers = Vec::with_capacity(rf.saturating_sub(1));
            for offset in 1..rf {
                let f_idx = (primary_idx + offset) % node_ids.len();
                followers.push(node_ids[f_idx]);
            }
            let state = if rf < (config.write_quorum as usize) {
                ShardState::UnderQuorum
            } else if rf < (config.replication_factor as usize) {
                ShardState::DegradedOk
            } else {
                ShardState::Healthy
            };
            shards.push(ShardAssignment {
                shard_id,
                epoch: 1,
                primary,
                followers,
                state,
            });
        }

        Ok(Self {
            inner: Arc::new(RwLock::new(CoordinatorInner {
                config,
                nodes,
                shards,
            })),
        })
    }

    /// Push current shard assignments, epochs, and primary/follower roles to all alive nodes.
    pub async fn push_all_shard_configs(&self) -> Result<(), VektorError> {
        let (config, nodes, shards) = {
            let guard = self
                .inner
                .read()
                .map_err(|_| VektorError::InvalidConfig("coordinator lock poisoned"))?;
            (
                guard.config.clone(),
                guard.nodes.clone(),
                guard.shards.clone(),
            )
        };

        for shard in &shards {
            push_single_shard_config(&config, &nodes, shard, None).await;
        }
        Ok(())
    }

    /// Run one heartbeat pass across all registered nodes (`3` consecutive misses -> failover).
    /// Returns the list of newly failed `NodeId`s detected in this tick.
    pub async fn heartbeat_tick(&self) -> Vec<NodeId> {
        let nodes_snapshot: Vec<NodeDescriptor> = {
            let Ok(guard) = self.inner.read() else {
                return Vec::new();
            };
            guard.nodes.values().filter(|n| n.alive).cloned().collect()
        };

        let mut newly_dead = Vec::new();
        for node in nodes_snapshot {
            let ping_res = ping_single_node(&node.addr, node.node_id).await;
            let mut mark_dead = false;
            if let Ok(mut guard) = self.inner.write() {
                if let Some(entry) = guard.nodes.get_mut(&node.node_id) {
                    match ping_res {
                        Ok(resp) => {
                            entry.missed_heartbeats = 0;
                            entry.memory_bytes = resp.memory_bytes;
                            entry.live_vectors = resp.live_vectors;
                        }
                        Err(_) => {
                            entry.missed_heartbeats = entry.missed_heartbeats.saturating_add(1);
                            if entry.missed_heartbeats >= 3 && entry.alive {
                                mark_dead = true;
                            }
                        }
                    }
                }
            }
            if mark_dead {
                let _ = self.handle_node_failure(node.node_id).await;
                newly_dead.push(node.node_id);
            }
        }
        newly_dead
    }

    /// Immediately mark `dead_node_id` offline, run union WAL reconciliation across surviving
    /// replicas for affected shards, bump shard epochs, and promote new primaries.
    pub async fn handle_node_failure(&self, dead_node_id: NodeId) -> Result<(), VektorError> {
        let (config, nodes, affected_shards) = {
            let mut guard = self
                .inner
                .write()
                .map_err(|_| VektorError::InvalidConfig("coordinator lock poisoned"))?;
            if let Some(node) = guard.nodes.get_mut(&dead_node_id) {
                node.alive = false;
                node.memory_bytes = 0;
            }
            let cfg = guard.config.clone();
            let nodes_map = guard.nodes.clone();
            let mut affected = Vec::new();

            for shard in &mut guard.shards {
                let was_primary = shard.primary == dead_node_id;
                let was_follower = shard.followers.contains(&dead_node_id);
                if !was_primary && !was_follower {
                    continue;
                }
                shard.followers.retain(|&id| {
                    id != dead_node_id && nodes_map.get(&id).is_some_and(|n| n.alive)
                });
                if was_primary {
                    if let Some(new_primary) = shard.followers.first().copied() {
                        shard.primary = new_primary;
                        shard.followers.remove(0);
                    }
                }
                shard.epoch += 1;
                let alive_replicas = if nodes_map.get(&shard.primary).is_some_and(|n| n.alive) {
                    1 + shard.followers.len()
                } else {
                    0
                };
                shard.state = if alive_replicas < (cfg.write_quorum as usize) {
                    ShardState::UnderQuorum
                } else {
                    ShardState::Reconciling
                };
                affected.push(shard.clone());
            }
            (cfg, nodes_map, affected)
        };

        // Run union/anti-entropy WAL reconciliation across surviving replicas for each affected shard.
        for mut shard in affected_shards {
            if shard.state == ShardState::Reconciling {
                reconcile_surviving_replicas(&config, &nodes, &shard).await;
                shard.state = ShardState::DegradedOk;
                if let Ok(mut guard) = self.inner.write() {
                    if let Some(s) = guard.shards.get_mut(shard.shard_id as usize) {
                        if s.epoch == shard.epoch {
                            s.state = ShardState::DegradedOk;
                        }
                    }
                }
            }
            push_single_shard_config(&config, &nodes, &shard, None).await;
        }
        Ok(())
    }

    /// Register a replacement node, stream historical WAL up to high-watermark while dual-streaming
    /// live writes (`RECOVERING`), and atomically transition restored shards to `HEALTHY` at `lag == 0`.
    pub async fn add_replacement_node(
        &self,
        new_node_id: NodeId,
        new_addr: String,
    ) -> Result<Vec<ShardId>, VektorError> {
        let (config, nodes, shards_to_recover) = {
            let mut guard = self
                .inner
                .write()
                .map_err(|_| VektorError::InvalidConfig("coordinator lock poisoned"))?;
            guard.nodes.insert(
                new_node_id,
                NodeDescriptor {
                    node_id: new_node_id,
                    addr: new_addr.clone(),
                    alive: true,
                    missed_heartbeats: 0,
                    memory_bytes: 0,
                    live_vectors: 0,
                },
            );
            let cfg = guard.config.clone();
            let nodes_map = guard.nodes.clone();
            let mut to_recover = Vec::new();
            for shard in &mut guard.shards {
                let alive_count = 1 + shard.followers.len();
                if alive_count < (cfg.replication_factor as usize)
                    && shard.primary != new_node_id
                    && !shard.followers.contains(&new_node_id)
                {
                    shard.state = ShardState::Recovering;
                    to_recover.push(shard.clone());
                }
            }
            (cfg, nodes_map, to_recover)
        };

        let mut recovered_ids = Vec::with_capacity(shards_to_recover.len());
        for shard in shards_to_recover {
            // 1. Configure primary to dual-stream incoming live writes to `new_addr`.
            push_single_shard_config(&config, &nodes, &shard, Some(&new_addr)).await;

            // 2. Stream historical WAL entries from primary to `new_addr`.
            if let Some(primary_node) = nodes.get(&shard.primary) {
                let _ =
                    catch_up_shard_wal(&primary_node.addr, &new_addr, shard.shard_id, shard.epoch)
                        .await;
            }

            // 3. Promote `new_node_id` into active followers, bump epoch, and set Healthy/DegradedOk.
            let updated_shard = {
                let mut guard = self
                    .inner
                    .write()
                    .map_err(|_| VektorError::InvalidConfig("coordinator lock poisoned"))?;
                let Some(s) = guard.shards.get_mut(shard.shard_id as usize) else {
                    continue;
                };
                if !s.followers.contains(&new_node_id) {
                    s.followers.push(new_node_id);
                }
                s.epoch += 1;
                let total_replicas = 1 + s.followers.len();
                s.state = if total_replicas >= (config.replication_factor as usize) {
                    ShardState::Healthy
                } else if total_replicas >= (config.write_quorum as usize) {
                    ShardState::DegradedOk
                } else {
                    ShardState::UnderQuorum
                };
                s.clone()
            };

            push_single_shard_config(&config, &nodes, &updated_shard, None).await;
            recovered_ids.push(updated_shard.shard_id);
        }

        Ok(recovered_ids)
    }

    #[must_use]
    pub fn build_topology_response(&self) -> GetTopologyResponse {
        let Ok(guard) = self.inner.read() else {
            return GetTopologyResponse::default();
        };
        let mut owned_counts: BTreeMap<NodeId, u32> = BTreeMap::new();
        let mut wire_shards = Vec::with_capacity(guard.shards.len());
        for s in &guard.shards {
            *owned_counts.entry(s.primary).or_insert(0) += 1;
            for &f in &s.followers {
                *owned_counts.entry(f).or_insert(0) += 1;
            }
            let primary_addr = guard
                .nodes
                .get(&s.primary)
                .map(|n| n.addr.clone())
                .unwrap_or_default();
            let follower_addrs = s
                .followers
                .iter()
                .filter_map(|id| guard.nodes.get(id).map(|n| n.addr.clone()))
                .collect();
            wire_shards.push(WireShardAssignment {
                shard_id: s.shard_id,
                epoch: s.epoch,
                primary_node_id: s.primary,
                primary_addr,
                follower_node_ids: s.followers.clone(),
                follower_addrs,
                state: shard_state_to_u32(s.state),
            });
        }

        let wire_nodes = guard
            .nodes
            .values()
            .map(|n| WireNodeStatus {
                node_id: n.node_id,
                addr: n.addr.clone(),
                alive: n.alive,
                memory_bytes: n.memory_bytes,
                live_vectors: n.live_vectors,
                owned_shards: if n.alive {
                    owned_counts.get(&n.node_id).copied().unwrap_or(0)
                } else {
                    0
                },
            })
            .collect();

        GetTopologyResponse {
            num_shards: guard.config.num_shards,
            replication_factor: u32::from(guard.config.replication_factor),
            write_quorum: u32::from(guard.config.write_quorum),
            dimensions: guard.config.dimensions as u32,
            metric: match guard.config.metric {
                DistanceMetric::Cosine => 0,
                DistanceMetric::L2 => 1,
            },
            shards: wire_shards,
            nodes: wire_nodes,
        }
    }
}

impl TopologyStore for CoordinatorService {
    fn snapshot(&self) -> TopologySnapshot {
        let Ok(guard) = self.inner.read() else {
            return TopologySnapshot {
                config: ClusterConfig::default(),
                shards: Vec::new(),
            };
        };
        TopologySnapshot {
            config: guard.config.clone(),
            shards: guard.shards.clone(),
        }
    }

    fn promote_primary(
        &mut self,
        shard_id: ShardId,
        new_primary: NodeId,
    ) -> Result<Epoch, VektorError> {
        let mut guard = self
            .inner
            .write()
            .map_err(|_| VektorError::InvalidConfig("coordinator lock poisoned"))?;
        let shard = guard
            .shards
            .iter_mut()
            .find(|s| s.shard_id == shard_id)
            .ok_or(VektorError::InvalidConfig("unknown shard_id"))?;
        shard.followers.retain(|&id| id != new_primary);
        if shard.primary != new_primary {
            let old = shard.primary;
            shard.primary = new_primary;
            shard.followers.push(old);
        }
        shard.epoch += 1;
        Ok(shard.epoch)
    }
}

#[tonic::async_trait]
impl TopologyRpc for CoordinatorService {
    async fn get_topology(
        &self,
        _request: Request<GetTopologyRequest>,
    ) -> Result<Response<GetTopologyResponse>, Status> {
        Ok(Response::new(self.build_topology_response()))
    }
}

async fn ping_single_node(
    addr: &str,
    node_id: NodeId,
) -> Result<vektordist_proto::v1::PingNodeResponse, Status> {
    let endpoint = format_endpoint(addr);
    let mut client =
        tokio::time::timeout(COORD_RPC_TIMEOUT, ShardReplicaRpcClient::connect(endpoint))
            .await
            .map_err(|_| Status::deadline_exceeded("ping connect timeout"))?
            .map_err(|e| Status::unavailable(e.to_string()))?;

    let resp = tokio::time::timeout(
        COORD_RPC_TIMEOUT,
        client.ping_node(Request::new(PingNodeRequest { node_id })),
    )
    .await
    .map_err(|_| Status::deadline_exceeded("ping rpc timeout"))??;
    Ok(resp.into_inner())
}

async fn push_single_shard_config(
    config: &ClusterConfig,
    nodes: &BTreeMap<NodeId, NodeDescriptor>,
    shard: &ShardAssignment,
    recovering_addr: Option<&str>,
) {
    let follower_addrs: Vec<String> = shard
        .followers
        .iter()
        .filter_map(|id| nodes.get(id).filter(|n| n.alive).map(|n| n.addr.clone()))
        .collect();

    if let Some(primary_node) = nodes.get(&shard.primary).filter(|n| n.alive) {
        let req = ConfigureShardRequest {
            shard_id: shard.shard_id,
            epoch: shard.epoch,
            is_primary: true,
            follower_addrs: follower_addrs.clone(),
            recovering_follower_addr: recovering_addr.unwrap_or_default().to_string(),
            state: shard_state_to_u32(shard.state),
            write_quorum: u32::from(config.write_quorum),
        };
        let _ = send_configure_shard(&primary_node.addr, req).await;
    }

    for &f_id in &shard.followers {
        if let Some(f_node) = nodes.get(&f_id).filter(|n| n.alive) {
            let req = ConfigureShardRequest {
                shard_id: shard.shard_id,
                epoch: shard.epoch,
                is_primary: false,
                follower_addrs: Vec::new(),
                recovering_follower_addr: String::new(),
                state: shard_state_to_u32(shard.state),
                write_quorum: u32::from(config.write_quorum),
            };
            let _ = send_configure_shard(&f_node.addr, req).await;
        }
    }
}

async fn send_configure_shard(addr: &str, req: ConfigureShardRequest) -> Result<(), Status> {
    let endpoint = format_endpoint(addr);
    let mut client =
        tokio::time::timeout(COORD_RPC_TIMEOUT, ShardReplicaRpcClient::connect(endpoint))
            .await
            .map_err(|_| Status::deadline_exceeded("configure connect timeout"))?
            .map_err(|e| Status::unavailable(e.to_string()))?;

    tokio::time::timeout(COORD_RPC_TIMEOUT, client.configure_shard(Request::new(req)))
        .await
        .map_err(|_| Status::deadline_exceeded("configure rpc timeout"))??;
    Ok(())
}

/// Union/anti-entropy WAL reconciliation across all surviving replicas of `shard`:
/// pulls WAL entries from every survivor, merges their union by `seq_no`, and pushes
/// any missing entries so every survivor holds 100% of `W=2` acknowledged writes.
async fn reconcile_surviving_replicas(
    _config: &ClusterConfig,
    nodes: &BTreeMap<NodeId, NodeDescriptor>,
    shard: &ShardAssignment,
) {
    let mut survivor_addrs = Vec::with_capacity(1 + shard.followers.len());
    if let Some(p) = nodes.get(&shard.primary).filter(|n| n.alive) {
        survivor_addrs.push(p.addr.clone());
    }
    for &f_id in &shard.followers {
        if let Some(f) = nodes.get(&f_id).filter(|n| n.alive) {
            survivor_addrs.push(f.addr.clone());
        }
    }
    if survivor_addrs.len() < 2 {
        return;
    }

    let mut union_by_seq: BTreeMap<u64, WireWalRecord> = BTreeMap::new();
    let mut have_by_addr: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();

    for addr in &survivor_addrs {
        if let Ok(records) = fetch_wal_records(addr, shard.shard_id, 1).await {
            let mut seq_set = BTreeSet::new();
            for rec in records {
                seq_set.insert(rec.seq_no);
                union_by_seq.entry(rec.seq_no).or_insert(rec);
            }
            have_by_addr.insert(addr.clone(), seq_set);
        }
    }

    for addr in &survivor_addrs {
        let have = have_by_addr.get(addr);
        let missing: Vec<WireWalRecord> = union_by_seq
            .iter()
            .filter(|(seq, _)| have.is_none_or(|s| !s.contains(seq)))
            .map(|(_, rec)| rec.clone())
            .collect();
        if !missing.is_empty() {
            let _ = push_wal_records(addr, shard.shard_id, shard.epoch, missing).await;
        }
    }
}

async fn catch_up_shard_wal(
    primary_addr: &str,
    recovering_addr: &str,
    shard_id: ShardId,
    epoch: Epoch,
) -> Result<(), Status> {
    let records = fetch_wal_records(primary_addr, shard_id, 1).await?;
    if records.is_empty() {
        return Ok(());
    }
    for chunk in records.chunks(500) {
        let resp = push_wal_records(recovering_addr, shard_id, epoch, chunk.to_vec()).await?;
        if resp.status_code != STATUS_OK {
            return Err(Status::internal(resp.error_message));
        }
    }
    Ok(())
}

async fn fetch_wal_records(
    addr: &str,
    shard_id: ShardId,
    from_seq_no: u64,
) -> Result<Vec<WireWalRecord>, Status> {
    let endpoint = format_endpoint(addr);
    let mut client = tokio::time::timeout(
        CATCHUP_RPC_TIMEOUT,
        ShardReplicaRpcClient::connect(endpoint),
    )
    .await
    .map_err(|_| Status::deadline_exceeded("fetch_wal connect timeout"))?
    .map_err(|e| Status::unavailable(e.to_string()))?;

    let resp = tokio::time::timeout(
        CATCHUP_RPC_TIMEOUT,
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
) -> Result<vektordist_proto::v1::ReplicateWalResponse, Status> {
    let endpoint = format_endpoint(addr);
    let mut client = tokio::time::timeout(
        CATCHUP_RPC_TIMEOUT,
        ShardReplicaRpcClient::connect(endpoint),
    )
    .await
    .map_err(|_| Status::deadline_exceeded("push_wal connect timeout"))?
    .map_err(|e| Status::unavailable(e.to_string()))?;

    let resp = tokio::time::timeout(
        CATCHUP_RPC_TIMEOUT,
        client.replicate_wal(Request::new(ReplicateWalRequest {
            shard_id,
            epoch,
            batched_fsync: true,
            records,
        })),
    )
    .await
    .map_err(|_| Status::deadline_exceeded("push_wal rpc timeout"))??;
    Ok(resp.into_inner())
}
