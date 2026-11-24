use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use tonic::Request;
use vektordist_core::exact::ScoredId;
use vektordist_core::wal::{WalOp, WalRecord};
use vektordist_core::{
    ClusterConfig, DistanceMetric, SeqNo, ShardId, VectorId, VektorError, WriteId,
};
use vektordist_proto::v1::shard_replica_rpc_client::ShardReplicaRpcClient;
use vektordist_proto::v1::topology_rpc_client::TopologyRpcClient;
use vektordist_proto::v1::{
    AppendQuorumRequest, GetTopologyRequest, GetTopologyResponse, GetWalRecordsRequest,
    SearchShardRequest, WireShardAssignment, WireWalRecord,
};
use vektordist_proto::{
    encode_f32_le, from_wire_scored_id, from_wire_wal_record, to_wire_wal_record, STATUS_OK,
    STATUS_QUORUM_UNAVAILABLE, STATUS_WRONG_EPOCH, WIRE_OP_UPSERT,
};

use crate::node::format_endpoint;
use crate::{u32_to_shard_state, ShardState};

const SEARCH_SHARD_TIMEOUT: Duration = Duration::from_millis(200);
const WRITE_QUORUM_TIMEOUT: Duration = Duration::from_millis(500);
const SEED_BATCH_TIMEOUT: Duration = Duration::from_secs(10);

/// Smart direct-to-primary cluster client (Shape 2) with cached `TopologySnapshot`
/// and automatic retry-on-`WRONG_EPOCH` / dead-replica fallback.
#[derive(Clone)]
pub struct ClusterClient {
    coord_addr: String,
    topology: Arc<RwLock<GetTopologyResponse>>,
}

impl ClusterClient {
    /// Connect to the coordinator at `coord_addr` and fetch the initial `GetTopologyResponse`.
    pub async fn connect(coord_addr: impl Into<String>) -> Result<Self, VektorError> {
        let addr = coord_addr.into();
        let topo = fetch_topology(&addr).await?;
        Ok(Self {
            coord_addr: addr,
            topology: Arc::new(RwLock::new(topo)),
        })
    }

    /// Refresh the cached topology snapshot from the coordinator.
    pub async fn refresh_topology(&self) -> Result<GetTopologyResponse, VektorError> {
        let topo = fetch_topology(&self.coord_addr).await?;
        if let Ok(mut guard) = self.topology.write() {
            *guard = topo.clone();
        }
        Ok(topo)
    }

    #[must_use]
    pub fn cached_topology(&self) -> GetTopologyResponse {
        self.topology.read().map(|g| g.clone()).unwrap_or_default()
    }

    #[must_use]
    pub fn cluster_config(&self) -> ClusterConfig {
        let topo = self.cached_topology();
        ClusterConfig {
            num_shards: topo.num_shards.max(1),
            replication_factor: topo.replication_factor.max(1) as u8,
            write_quorum: topo.write_quorum.max(1) as u8,
            dimensions: topo.dimensions.max(1) as usize,
            metric: if topo.metric == 1 {
                DistanceMetric::L2
            } else {
                DistanceMetric::Cosine
            },
        }
    }

    /// Perform a single durable `W=2` quorum upsert routed directly to the shard primary.
    /// Automatically refreshes topology and retries once on `WRONG_EPOCH` or primary failure.
    pub async fn upsert(
        &self,
        id: VectorId,
        write_id: WriteId,
        vector: &[f32],
    ) -> Result<(ShardId, SeqNo), VektorError> {
        let cfg = self.cluster_config();
        let shard_id = cfg.shard_for(id);
        let wire_rec = WireWalRecord {
            seq_no: 0,
            write_id,
            op_kind: WIRE_OP_UPSERT,
            vector_id: id,
            vector_f32_le: encode_f32_le(vector),
        };

        for attempt in 0..2 {
            let Some(shard) = self.get_shard_assignment(shard_id) else {
                let _ = self.refresh_topology().await;
                continue;
            };
            if u32_to_shard_state(shard.state) == ShardState::UnderQuorum {
                return Err(VektorError::QuorumUnavailable {
                    shard_id,
                    required: cfg.write_quorum,
                    alive: 1,
                });
            }

            match send_append_quorum(
                &shard.primary_addr,
                shard_id,
                shard.epoch,
                false,
                vec![wire_rec.clone()],
                WRITE_QUORUM_TIMEOUT,
            )
            .await
            {
                Ok(resp) if resp.status_code == STATUS_OK => {
                    let seq = resp.assigned_seq_nos.first().copied().unwrap_or(0);
                    return Ok((shard_id, seq));
                }
                Ok(resp) if resp.status_code == STATUS_QUORUM_UNAVAILABLE => {
                    return Err(VektorError::QuorumUnavailable {
                        shard_id,
                        required: cfg.write_quorum,
                        alive: 1,
                    });
                }
                Ok(resp) if resp.status_code == STATUS_WRONG_EPOCH && attempt == 0 => {
                    let _ = self.refresh_topology().await;
                }
                Err(_) if attempt == 0 => {
                    let _ = self.refresh_topology().await;
                }
                Ok(resp) => {
                    return Err(VektorError::WalIo(resp.error_message));
                }
                Err(e) => {
                    return Err(VektorError::WalIo(e));
                }
            }
        }

        Err(VektorError::WalIo(format!(
            "upsert failed after topology refresh on shard {shard_id}"
        )))
    }

    /// Seed a batch of vectors for `shard_id` using batched WAL `fsync` (`1 fsync` per batch).
    pub async fn seed_shard_batch(
        &self,
        shard_id: ShardId,
        batch: &[(VectorId, WriteId, Vec<f32>)],
    ) -> Result<Vec<SeqNo>, VektorError> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        let shard = self
            .get_shard_assignment(shard_id)
            .ok_or(VektorError::InvalidConfig("unknown shard_id"))?;
        let wire_records: Vec<WireWalRecord> = batch
            .iter()
            .map(|(id, write_id, vec)| {
                to_wire_wal_record(&WalRecord {
                    seq_no: 0,
                    write_id: *write_id,
                    op: WalOp::Upsert {
                        id: *id,
                        vector: vec.clone(),
                    },
                })
            })
            .collect();

        let resp = send_append_quorum(
            &shard.primary_addr,
            shard_id,
            shard.epoch,
            true,
            wire_records,
            SEED_BATCH_TIMEOUT,
        )
        .await
        .map_err(VektorError::WalIo)?;

        if resp.status_code == STATUS_OK {
            Ok(resp.assigned_seq_nos)
        } else {
            Err(VektorError::WalIo(resp.error_message))
        }
    }

    /// Distributed scatter-gather top-`k` search across all shards.
    ///
    /// - Steady state: queries `R=1` per shard (primary first, falling back to any follower on timeout/error).
    /// - Reconciling state: queries both surviving replicas (`R=2`) and merges hits by `VectorId`.
    pub async fn search(
        &self,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
    ) -> Result<Vec<ScoredId>, VektorError> {
        let topo = self.cached_topology();
        let metric = if topo.metric == 1 {
            DistanceMetric::L2
        } else {
            DistanceMetric::Cosine
        };
        let query_bytes = encode_f32_le(query);

        let mut tasks = Vec::with_capacity(topo.shards.len());
        for shard in topo.shards {
            let q_bytes = query_bytes.clone();
            let k_u32 = top_k as u32;
            let ef_u32 = ef_search as u32;
            tasks.push(tokio::spawn(async move {
                search_single_shard(shard, q_bytes, k_u32, ef_u32).await
            }));
        }

        let mut merged_by_id: HashMap<VectorId, f32> =
            HashMap::with_capacity(topo.num_shards as usize * top_k);
        let mut need_refresh = false;
        for task in tasks {
            if let Ok((hits, refresh)) = task.await {
                if refresh {
                    need_refresh = true;
                }
                for hit in hits {
                    merged_by_id.entry(hit.id).or_insert(hit.score);
                }
            }
        }
        if need_refresh {
            let _ = self.refresh_topology().await;
        }

        let mut all_hits: Vec<ScoredId> = merged_by_id
            .into_iter()
            .map(|(id, score)| ScoredId { id, score })
            .collect();
        all_hits.sort_by(|a, b| match metric {
            DistanceMetric::Cosine => b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)),
            DistanceMetric::L2 => a.score.total_cmp(&b.score).then_with(|| a.id.cmp(&b.id)),
        });
        all_hits.truncate(top_k);
        Ok(all_hits)
    }

    /// Fetch all durable WAL records across surviving replicas of `shard_id` for external write-ID verification.
    pub async fn verify_shard_wal_write_ids(
        &self,
        shard_id: ShardId,
    ) -> Result<Vec<WalRecord>, VektorError> {
        let cfg = self.cluster_config();
        let shard = self
            .get_shard_assignment(shard_id)
            .ok_or(VektorError::InvalidConfig("unknown shard_id"))?;
        let mut candidate_addrs = vec![shard.primary_addr];
        candidate_addrs.extend(shard.follower_addrs);

        let mut by_seq: HashMap<u64, WalRecord> = HashMap::new();
        for addr in candidate_addrs {
            if addr.is_empty() {
                continue;
            }
            if let Ok(wire_recs) = query_shard_wal(&addr, shard_id).await {
                for w in &wire_recs {
                    if let Ok(rec) = from_wire_wal_record(w, cfg.dimensions) {
                        by_seq.entry(rec.seq_no).or_insert(rec);
                    }
                }
            }
        }
        let mut out: Vec<WalRecord> = by_seq.into_values().collect();
        out.sort_by_key(|r| r.seq_no);
        Ok(out)
    }

    fn get_shard_assignment(&self, shard_id: ShardId) -> Option<WireShardAssignment> {
        let guard = self.topology.read().ok()?;
        guard
            .shards
            .iter()
            .find(|s| s.shard_id == shard_id)
            .cloned()
    }
}

async fn fetch_topology(coord_addr: &str) -> Result<GetTopologyResponse, VektorError> {
    let endpoint = format_endpoint(coord_addr);
    let mut client =
        tokio::time::timeout(WRITE_QUORUM_TIMEOUT, TopologyRpcClient::connect(endpoint))
            .await
            .map_err(|_| VektorError::WalIo("coordinator connect timeout".to_string()))?
            .map_err(|e| VektorError::WalIo(e.to_string()))?;

    let resp = tokio::time::timeout(
        WRITE_QUORUM_TIMEOUT,
        client.get_topology(Request::new(GetTopologyRequest {})),
    )
    .await
    .map_err(|_| VektorError::WalIo("get_topology timeout".to_string()))?
    .map_err(|e| VektorError::WalIo(e.to_string()))?;
    Ok(resp.into_inner())
}

async fn send_append_quorum(
    addr: &str,
    shard_id: ShardId,
    expected_epoch: u64,
    batched_fsync: bool,
    records: Vec<WireWalRecord>,
    timeout: Duration,
) -> Result<vektordist_proto::v1::AppendQuorumResponse, String> {
    let endpoint = format_endpoint(addr);
    let mut client = tokio::time::timeout(timeout, ShardReplicaRpcClient::connect(endpoint))
        .await
        .map_err(|_| "connect timeout".to_string())?
        .map_err(|e| e.to_string())?;

    let resp = tokio::time::timeout(
        timeout,
        client.append_quorum(Request::new(AppendQuorumRequest {
            shard_id,
            expected_epoch,
            batched_fsync,
            records,
        })),
    )
    .await
    .map_err(|_| "append_quorum timeout".to_string())?
    .map_err(|e| e.to_string())?;
    Ok(resp.into_inner())
}

async fn search_single_shard(
    shard: WireShardAssignment,
    query_f32_le: Vec<u8>,
    top_k: u32,
    ef_search: u32,
) -> (Vec<ScoredId>, bool) {
    let state = u32_to_shard_state(shard.state);
    let mut candidates = Vec::with_capacity(1 + shard.follower_addrs.len());
    if !shard.primary_addr.is_empty() {
        candidates.push(shard.primary_addr);
    }
    for f in shard.follower_addrs {
        if !f.is_empty() {
            candidates.push(f);
        }
    }

    // In Reconciling state, query up to 2 surviving replicas (R=2 union merge); otherwise R=1 with fallback.
    let target_reads = if state == ShardState::Reconciling {
        2
    } else {
        1
    };
    let mut collected = Vec::new();
    let mut saw_failure = false;

    for addr in candidates {
        match rpc_search_shard(
            &addr,
            shard.shard_id,
            shard.epoch,
            top_k,
            ef_search,
            query_f32_le.clone(),
        )
        .await
        {
            Ok(hits) => {
                collected.extend(hits);
                if collected.len() >= (top_k as usize) || target_reads == 1 {
                    break;
                }
            }
            Err(_) => {
                saw_failure = true;
            }
        }
    }
    (collected, saw_failure)
}

async fn rpc_search_shard(
    addr: &str,
    shard_id: ShardId,
    expected_epoch: u64,
    top_k: u32,
    ef_search: u32,
    query_f32_le: Vec<u8>,
) -> Result<Vec<ScoredId>, String> {
    let endpoint = format_endpoint(addr);
    let mut client = tokio::time::timeout(
        SEARCH_SHARD_TIMEOUT,
        ShardReplicaRpcClient::connect(endpoint),
    )
    .await
    .map_err(|_| "search connect timeout".to_string())?
    .map_err(|e| e.to_string())?;

    let resp = tokio::time::timeout(
        SEARCH_SHARD_TIMEOUT,
        client.search_shard(Request::new(SearchShardRequest {
            shard_id,
            expected_epoch,
            top_k,
            ef_search,
            query_f32_le,
        })),
    )
    .await
    .map_err(|_| "search_shard timeout".to_string())?
    .map_err(|e| e.to_string())?
    .into_inner();

    if resp.status_code == STATUS_OK {
        Ok(resp.hits.iter().map(from_wire_scored_id).collect())
    } else {
        Err(resp.error_message)
    }
}

async fn query_shard_wal(addr: &str, shard_id: ShardId) -> Result<Vec<WireWalRecord>, String> {
    let endpoint = format_endpoint(addr);
    let mut client = tokio::time::timeout(
        WRITE_QUORUM_TIMEOUT,
        ShardReplicaRpcClient::connect(endpoint),
    )
    .await
    .map_err(|_| "wal query connect timeout".to_string())?
    .map_err(|e| e.to_string())?;

    let resp = tokio::time::timeout(
        WRITE_QUORUM_TIMEOUT,
        client.get_wal_records(Request::new(GetWalRecordsRequest {
            shard_id,
            from_seq_no: 1,
        })),
    )
    .await
    .map_err(|_| "get_wal_records timeout".to_string())?
    .map_err(|e| e.to_string())?
    .into_inner();
    Ok(resp.records)
}
