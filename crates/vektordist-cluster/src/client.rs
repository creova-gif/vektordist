use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use tonic::transport::{Channel, Endpoint};
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
    STATUS_QUORUM_UNAVAILABLE, STATUS_WRONG_EPOCH, WIRE_OP_DELETE, WIRE_OP_UPSERT,
};

use crate::node::format_endpoint;
use crate::{u32_to_shard_state, ShardState};

const CHANNEL_CONNECT_TIMEOUT: Duration = Duration::from_millis(500);
const SEARCH_SHARD_TIMEOUT: Duration = Duration::from_millis(200);
const WRITE_QUORUM_TIMEOUT: Duration = Duration::from_millis(500);
const SEED_BATCH_TIMEOUT: Duration = Duration::from_secs(10);

/// Thread-safe HTTP/2 gRPC channel pool caching connected `tonic::transport::Channel` handles
/// per normalized endpoint (`http://...`) to eliminate per-RPC TCP/HTTP2 handshake churn.
#[derive(Debug, Clone, Default)]
pub struct GrpcChannelPool {
    channels: Arc<RwLock<HashMap<String, Channel>>>,
}

impl GrpcChannelPool {
    /// Construct an empty `GrpcChannelPool`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of currently cached endpoint channels in the pool.
    #[must_use]
    pub fn len(&self) -> usize {
        self.channels.read().map_or(0, |g| g.len())
    }

    /// Returns `true` if no channels are currently cached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return a cached `Channel` for `endpoint` or establish and cache a new HTTP/2 connection
    /// bounded by `connect_timeout(500ms)` and `rpc_timeout`.
    pub async fn get_or_connect(
        &self,
        endpoint: &str,
        rpc_timeout: Duration,
    ) -> Result<Channel, String> {
        let normalized = format_endpoint(endpoint);
        if let Ok(guard) = self.channels.read() {
            if let Some(existing) = guard.get(&normalized) {
                return Ok(existing.clone());
            }
        }

        let ep = Endpoint::from_shared(normalized.clone())
            .map_err(|e| e.to_string())?
            .connect_timeout(CHANNEL_CONNECT_TIMEOUT)
            .tcp_nodelay(true);

        let connect_cap = rpc_timeout.min(CHANNEL_CONNECT_TIMEOUT);
        let connected = tokio::time::timeout(connect_cap, ep.connect())
            .await
            .map_err(|_| "connect timeout".to_string())?
            .map_err(|e| e.to_string())?;

        if let Ok(mut guard) = self.channels.write() {
            let ch = guard
                .entry(normalized)
                .or_insert_with(|| connected.clone())
                .clone();
            return Ok(ch);
        }
        Ok(connected)
    }

    /// Evict any cached channel for `endpoint` after a transport failure or timeout so subsequent
    /// retries establish a fresh connection.
    pub fn evict(&self, endpoint: &str) {
        let normalized = format_endpoint(endpoint);
        if let Ok(mut guard) = self.channels.write() {
            guard.remove(&normalized);
            guard.remove(endpoint);
        }
    }
}

/// Smart direct-to-primary cluster client (Shape 2) with cached `TopologySnapshot`,
/// pooled HTTP/2 gRPC channels (`GrpcChannelPool`), and automatic retry-on-`WRONG_EPOCH` /
/// dead-replica fallback.
#[derive(Clone)]
pub struct ClusterClient {
    coord_addr: String,
    topology: Arc<RwLock<GetTopologyResponse>>,
    pool: GrpcChannelPool,
}

impl ClusterClient {
    /// Connect to the coordinator at `coord_addr` using a fresh `GrpcChannelPool` and fetch the
    /// initial `GetTopologyResponse`.
    pub async fn connect(coord_addr: impl Into<String>) -> Result<Self, VektorError> {
        Self::connect_with_pool(coord_addr, GrpcChannelPool::new()).await
    }

    /// Connect to the coordinator at `coord_addr` sharing an explicit `GrpcChannelPool`.
    pub async fn connect_with_pool(
        coord_addr: impl Into<String>,
        pool: GrpcChannelPool,
    ) -> Result<Self, VektorError> {
        let addr = coord_addr.into();
        let topo = fetch_topology(&pool, &addr).await?;
        Ok(Self {
            coord_addr: addr,
            topology: Arc::new(RwLock::new(topo)),
            pool,
        })
    }

    /// Borrow the underlying HTTP/2 `GrpcChannelPool`.
    #[must_use]
    pub fn channel_pool(&self) -> &GrpcChannelPool {
        &self.pool
    }

    /// Refresh the cached topology snapshot from the coordinator.
    pub async fn refresh_topology(&self) -> Result<GetTopologyResponse, VektorError> {
        let topo = fetch_topology(&self.pool, &self.coord_addr).await?;
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
        let wire_rec = WireWalRecord {
            seq_no: 0,
            write_id,
            op_kind: WIRE_OP_UPSERT,
            vector_id: id,
            vector_f32_le: encode_f32_le(vector),
        };
        self.append_single_quorum_record(id, wire_rec, "upsert")
            .await
    }

    /// Perform a single durable `W=2` quorum delete routed directly to the shard primary.
    /// Automatically refreshes topology and retries once on `WRONG_EPOCH` or primary failure.
    pub async fn delete(
        &self,
        id: VectorId,
        write_id: WriteId,
    ) -> Result<(ShardId, SeqNo), VektorError> {
        let wire_rec = WireWalRecord {
            seq_no: 0,
            write_id,
            op_kind: WIRE_OP_DELETE,
            vector_id: id,
            vector_f32_le: Vec::new(),
        };
        self.append_single_quorum_record(id, wire_rec, "delete")
            .await
    }

    /// Validate vector dimensions, partition `records` across logical shards via `cfg.shard_for(id)`,
    /// and dispatch batched `W=2` quorum writes (`batched_fsync = true`) concurrently across target
    /// shard primaries with automatic topology-refresh retry on `STATUS_WRONG_EPOCH` or transport
    /// failure. Returns the total count of durably acknowledged vectors.
    pub async fn upsert_batch(
        &self,
        records: &[(VectorId, WriteId, Vec<f32>)],
    ) -> Result<usize, VektorError> {
        if records.is_empty() {
            return Ok(0);
        }

        let cfg = self.cluster_config();
        let mut by_shard: BTreeMap<ShardId, Vec<WireWalRecord>> = BTreeMap::new();
        for (id, write_id, vector) in records {
            if vector.len() != cfg.dimensions {
                return Err(VektorError::DimensionMismatch {
                    expected: cfg.dimensions,
                    actual: vector.len(),
                });
            }
            if vector.iter().any(|v| !v.is_finite()) {
                return Err(VektorError::InvalidVector("vector contains NaN or Inf"));
            }
            let shard_id = cfg.shard_for(*id);
            by_shard.entry(shard_id).or_default().push(WireWalRecord {
                seq_no: 0,
                write_id: *write_id,
                op_kind: WIRE_OP_UPSERT,
                vector_id: *id,
                vector_f32_le: encode_f32_le(vector),
            });
        }

        let mut tasks = Vec::with_capacity(by_shard.len());
        for (shard_id, wire_records) in by_shard {
            let client = self.clone();
            let write_quorum = cfg.write_quorum;
            tasks.push(tokio::spawn(async move {
                client
                    .append_batch_quorum_for_shard(shard_id, wire_records, write_quorum)
                    .await
            }));
        }

        let mut total_acked = 0_usize;
        for task in tasks {
            let acked = task
                .await
                .map_err(|e| VektorError::WalIo(format!("upsert_batch task join error: {e}")))??;
            total_acked = total_acked.saturating_add(acked);
        }
        Ok(total_acked)
    }

    async fn append_batch_quorum_for_shard(
        &self,
        shard_id: ShardId,
        wire_records: Vec<WireWalRecord>,
        write_quorum: u8,
    ) -> Result<usize, VektorError> {
        for attempt in 0..2 {
            let Some(shard) = self.get_shard_assignment(shard_id) else {
                let _ = self.refresh_topology().await;
                continue;
            };
            if u32_to_shard_state(shard.state) == ShardState::UnderQuorum {
                return Err(VektorError::QuorumUnavailable {
                    shard_id,
                    required: write_quorum,
                    alive: 1,
                });
            }

            match send_append_quorum(
                &self.pool,
                &shard.primary_addr,
                shard_id,
                shard.epoch,
                true,
                wire_records.clone(),
                SEED_BATCH_TIMEOUT,
            )
            .await
            {
                Ok(resp) if resp.status_code == STATUS_OK => {
                    return Ok(resp.assigned_seq_nos.len());
                }
                Ok(resp) if resp.status_code == STATUS_QUORUM_UNAVAILABLE => {
                    return Err(VektorError::QuorumUnavailable {
                        shard_id,
                        required: write_quorum,
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
            "upsert_batch failed after topology refresh on shard {shard_id}"
        )))
    }

    async fn append_single_quorum_record(
        &self,
        id: VectorId,
        wire_rec: WireWalRecord,
        op_name: &str,
    ) -> Result<(ShardId, SeqNo), VektorError> {
        let cfg = self.cluster_config();
        let shard_id = cfg.shard_for(id);

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
                &self.pool,
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
            "{op_name} failed after topology refresh on shard {shard_id}"
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
            &self.pool,
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
            let pool = self.pool.clone();
            let q_bytes = query_bytes.clone();
            let k_u32 = top_k as u32;
            let ef_u32 = ef_search as u32;
            tasks.push(tokio::spawn(async move {
                search_single_shard(pool, shard, q_bytes, k_u32, ef_u32).await
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
            if let Ok(wire_recs) = query_shard_wal(&self.pool, &addr, shard_id).await {
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

async fn fetch_topology(
    pool: &GrpcChannelPool,
    coord_addr: &str,
) -> Result<GetTopologyResponse, VektorError> {
    let channel = pool
        .get_or_connect(coord_addr, WRITE_QUORUM_TIMEOUT)
        .await
        .map_err(VektorError::WalIo)?;
    let mut client = TopologyRpcClient::new(channel);

    let resp = match tokio::time::timeout(
        WRITE_QUORUM_TIMEOUT,
        client.get_topology(Request::new(GetTopologyRequest {})),
    )
    .await
    {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            pool.evict(coord_addr);
            return Err(VektorError::WalIo(e.to_string()));
        }
        Err(_) => {
            pool.evict(coord_addr);
            return Err(VektorError::WalIo("get_topology timeout".to_string()));
        }
    };
    Ok(resp.into_inner())
}

async fn send_append_quorum(
    pool: &GrpcChannelPool,
    addr: &str,
    shard_id: ShardId,
    expected_epoch: u64,
    batched_fsync: bool,
    records: Vec<WireWalRecord>,
    timeout: Duration,
) -> Result<vektordist_proto::v1::AppendQuorumResponse, String> {
    let channel = pool.get_or_connect(addr, timeout).await?;
    let mut client = ShardReplicaRpcClient::new(channel);

    match tokio::time::timeout(
        timeout,
        client.append_quorum(Request::new(AppendQuorumRequest {
            shard_id,
            expected_epoch,
            batched_fsync,
            records,
        })),
    )
    .await
    {
        Ok(Ok(resp)) => Ok(resp.into_inner()),
        Ok(Err(e)) => {
            pool.evict(addr);
            Err(e.to_string())
        }
        Err(_) => {
            pool.evict(addr);
            Err("append_quorum timeout".to_string())
        }
    }
}

async fn search_single_shard(
    pool: GrpcChannelPool,
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
            &pool,
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
    pool: &GrpcChannelPool,
    addr: &str,
    shard_id: ShardId,
    expected_epoch: u64,
    top_k: u32,
    ef_search: u32,
    query_f32_le: Vec<u8>,
) -> Result<Vec<ScoredId>, String> {
    let channel = pool.get_or_connect(addr, SEARCH_SHARD_TIMEOUT).await?;
    let mut client = ShardReplicaRpcClient::new(channel);

    let resp = match tokio::time::timeout(
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
    {
        Ok(Ok(r)) => r.into_inner(),
        Ok(Err(e)) => {
            pool.evict(addr);
            return Err(e.to_string());
        }
        Err(_) => {
            pool.evict(addr);
            return Err("search_shard timeout".to_string());
        }
    };

    if resp.status_code == STATUS_OK {
        Ok(resp.hits.iter().map(from_wire_scored_id).collect())
    } else {
        Err(resp.error_message)
    }
}

async fn query_shard_wal(
    pool: &GrpcChannelPool,
    addr: &str,
    shard_id: ShardId,
) -> Result<Vec<WireWalRecord>, String> {
    let channel = pool.get_or_connect(addr, WRITE_QUORUM_TIMEOUT).await?;
    let mut client = ShardReplicaRpcClient::new(channel);

    let resp = match tokio::time::timeout(
        WRITE_QUORUM_TIMEOUT,
        client.get_wal_records(Request::new(GetWalRecordsRequest {
            shard_id,
            from_seq_no: 1,
        })),
    )
    .await
    {
        Ok(Ok(r)) => r.into_inner(),
        Ok(Err(e)) => {
            pool.evict(addr);
            return Err(e.to_string());
        }
        Err(_) => {
            pool.evict(addr);
            return Err("get_wal_records timeout".to_string());
        }
    };
    Ok(resp.records)
}
