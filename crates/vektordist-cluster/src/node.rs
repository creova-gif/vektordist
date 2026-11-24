use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use tonic::{Request, Response, Status};
use vektordist_core::hnsw::{HnswIndex, HnswParams};
use vektordist_core::wal::{replay_into_hnsw, WalLog, WalRecord};
use vektordist_core::{ClusterConfig, Epoch, NodeId, SeqNo, ShardId, VektorError};
use vektordist_proto::v1::shard_replica_rpc_client::ShardReplicaRpcClient;
use vektordist_proto::v1::shard_replica_rpc_server::ShardReplicaRpc;
use vektordist_proto::v1::{
    AppendQuorumRequest, AppendQuorumResponse, ConfigureShardRequest, ConfigureShardResponse,
    GetWalRecordsRequest, GetWalRecordsResponse, PingNodeRequest, PingNodeResponse,
    ReplicateWalRequest, ReplicateWalResponse, SearchShardRequest, SearchShardResponse,
};
use vektordist_proto::{
    decode_f32_le, from_wire_wal_record, to_wire_scored_id, to_wire_wal_record, STATUS_INTERNAL,
    STATUS_INVALID_ARGUMENT, STATUS_OK, STATUS_QUORUM_UNAVAILABLE, STATUS_WRONG_EPOCH,
};

use crate::{u32_to_shard_state, ShardState};

const REPLICATE_RPC_TIMEOUT: Duration = Duration::from_millis(500);
const REPLICATE_BATCH_RPC_TIMEOUT: Duration = Duration::from_secs(5);

/// Single owner for one logical shard's WAL, HNSW index, epoch, and replication peers.
pub struct ShardEngine {
    pub shard_id: ShardId,
    pub epoch: Epoch,
    pub is_primary: bool,
    pub state: ShardState,
    pub write_quorum: u8,
    pub follower_addrs: Vec<String>,
    pub recovering_follower_addr: Option<String>,
    pub hnsw: HnswIndex,
    pub wal: WalLog,
    pub records_by_seq: Vec<WalRecord>,
}

impl ShardEngine {
    pub fn open(
        shard_id: ShardId,
        data_dir: &Path,
        config: &ClusterConfig,
        params: HnswParams,
    ) -> Result<Self, VektorError> {
        let wal_path = data_dir.join(format!("shard-{shard_id:04}.wal"));
        let (wal, replay) = WalLog::open(&wal_path, config.dimensions)?;
        let mut hnsw = HnswIndex::new(config.dimensions, config.metric, params)?;
        replay_into_hnsw(&replay.records, &mut hnsw)?;
        Ok(Self {
            shard_id,
            epoch: 1,
            is_primary: false,
            state: ShardState::Healthy,
            write_quorum: config.write_quorum,
            follower_addrs: Vec::new(),
            recovering_follower_addr: None,
            hnsw,
            wal,
            records_by_seq: replay.records,
        })
    }

    /// Apply records to WAL + HNSW if `seq_no > hnsw.last_seq_no()`.
    pub fn apply_records(
        &mut self,
        records: &[WalRecord],
        batched_fsync: bool,
    ) -> Result<SeqNo, VektorError> {
        let mut new_records = Vec::with_capacity(records.len());
        let mut cur_last = self.hnsw.last_seq_no();
        for rec in records {
            if rec.seq_no > cur_last {
                cur_last = rec.seq_no;
                new_records.push(rec.clone());
            }
        }
        if new_records.is_empty() {
            return Ok(self.hnsw.last_seq_no());
        }
        if batched_fsync || new_records.len() > 1 {
            self.wal.append_batch_and_sync(&new_records)?;
        } else if let Some(first) = new_records.first() {
            self.wal.append_and_sync(first)?;
        }
        replay_into_hnsw(&new_records, &mut self.hnsw)?;
        self.records_by_seq.extend(new_records);
        Ok(self.hnsw.last_seq_no())
    }
}

/// Multi-shard gRPC node service hosting 16 independent `RwLock<ShardEngine>` instances.
#[derive(Clone)]
pub struct NodeService {
    pub node_id: NodeId,
    pub config: ClusterConfig,
    pub shards: Arc<HashMap<ShardId, Arc<RwLock<ShardEngine>>>>,
}

impl NodeService {
    pub fn open(
        node_id: NodeId,
        data_dir: &Path,
        config: ClusterConfig,
        params: HnswParams,
    ) -> Result<Self, VektorError> {
        std::fs::create_dir_all(data_dir).map_err(|e| VektorError::WalIo(e.to_string()))?;
        let mut map = HashMap::with_capacity(config.num_shards as usize);
        for shard_id in 0..config.num_shards {
            let engine = ShardEngine::open(shard_id, data_dir, &config, params)?;
            map.insert(shard_id, Arc::new(RwLock::new(engine)));
        }
        Ok(Self {
            node_id,
            config,
            shards: Arc::new(map),
        })
    }

    fn get_shard(&self, shard_id: ShardId) -> Result<&Arc<RwLock<ShardEngine>>, Status> {
        self.shards
            .get(&shard_id)
            .ok_or_else(|| Status::not_found(format!("unknown shard {shard_id}")))
    }
}

#[tonic::async_trait]
impl ShardReplicaRpc for NodeService {
    async fn append_quorum(
        &self,
        request: Request<AppendQuorumRequest>,
    ) -> Result<Response<AppendQuorumResponse>, Status> {
        let req = request.into_inner();
        let shard_lock = self.get_shard(req.shard_id)?;

        // Parse & validate wire records at the boundary before taking the write lock.
        let mut parsed_records = Vec::with_capacity(req.records.len());
        for w in &req.records {
            match from_wire_wal_record(w, self.config.dimensions) {
                Ok(rec) => parsed_records.push(rec),
                Err(e) => {
                    return Ok(Response::new(AppendQuorumResponse {
                        status_code: STATUS_INVALID_ARGUMENT,
                        current_epoch: 0,
                        assigned_seq_nos: Vec::new(),
                        error_message: e.to_string(),
                    }));
                }
            }
        }

        let (epoch, write_quorum, followers, recovering_peer, sequenced_records, assigned_seqs) = {
            let mut engine = shard_lock.write().await;
            if req.expected_epoch != engine.epoch || !engine.is_primary {
                return Ok(Response::new(AppendQuorumResponse {
                    status_code: STATUS_WRONG_EPOCH,
                    current_epoch: engine.epoch,
                    assigned_seq_nos: Vec::new(),
                    error_message: format!(
                        "node {} shard {} epoch={} is_primary={} (expected epoch={})",
                        self.node_id,
                        req.shard_id,
                        engine.epoch,
                        engine.is_primary,
                        req.expected_epoch
                    ),
                }));
            }
            if engine.state == ShardState::UnderQuorum {
                return Ok(Response::new(AppendQuorumResponse {
                    status_code: STATUS_QUORUM_UNAVAILABLE,
                    current_epoch: engine.epoch,
                    assigned_seq_nos: Vec::new(),
                    error_message: "shard is under write quorum".to_string(),
                }));
            }
            // Check that 1 (primary) + configured followers can even reach write_quorum.
            let max_possible = 1_u8.saturating_add(engine.follower_addrs.len() as u8);
            if max_possible < engine.write_quorum {
                return Ok(Response::new(AppendQuorumResponse {
                    status_code: STATUS_QUORUM_UNAVAILABLE,
                    current_epoch: engine.epoch,
                    assigned_seq_nos: Vec::new(),
                    error_message: format!(
                        "only {max_possible} replicas configured, need {}",
                        engine.write_quorum
                    ),
                }));
            }

            let mut next_seq = engine.hnsw.last_seq_no();
            let mut assigned = Vec::with_capacity(parsed_records.len());
            for rec in &mut parsed_records {
                if rec.seq_no == 0 {
                    next_seq += 1;
                    rec.seq_no = next_seq;
                } else if rec.seq_no > next_seq {
                    next_seq = rec.seq_no;
                }
                assigned.push(rec.seq_no);
            }

            if let Err(e) = engine.apply_records(&parsed_records, req.batched_fsync) {
                return Ok(Response::new(AppendQuorumResponse {
                    status_code: STATUS_INTERNAL,
                    current_epoch: engine.epoch,
                    assigned_seq_nos: Vec::new(),
                    error_message: e.to_string(),
                }));
            }

            (
                engine.epoch,
                engine.write_quorum,
                engine.follower_addrs.clone(),
                engine.recovering_follower_addr.clone(),
                parsed_records,
                assigned,
            )
        };

        let wire_records: Vec<_> = sequenced_records.iter().map(to_wire_wal_record).collect();
        let timeout = if req.batched_fsync || wire_records.len() > 1 {
            REPLICATE_BATCH_RPC_TIMEOUT
        } else {
            REPLICATE_RPC_TIMEOUT
        };

        // Replicate in parallel to all active followers (and dual-stream to recovering follower if present).
        let mut tasks = Vec::with_capacity(followers.len());
        for addr in followers {
            let wire_clone = wire_records.clone();
            let shard_id = req.shard_id;
            let batched = req.batched_fsync;
            tasks.push(tokio::spawn(async move {
                replicate_to_peer(&addr, shard_id, epoch, batched, wire_clone, timeout).await
            }));
        }

        if let Some(rec_addr) = recovering_peer {
            let wire_clone = wire_records;
            let shard_id = req.shard_id;
            let batched = req.batched_fsync;
            tokio::spawn(async move {
                let _ = replicate_to_peer(&rec_addr, shard_id, epoch, batched, wire_clone, timeout)
                    .await;
            });
        }

        let mut acks: u8 = 1; // Primary already fsynced.
        let mut saw_higher_epoch: Option<Epoch> = None;
        for task in tasks {
            if let Ok(Ok(resp)) = task.await {
                if resp.status_code == STATUS_OK {
                    acks = acks.saturating_add(1);
                } else if resp.status_code == STATUS_WRONG_EPOCH && resp.current_epoch > epoch {
                    saw_higher_epoch = Some(resp.current_epoch);
                }
            }
        }

        if let Some(higher) = saw_higher_epoch {
            return Ok(Response::new(AppendQuorumResponse {
                status_code: STATUS_WRONG_EPOCH,
                current_epoch: higher,
                assigned_seq_nos: Vec::new(),
                error_message: "fenced by higher follower epoch".to_string(),
            }));
        }

        if acks >= write_quorum {
            Ok(Response::new(AppendQuorumResponse {
                status_code: STATUS_OK,
                current_epoch: epoch,
                assigned_seq_nos: assigned_seqs,
                error_message: String::new(),
            }))
        } else {
            Ok(Response::new(AppendQuorumResponse {
                status_code: STATUS_QUORUM_UNAVAILABLE,
                current_epoch: epoch,
                assigned_seq_nos: Vec::new(),
                error_message: format!(
                    "quorum not reached for shard {}: acks={acks}, required={write_quorum}",
                    req.shard_id
                ),
            }))
        }
    }

    async fn replicate_wal(
        &self,
        request: Request<ReplicateWalRequest>,
    ) -> Result<Response<ReplicateWalResponse>, Status> {
        let req = request.into_inner();
        let shard_lock = self.get_shard(req.shard_id)?;

        let mut parsed_records = Vec::with_capacity(req.records.len());
        for w in &req.records {
            match from_wire_wal_record(w, self.config.dimensions) {
                Ok(rec) => parsed_records.push(rec),
                Err(e) => {
                    return Ok(Response::new(ReplicateWalResponse {
                        status_code: STATUS_INVALID_ARGUMENT,
                        current_epoch: 0,
                        last_seq_no: 0,
                        error_message: e.to_string(),
                    }));
                }
            }
        }

        let mut engine = shard_lock.write().await;
        if req.epoch < engine.epoch {
            return Ok(Response::new(ReplicateWalResponse {
                status_code: STATUS_WRONG_EPOCH,
                current_epoch: engine.epoch,
                last_seq_no: engine.hnsw.last_seq_no(),
                error_message: format!(
                    "stale epoch: received {}, local {}",
                    req.epoch, engine.epoch
                ),
            }));
        }
        if req.epoch > engine.epoch {
            engine.epoch = req.epoch;
            engine.is_primary = false;
        }

        match engine.apply_records(&parsed_records, req.batched_fsync) {
            Ok(last_seq) => Ok(Response::new(ReplicateWalResponse {
                status_code: STATUS_OK,
                current_epoch: engine.epoch,
                last_seq_no: last_seq,
                error_message: String::new(),
            })),
            Err(e) => Ok(Response::new(ReplicateWalResponse {
                status_code: STATUS_INTERNAL,
                current_epoch: engine.epoch,
                last_seq_no: engine.hnsw.last_seq_no(),
                error_message: e.to_string(),
            })),
        }
    }

    async fn search_shard(
        &self,
        request: Request<SearchShardRequest>,
    ) -> Result<Response<SearchShardResponse>, Status> {
        let req = request.into_inner();
        let shard_lock = self.get_shard(req.shard_id)?;
        let query = match decode_f32_le(&req.query_f32_le, self.config.dimensions) {
            Ok(q) => q,
            Err(e) => {
                return Ok(Response::new(SearchShardResponse {
                    status_code: STATUS_INVALID_ARGUMENT,
                    current_epoch: 0,
                    hits: Vec::new(),
                    error_message: e.to_string(),
                }));
            }
        };

        let engine = shard_lock.read().await;
        if req.expected_epoch > 0 && req.expected_epoch < engine.epoch {
            return Ok(Response::new(SearchShardResponse {
                status_code: STATUS_WRONG_EPOCH,
                current_epoch: engine.epoch,
                hits: Vec::new(),
                error_message: "stale search epoch".to_string(),
            }));
        }

        let ef = if req.ef_search == 0 {
            engine.hnsw.params().ef_search
        } else {
            req.ef_search as usize
        };
        match engine.hnsw.search_with_ef(&query, req.top_k as usize, ef) {
            Ok(hits) => Ok(Response::new(SearchShardResponse {
                status_code: STATUS_OK,
                current_epoch: engine.epoch,
                hits: hits.into_iter().map(to_wire_scored_id).collect(),
                error_message: String::new(),
            })),
            Err(e) => Ok(Response::new(SearchShardResponse {
                status_code: STATUS_INTERNAL,
                current_epoch: engine.epoch,
                hits: Vec::new(),
                error_message: e.to_string(),
            })),
        }
    }

    async fn get_wal_records(
        &self,
        request: Request<GetWalRecordsRequest>,
    ) -> Result<Response<GetWalRecordsResponse>, Status> {
        let req = request.into_inner();
        let shard_lock = self.get_shard(req.shard_id)?;
        let engine = shard_lock.read().await;
        let records: Vec<_> = engine
            .records_by_seq
            .iter()
            .filter(|r| r.seq_no >= req.from_seq_no)
            .map(to_wire_wal_record)
            .collect();
        Ok(Response::new(GetWalRecordsResponse {
            status_code: STATUS_OK,
            current_epoch: engine.epoch,
            last_seq_no: engine.hnsw.last_seq_no(),
            records,
        }))
    }

    async fn configure_shard(
        &self,
        request: Request<ConfigureShardRequest>,
    ) -> Result<Response<ConfigureShardResponse>, Status> {
        let req = request.into_inner();
        let shard_lock = self.get_shard(req.shard_id)?;
        let mut engine = shard_lock.write().await;
        if req.epoch < engine.epoch {
            return Ok(Response::new(ConfigureShardResponse {
                status_code: STATUS_WRONG_EPOCH,
                current_epoch: engine.epoch,
                last_seq_no: engine.hnsw.last_seq_no(),
            }));
        }
        engine.epoch = req.epoch;
        engine.is_primary = req.is_primary;
        engine.follower_addrs = req.follower_addrs;
        engine.recovering_follower_addr = if req.recovering_follower_addr.is_empty() {
            None
        } else {
            Some(req.recovering_follower_addr)
        };
        engine.state = u32_to_shard_state(req.state);
        if req.write_quorum > 0 {
            engine.write_quorum = req.write_quorum as u8;
        }
        Ok(Response::new(ConfigureShardResponse {
            status_code: STATUS_OK,
            current_epoch: engine.epoch,
            last_seq_no: engine.hnsw.last_seq_no(),
        }))
    }

    async fn ping_node(
        &self,
        _request: Request<PingNodeRequest>,
    ) -> Result<Response<PingNodeResponse>, Status> {
        let mut total_mem = 0_u64;
        let mut total_vecs = 0_u64;
        let mut seq_map = HashMap::with_capacity(self.shards.len());
        for (&shard_id, lock) in self.shards.iter() {
            let engine = lock.read().await;
            total_mem += engine.hnsw.memory_bytes() as u64;
            total_vecs += engine.hnsw.len() as u64;
            seq_map.insert(shard_id, engine.hnsw.last_seq_no());
        }
        Ok(Response::new(PingNodeResponse {
            node_id: self.node_id,
            alive: true,
            memory_bytes: total_mem,
            live_vectors: total_vecs,
            shard_last_seq_no: seq_map,
        }))
    }
}

async fn replicate_to_peer(
    addr: &str,
    shard_id: ShardId,
    epoch: Epoch,
    batched_fsync: bool,
    records: Vec<vektordist_proto::v1::WireWalRecord>,
    timeout: Duration,
) -> Result<ReplicateWalResponse, Status> {
    let endpoint = format_endpoint(addr);
    let connect_fut = ShardReplicaRpcClient::connect(endpoint);
    let mut client = tokio::time::timeout(timeout, connect_fut)
        .await
        .map_err(|_| Status::deadline_exceeded("connect timeout"))?
        .map_err(|e| Status::unavailable(e.to_string()))?;

    let rpc_fut = client.replicate_wal(Request::new(ReplicateWalRequest {
        shard_id,
        epoch,
        batched_fsync,
        records,
    }));
    let resp = tokio::time::timeout(timeout, rpc_fut)
        .await
        .map_err(|_| Status::deadline_exceeded("replicate_wal timeout"))??;
    Ok(resp.into_inner())
}

#[must_use]
pub fn format_endpoint(addr: &str) -> String {
    if addr.starts_with("http://") || addr.starts_with("https://") {
        addr.to_string()
    } else {
        format!("http://{addr}")
    }
}
