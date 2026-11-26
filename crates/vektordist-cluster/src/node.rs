use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use tonic::{Request, Response, Status};
use vektordist_core::checkpoint::{CheckpointManifest, ShardCheckpointManager};
use vektordist_core::hnsw::{HnswIndex, HnswParams};
use vektordist_core::wal::{replay_into_hnsw, WalLog, WalOp, WalRecord};
use vektordist_core::{ClusterConfig, Epoch, NodeId, SeqNo, ShardId, VektorError, WriteId};
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

use crate::client::GrpcChannelPool;
use crate::{u32_to_shard_state, ShardState};

const REPLICATE_RPC_TIMEOUT: Duration = Duration::from_millis(500);
const REPLICATE_BATCH_RPC_TIMEOUT: Duration = Duration::from_secs(5);

/// Default bounded capacity of [`WriteIdDeduplicator`] (`4,096` committed `WriteId -> SeqNo` entries).
pub const DEFAULT_WRITE_ID_DEDUP_CAPACITY: usize = 4_096;

/// Bounded FIFO deduplication cache mapping caller-supplied non-zero `(WriteId, op_tag)` keys to
/// their originally assigned `SeqNo`, preventing duplicate WAL frames across client/network retries
/// while bounding per-shard memory to `O(capacity)`.
#[derive(Debug, Clone)]
pub struct WriteIdDeduplicator {
    entries: HashMap<(WriteId, u8), SeqNo>,
    order: VecDeque<(WriteId, u8)>,
    capacity: usize,
}

impl Default for WriteIdDeduplicator {
    fn default() -> Self {
        Self::new(DEFAULT_WRITE_ID_DEDUP_CAPACITY)
    }
}

#[inline]
const fn wal_op_tag(op: &WalOp) -> u8 {
    match op {
        WalOp::Upsert { .. } => 1,
        WalOp::Delete { .. } => 2,
    }
}

impl WriteIdDeduplicator {
    /// Construct a bounded deduplicator retaining at most `capacity.max(1)` non-zero `(WriteId, op_tag)` keys.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            entries: HashMap::with_capacity(cap),
            order: VecDeque::with_capacity(cap),
            capacity: cap,
        }
    }

    /// Look up the previously committed `SeqNo` for `(write_id, op_tag)` (`write_id == 0` is never deduplicated).
    #[must_use]
    pub fn get_op(&self, write_id: WriteId, op_tag: u8) -> Option<SeqNo> {
        if write_id == 0 {
            return None;
        }
        self.entries.get(&(write_id, op_tag)).copied()
    }

    /// Look up the previously committed `SeqNo` for an upsert `write_id`.
    #[must_use]
    pub fn get(&self, write_id: WriteId) -> Option<SeqNo> {
        self.get_op(write_id, 1)
    }

    /// Record a committed `(write_id, op_tag, seq_no)` entry, evicting the oldest entry in FIFO order if full.
    pub fn record_op(&mut self, write_id: WriteId, op_tag: u8, seq_no: SeqNo) {
        if write_id == 0 {
            return;
        }
        let key = (write_id, op_tag);
        if let std::collections::hash_map::Entry::Occupied(mut occ) = self.entries.entry(key) {
            *occ.get_mut() = seq_no;
            return;
        }
        while self.order.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            } else {
                break;
            }
        }
        self.entries.insert(key, seq_no);
        self.order.push_back(key);
    }

    /// Record a committed upsert `(write_id, seq_no)` pair.
    pub fn record(&mut self, write_id: WriteId, seq_no: SeqNo) {
        self.record_op(write_id, 1, seq_no);
    }

    /// Number of non-zero `WriteId` entries currently cached.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the deduplicator cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Maximum number of entries retained by this deduplicator.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Single owner for one logical shard's WAL, HNSW index, epoch, idempotency table, and replication peers.
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
    pub dedup: WriteIdDeduplicator,
}

impl ShardEngine {
    pub fn open(
        shard_id: ShardId,
        data_dir: &Path,
        config: &ClusterConfig,
        params: HnswParams,
    ) -> Result<Self, VektorError> {
        let (hnsw, wal, replay, _ckpt) = ShardCheckpointManager::recover_shard(
            data_dir,
            shard_id,
            config.dimensions,
            config.metric,
            params,
        )?;
        let mut dedup = WriteIdDeduplicator::default();
        for rec in &replay.records {
            if rec.write_id != 0 {
                dedup.record_op(rec.write_id, wal_op_tag(&rec.op), rec.seq_no);
            }
        }
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
            dedup,
        })
    }

    /// Atomically checkpoint live `HnswIndex` vectors into a zero-tombstone `b"VDSG"` segment +
    /// `b"VDCK"` manifest, compact the shard WAL on disk, and deduplicate `records_by_seq` in RAM.
    pub fn checkpoint_and_compact(
        &mut self,
        data_dir: &Path,
    ) -> Result<CheckpointManifest, VektorError> {
        let manifest = ShardCheckpointManager::create_checkpoint(
            data_dir,
            self.shard_id,
            &self.hnsw,
            &mut self.wal,
            &[],
        )?;
        self.compact_history();
        Ok(manifest)
    }

    /// Apply records to WAL + HNSW if `seq_no > hnsw.last_seq_no()`, updating `self.dedup`.
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
        for rec in &new_records {
            if rec.write_id != 0 {
                self.dedup
                    .record_op(rec.write_id, wal_op_tag(&rec.op), rec.seq_no);
            }
        }
        self.records_by_seq.extend(new_records);
        Ok(self.hnsw.last_seq_no())
    }

    /// Deduplicate `self.records_by_seq` in-place to retain only the latest mutation (`Upsert` or
    /// `Delete` tombstone) per `VectorId` in strictly ascending `seq_no` order, bounding memory
    /// across millions of overwrites.
    pub fn compact_history(&mut self) {
        if self.records_by_seq.len() <= 1 {
            return;
        }
        let mut seen_ids = HashSet::with_capacity(self.records_by_seq.len());
        let mut compacted_rev = Vec::with_capacity(self.hnsw.len().saturating_add(16));
        for rec in self.records_by_seq.iter().rev() {
            let id = match &rec.op {
                WalOp::Upsert { id, .. } | WalOp::Delete { id } => *id,
            };
            if seen_ids.insert(id) {
                compacted_rev.push(rec.clone());
            }
        }
        compacted_rev.reverse();
        compacted_rev.shrink_to_fit();
        self.records_by_seq = compacted_rev;
    }
}

/// Multi-shard gRPC node service hosting independent `RwLock<ShardEngine>` instances and a
/// shared `GrpcChannelPool` for follower WAL replication.
#[derive(Clone)]
pub struct NodeService {
    pub node_id: NodeId,
    pub config: ClusterConfig,
    pub shards: Arc<HashMap<ShardId, Arc<RwLock<ShardEngine>>>>,
    pub channel_pool: GrpcChannelPool,
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
            channel_pool: GrpcChannelPool::new(),
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
            let mut records_to_commit = Vec::with_capacity(parsed_records.len());
            let mut batch_seen: HashMap<(WriteId, u8), SeqNo> = HashMap::new();

            for mut rec in parsed_records {
                let op_tag = wal_op_tag(&rec.op);
                if rec.write_id != 0 {
                    if let Some(existing_seq) = engine
                        .dedup
                        .get_op(rec.write_id, op_tag)
                        .or_else(|| batch_seen.get(&(rec.write_id, op_tag)).copied())
                    {
                        assigned.push(existing_seq);
                        continue;
                    }
                }

                if rec.seq_no == 0 {
                    next_seq += 1;
                    rec.seq_no = next_seq;
                } else if rec.seq_no > next_seq {
                    next_seq = rec.seq_no;
                }
                if rec.write_id != 0 {
                    batch_seen.insert((rec.write_id, op_tag), rec.seq_no);
                }
                assigned.push(rec.seq_no);
                records_to_commit.push(rec);
            }

            if records_to_commit.is_empty() {
                return Ok(Response::new(AppendQuorumResponse {
                    status_code: STATUS_OK,
                    current_epoch: engine.epoch,
                    assigned_seq_nos: assigned,
                    error_message: String::new(),
                }));
            }

            if let Err(e) = engine.apply_records(&records_to_commit, req.batched_fsync) {
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
                records_to_commit,
                assigned,
            )
        };

        let wire_records: Vec<_> = sequenced_records.iter().map(to_wire_wal_record).collect();
        let timeout = if req.batched_fsync || wire_records.len() > 1 {
            REPLICATE_BATCH_RPC_TIMEOUT
        } else {
            REPLICATE_RPC_TIMEOUT
        };

        // Replicate in parallel to all active followers via pooled HTTP/2 channels.
        let mut tasks = Vec::with_capacity(followers.len());
        for addr in followers {
            let pool = self.channel_pool.clone();
            let wire_clone = wire_records.clone();
            let shard_id = req.shard_id;
            let batched = req.batched_fsync;
            tasks.push(tokio::spawn(async move {
                replicate_to_peer(&pool, &addr, shard_id, epoch, batched, wire_clone, timeout).await
            }));
        }

        if let Some(rec_addr) = recovering_peer {
            let pool = self.channel_pool.clone();
            let wire_clone = wire_records;
            let shard_id = req.shard_id;
            let batched = req.batched_fsync;
            tokio::spawn(async move {
                let _ = replicate_to_peer(
                    &pool, &rec_addr, shard_id, epoch, batched, wire_clone, timeout,
                )
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
    pool: &GrpcChannelPool,
    addr: &str,
    shard_id: ShardId,
    epoch: Epoch,
    batched_fsync: bool,
    records: Vec<vektordist_proto::v1::WireWalRecord>,
    timeout: Duration,
) -> Result<ReplicateWalResponse, Status> {
    let channel = pool
        .get_or_connect(addr, timeout)
        .await
        .map_err(Status::unavailable)?;
    let mut client = ShardReplicaRpcClient::new(channel);

    let rpc_fut = client.replicate_wal(Request::new(ReplicateWalRequest {
        shard_id,
        epoch,
        batched_fsync,
        records,
    }));
    match tokio::time::timeout(timeout, rpc_fut).await {
        Ok(Ok(resp)) => Ok(resp.into_inner()),
        Ok(Err(status)) => {
            pool.evict(addr);
            Err(status)
        }
        Err(_) => {
            pool.evict(addr);
            Err(Status::deadline_exceeded("replicate_wal timeout"))
        }
    }
}

#[must_use]
pub fn format_endpoint(addr: &str) -> String {
    if addr.starts_with("http://") || addr.starts_with("https://") {
        addr.to_string()
    } else {
        format!("http://{addr}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_id_deduplicator_enforces_fifo_eviction_and_ignores_zero() {
        let mut dedup = WriteIdDeduplicator::new(3);
        assert_eq!(dedup.capacity(), 3);
        assert!(dedup.is_empty());

        dedup.record(0, 99);
        assert_eq!(dedup.get(0), None);
        assert!(dedup.is_empty());

        dedup.record(101, 1);
        dedup.record(102, 2);
        dedup.record(103, 3);
        assert_eq!(dedup.len(), 3);
        assert_eq!(dedup.get(101), Some(1));

        // Recording a 4th distinct WriteId evicts oldest (101).
        dedup.record(104, 4);
        assert_eq!(dedup.len(), 3);
        assert_eq!(dedup.get(101), None);
        assert_eq!(dedup.get(102), Some(2));
        assert_eq!(dedup.get(104), Some(4));
    }
}
