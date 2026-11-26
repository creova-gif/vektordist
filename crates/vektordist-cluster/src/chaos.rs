//! Automated multi-step fault-injection and SLA verification runner (`ChaosCampaign`).

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tonic::transport::Server;
use vektordist_core::exact::{ExactIndex, SeededRng};
use vektordist_core::hnsw::HnswParams;
use vektordist_core::{ClusterConfig, NodeId, ShardId, VectorId, VektorError, WriteId};
use vektordist_proto::v1::shard_replica_rpc_server::ShardReplicaRpcServer;
use vektordist_proto::v1::topology_rpc_server::TopologyRpcServer;

use crate::client::ClusterClient;
use crate::coord::CoordinatorService;
use crate::node::NodeService;
use crate::telemetry::compute_percentiles;

static CHAOS_DIR_COUNTER: AtomicU64 = AtomicU64::new(1);

/// A single programmable step in an automated chaos campaign.
#[derive(Debug, Clone)]
pub enum ChaosStep {
    /// Seed `count` deterministic unit vectors across all shards using batched WAL fsync.
    SeedVectors {
        count: usize,
        batch_size: usize,
        seed: u64,
    },
    /// Execute concurrent `W=2` quorum writes and scatter-gather reads.
    ConcurrentWorkload {
        writes: usize,
        reads: usize,
        top_k: usize,
        ef_search: usize,
        seed: u64,
    },
    /// Abruptly terminate `node_id` (simulating `SIGKILL`) and trigger coordinator failover + union WAL reconciliation.
    KillNode { node_id: NodeId },
    /// Audit all surviving shard WALs against the external acknowledged-write ledger (`acked_writes_lost == 0`).
    VerifyZeroWriteLoss,
    /// Spawn a replacement node `node_id`, stream historical + live WAL catch-up (`RECOVERING` -> `HEALTHY`).
    AddReplacementNode { node_id: NodeId },
    /// Measure distributed `recall@10` against an exact brute-force oracle and verify `p95` latency SLA.
    AssertRecallAndLatencySla {
        queries: usize,
        top_k: usize,
        ef_search: usize,
        min_recall_at_10: f64,
        max_p95_ms: f64,
        seed: u64,
    },
}

/// Falsifiable report produced by `ChaosCampaign::run`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChaosCampaignReport {
    pub steps_executed: usize,
    pub acked_writes_verified: usize,
    pub acked_writes_lost: usize,
    pub measured_recall_at_10: f64,
    pub measured_p95_ms: f64,
    pub sla_passed: bool,
    pub step_logs: Vec<String>,
}

/// Automated multi-step fault-injection and SLA verification runner against a live `CoordinatorService` + `NodeService` cluster.
#[derive(Debug, Clone)]
pub struct ChaosCampaign {
    pub config: ClusterConfig,
    pub hnsw_params: HnswParams,
    pub initial_nodes: Vec<NodeId>,
    pub steps: Vec<ChaosStep>,
}

impl ChaosCampaign {
    /// Construct a new `ChaosCampaign` with default `HnswParams`.
    #[must_use]
    pub fn new(config: ClusterConfig, initial_nodes: Vec<NodeId>, steps: Vec<ChaosStep>) -> Self {
        Self {
            config,
            hnsw_params: HnswParams::default(),
            initial_nodes,
            steps,
        }
    }

    /// Override `HnswParams` used by cluster nodes during the chaos campaign.
    #[must_use]
    pub fn with_hnsw_params(mut self, params: HnswParams) -> Self {
        self.hnsw_params = params;
        self
    }

    /// Execute the entire chaos schedule against real localhost gRPC node & coordinator servers.
    pub async fn run(&self) -> Result<ChaosCampaignReport, VektorError> {
        self.config.validate()?;
        if self.initial_nodes.is_empty() {
            return Err(VektorError::InvalidConfig(
                "chaos campaign requires at least 1 initial node",
            ));
        }

        let mut node_handles: BTreeMap<NodeId, (String, PathBuf, JoinHandle<()>)> = BTreeMap::new();
        let mut initial_descriptors = Vec::with_capacity(self.initial_nodes.len());

        for &node_id in &self.initial_nodes {
            let (addr, dir, handle) =
                spawn_chaos_node(node_id, self.config.clone(), self.hnsw_params).await?;
            initial_descriptors.push((node_id, addr.clone()));
            node_handles.insert(node_id, (addr, dir, handle));
        }

        let coord = CoordinatorService::new(self.config.clone(), initial_descriptors)?;
        coord.push_all_shard_configs().await?;
        let (coord_addr, coord_handle) = spawn_chaos_coord(coord.clone()).await?;
        let client = ClusterClient::connect(&coord_addr).await?;

        let mut exact_oracle = ExactIndex::new(self.config.dimensions, self.config.metric)?;
        let mut acked_ledger: Vec<(WriteId, VectorId, ShardId)> = Vec::new();
        let mut next_vec_id: VectorId = 1;
        let mut next_write_id: WriteId = 10_000;

        let mut steps_executed = 0_usize;
        let mut acked_writes_verified = 0_usize;
        let mut acked_writes_lost = 0_usize;
        let mut measured_recall_at_10 = 1.0_f64;
        let mut measured_p95_ms = 0.0_f64;
        let mut sla_passed = true;
        let mut step_logs = Vec::with_capacity(self.steps.len());

        for step in &self.steps {
            match step {
                ChaosStep::SeedVectors {
                    count,
                    batch_size,
                    seed,
                } => {
                    let mut rng = SeededRng::new(*seed);
                    let effective_batch = (*batch_size).max(1);
                    let mut by_shard: BTreeMap<ShardId, Vec<(VectorId, WriteId, Vec<f32>)>> =
                        BTreeMap::new();

                    for _ in 0..*count {
                        let mut vec_buf = vec![0.0_f32; self.config.dimensions];
                        rng.fill_unit_vector(&mut vec_buf);
                        let id = next_vec_id;
                        let write_id = next_write_id;
                        next_vec_id += 1;
                        next_write_id += 1;
                        let shard_id = self.config.shard_for(id);
                        by_shard
                            .entry(shard_id)
                            .or_default()
                            .push((id, write_id, vec_buf));
                    }

                    for (shard_id, records) in by_shard {
                        for chunk in records.chunks(effective_batch) {
                            let assigned = client.seed_shard_batch(shard_id, chunk).await?;
                            if assigned.len() == chunk.len() {
                                for (id, write_id, vec_buf) in chunk {
                                    exact_oracle.upsert(*id, vec_buf)?;
                                    acked_ledger.push((*write_id, *id, shard_id));
                                }
                            }
                        }
                    }
                    steps_executed += 1;
                    step_logs.push(format!("SeedVectors(count={count})"));
                }
                ChaosStep::ConcurrentWorkload {
                    writes,
                    reads,
                    top_k,
                    ef_search,
                    seed,
                } => {
                    let base_id = next_vec_id;
                    let base_write_id = next_write_id;
                    next_vec_id += *writes as u64;
                    next_write_id += *writes as u64;

                    let dims = self.config.dimensions;
                    let write_client = client.clone();
                    let write_count = *writes;
                    let write_seed = *seed;

                    let writer_task = tokio::spawn(async move {
                        let mut rng = SeededRng::new(write_seed);
                        let mut acked = Vec::with_capacity(write_count);
                        for offset in 0..write_count {
                            let mut v = vec![0.0_f32; dims];
                            rng.fill_unit_vector(&mut v);
                            let id = base_id + offset as u64;
                            let wid = base_write_id + offset as u64;
                            if let Ok((shard_id, _seq)) = write_client.upsert(id, wid, &v).await {
                                acked.push((wid, id, shard_id, v));
                            }
                        }
                        acked
                    });

                    let read_client = client.clone();
                    let read_count = *reads;
                    let k = (*top_k).max(1);
                    let ef = (*ef_search).max(k);
                    let read_seed = seed.wrapping_add(7919);

                    let reader_task = tokio::spawn(async move {
                        let mut rng = SeededRng::new(read_seed);
                        let mut completed_reads = 0_usize;
                        for _ in 0..read_count {
                            let mut q = vec![0.0_f32; dims];
                            rng.fill_unit_vector(&mut q);
                            if let Ok(hits) = read_client.search(&q, k, ef).await {
                                if !hits.is_empty() {
                                    completed_reads += 1;
                                }
                            }
                        }
                        completed_reads
                    });

                    let acked_batch = writer_task
                        .await
                        .map_err(|e| VektorError::WalIo(format!("writer task join error: {e}")))?;
                    let completed_reads = reader_task
                        .await
                        .map_err(|e| VektorError::WalIo(format!("reader task join error: {e}")))?;

                    for (wid, id, shard_id, v) in acked_batch {
                        exact_oracle.upsert(id, &v)?;
                        acked_ledger.push((wid, id, shard_id));
                    }

                    steps_executed += 1;
                    step_logs.push(format!(
                        "ConcurrentWorkload(writes={writes}, reads_ok={completed_reads}/{reads})"
                    ));
                }
                ChaosStep::KillNode { node_id } => {
                    if let Some((_addr, _dir, handle)) = node_handles.get(node_id) {
                        handle.abort();
                    }
                    coord.handle_node_failure(*node_id).await?;
                    let _ = client.refresh_topology().await?;
                    steps_executed += 1;
                    step_logs.push(format!("KillNode(node_id={node_id})"));
                }
                ChaosStep::VerifyZeroWriteLoss => {
                    let mut durable_write_ids: HashSet<WriteId> =
                        HashSet::with_capacity(acked_ledger.len());
                    for shard_id in 0..self.config.num_shards {
                        let recs = client.verify_shard_wal_write_ids(shard_id).await?;
                        for rec in recs {
                            durable_write_ids.insert(rec.write_id);
                        }
                    }

                    let mut verified = 0_usize;
                    let mut lost = 0_usize;
                    for &(wid, _vec_id, _shard_id) in &acked_ledger {
                        if durable_write_ids.contains(&wid) {
                            verified += 1;
                        } else {
                            lost += 1;
                        }
                    }
                    acked_writes_verified = verified;
                    acked_writes_lost += lost;
                    if lost > 0 {
                        sla_passed = false;
                    }
                    steps_executed += 1;
                    step_logs.push(format!(
                        "VerifyZeroWriteLoss(verified={verified}, lost={lost})"
                    ));
                }
                ChaosStep::AddReplacementNode { node_id } => {
                    let (addr, dir, handle) =
                        spawn_chaos_node(*node_id, self.config.clone(), self.hnsw_params).await?;
                    node_handles.insert(*node_id, (addr.clone(), dir, handle));
                    let recovered_shards = coord.add_replacement_node(*node_id, addr).await?;
                    let _ = client.refresh_topology().await?;
                    steps_executed += 1;
                    step_logs.push(format!(
                        "AddReplacementNode(node_id={node_id}, recovered_shards={})",
                        recovered_shards.len()
                    ));
                }
                ChaosStep::AssertRecallAndLatencySla {
                    queries,
                    top_k,
                    ef_search,
                    min_recall_at_10,
                    max_p95_ms,
                    seed,
                } => {
                    let k = (*top_k).max(1);
                    let ef = (*ef_search).max(k);
                    let q_count = (*queries).max(1);
                    let mut rng = SeededRng::new(*seed);
                    let mut latencies_ms = Vec::with_capacity(q_count);
                    let mut recall_sum = 0.0_f64;

                    for _ in 0..q_count {
                        let mut q = vec![0.0_f32; self.config.dimensions];
                        rng.fill_unit_vector(&mut q);
                        let exact_hits = exact_oracle.search(&q, k)?;
                        let expected_ids: HashSet<VectorId> =
                            exact_hits.iter().map(|h| h.id).collect();

                        let t0 = Instant::now();
                        let approx_hits = client.search(&q, k, ef).await?;
                        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                        latencies_ms.push(elapsed_ms);

                        if expected_ids.is_empty() {
                            recall_sum += 1.0;
                        } else {
                            let matched = approx_hits
                                .iter()
                                .filter(|h| expected_ids.contains(&h.id))
                                .count();
                            recall_sum += (matched as f64) / (expected_ids.len() as f64);
                        }
                    }

                    measured_recall_at_10 = recall_sum / (q_count as f64);
                    let (_p50, p95, _p99) = compute_percentiles(&latencies_ms);
                    measured_p95_ms = p95;

                    let step_ok = acked_writes_lost == 0
                        && measured_recall_at_10 >= *min_recall_at_10
                        && measured_p95_ms <= *max_p95_ms;
                    if !step_ok {
                        sla_passed = false;
                    }
                    steps_executed += 1;
                    step_logs.push(format!(
                        "AssertRecallAndLatencySla(recall={measured_recall_at_10:.4}, p95_ms={measured_p95_ms:.3}, passed={step_ok})"
                    ));
                }
            }
        }

        coord_handle.abort();
        for (_id, (_addr, dir, handle)) in node_handles {
            handle.abort();
            let _ = std::fs::remove_dir_all(&dir);
        }

        Ok(ChaosCampaignReport {
            steps_executed,
            acked_writes_verified,
            acked_writes_lost,
            measured_recall_at_10,
            measured_p95_ms,
            sla_passed,
            step_logs,
        })
    }
}

pub(crate) async fn spawn_chaos_node(
    node_id: NodeId,
    config: ClusterConfig,
    params: HnswParams,
) -> Result<(String, PathBuf, JoinHandle<()>), VektorError> {
    let seq = CHAOS_DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("vektordist-m10-chaos-{pid}-{seq}-n{node_id}"));
    let _ = std::fs::remove_dir_all(&dir);
    let svc = NodeService::open(node_id, &dir, config, params)?;

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| VektorError::WalIo(e.to_string()))?;
    let addr = listener
        .local_addr()
        .map_err(|e| VektorError::WalIo(e.to_string()))?
        .to_string();
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);

    let handle = tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(ShardReplicaRpcServer::new(svc))
            .serve_with_incoming(incoming)
            .await;
    });

    Ok((addr, dir, handle))
}

async fn spawn_chaos_coord(
    coord: CoordinatorService,
) -> Result<(String, JoinHandle<()>), VektorError> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| VektorError::WalIo(e.to_string()))?;
    let addr = listener
        .local_addr()
        .map_err(|e| VektorError::WalIo(e.to_string()))?
        .to_string();
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);

    let handle = tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(TopologyRpcServer::new(coord))
            .serve_with_incoming(incoming)
            .await;
    });

    Ok((addr, handle))
}
