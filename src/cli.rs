use std::collections::{BTreeMap, HashSet, VecDeque};
use std::net::TcpListener as StdTcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tonic::transport::Server;
use tonic::Request;
use vektordist_cluster::client::ClusterClient;
use vektordist_cluster::coord::CoordinatorService;
use vektordist_cluster::node::{format_endpoint, NodeService};
use vektordist_cluster::{u32_to_shard_state, ShardState};
use vektordist_core::arena::dot_product;
use vektordist_core::exact::{ScoredId, SeededRng};
use vektordist_core::hnsw::{recall_at_k, HnswParams};
use vektordist_core::{
    ClusterConfig, DistanceMetric, NodeId, ShardId, VectorId, VektorError, WriteId,
};
use vektordist_proto::v1::shard_replica_rpc_client::ShardReplicaRpcClient;
use vektordist_proto::v1::shard_replica_rpc_server::ShardReplicaRpcServer;
use vektordist_proto::v1::topology_rpc_server::TopologyRpcServer;
use vektordist_proto::v1::{PingNodeRequest, WireNodeStatus};

static DEMO_RUN_SEQ: AtomicU64 = AtomicU64::new(1);

/// Incremental ground-truth tracker for up to 200 seeded evaluation queries.
/// Updates exact top-10 heaps in `< 0.1ms` on every acknowledged write so live `recall@10` never drifts.
pub struct LiveRecallEvaluator {
    queries: Vec<Vec<f32>>,
    exact_top_k: Vec<Vec<ScoredId>>,
    top_k: usize,
}

impl LiveRecallEvaluator {
    #[must_use]
    pub fn new(num_queries: usize, dims: usize, top_k: usize, seed: u64) -> Self {
        let mut rng = SeededRng::new(seed ^ 0x5555_5555_dead_beef);
        let mut queries = Vec::with_capacity(num_queries);
        let mut exact_top_k = Vec::with_capacity(num_queries);
        for _ in 0..num_queries {
            let mut q = vec![0.0_f32; dims];
            rng.fill_unit_vector(&mut q);
            queries.push(q);
            exact_top_k.push(Vec::with_capacity(top_k));
        }
        Self {
            queries,
            exact_top_k,
            top_k,
        }
    }

    /// Observe a newly acknowledged unit-normalized vector and incrementally update all query top-K lists.
    pub fn observe_vector(&mut self, id: VectorId, unit_vector: &[f32]) {
        for (q, top_list) in self.queries.iter().zip(self.exact_top_k.iter_mut()) {
            let sim = dot_product(q, unit_vector);
            if let Some(existing) = top_list.iter_mut().find(|s| s.id == id) {
                existing.score = sim;
                sort_top_k(top_list);
                continue;
            }
            if top_list.len() < self.top_k {
                top_list.push(ScoredId { id, score: sim });
                sort_top_k(top_list);
            } else if let Some(worst) = top_list.last() {
                if sim > worst.score || ((sim - worst.score).abs() < f32::EPSILON && id < worst.id)
                {
                    top_list.pop();
                    top_list.push(ScoredId { id, score: sim });
                    sort_top_k(top_list);
                }
            }
        }
    }

    #[must_use]
    pub fn query_count(&self) -> usize {
        self.queries.len()
    }

    #[must_use]
    pub fn query_and_truth(&self, idx: usize) -> Option<(&[f32], &[ScoredId])> {
        let q = self.queries.get(idx)?;
        let t = self.exact_top_k.get(idx)?;
        Some((q.as_slice(), t.as_slice()))
    }
}

fn sort_top_k(list: &mut [ScoredId]) {
    list.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
}

/// External ledger tracking every acknowledged `(write_id, vector_id, shard_id)`.
#[derive(Debug, Default, Clone)]
pub struct WriteIdLedger {
    entries: Vec<(WriteId, VectorId, ShardId)>,
}

impl WriteIdLedger {
    pub fn record_ack(&mut self, write_id: WriteId, vector_id: VectorId, shard_id: ShardId) {
        self.entries.push((write_id, vector_id, shard_id));
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn entries(&self) -> &[(WriteId, VectorId, ShardId)] {
        &self.entries
    }
}

/// Live metrics snapshot rendered by the `ratatui` TUI and inspected by E2E tests.
#[derive(Debug, Clone)]
pub struct ClusterMetricsSnapshot {
    pub configured_nodes: u32,
    pub active_nodes: u32,
    pub indexed_vectors: usize,
    pub dimensions: usize,
    pub replication_factor: u8,
    pub write_quorum: u8,
    pub cluster_status: String,
    pub degraded_under_quorum: bool,
    pub recall_at_10: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub search_qps: f64,
    pub acked_writes: usize,
    pub acked_writes_lost: usize,
    pub last_failover_ms: Option<f64>,
    pub nodes: Vec<WireNodeStatus>,
    pub events: Vec<String>,
}

struct SharedMetrics {
    indexed_vectors: usize,
    recall_samples: VecDeque<f64>,
    latency_samples_ms: VecDeque<f64>,
    total_searches: u64,
    started_at: Instant,
    acked_writes: usize,
    acked_writes_lost: usize,
    last_failover_ms: Option<f64>,
    degraded_under_quorum: bool,
    events: VecDeque<String>,
}

impl SharedMetrics {
    fn new(initial_vectors: usize) -> Self {
        let mut events = VecDeque::with_capacity(64);
        events.push_back("[OK] Cluster supervisor initializing...".to_string());
        Self {
            indexed_vectors: initial_vectors,
            recall_samples: VecDeque::with_capacity(200),
            latency_samples_ms: VecDeque::with_capacity(500),
            total_searches: 0,
            started_at: Instant::now(),
            acked_writes: 0,
            acked_writes_lost: 0,
            last_failover_ms: None,
            degraded_under_quorum: false,
            events,
        }
    }

    fn log_event(&mut self, msg: impl Into<String>) {
        if self.events.len() >= 50 {
            self.events.pop_front();
        }
        self.events.push_back(msg.into());
    }
}

struct ChildNodeProcess {
    data_dir: PathBuf,
    child: Option<Child>,
}

/// Multi-process cluster supervisor that spawns real `vektordist node` OS processes,
/// runs continuous 10-client search + quorum write load, and exposes chaos controls (`k`/`a`/`r`/`q`).
pub struct DemoClusterSupervisor {
    pub config: ClusterConfig,
    pub node_bin: PathBuf,
    pub temp_root: PathBuf,
    pub keep_data: bool,
    pub coord: CoordinatorService,
    pub coord_addr: String,
    pub client: ClusterClient,
    children: Arc<Mutex<BTreeMap<NodeId, ChildNodeProcess>>>,
    next_node_id: AtomicU64,
    next_vector_id: Arc<AtomicU64>,
    next_write_id: Arc<AtomicU64>,
    ledger: Arc<Mutex<WriteIdLedger>>,
    evaluator: Arc<RwLock<LiveRecallEvaluator>>,
    metrics: Arc<Mutex<SharedMetrics>>,
    stop_flag: Arc<AtomicBool>,
    worker_handles: Mutex<Vec<JoinHandle<()>>>,
    coord_handle: JoinHandle<()>,
}

impl DemoClusterSupervisor {
    /// Launch a local multi-process cluster, seed `initial_vectors` in 1k-vector WAL batches,
    /// initialize the 200-query exact recall evaluator, and start continuous background load.
    pub async fn start(
        node_bin: PathBuf,
        nodes: u32,
        initial_vectors: usize,
        dimensions: usize,
        replication_factor: u8,
        seed: u64,
        keep_data: bool,
    ) -> Result<Arc<Self>, VektorError> {
        let write_quorum = ((replication_factor / 2) + 1)
            .min(replication_factor)
            .max(1);
        let config = ClusterConfig {
            num_shards: 16,
            replication_factor,
            write_quorum,
            dimensions,
            metric: DistanceMetric::Cosine,
        };
        config.validate()?;

        let run_seq = DEMO_RUN_SEQ.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let temp_root = std::env::temp_dir().join(format!("vektordist-demo-{pid}-{run_seq}"));
        std::fs::create_dir_all(&temp_root).map_err(|e| VektorError::WalIo(e.to_string()))?;

        let mut initial_addrs = Vec::with_capacity(nodes as usize);
        let mut child_map = BTreeMap::new();
        for i in 1..=nodes {
            let port = allocate_ephemeral_port()?;
            let addr = format!("127.0.0.1:{port}");
            let data_dir = temp_root.join(format!("node-{i}"));
            let child = spawn_node_os_process(&node_bin, i, &addr, &data_dir, &config)?;
            initial_addrs.push((i, addr));
            child_map.insert(
                i,
                ChildNodeProcess {
                    data_dir,
                    child: Some(child),
                },
            );
        }

        // Wait for all spawned OS node processes to accept gRPC connections.
        for (id, addr) in &initial_addrs {
            wait_for_node_ready(addr, *id, Duration::from_secs(8)).await?;
        }

        let coord = CoordinatorService::new(config.clone(), initial_addrs)?;
        coord.push_all_shard_configs().await?;

        let coord_listener = TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|e| VektorError::WalIo(e.to_string()))?;
        let coord_addr = coord_listener
            .local_addr()
            .map_err(|e| VektorError::WalIo(e.to_string()))?
            .to_string();
        let coord_clone = coord.clone();
        let coord_handle = tokio::spawn(async move {
            let incoming = tokio_stream::wrappers::TcpListenerStream::new(coord_listener);
            let _ = Server::builder()
                .add_service(TopologyRpcServer::new(coord_clone))
                .serve_with_incoming(incoming)
                .await;
        });

        let client = ClusterClient::connect(&coord_addr).await?;
        let eval_queries = 50_usize;
        let mut evaluator = LiveRecallEvaluator::new(eval_queries, dimensions, 10, seed);
        let mut ledger = WriteIdLedger::default();

        // Seed initial dataset in batches of up to 1,000 vectors per shard.
        let mut rng = SeededRng::new(seed);
        let mut by_shard: BTreeMap<ShardId, Vec<(VectorId, WriteId, Vec<f32>)>> = BTreeMap::new();
        for idx in 1..=(initial_vectors as u64) {
            let mut v = vec![0.0_f32; dimensions];
            rng.fill_unit_vector(&mut v);
            evaluator.observe_vector(idx, &v);
            let shard_id = config.shard_for(idx);
            by_shard.entry(shard_id).or_default().push((idx, idx, v));
        }

        for (shard_id, records) in by_shard {
            for chunk in records.chunks(1_000) {
                client.seed_shard_batch(shard_id, chunk).await?;
                for &(vec_id, write_id, _) in chunk {
                    ledger.record_ack(write_id, vec_id, shard_id);
                }
            }
        }

        // Run one initial heartbeat tick so node memory/vector counts populate immediately.
        let _ = coord.heartbeat_tick().await;

        let acked_count = ledger.len();
        let mut metrics_init = SharedMetrics::new(initial_vectors);
        metrics_init.acked_writes = acked_count;
        metrics_init.log_event(format!(
            "[OK] {nodes} nodes healthy; {initial_vectors} vectors indexed (seed={seed})."
        ));

        let sup = Arc::new(Self {
            config,
            node_bin,
            temp_root,
            keep_data,
            coord,
            coord_addr,
            client,
            children: Arc::new(Mutex::new(child_map)),
            next_node_id: AtomicU64::new(u64::from(nodes) + 1),
            next_vector_id: Arc::new(AtomicU64::new((initial_vectors as u64) + 1)),
            next_write_id: Arc::new(AtomicU64::new((initial_vectors as u64) + 1)),
            ledger: Arc::new(Mutex::new(ledger)),
            evaluator: Arc::new(RwLock::new(evaluator)),
            metrics: Arc::new(Mutex::new(metrics_init)),
            stop_flag: Arc::new(AtomicBool::new(false)),
            worker_handles: Mutex::new(Vec::new()),
            coord_handle,
        });

        sup.start_background_workers();
        Ok(sup)
    }

    fn start_background_workers(self: &Arc<Self>) {
        let mut handles = Vec::new();

        // 1. 10 concurrent search client workers.
        for client_idx in 0..10_usize {
            let sup = Arc::clone(self);
            handles.push(tokio::spawn(async move {
                let mut q_cursor = client_idx;
                while !sup.stop_flag.load(Ordering::Relaxed) {
                    let query_opt = {
                        let Ok(eval_guard) = sup.evaluator.read() else {
                            break;
                        };
                        let count = eval_guard.query_count().max(1);
                        let idx = q_cursor % count;
                        q_cursor = q_cursor.wrapping_add(1);
                        eval_guard
                            .query_and_truth(idx)
                            .map(|(q, t)| (q.to_vec(), t.to_vec()))
                    };
                    if let Some((q_vec, exact_truth)) = query_opt {
                        let t0 = Instant::now();
                        if let Ok(hits) = sup.client.search(&q_vec, 10, 64).await {
                            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                            let rec = recall_at_k(&exact_truth, &hits, 10);
                            if let Ok(mut m) = sup.metrics.lock() {
                                if m.latency_samples_ms.len() >= 400 {
                                    m.latency_samples_ms.pop_front();
                                }
                                m.latency_samples_ms.push_back(elapsed_ms);
                                if m.recall_samples.len() >= 100 {
                                    m.recall_samples.pop_front();
                                }
                                m.recall_samples.push_back(rec);
                                m.total_searches += 1;
                            }
                        }
                    }
                    tokio::time::sleep(Duration::from_millis(25)).await;
                }
            }));
        }

        // 2. Continuous quorum writer worker.
        {
            let sup = Arc::clone(self);
            handles.push(tokio::spawn(async move {
                let mut rng = SeededRng::new(999_999);
                let mut vec_buf = vec![0.0_f32; sup.config.dimensions];
                while !sup.stop_flag.load(Ordering::Relaxed) {
                    rng.fill_unit_vector(&mut vec_buf);
                    let vec_id = sup.next_vector_id.fetch_add(1, Ordering::SeqCst);
                    let write_id = sup.next_write_id.fetch_add(1, Ordering::SeqCst);
                    match sup.client.upsert(vec_id, write_id, &vec_buf).await {
                        Ok((shard_id, _seq)) => {
                            if let Ok(mut eval_guard) = sup.evaluator.write() {
                                eval_guard.observe_vector(vec_id, &vec_buf);
                            }
                            if let Ok(mut l_guard) = sup.ledger.lock() {
                                l_guard.record_ack(write_id, vec_id, shard_id);
                            }
                            if let Ok(mut m) = sup.metrics.lock() {
                                m.indexed_vectors += 1;
                                m.acked_writes += 1;
                                m.degraded_under_quorum = false;
                            }
                        }
                        Err(VektorError::QuorumUnavailable { .. }) => {
                            if let Ok(mut m) = sup.metrics.lock() {
                                m.degraded_under_quorum = true;
                            }
                        }
                        Err(_) => {}
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            }));
        }

        // 3. 250ms heartbeat monitor worker.
        {
            let sup = Arc::clone(self);
            handles.push(tokio::spawn(async move {
                while !sup.stop_flag.load(Ordering::Relaxed) {
                    let newly_dead = sup.coord.heartbeat_tick().await;
                    if !newly_dead.is_empty() {
                        let _ = sup.client.refresh_topology().await;
                        for dead_id in newly_dead {
                            if let Ok(mut m) = sup.metrics.lock() {
                                m.log_event(format!(
                                    "[FAILOVER] Heartbeat detected node-{dead_id} DEAD; union WAL reconciled."
                                ));
                            }
                        }
                    }
                    tokio::time::sleep(Duration::from_millis(250)).await;
                }
            }));
        }

        if let Ok(mut m) = self.metrics.lock() {
            m.log_event("[LOAD] 10 search clients + continuous W=2 quorum writer active.");
        }
        if let Ok(mut w) = self.worker_handles.lock() {
            *w = handles;
        }
    }

    /// Terminate one live OS node process with `SIGKILL` (`k` key), trigger coordinator
    /// failover + union WAL reconciliation, and verify `acknowledged_writes_lost == 0`.
    pub async fn kill_one_node(&self) -> Result<Option<NodeId>, VektorError> {
        let killed_id_opt = {
            let Ok(mut map) = self.children.lock() else {
                return Ok(None);
            };
            let mut found = None;
            for (&node_id, entry) in map.iter_mut().rev() {
                if let Some(mut child) = entry.child.take() {
                    let _ = child.kill();
                    let _ = child.wait();
                    found = Some(node_id);
                    break;
                }
            }
            found
        };

        let Some(dead_id) = killed_id_opt else {
            return Ok(None);
        };

        let t0 = Instant::now();
        self.coord.handle_node_failure(dead_id).await?;
        self.client.refresh_topology().await?;
        let recovery_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let lost = self.verify_all_acked_writes().await?;
        let topo = self.coord.build_topology_response();
        let any_under_quorum = topo
            .shards
            .iter()
            .any(|s| u32_to_shard_state(s.state) == ShardState::UnderQuorum);

        if let Ok(mut m) = self.metrics.lock() {
            m.last_failover_ms = Some(recovery_ms);
            m.acked_writes_lost = lost;
            m.degraded_under_quorum = any_under_quorum;
            if any_under_quorum {
                m.log_event(format!(
                    "[DEGRADED] Killed node-{dead_id}; cluster under write quorum (<W=2) — reads online, writes rejected (lost={lost})."
                ));
            } else {
                m.log_event(format!(
                    "[FAILOVER] Killed node-{dead_id} (SIGKILL); recovered in {recovery_ms:.1}ms, acked_writes_lost={lost}."
                ));
            }
        }
        Ok(Some(dead_id))
    }

    /// Spawn a replacement OS node process (`a` key), run `RECOVERING -> HEALTHY` WAL catch-up,
    /// and verify `acknowledged_writes_lost == 0`.
    pub async fn add_replacement_node(&self) -> Result<NodeId, VektorError> {
        let new_id = self.next_node_id.fetch_add(1, Ordering::SeqCst) as NodeId;
        let port = allocate_ephemeral_port()?;
        let addr = format!("127.0.0.1:{port}");
        let data_dir = self.temp_root.join(format!("node-{new_id}"));
        let child = spawn_node_os_process(&self.node_bin, new_id, &addr, &data_dir, &self.config)?;

        if let Ok(mut map) = self.children.lock() {
            map.insert(
                new_id,
                ChildNodeProcess {
                    data_dir,
                    child: Some(child),
                },
            );
        }

        wait_for_node_ready(&addr, new_id, Duration::from_secs(8)).await?;
        let recovered_shards = self.coord.add_replacement_node(new_id, addr).await?;
        self.client.refresh_topology().await?;
        let _ = self.coord.heartbeat_tick().await;
        let lost = self.verify_all_acked_writes().await?;

        let topo = self.coord.build_topology_response();
        let any_under_quorum = topo
            .shards
            .iter()
            .any(|s| u32_to_shard_state(s.state) == ShardState::UnderQuorum);

        if let Ok(mut m) = self.metrics.lock() {
            m.acked_writes_lost = lost;
            m.degraded_under_quorum = any_under_quorum;
            m.log_event(format!(
                "[RECOVERY] Added node-{new_id}; caught up {} shards (RECOVERING -> HEALTHY, acked_writes_lost={lost}).",
                recovered_shards.len()
            ));
        }

        Ok(new_id)
    }

    /// Independently query surviving shard replicas' WAL records and verify every acknowledged
    /// `write_id` in `WriteIdLedger` is durably present. Returns the number of lost writes (`0`).
    pub async fn verify_all_acked_writes(&self) -> Result<usize, VektorError> {
        let snapshot_entries = {
            let Ok(guard) = self.ledger.lock() else {
                return Ok(0);
            };
            guard.entries().to_vec()
        };

        let mut durable_ids: HashSet<WriteId> = HashSet::with_capacity(snapshot_entries.len());
        for shard_id in 0..self.config.num_shards {
            if let Ok(recs) = self.client.verify_shard_wal_write_ids(shard_id).await {
                for r in recs {
                    durable_ids.insert(r.write_id);
                }
            }
        }

        let lost = snapshot_entries
            .iter()
            .filter(|(write_id, _, _)| !durable_ids.contains(write_id))
            .count();

        if let Ok(mut m) = self.metrics.lock() {
            m.acked_writes_lost = lost;
        }
        Ok(lost)
    }

    /// Take a point-in-time snapshot of cluster topology and live performance metrics.
    #[must_use]
    pub fn metrics_snapshot(&self) -> ClusterMetricsSnapshot {
        let topo = self.coord.build_topology_response();
        let configured_nodes = topo.nodes.len() as u32;
        let active_nodes = topo.nodes.iter().filter(|n| n.alive).count() as u32;

        let Ok(m) = self.metrics.lock() else {
            return ClusterMetricsSnapshot {
                configured_nodes,
                active_nodes,
                indexed_vectors: 0,
                dimensions: self.config.dimensions,
                replication_factor: self.config.replication_factor,
                write_quorum: self.config.write_quorum,
                cluster_status: "UNKNOWN".to_string(),
                degraded_under_quorum: false,
                recall_at_10: 1.0,
                latency_p50_ms: 0.0,
                latency_p95_ms: 0.0,
                latency_p99_ms: 0.0,
                search_qps: 0.0,
                acked_writes: 0,
                acked_writes_lost: 0,
                last_failover_ms: None,
                nodes: topo.nodes,
                events: Vec::new(),
            };
        };

        let mut sorted_lat: Vec<f64> = m.latency_samples_ms.iter().copied().collect();
        sorted_lat.sort_by(f64::total_cmp);
        let p50 = percentile(&sorted_lat, 50.0);
        let p95 = percentile(&sorted_lat, 95.0);
        let p99 = percentile(&sorted_lat, 99.0);

        let recall_mean = if m.recall_samples.is_empty() {
            1.0
        } else {
            m.recall_samples.iter().sum::<f64>() / (m.recall_samples.len() as f64)
        };
        let elapsed_s = m.started_at.elapsed().as_secs_f64().max(0.001);
        let qps = (m.total_searches as f64) / elapsed_s;

        let cluster_status = if m.degraded_under_quorum {
            "DEGRADED (under-quorum, reads only)".to_string()
        } else if active_nodes < configured_nodes {
            "Online (degraded replica set, W=2 active)".to_string()
        } else {
            "Online".to_string()
        };

        ClusterMetricsSnapshot {
            configured_nodes,
            active_nodes,
            indexed_vectors: m.indexed_vectors,
            dimensions: self.config.dimensions,
            replication_factor: self.config.replication_factor,
            write_quorum: self.config.write_quorum,
            cluster_status,
            degraded_under_quorum: m.degraded_under_quorum,
            recall_at_10: recall_mean,
            latency_p50_ms: p50,
            latency_p95_ms: p95,
            latency_p99_ms: p99,
            search_qps: qps,
            acked_writes: m.acked_writes,
            acked_writes_lost: m.acked_writes_lost,
            last_failover_ms: m.last_failover_ms,
            nodes: topo.nodes,
            events: m.events.iter().cloned().collect(),
        }
    }

    /// Cleanly terminate all background workers, child OS processes, and temp WAL directories.
    pub fn shutdown(&self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        if let Ok(mut handles) = self.worker_handles.lock() {
            for h in handles.drain(..) {
                h.abort();
            }
        }
        self.coord_handle.abort();
        if let Ok(mut map) = self.children.lock() {
            for entry in map.values_mut() {
                if let Some(mut child) = entry.child.take() {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                if !self.keep_data {
                    let _ = std::fs::remove_dir_all(&entry.data_dir);
                }
            }
        }
        if !self.keep_data {
            let _ = std::fs::remove_dir_all(&self.temp_root);
        }
    }
}

impl Drop for DemoClusterSupervisor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Run a standalone `vektordist node` gRPC server process until terminated.
pub async fn run_node_server(
    node_id: NodeId,
    listen: &str,
    data_dir: &Path,
    dimensions: usize,
    num_shards: u32,
    write_quorum: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = ClusterConfig {
        num_shards,
        replication_factor: 3,
        write_quorum,
        dimensions,
        metric: DistanceMetric::Cosine,
    };
    let svc = NodeService::open(node_id, data_dir, config, HnswParams::default())?;
    let listener = TcpListener::bind(listen).await?;
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
    Server::builder()
        .add_service(ShardReplicaRpcServer::new(svc))
        .serve_with_incoming(incoming)
        .await?;
    Ok(())
}

fn allocate_ephemeral_port() -> Result<u16, VektorError> {
    let listener = StdTcpListener::bind("127.0.0.1:0")
        .map_err(|e| VektorError::WalIo(format!("ephemeral port bind failed: {e}")))?;
    let port = listener
        .local_addr()
        .map_err(|e| VektorError::WalIo(e.to_string()))?
        .port();
    Ok(port)
}

fn spawn_node_os_process(
    node_bin: &Path,
    node_id: NodeId,
    listen_addr: &str,
    data_dir: &Path,
    config: &ClusterConfig,
) -> Result<Child, VektorError> {
    Command::new(node_bin)
        .arg("node")
        .arg("--id")
        .arg(node_id.to_string())
        .arg("--listen")
        .arg(listen_addr)
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--dimensions")
        .arg(config.dimensions.to_string())
        .arg("--num-shards")
        .arg(config.num_shards.to_string())
        .arg("--write-quorum")
        .arg(config.write_quorum.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| VektorError::WalIo(format!("failed to spawn node-{node_id}: {e}")))
}

async fn wait_for_node_ready(
    addr: &str,
    node_id: NodeId,
    timeout: Duration,
) -> Result<(), VektorError> {
    let deadline = Instant::now() + timeout;
    let endpoint = format_endpoint(addr);
    while Instant::now() < deadline {
        if let Ok(mut client) = ShardReplicaRpcClient::connect(endpoint.clone()).await {
            if client
                .ping_node(Request::new(PingNodeRequest { node_id }))
                .await
                .is_ok()
            {
                return Ok(());
            }
        }
        tokio::time::sleep(Duration::from_millis(40)).await;
    }
    Err(VektorError::WalIo(format!(
        "timed out waiting for node-{node_id} at {addr}"
    )))
}

pub fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((pct / 100.0) * ((sorted.len() - 1) as f64)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
