#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use clap::{Parser, Subcommand};
use tokio::net::TcpListener;
use tonic::transport::Server;
use vektordist::cli::{percentile, run_node_server};
use vektordist::config::{derive_master_secret, VektorNodeConfig};
use vektordist::tui::run_demo_ui;
use vektordist_cluster::backup::SnapshotManager;
use vektordist_cluster::client::ClusterClient;
use vektordist_cluster::coord::CoordinatorService;
use vektordist_cluster::gateway::{
    spawn_http_gateway, ChaosCampaign, ChaosStep, HttpGatewayServer,
};
use vektordist_cluster::node::NodeService;
use vektordist_cluster::security::{OrgId, SecurityGate, TenantQuotaPolicy, TenantRole};
use vektordist_core::adaptive::{AdaptiveSearchConfig, AdaptiveSearchController};
use vektordist_core::exact::{ExactIndex, SeededRng};
use vektordist_core::hnsw::{recall_at_k, HnswIndex, HnswParams};
use vektordist_core::pq_bq::BqIndex;
use vektordist_core::quant::{Sq8HnswIndex, Sq8Quantizer};
use vektordist_core::{ClusterConfig, DistanceMetric};
use vektordist_proto::v1::topology_rpc_server::TopologyRpcServer;

#[derive(Debug, Parser)]
#[command(
    name = "vektordist",
    version,
    about = "Laptop-first distributed sharded HNSW vector search engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Launch a local multi-process cluster with interactive chaos TUI.
    Demo {
        #[arg(long, default_value_t = 5)]
        nodes: u32,
        #[arg(long, default_value_t = 100_000)]
        vectors: usize,
        #[arg(long, default_value_t = 768)]
        dimensions: usize,
        #[arg(long, default_value_t = 3)]
        replication_factor: u8,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = false)]
        keep_data: bool,
        #[arg(long, default_value_t = false)]
        headless: bool,
    },
    /// Run a single shard replica node process (via `--config` file, `VEKTORDIST_*` env vars, or flags).
    Node {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        id: Option<u32>,
        #[arg(long)]
        listen: Option<String>,
        #[arg(long)]
        data_dir: Option<PathBuf>,
        #[arg(long)]
        dimensions: Option<usize>,
        #[arg(long)]
        num_shards: Option<u32>,
        #[arg(long)]
        write_quorum: Option<u8>,
    },
    /// Run a standalone M1 authoritative topology coordinator.
    Coord {
        #[arg(long, default_value = "127.0.0.1:50051")]
        listen: String,
    },
    /// Run local indexing and search benchmarks (Exact, HNSW f32, SQ8 u8, BQ 1-bit, and M6 Adaptive).
    Bench {
        #[arg(long, default_value_t = 100_000)]
        vectors: usize,
        #[arg(long, default_value_t = 768)]
        dimensions: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 20)]
        queries: usize,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
    },
    /// Manage point-in-time cluster snapshots (create, verify, restore).
    Snapshot {
        #[command(subcommand)]
        action: SnapshotAction,
    },
    /// Run the HTTP/1.1 REST (`/v1/upsert`, `/v1/search`, `/v1/delete`, `/healthz`) and Prometheus `/metrics` gateway.
    Gateway {
        #[arg(long, default_value = "127.0.0.1:50051")]
        coord: String,
        #[arg(long, default_value = "127.0.0.1:8080")]
        listen: String,
        #[arg(long)]
        master_secret: Option<String>,
    },
    /// Issue a tenant-scoped zero-trust `vdk.<key_id>.<hex_tag>` bearer token.
    Keygen {
        #[arg(long)]
        secret: Option<String>,
        #[arg(long, default_value = "default")]
        org: String,
        #[arg(long, default_value = "writer")]
        role: String,
        #[arg(long, default_value = "key-01")]
        key_id: String,
        #[arg(long, default_value_t = 1)]
        nonce: u64,
    },
    /// Run an automated multi-step Jepsen-style fault-injection & SLA campaign.
    Chaos {
        #[arg(long, default_value_t = 5)]
        nodes: u32,
        #[arg(long, default_value_t = 1_200)]
        vectors: usize,
        #[arg(long, default_value_t = 128)]
        dimensions: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
}

#[derive(Debug, Subcommand)]
enum SnapshotAction {
    /// Capture a point-in-time CRC32-verified snapshot from a node data directory.
    Create {
        #[arg(long, default_value_t = 1)]
        node_id: u32,
        #[arg(long)]
        node_dir: PathBuf,
        #[arg(long)]
        snapshot_dir: PathBuf,
        #[arg(long, default_value_t = 768)]
        dimensions: usize,
        #[arg(long, default_value_t = 16)]
        num_shards: u32,
    },
    /// Verify the manifest and per-shard IEEE CRC32 checksums of a snapshot directory.
    Verify {
        #[arg(long)]
        snapshot_dir: PathBuf,
    },
    /// Restore a verified snapshot into a target node directory and rebuild HNSW indexes.
    Restore {
        #[arg(long)]
        snapshot_dir: PathBuf,
        #[arg(long)]
        target_dir: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Demo {
            nodes,
            vectors,
            dimensions,
            replication_factor,
            seed,
            keep_data,
            headless,
        } => {
            let exe = std::env::current_exe()?;
            run_demo_ui(
                exe,
                nodes,
                vectors,
                dimensions,
                replication_factor,
                seed,
                keep_data,
                headless,
            )
            .await?;
        }
        Commands::Node {
            config,
            id,
            listen,
            data_dir,
            dimensions,
            num_shards,
            write_quorum,
        } => {
            let mut node_cfg = match config {
                Some(ref path) => VektorNodeConfig::from_file(path)?,
                None => VektorNodeConfig::default(),
            };
            node_cfg.apply_env_overrides()?;
            if let Some(v) = id {
                node_cfg.node_id = v;
            }
            if let Some(v) = listen {
                node_cfg.listen_addr = v;
            }
            if let Some(v) = data_dir {
                node_cfg.data_dir = v;
            }
            if let Some(v) = dimensions {
                node_cfg.dimensions = v;
            }
            if let Some(v) = num_shards {
                node_cfg.num_shards = v;
            }
            if let Some(v) = write_quorum {
                node_cfg.write_quorum = v;
            }
            node_cfg.validate()?;
            run_node_server(
                node_cfg.node_id,
                &node_cfg.listen_addr,
                &node_cfg.data_dir,
                node_cfg.dimensions,
                node_cfg.num_shards,
                node_cfg.write_quorum,
            )
            .await?;
        }
        Commands::Coord { listen } => {
            let coord = CoordinatorService::new(
                ClusterConfig::default(),
                vec![(1, "127.0.0.1:50052".to_string())],
            )?;
            let listener = TcpListener::bind(&listen).await?;
            let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
            println!("vektordist coordinator listening on {listen}");
            Server::builder()
                .add_service(TopologyRpcServer::new(coord))
                .serve_with_incoming(incoming)
                .await?;
        }
        Commands::Bench {
            vectors,
            dimensions,
            seed,
            queries,
            top_k,
        } => {
            run_exact_and_hnsw_benchmark(vectors, dimensions, seed, queries, top_k)?;
        }
        Commands::Snapshot { action } => match action {
            SnapshotAction::Create {
                node_id,
                node_dir,
                snapshot_dir,
                dimensions,
                num_shards,
            } => {
                let config = ClusterConfig {
                    num_shards,
                    dimensions,
                    ..ClusterConfig::default()
                };
                let node = NodeService::open(node_id, &node_dir, config, HnswParams::default())?;
                let manifest = SnapshotManager::create_snapshot(&node, &snapshot_dir).await?;
                println!(
                    "Snapshot created at {}: node_id={} shards={}",
                    snapshot_dir.display(),
                    manifest.node_id,
                    manifest.shards.len()
                );
            }
            SnapshotAction::Verify { snapshot_dir } => {
                let report = SnapshotManager::verify_snapshot(&snapshot_dir)?;
                println!(
                    "Snapshot verified: shards={} vectors={} bytes={} corrupt_shards={:?}",
                    report.verified_shards,
                    report.total_vectors,
                    report.total_bytes,
                    report.corrupt_shards
                );
            }
            SnapshotAction::Restore {
                snapshot_dir,
                target_dir,
            } => {
                let restored = SnapshotManager::restore_node_from_snapshot(
                    &snapshot_dir,
                    &target_dir,
                    ClusterConfig::default(),
                    HnswParams::default(),
                )?;
                println!(
                    "Restored node-{} into {} ({} shards)",
                    restored.node_id,
                    target_dir.display(),
                    restored.shards.len()
                );
            }
        },
        Commands::Gateway {
            coord,
            listen,
            master_secret,
        } => {
            let client = ClusterClient::connect(&coord).await?;
            let secret_opt = master_secret
                .or_else(|| std::env::var("VEKTORDIST_MASTER_SECRET").ok())
                .filter(|s| !s.trim().is_empty());
            let gw = if let Some(secret_str) = secret_opt {
                let gate = SecurityGate::new(
                    derive_master_secret(&secret_str),
                    TenantQuotaPolicy::default(),
                );
                HttpGatewayServer::new(client).with_security_gate(Arc::new(Mutex::new(gate)))
            } else {
                HttpGatewayServer::new(client)
            };
            let (bound_addr, handle) = spawn_http_gateway(&listen, gw).await?;
            println!(
                "vektordist HTTP/1.1 REST & Prometheus gateway listening on http://{bound_addr} (upstream coord={coord})"
            );
            let _ = handle.await;
        }
        Commands::Keygen {
            secret,
            org,
            role,
            key_id,
            nonce,
        } => {
            let secret_raw = secret
                .or_else(|| std::env::var("VEKTORDIST_MASTER_SECRET").ok())
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| "vektordist-dev-master-secret".to_string());
            let parsed_role = match role.to_ascii_lowercase().as_str() {
                "reader" => TenantRole::Reader,
                "writer" => TenantRole::Writer,
                "admin" => TenantRole::Admin,
                _ => {
                    return Err("invalid --role: expected reader, writer, or admin".into());
                }
            };
            let org_id = OrgId::new(org)?;
            let mut gate = SecurityGate::new(
                derive_master_secret(&secret_raw),
                TenantQuotaPolicy::default(),
            );
            let (raw_token, record) = gate.issue_api_key(&key_id, org_id, parsed_role, nonce);
            println!("token={raw_token}");
            println!(
                "key_id={} org={} role={}",
                record.key_id,
                record.org_id.as_str(),
                record.role
            );
        }
        Commands::Chaos {
            nodes,
            vectors,
            dimensions,
            seed,
        } => {
            let node_count = nodes.max(3);
            let initial_nodes: Vec<u32> = (1..=node_count).collect();
            let config = ClusterConfig {
                num_shards: 8,
                replication_factor: 3,
                write_quorum: 2,
                dimensions,
                metric: DistanceMetric::Cosine,
            };
            let steps = vec![
                ChaosStep::SeedVectors {
                    count: vectors,
                    batch_size: 400,
                    seed,
                },
                ChaosStep::ConcurrentWorkload {
                    writes: 120,
                    reads: 40,
                    top_k: 10,
                    ef_search: 64,
                    seed: seed.wrapping_add(1),
                },
                ChaosStep::KillNode { node_id: 2 },
                ChaosStep::VerifyZeroWriteLoss,
                ChaosStep::AddReplacementNode {
                    node_id: node_count + 1,
                },
                ChaosStep::VerifyZeroWriteLoss,
                ChaosStep::AssertRecallAndLatencySla {
                    queries: 25,
                    top_k: 10,
                    ef_search: 64,
                    min_recall_at_10: 0.95,
                    max_p95_ms: 50.0,
                    seed: seed.wrapping_add(2),
                },
            ];
            let campaign = ChaosCampaign::new(config, initial_nodes, steps);
            let report = campaign.run().await?;
            println!("=== VektorDist M10 Chaos Campaign Report ===");
            for line in &report.step_logs {
                println!("  - {line}");
            }
            println!(
                "summary: steps={} acked_verified={} acked_lost={} recall@10={:.2}% p95_ms={:.2} sla_passed={}",
                report.steps_executed,
                report.acked_writes_verified,
                report.acked_writes_lost,
                report.measured_recall_at_10 * 100.0,
                report.measured_p95_ms,
                report.sla_passed
            );
            if !report.sla_passed {
                return Err("chaos campaign failed SLA verification".into());
            }
        }
    }
    Ok(())
}

fn run_exact_and_hnsw_benchmark(
    vectors: usize,
    dimensions: usize,
    seed: u64,
    queries: usize,
    top_k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let build_start = Instant::now();
    let mut exact = ExactIndex::with_capacity(dimensions, vectors, DistanceMetric::Cosine)?;
    let mut rng = SeededRng::new(seed);
    let mut buf = vec![0.0_f32; dimensions];
    for id in 0..vectors {
        rng.fill_unit_vector(&mut buf);
        exact.upsert(id as u64, &buf)?;
    }
    let exact_build_ms = build_start.elapsed().as_secs_f64() * 1000.0;

    let mut exact_latencies_ms = Vec::with_capacity(queries);
    for _ in 0..queries {
        let mut q = vec![0.0_f32; dimensions];
        rng.fill_unit_vector(&mut q);
        let q_start = Instant::now();
        let _ = exact.search(&q, top_k)?;
        exact_latencies_ms.push(q_start.elapsed().as_secs_f64() * 1000.0);
    }

    exact_latencies_ms.sort_by(f64::total_cmp);
    let exact_p50 = percentile(&exact_latencies_ms, 50.0);
    let exact_p95 = percentile(&exact_latencies_ms, 95.0);
    let arena_mb = (exact.arena().memory_bytes() as f64) / (1024.0 * 1024.0);

    println!("=== VektorDist M1–M10 Benchmark Suite ===");
    println!("vectors={vectors} dims={dimensions} seed={seed} queries={queries} top_k={top_k}");
    println!("exact (f32): build_ms={exact_build_ms:.2} arena_mem_mb={arena_mb:.2} p50_ms={exact_p50:.2} p95_ms={exact_p95:.2}");

    let hnsw_sample = vectors.min(5_000);
    let hnsw_start = Instant::now();
    let mut hnsw = HnswIndex::with_capacity(
        dimensions,
        hnsw_sample,
        DistanceMetric::Cosine,
        HnswParams::default(),
    )?;
    let mut sq8 = Sq8HnswIndex::with_capacity(
        Sq8Quantizer::for_unit_vectors(dimensions)?,
        hnsw_sample,
        DistanceMetric::Cosine,
        HnswParams::default(),
    )?;
    let mut bq = BqIndex::with_capacity(dimensions, hnsw_sample, DistanceMetric::Cosine)?;
    let mut exact_sub = ExactIndex::with_capacity(dimensions, hnsw_sample, DistanceMetric::Cosine)?;
    let mut sub_rng = SeededRng::new(seed);
    let num_clusters = (hnsw_sample / 12).max(16);
    let mut centroids = vec![vec![0.0_f32; dimensions]; num_clusters];
    for c in &mut centroids {
        sub_rng.fill_unit_vector(c);
    }
    let mut noise = vec![0.0_f32; dimensions];
    for id in 0..hnsw_sample {
        let c = &centroids[id % num_clusters];
        sub_rng.fill_unit_vector(&mut noise);
        for ((d, &cv), &nv) in buf.iter_mut().zip(c.iter()).zip(noise.iter()) {
            *d = 0.75 * cv + 0.25 * nv;
        }
        exact_sub.upsert(id as u64, &buf)?;
        hnsw.upsert(id as u64, &buf)?;
        sq8.upsert(id as u64, &buf)?;
        bq.upsert(id as u64, &buf)?;
    }
    let hnsw_build_ms = hnsw_start.elapsed().as_secs_f64() * 1000.0;

    let controller = AdaptiveSearchController::new(AdaptiveSearchConfig::default())?;
    let mut hnsw_lat = Vec::with_capacity(queries);
    let mut sq8_lat = Vec::with_capacity(queries);
    let mut bq_lat = Vec::with_capacity(queries);
    let mut adaptive_lat = Vec::with_capacity(queries);
    let mut recall_sum = 0.0_f64;
    let mut sq8_recall_sum = 0.0_f64;
    let mut bq_recall_sum = 0.0_f64;
    let mut adaptive_recall_sum = 0.0_f64;

    for q_idx in 0..queries {
        let c = &centroids[q_idx % num_clusters];
        sub_rng.fill_unit_vector(&mut noise);
        for ((d, &cv), &nv) in buf.iter_mut().zip(c.iter()).zip(noise.iter()) {
            *d = 0.75 * cv + 0.25 * nv;
        }
        let truth = exact_sub.search(&buf, top_k)?;

        let t0 = Instant::now();
        let approx = hnsw.search(&buf, top_k)?;
        hnsw_lat.push(t0.elapsed().as_secs_f64() * 1000.0);
        recall_sum += recall_at_k(&truth, &approx, top_k);

        let t1 = Instant::now();
        let sq8_hits = sq8.search_reranked_with_arena(&buf, top_k, 64, 3, Some(hnsw.arena()))?;
        sq8_lat.push(t1.elapsed().as_secs_f64() * 1000.0);
        sq8_recall_sum += recall_at_k(&truth, &sq8_hits, top_k);

        let t_bq = Instant::now();
        let bq_hits = bq.search_reranked(&buf, top_k, 10, exact_sub.arena())?;
        bq_lat.push(t_bq.elapsed().as_secs_f64() * 1000.0);
        bq_recall_sum += recall_at_k(&truth, &bq_hits, top_k);

        let t2 = Instant::now();
        let (adap_hits, _) = controller.search_adaptive(&hnsw, &buf, top_k)?;
        adaptive_lat.push(t2.elapsed().as_secs_f64() * 1000.0);
        adaptive_recall_sum += recall_at_k(&truth, &adap_hits, top_k);
    }
    hnsw_lat.sort_by(f64::total_cmp);
    sq8_lat.sort_by(f64::total_cmp);
    bq_lat.sort_by(f64::total_cmp);
    adaptive_lat.sort_by(f64::total_cmp);

    let hnsw_mb = (hnsw.memory_bytes() as f64) / (1024.0 * 1024.0);
    let sq8_mb = (sq8.memory_bytes() as f64) / (1024.0 * 1024.0);
    let bq_payload_kb = (bq.arena().vector_bytes() as f64) / 1024.0;
    println!(
        "hnsw (f32, n={hnsw_sample}): build_ms={hnsw_build_ms:.2} mem_mb={hnsw_mb:.2} p50_ms={:.2} p95_ms={:.2} recall@{top_k}={:.2}%",
        percentile(&hnsw_lat, 50.0),
        percentile(&hnsw_lat, 95.0),
        (recall_sum / (queries as f64)) * 100.0
    );
    println!(
        "sq8  (u8,  n={hnsw_sample}): mem_mb={sq8_mb:.2} p50_ms={:.2} p95_ms={:.2} recall@{top_k}={:.2}%",
        percentile(&sq8_lat, 50.0),
        percentile(&sq8_lat, 95.0),
        (sq8_recall_sum / (queries as f64)) * 100.0
    );
    println!(
        "bq   (1b,  n={hnsw_sample}): payload_kb={bq_payload_kb:.1} (32x) p50_ms={:.2} p95_ms={:.2} recall@{top_k}={:.2}%",
        percentile(&bq_lat, 50.0),
        percentile(&bq_lat, 95.0),
        (bq_recall_sum / (queries as f64)) * 100.0
    );
    println!(
        "adaptive (M6, n={hnsw_sample}): p50_ms={:.2} p95_ms={:.2} recall@{top_k}={:.2}%",
        percentile(&adaptive_lat, 50.0),
        percentile(&adaptive_lat, 95.0),
        (adaptive_recall_sum / (queries as f64)) * 100.0
    );

    Ok(())
}
