#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};
use tokio::net::TcpListener;
use tonic::transport::Server;
use vektordist::cli::{percentile, run_node_server};
use vektordist::tui::run_demo_ui;
use vektordist_cluster::coord::CoordinatorService;
use vektordist_core::exact::{ExactIndex, SeededRng};
use vektordist_core::hnsw::{recall_at_k, HnswIndex, HnswParams};
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
    /// Run a single shard replica node process.
    Node {
        #[arg(long)]
        id: u32,
        #[arg(long, default_value = "127.0.0.1:0")]
        listen: String,
        #[arg(long, default_value = "/tmp/vektordist-node")]
        data_dir: PathBuf,
        #[arg(long, default_value_t = 768)]
        dimensions: usize,
        #[arg(long, default_value_t = 16)]
        num_shards: u32,
        #[arg(long, default_value_t = 2)]
        write_quorum: u8,
    },
    /// Run a standalone M1 authoritative topology coordinator.
    Coord {
        #[arg(long, default_value = "127.0.0.1:50051")]
        listen: String,
    },
    /// Run local indexing and search benchmarks.
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
            id,
            listen,
            data_dir,
            dimensions,
            num_shards,
            write_quorum,
        } => {
            run_node_server(id, &listen, &data_dir, dimensions, num_shards, write_quorum).await?;
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
    let mut query_vecs = Vec::with_capacity(queries);
    let mut exact_truth = Vec::with_capacity(queries);
    for _ in 0..queries {
        let mut q = vec![0.0_f32; dimensions];
        rng.fill_unit_vector(&mut q);
        let q_start = Instant::now();
        let hits = exact.search(&q, top_k)?;
        exact_latencies_ms.push(q_start.elapsed().as_secs_f64() * 1000.0);
        query_vecs.push(q);
        exact_truth.push(hits);
    }

    exact_latencies_ms.sort_by(f64::total_cmp);
    let exact_p50 = percentile(&exact_latencies_ms, 50.0);
    let exact_p95 = percentile(&exact_latencies_ms, 95.0);
    let arena_mb = (exact.arena().memory_bytes() as f64) / (1024.0 * 1024.0);

    println!("=== VektorDist M1.1-M1.2 Benchmark ===");
    println!("vectors={vectors} dims={dimensions} seed={seed} queries={queries} top_k={top_k}");
    println!("exact: build_ms={exact_build_ms:.2} arena_mem_mb={arena_mb:.2} p50_ms={exact_p50:.2} p95_ms={exact_p95:.2}");

    // Also run single-shard HNSW sample (capped at 5,000 vectors in CLI bench if vectors > 5,000 for quick CLI feedback, or full vectors when <= 5,000).
    let hnsw_sample = vectors.min(5_000);
    let hnsw_start = Instant::now();
    let mut hnsw = HnswIndex::with_capacity(
        dimensions,
        hnsw_sample,
        DistanceMetric::Cosine,
        HnswParams::default(),
    )?;
    let mut exact_sub = ExactIndex::with_capacity(dimensions, hnsw_sample, DistanceMetric::Cosine)?;
    let mut sub_rng = SeededRng::new(seed);
    let num_clusters = 16;
    let mut centroids = vec![vec![0.0_f32; dimensions]; num_clusters];
    for c in &mut centroids {
        sub_rng.fill_unit_vector(c);
    }
    for id in 0..hnsw_sample {
        let c = &centroids[id % num_clusters];
        for (d, &cv) in buf.iter_mut().zip(c.iter()) {
            *d = cv + 0.30 * sub_rng.next_f32_signed();
        }
        exact_sub.upsert(id as u64, &buf)?;
        hnsw.upsert(id as u64, &buf)?;
    }
    let hnsw_build_ms = hnsw_start.elapsed().as_secs_f64() * 1000.0;
    let mut hnsw_lat = Vec::with_capacity(queries);
    let mut recall_sum = 0.0_f64;
    for q_idx in 0..queries {
        let c = &centroids[q_idx % num_clusters];
        for (d, &cv) in buf.iter_mut().zip(c.iter()) {
            *d = cv + 0.30 * sub_rng.next_f32_signed();
        }
        let truth = exact_sub.search(&buf, top_k)?;
        let t0 = Instant::now();
        let approx = hnsw.search(&buf, top_k)?;
        hnsw_lat.push(t0.elapsed().as_secs_f64() * 1000.0);
        recall_sum += recall_at_k(&truth, &approx, top_k);
    }
    hnsw_lat.sort_by(f64::total_cmp);
    let hnsw_p50 = percentile(&hnsw_lat, 50.0);
    let hnsw_p95 = percentile(&hnsw_lat, 95.0);
    let hnsw_recall = recall_sum / (queries as f64);
    let hnsw_mb = (hnsw.memory_bytes() as f64) / (1024.0 * 1024.0);
    println!(
        "hnsw (n={hnsw_sample}): build_ms={hnsw_build_ms:.2} mem_mb={hnsw_mb:.2} p50_ms={hnsw_p50:.2} p95_ms={hnsw_p95:.2} recall@{top_k}={:.2}%",
        hnsw_recall * 100.0
    );

    Ok(())
}
