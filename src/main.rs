#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::time::Instant;

use clap::{Parser, Subcommand};
use vektordist_core::exact::{ExactIndex, SeededRng};
use vektordist_core::{ClusterConfig, DistanceMetric};

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
    },
    /// Run a single shard replica node process.
    Node {
        #[arg(long)]
        id: u32,
        #[arg(long, default_value = "127.0.0.1:0")]
        listen: String,
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Demo {
            nodes,
            vectors,
            dimensions,
            replication_factor,
            seed,
        } => {
            let cfg = ClusterConfig {
                dimensions,
                replication_factor,
                ..ClusterConfig::default()
            };
            cfg.validate()?;
            println!(
                "vektordist demo scaffold ready (nodes={nodes}, vectors={vectors}, dims={dimensions}, rf={replication_factor}, seed={seed}); runtime lands in T6-T7."
            );
        }
        Commands::Node { id, listen } => {
            println!("vektordist node {id} on {listen} (runtime lands in T6).");
        }
        Commands::Coord { listen } => {
            println!("vektordist coord on {listen} (runtime lands in T6).");
        }
        Commands::Bench {
            vectors,
            dimensions,
            seed,
            queries,
            top_k,
        } => {
            run_exact_baseline(vectors, dimensions, seed, queries, top_k)?;
        }
    }
    Ok(())
}

fn run_exact_baseline(
    vectors: usize,
    dimensions: usize,
    seed: u64,
    queries: usize,
    top_k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let build_start = Instant::now();
    let mut index = ExactIndex::with_capacity(dimensions, vectors, DistanceMetric::Cosine)?;
    let mut rng = SeededRng::new(seed);
    let mut buf = vec![0.0_f32; dimensions];
    for id in 0..vectors {
        rng.fill_unit_vector(&mut buf);
        index.upsert(id as u64, &buf)?;
    }
    let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;

    let mut latencies_ms = Vec::with_capacity(queries);
    let mut query_buf = vec![0.0_f32; dimensions];
    let mut last_top_id = 0_u64;
    for _ in 0..queries {
        rng.fill_unit_vector(&mut query_buf);
        let q_start = Instant::now();
        let hits = index.search(&query_buf, top_k)?;
        let elapsed_ms = q_start.elapsed().as_secs_f64() * 1000.0;
        if let Some(first) = hits.first() {
            last_top_id = first.id;
        }
        latencies_ms.push(elapsed_ms);
    }

    latencies_ms.sort_by(f64::total_cmp);
    let p50 = percentile(&latencies_ms, 50.0);
    let p95 = percentile(&latencies_ms, 95.0);
    let mean = if latencies_ms.is_empty() {
        0.0
    } else {
        latencies_ms.iter().sum::<f64>() / (latencies_ms.len() as f64)
    };
    let arena_mb = (index.arena().memory_bytes() as f64) / (1024.0 * 1024.0);

    println!("=== VektorDist M1.1 Exact Cosine Baseline ===");
    println!("vectors={vectors} dims={dimensions} seed={seed} queries={queries} top_k={top_k}");
    println!("build_ms={build_ms:.2} arena_mem_mb={arena_mb:.2}");
    println!("exact_scan_ms: mean={mean:.2} p50={p50:.2} p95={p95:.2} (last_top_id={last_top_id})");
    Ok(())
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((pct / 100.0) * ((sorted.len() - 1) as f64)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
