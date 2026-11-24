#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use clap::{Parser, Subcommand};
use vektordist_core::ClusterConfig;

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
        } => {
            println!(
                "vektordist bench (vectors={vectors}, dims={dimensions}, seed={seed}; M1.0 baseline lands in T2)."
            );
        }
    }
    Ok(())
}
