# VektorDist

**VektorDist** is a laptop-first, distributed sharded HNSW vector search engine written in Rust from first principles.

Milestone 1 delivers a self-contained local 5-node cluster with an interactive terminal dashboard where you can ingest 100,000 768-dimensional vectors, kill a live node (`SIGKILL`), add a replacement node, and independently verify that acknowledged writes survive (`acknowledged_writes_lost == 0`) while distributed search recovers.

## Quickstart (Milestone 1 Target)

```bash
cargo install --path .
vektordist demo \
  --nodes 5 \
  --vectors 100000 \
  --dimensions 768 \
  --replication-factor 3 \
  --seed 42
```

## Workspace Structure

- `src/` — Root `vektordist` CLI binary (`demo`, `node`, `coord`, `bench`) and `ratatui` TUI
- `crates/vektordist-core` — Contiguous `VectorArena`, exact cosine/L2 search, HNSW + tombstones, CRC32 WAL
- `crates/vektordist-proto` — Packed little-endian `f32` wire codecs and checked-in Tonic gRPC stubs
- `crates/vektordist-cluster` — `TopologyStore` trait, M1 authoritative coordinator, `ShardEngine`, and `ClusterClient`

## Documentation & Standards

- **Engineering Standards:** [AGENTS.md](AGENTS.md)
- **Milestone 1 Design & Eng Review:** [docs/designs/vektordist-milestone-1.md](docs/designs/vektordist-milestone-1.md)
- **Architecture Decision Records:** [docs/adr/001-m1-workspace-and-coordinator.md](docs/adr/001-m1-workspace-and-coordinator.md)
- **Threat Model:** [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md)
