# ADR-005: Point-in-Time Cluster Snapshots, CRC32 Restore Verification, and Production Packaging
- **Status:** Accepted
- **Owner:** Justin Mafie

## Context
Running a distributed sharded vector database in production requires deterministic disaster recovery (point-in-time backup and restore across all shards with bit-rot detection) alongside reproducible container, Kubernetes, cloud infrastructure, and multi-architecture binary release packaging.

## Decision
1. **Point-in-Time Multi-Shard Snapshots (`b"VDSN"`):**
   - `SnapshotManager::create_snapshot` acquires read locks across all shards on a `NodeService` in deterministic `shard_id` order (`0..num_shards`) so no write can interleave across shards during snapshot capture.
   - Each shard's durable `.wal` file is copied and `fsync`ed into the snapshot directory (`shard-XXXX.wal`), accompanied by a binary `manifest.bin` (`magic: b"VDSN"`, format version `1`, `node_id`, `ClusterConfig`, per-shard `(shard_id, epoch, last_seq_no, vector_count, wal_file_bytes, wal_crc32)`, and a trailing whole-manifest IEEE CRC32 checksum).
2. **Pre-Restore Integrity Verification (`verify_snapshot` & `restore_node_from_snapshot`):**
   - `verify_snapshot` validates the `b"VDSN"` magic, format version, whole-manifest CRC32, and every shard's `.wal` byte length and IEEE CRC32 checksum before any restore touches the target node directory.
   - Any 1-byte tamper or bit-rot in either `manifest.bin` or a shard `.wal` file is deterministically rejected with `VektorError::WalCorruption`.
   - `restore_node_from_snapshot` copies verified shard WALs into `target_node_dir`, replays them into `ShardEngine` HNSW indexes, restores each shard's leadership `epoch`, and verifies that reconstructed `last_seq_no` and live vector counts match `manifest.bin`.
3. **Production Deployment & Release Artifacts:**
   - Provide a multi-stage distroless `deploy/Dockerfile`, a 5-replica Kubernetes `StatefulSet` + Headless Service (`deploy/k8s/statefulset.yaml`), a 5-node GCP Compute Engine + `pd-ssd` Terraform module (`deploy/gcp/main.tf`), and a tag-triggered GitHub Actions release workflow (`.github/workflows/release.yml`) producing `aarch64-apple-darwin`, `x86_64-unknown-linux-gnu`, and `aarch64-unknown-linux-gnu` tarballs with SHA256 checksums.

## Alternatives Considered
- **Uncoordinated Per-Shard File Copy:** Copying `shard-XXXX.wal` files from disk without holding ordered shard read locks or recording expected `(last_seq_no, vector_count, wal_crc32)` in a signed manifest risks torn multi-shard state and silent bit-rot during restore.
- **Snapshotting Raw In-Memory HNSW Pointers:** Significantly larger on disk than compact WAL frames and couples backup portability to internal memory layout versions.

## Consequences & Failure Modes
- **Bit-Rot Detection Before Restore:** Corrupted backups fail fast during `verify_snapshot` with `VektorError::WalCorruption` before overwriting or initializing target node directories.
- **Deterministic HNSW Reconstruction:** Replaying the verified WAL frames deterministically reconstructs the exact HNSW graph, preserving 100% of acknowledged `write_id`s, `last_seq_no`, and top-`k` search rankings.
