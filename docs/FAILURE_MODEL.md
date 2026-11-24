# VektorDist Milestone 1 — Failure Model & Consistency Guarantees

This document specifies the exact failure modes, quorum durability rules, and explicit architectural boundaries supported in **VektorDist Milestone 1 (M1)**.

## 1. Replication & Quorum Durability (`RF=3`, `W=2`)

- **Shard Partitioning:** Every `VectorId` maps deterministically via SplitMix64 hashing (`ClusterConfig::shard_for`) to one of `16` logical shards.
- **Replica Placement:** Each shard has `RF=3` replicas (1 coordinator-assigned primary + 2 followers).
- **Primary-Ordered Writes:** Live writes route directly to the shard primary (`ClusterClient::upsert`) with `expected_epoch`. The primary assigns a gap-free monotonic `(shard_id, seq_no, write_id)`, writes a length-prefixed, IEEE CRC32-checksummed record to its local WAL, calls `sync_data()` (`fsync`), and replicates in parallel to both followers.
- **Quorum Acknowledgement (`W=2`):** A write is acknowledged to the caller **only after at least 2 replicas (the primary + at least 1 follower) confirm durable `fsync`** and apply the vector to their in-memory HNSW index.
- **Unacknowledged In-Flight Writes:** A write that is in flight when a node receives `SIGKILL` and has not yet returned a quorum acknowledgement may return an error or retry; every write that returns `Ok` is recorded in the external `WriteIdLedger` and must survive supported failures (`acknowledged_writes_lost == 0`).

## 2. Read Consistency & Failover Reconciliation

- **Steady-State Reads (`R=1`):** In steady state (`ShardState::Healthy` or `ShardState::DegradedOk`), scatter-gather search queries one healthy replica per shard (`R=1`), falling back automatically to another replica on timeout (`200ms`) or transport error. Under `W=2` of `RF=3`, the third follower may briefly lag by 1 write; the external Write-ID verifier and failover reconciliation inspect WAL records across surviving replicas.
- **Union/Anti-Entropy WAL Reconciliation (`ShardState::Reconciling`):** When a node fails (either triggered immediately via `k` in `vektordist demo` or detected within `~1.5s` via 3 missed `500ms` heartbeats), the coordinator increments the shard's `epoch` and runs a union WAL reconciliation pass across the 2 surviving replicas (`GetWalRecords` -> union merge by `seq_no` -> `ReplicateWal`) before promoting the new primary. Because `W=2` overlaps any 2-of-3 surviving replica subset, the union of the 2 survivors is mathematically guaranteed to contain 100% of acknowledged writes.
- **Epoch Fencing (`STATUS_WRONG_EPOCH`):** Every primary write and follower replication RPC carries `epoch: u64`. Any RPC bearing `epoch < local_epoch` is rejected with `STATUS_WRONG_EPOCH`, preventing a slow or stalled old primary from corrupting a newly promoted primary.

## 3. Supported vs Unsupported Failure Modes in Milestone 1

| Failure Scenario | Milestone 1 Support | Expected System Behavior |
|---|---|---|
| Single data-node `SIGKILL` (`k` key) | **Supported & Tested** | Detected in `<= 5s` (`~1.5s` heartbeat or immediate in demo); union WAL reconciliation runs across 2 survivors; search and `W=2` writes continue with `acknowledged_writes_lost == 0`. |
| Replacement node join (`a` key) | **Supported & Tested** | New node spawns (`ShardState::Recovering`), primary dual-streams live writes while coordinator streams historical WAL records; transitions to `ShardState::Healthy` at `lag == 0`. |
| Double data-node `SIGKILL` (`< W=2` survivors) | **Supported & Tested (Degraded Read-Only)** | Shards with `< 2` surviving replicas enter `ShardState::UnderQuorum`: search continues on the remaining replica, while writes are explicitly rejected with `QUORUM_UNAVAILABLE` (never falsely acknowledged). |
| Mid-write `SIGKILL` leaving truncated WAL tail | **Supported & Tested** | `WalLog::open` detects incomplete trailing header/payload at EOF, truncates the torn tail frame safely, and preserves all prior `fsync`ed records. |
| Mid-file WAL bit-rot / CRC32 mismatch | **Supported & Tested** | `WalLog::open` returns `Err(VektorError::WalCorruption { offset })` rather than silently skipping corrupt records. |
| Stale primary write after epoch bump | **Supported & Tested** | Rejected by primary and followers via `epoch < local_epoch` (`STATUS_WRONG_EPOCH`); `ClusterClient` refreshes topology and retries against the new primary. |
| Coordinator process crash / SPOF | **Deferred to M2** | M1 uses a single authoritative coordinator (`CoordinatorService`) behind the `TopologyStore` trait. Full Raft consensus for coordinator HA is scheduled for Milestone 2. |
| Asymmetric network partitions (split-brain) | **Deferred to M2** | Requires Raft quorum leases and majority partition fencing scheduled for Milestone 2. |
| Dynamic shard splitting / rebalancing | **Deferred to M2** | M1 uses a fixed 16-shard ring with replica catch-up; online shard migration is scheduled for Milestone 2. |
