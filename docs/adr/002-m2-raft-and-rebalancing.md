# ADR-002: Milestone 2 Raft Topology Consensus, Fencing, and Dynamic Online Shard Rebalancing

- **Status:** Accepted
- **Owner:** Justin Mafie

## Context

Milestone 1 established VektorDist's direct-to-primary data plane (`ClusterClient` -> `NodeService`) with `RF=3`, `W=2` quorum WAL durability and monotonic `shard_epoch` fencing, backed by a single authoritative `CoordinatorService` implementing the `TopologyStore` trait. While `CoordinatorService` enabled rapid M1 delivery, a single coordinator represents a control-plane single point of failure and lacks automated shard rebalancing when scaling cluster capacity (e.g., expanding from 3 to 5 nodes or decommissioning nodes).

Milestone 2 requires:
1. Replacing the single coordinator point of failure with a deterministic, first-principles Raft consensus state machine (`RaftNode`, `RaftCluster`, `LogEntry { term, index, cmd: TopologyCommand }`) implementing `TopologyStore`.
2. Strict term and leader-lease fencing (`validate_leader_lease(term)`) so that an isolated leader in term `T` cannot commit topology changes or overwrite a newly elected leader in term `T+1`.
3. Dynamic online shard rebalancing (`compute_rebalance_plan` and `execute_online_rebalance`) that computes minimal replica and primary movements when scaling nodes (e.g., `3 -> 5` nodes) and streams historical WAL entries prior to switching shard ownership so zero acknowledged writes are lost.

## Decision

1. **First-Principles Replicated Topology State Machine (`crates/vektordist-cluster/src/raft.rs`):**
   - `RaftNode` implements standard Raft leader election (`start_election`, `handle_request_vote` with §5.4.1 up-to-date log comparisons on `(last_log_term, last_log_index)`) and log replication (`append_leader_entry`, `handle_append_entries`, `advance_leader_commit_index` enforcing §5.4.2 current-term majority commit rules).
   - Committed `TopologyCommand` entries (`PromotePrimary`, `UpdateShard`, `ApplyRebalance`, `RegisterNode`, `RemoveNode`) are deterministically applied to each node's `TopologySnapshot`.
   - Both `RaftNode` and `RaftCluster` implement `TopologyStore`, and `RaftTopologyService` implements the gRPC `TopologyRpc` trait so `ClusterClient` operates unchanged against a Raft-backed control plane.
2. **Term & Majority Leader-Lease Fencing:**
   - Every `RaftNode` tracks `current_term` and `lease_acks` (the set of voting peers that have acknowledged the leader in the current term/heartbeat round).
   - `validate_leader_lease(expected_term)` rejects stale terms with `VektorError::WrongEpoch` and rejects partitioned leaders lacking majority reachability with `VektorError::QuorumUnavailable`.
   - When a network partition heals, any stale leader discovering a higher term immediately steps down to `RaftRole::Follower`, truncates uncommitted suffix entries from the stale term, and converges to the majority-committed log.
3. **Minimal-Movement Online Shard Rebalancing with Pre-Cutover WAL Streaming:**
   - `compute_rebalance_plan(current_shards, active_nodes, rf)` retains existing valid replicas from `current_shards`, fills under-replicated shards using the least-loaded active nodes, greedily shifts follower replicas from overloaded to underloaded nodes until `max(owned_shards) - min(owned_shards) <= 1`, and balances shard primaries via in-place replica role swaps so `max(primaries) - min(primaries) <= 1`.
   - `execute_online_rebalance` configures the existing shard primary to dual-stream live writes (`recovering_follower_addr`) to newly added replica targets, transfers the union of historical WAL records via `GetWalRecords` / `ReplicateWal`, commits `TopologyCommand::ApplyRebalance` through Raft majority consensus with bumped shard epochs, and pushes the updated `ConfigureShardRequest` roles to all active and removed nodes.

## Alternatives Considered

- **External `etcd` / `consul` sidecar dependency:** Violates VektorDist's zero-Docker, single-binary laptop architecture (`cargo install --path .`).
- **Third-party async Raft crate (`openraft` / `tikv-raft`):** Introduces heavy trait/storage boilerplate and non-deterministic background timers when our topology state machine is compact and benefits from deterministic partition/split-brain testing.
- **Full consistent-hash ring recomputation on node join:** Naively recomputing `shard_id % N` moves nearly all shards unnecessarily; greedy minimal-movement balancing transfers only the mathematical minimum `|S * RF / N_old - S * RF / N_new|` replicas.
- **Immediate ownership switch before WAL catch-up:** Would expose empty or lagging replicas to read/quorum traffic and risk losing acknowledged `W=2` writes if another replica fails during migration.

## Consequences & Failure Modes

- **Guaranteed invariants:** Split-brain leaders in stale terms `T < T_current` are fenced at both the Raft control plane (`validate_leader_lease`, `AppendEntries` term checks) and the data plane (`shard_epoch` monotonicity). Online `3 -> 5` node scaling achieves uniform replica (`6` per node for `10` shards at `RF=3`) and primary (`2` per node) balance with `acknowledged_writes_lost == 0`.
- **Tradeoffs:** Pre-cutover WAL catch-up temporarily increases network bandwidth between existing primaries and joining nodes proportional to shard WAL size, bounded by 500-record batched fsync chunks.
