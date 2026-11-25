# ADR-006: Adaptive `ef_search` Latency-Budget Controller and Workload-Aware Shard Placement
- **Status:** Accepted
- **Owner:** Justin Mafie

## Context
Fixed `ef_search` beam widths and static modulo primary placement (`shard_id % num_nodes`) suffer from two fundamental production pathologies under skewed real-world workloads:
1. **Recall-Latency Tradeoff Across Heterogeneous Queries:** Queries located near a dense cluster center or matching a known document have a sharp nearest-neighbor distance contrast (`d_1 << d_k`) and converge to `recall@10 >= 0.99` even at `ef_search = 20`. Conversely, high-dimensional boundary queries equidistant to many clusters (`d_1 / d_k -> 1.0`) require a much wider search beam (`ef_search = 100..160`) to maintain `recall@10 >= 0.95`. Using a single static `ef_search` either wastes CPU on easy queries or degrades recall on hard boundary queries, and cannot self-throttle when a burst pushes p95 latency toward the 50 ms SLA budget.
2. **Zipfian Hot-Shard Skew and Tail-Latency Amplification:** Modulo shard placement distributes shard *counts* evenly, but under Zipfian query/write distributions a single node can own multiple hot shard primaries simultaneously while its peers sit nearly idle. Routing every steady-state read exclusively to the shard primary further concentrates CPU load on hot primaries.

## Decision
1. **Contrast-Driven Adaptive `ef_search` with Closed-Loop Latency Feedback (`vektordist_core::adaptive`):**
   - `AdaptiveSearchController` runs a fast pilot probe (`pilot_ef = min_ef`) and measures the local distance contrast ratio `d_k / d_1` (using `1.0 - cosine_similarity` for Cosine and squared distance for L2).
   - High-contrast queries (`d_1 / d_k <= 0.35`, difficulty `0.0`) use `min_ef`, whereas flat boundary queries (`d_1 / d_k -> 0.95`, difficulty `1.0`) scale up to `max_ef`.
   - A sliding-window p95 latency tracker (`observe_latency_ms`) computes a closed-loop budget scale in `[0.0, 1.0]`: when `current_p95_ms <= 0.70 * target_p95_ms`, full expansion up to `max_ef` is permitted; as `current_p95_ms` approaches `target_p95_ms`, the controller smoothly throttles `ef_search` toward `min_ef`.
2. **Power-of-Two-Choices Read Routing and Zero-Copy Primary Rebalancing (`vektordist_cluster::placement`):**
   - `ShardLoadTracker` maintains per-shard and per-node EWMA QPS, EWMA latency (`ms`), and in-flight request counts.
   - `WorkloadAwarePlacer::select_read_replica` scores healthy shard replicas using `5.0 * in_flight_requests + ewma_latency_ms + 0.02 * ewma_qps`, routing reads around congested or tail-latency-spiking nodes.
   - `WorkloadAwarePlacer::plan_hot_shard_transfers` detects nodes whose combined primary QPS exceeds `max_skew_ratio * cluster_mean_qps` and emits `PrimaryTransferOp` promotions that transfer primary leadership to the coolest existing warm follower of a hot shard—eliminating primary skew with zero vector copying over the network.

## Alternatives Considered
- **ML-Based Query Difficulty Predictors:** Requires training and storing per-dataset neural or GBDT regressors, adding inference latency and operational complexity. Local pilot distance contrast (`d_k / d_1`) is zero-training, metric-agnostic, and reuses the HNSW entry descent.
- **Full Shard Migration for Load Balancing:** Streaming millions of vectors to a non-replica node during a traffic spike saturates network bandwidth and worsens tail latency during the spike. Promoting an already-synchronized follower (`RF=3`) shifts primary load instantaneously via an epoch bump.

## Consequences & Failure Modes
- **Pilot Probe Overhead on Hard Queries:** When a query is classified as hard (`chosen_ef > min_ef`), the full layer-0 search re-runs with `chosen_ef`. Because `min_ef` is small (`16..24`), the pilot probe touches only a tiny fraction of nodes and keeps easy queries at `min_ef` in a single pass.
- **Oscillation Under Rapidly Shifting Hotspots:** `plan_hot_shard_transfers` only triggers when a node exceeds `max_skew_ratio` (`1.25x` cluster mean) and transfers each shard at most once per planning pass, preventing leadership ping-pong.
