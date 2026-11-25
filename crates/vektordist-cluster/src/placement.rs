use std::collections::{BTreeMap, BTreeSet};

use vektordist_core::{Epoch, NodeId, ShardId, VektorError};

use crate::{ShardAssignment, ShardState, TopologyStore};

/// Per-shard EWMA telemetry tracked by the control and routing planes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShardTelemetry {
    pub shard_id: ShardId,
    pub ewma_qps: f64,
    pub ewma_latency_ms: f64,
    pub memory_bytes: u64,
}

/// Per-node EWMA telemetry and instantaneous in-flight load.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeTelemetry {
    pub node_id: NodeId,
    pub alive: bool,
    pub in_flight_requests: u32,
    pub ewma_qps: f64,
    pub ewma_latency_ms: f64,
    pub memory_bytes: u64,
}

/// Zero-copy primary leadership transfer operation that shifts primary write/search load
/// from an overloaded node onto an existing warm follower replica without moving vector data.
#[derive(Debug, Clone, PartialEq)]
pub struct PromoteFollowerToPrimary {
    pub shard_id: ShardId,
    pub from_node: NodeId,
    pub to_follower: NodeId,
    pub estimated_qps_shifted: f64,
}

/// Type alias for `PromoteFollowerToPrimary`.
pub type PrimaryTransferOp = PromoteFollowerToPrimary;

/// Sliding EWMA load tracker for shards and physical cluster nodes.
#[derive(Debug, Clone)]
pub struct ShardLoadTracker {
    alpha: f64,
    shards: BTreeMap<ShardId, ShardTelemetry>,
    nodes: BTreeMap<NodeId, NodeTelemetry>,
    rr_counter: u64,
}

impl Default for ShardLoadTracker {
    fn default() -> Self {
        Self::new(0.3)
    }
}

impl ShardLoadTracker {
    /// Create a new `ShardLoadTracker` with EWMA smoothing factor `alpha` in `(0.0, 1.0]`.
    #[must_use]
    pub fn new(alpha: f64) -> Self {
        let clamped_alpha = if alpha.is_finite() {
            alpha.clamp(0.05, 1.0)
        } else {
            0.3
        };
        Self {
            alpha: clamped_alpha,
            shards: BTreeMap::new(),
            nodes: BTreeMap::new(),
            rr_counter: 0,
        }
    }

    /// Register or update a node's liveness, in-flight request count, and memory footprint.
    pub fn update_node_state(
        &mut self,
        node_id: NodeId,
        alive: bool,
        in_flight_requests: u32,
        memory_bytes: u64,
    ) {
        let entry = self.nodes.entry(node_id).or_insert(NodeTelemetry {
            node_id,
            alive,
            in_flight_requests,
            ewma_qps: 0.0,
            ewma_latency_ms: 1.0,
            memory_bytes,
        });
        entry.alive = alive;
        entry.in_flight_requests = in_flight_requests;
        entry.memory_bytes = memory_bytes;
    }

    /// Observe a shard query rate and latency sample on `serving_node`.
    pub fn observe_shard_load(
        &mut self,
        shard_id: ShardId,
        serving_node: NodeId,
        instant_qps: f64,
        latency_ms: f64,
        shard_memory_bytes: u64,
    ) {
        let qps = instant_qps.max(0.0);
        let lat = latency_ms.max(0.0);
        let a = self.alpha;

        let s_entry = self.shards.entry(shard_id).or_insert(ShardTelemetry {
            shard_id,
            ewma_qps: qps,
            ewma_latency_ms: lat,
            memory_bytes: shard_memory_bytes,
        });
        s_entry.ewma_qps = a * qps + (1.0 - a) * s_entry.ewma_qps;
        s_entry.ewma_latency_ms = a * lat + (1.0 - a) * s_entry.ewma_latency_ms;
        s_entry.memory_bytes = shard_memory_bytes;

        let n_entry = self.nodes.entry(serving_node).or_insert(NodeTelemetry {
            node_id: serving_node,
            alive: true,
            in_flight_requests: 0,
            ewma_qps: qps,
            ewma_latency_ms: lat,
            memory_bytes: shard_memory_bytes,
        });
        n_entry.ewma_qps = a * qps + (1.0 - a) * n_entry.ewma_qps;
        n_entry.ewma_latency_ms = a * lat + (1.0 - a) * n_entry.ewma_latency_ms;
    }

    /// Directly set a node's EWMA latency and in-flight load (useful for deterministic simulation).
    pub fn set_node_load(
        &mut self,
        node_id: NodeId,
        in_flight_requests: u32,
        ewma_latency_ms: f64,
        ewma_qps: f64,
    ) {
        let entry = self.nodes.entry(node_id).or_insert(NodeTelemetry {
            node_id,
            alive: true,
            in_flight_requests,
            ewma_qps,
            ewma_latency_ms,
            memory_bytes: 0,
        });
        entry.in_flight_requests = in_flight_requests;
        entry.ewma_latency_ms = ewma_latency_ms.max(0.0);
        entry.ewma_qps = ewma_qps.max(0.0);
    }

    /// Directly set a shard's EWMA QPS and latency.
    pub fn set_shard_qps(&mut self, shard_id: ShardId, ewma_qps: f64, ewma_latency_ms: f64) {
        let entry = self.shards.entry(shard_id).or_insert(ShardTelemetry {
            shard_id,
            ewma_qps: 0.0,
            ewma_latency_ms: 1.0,
            memory_bytes: 0,
        });
        entry.ewma_qps = ewma_qps.max(0.0);
        entry.ewma_latency_ms = ewma_latency_ms.max(0.0);
    }

    #[must_use]
    pub fn shard_telemetry(&self, shard_id: ShardId) -> Option<ShardTelemetry> {
        self.shards.get(&shard_id).copied()
    }

    #[must_use]
    pub fn node_telemetry(&self, node_id: NodeId) -> Option<NodeTelemetry> {
        self.nodes.get(&node_id).copied()
    }

    /// Compute the composite routing load score for `node_id`:
    /// `in_flight_weight (5.0 * in_flight) + ewma_latency_ms + 0.02 * ewma_qps`.
    #[must_use]
    pub fn composite_node_score(&self, node_id: NodeId) -> f64 {
        let Some(node) = self.nodes.get(&node_id) else {
            return 1.0;
        };
        if !node.alive {
            return f64::INFINITY;
        }
        let in_flight_weight = f64::from(node.in_flight_requests) * 5.0;
        in_flight_weight + node.ewma_latency_ms + 0.02 * node.ewma_qps
    }

    fn next_choice_seed(&mut self) -> u64 {
        self.rr_counter = self.rr_counter.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.rr_counter;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Workload-aware replica router and zero-copy hot-primary rebalancer.
#[derive(Debug, Clone)]
pub struct WorkloadAwarePlacer {
    /// Maximum allowed ratio of a node's primary QPS to the cluster mean primary QPS before rebalancing.
    pub max_skew_ratio: f64,
    /// Penalty per millisecond of EWMA latency when scoring candidate replicas.
    pub in_flight_weight: f64,
}

impl Default for WorkloadAwarePlacer {
    fn default() -> Self {
        Self {
            max_skew_ratio: 1.25,
            in_flight_weight: 5.0,
        }
    }
}

impl WorkloadAwarePlacer {
    #[must_use]
    pub fn new(max_skew_ratio: f64) -> Self {
        Self {
            max_skew_ratio: max_skew_ratio.max(1.05),
            in_flight_weight: 5.0,
        }
    }

    /// Power-of-Two-Choices replica router: given a shard's replica set (`[primary, followers...]`),
    /// samples two distinct healthy replicas (or evaluates the healthy replica pair/triplet) and returns
    /// the `NodeId` with the lowest composite load score (`in_flight_weight * in_flight + ewma_latency_ms`).
    pub fn select_read_replica(
        &self,
        shard: &ShardAssignment,
        tracker: &mut ShardLoadTracker,
    ) -> Result<NodeId, VektorError> {
        if shard.state == ShardState::UnderQuorum {
            return Err(VektorError::QuorumUnavailable {
                shard_id: shard.shard_id,
                required: 2,
                alive: 1,
            });
        }

        let mut candidates = Vec::with_capacity(1 + shard.followers.len());
        if tracker
            .node_telemetry(shard.primary)
            .is_none_or(|n| n.alive)
        {
            candidates.push(shard.primary);
        }
        for &f in &shard.followers {
            if !candidates.contains(&f) && tracker.node_telemetry(f).is_none_or(|n| n.alive) {
                candidates.push(f);
            }
        }

        if candidates.is_empty() {
            return Ok(shard.primary);
        }
        if candidates.len() == 1 {
            return Ok(candidates[0]);
        }

        // Power-of-Two-Choices: pick two distinct candidates using SplitMix64 counter,
        // and also compare against the third replica if one of the two sampled nodes is degraded.
        let seed = tracker.next_choice_seed();
        let idx_a = (seed as usize) % candidates.len();
        let idx_b =
            (idx_a + 1 + ((seed >> 16) as usize % (candidates.len() - 1))) % candidates.len();
        let cand_a = candidates[idx_a];
        let cand_b = candidates[idx_b];

        let score_a = self.score_replica(cand_a, tracker);
        let score_b = self.score_replica(cand_b, tracker);
        let best_two = if score_a <= score_b { cand_a } else { cand_b };
        let best_two_score = score_a.min(score_b);

        // Guard against the case where RF=3 and a third replica is significantly cooler than both sampled ones.
        let mut best_node = best_two;
        let mut best_score = best_two_score;
        for &cand in &candidates {
            let s = self.score_replica(cand, tracker);
            if s + 1.0 < best_score {
                best_score = s;
                best_node = cand;
            }
        }

        Ok(best_node)
    }

    fn score_replica(&self, node_id: NodeId, tracker: &ShardLoadTracker) -> f64 {
        let Some(node) = tracker.node_telemetry(node_id) else {
            return 1.0;
        };
        if !node.alive {
            return f64::INFINITY;
        }
        f64::from(node.in_flight_requests) * self.in_flight_weight
            + node.ewma_latency_ms
            + 0.02 * node.ewma_qps
    }

    /// Compute the per-node primary QPS distribution across all active nodes in `shards`.
    #[must_use]
    pub fn primary_qps_by_node(
        &self,
        shards: &[ShardAssignment],
        tracker: &ShardLoadTracker,
    ) -> BTreeMap<NodeId, f64> {
        let mut node_qps: BTreeMap<NodeId, f64> = BTreeMap::new();
        for s in shards {
            node_qps.entry(s.primary).or_insert(0.0);
            for &f in &s.followers {
                node_qps.entry(f).or_insert(0.0);
            }
        }
        for s in shards {
            let qps = tracker
                .shard_telemetry(s.shard_id)
                .map_or(1.0, |t| t.ewma_qps);
            *node_qps.entry(s.primary).or_insert(0.0) += qps;
        }
        node_qps
    }

    /// Detect nodes whose primary QPS exceeds `max_skew_ratio * cluster_mean_qps` and generate
    /// zero-data-loss primary transfer operations (`PrimaryTransferOp`) that promote cooler existing
    /// followers to primary without copying any vectors over the network.
    #[must_use]
    pub fn plan_hot_shard_transfers(
        &self,
        shards: &[ShardAssignment],
        tracker: &ShardLoadTracker,
    ) -> Vec<PrimaryTransferOp> {
        if shards.is_empty() {
            return Vec::new();
        }

        let mut working_shards = shards.to_vec();
        let mut node_primary_qps = self.primary_qps_by_node(&working_shards, tracker);

        // Filter to alive nodes.
        let alive_nodes: BTreeSet<NodeId> = node_primary_qps
            .keys()
            .copied()
            .filter(|&id| tracker.node_telemetry(id).is_none_or(|n| n.alive))
            .collect();
        if alive_nodes.len() < 2 {
            return Vec::new();
        }

        let total_qps: f64 = node_primary_qps
            .iter()
            .filter(|(id, _)| alive_nodes.contains(id))
            .map(|(_, &q)| q)
            .sum();
        let mean_qps = total_qps / (alive_nodes.len() as f64);
        if mean_qps <= 1e-6 {
            return Vec::new();
        }
        let mean_shard_qps = total_qps / (working_shards.len() as f64);

        let threshold_qps = mean_qps * self.max_skew_ratio;
        let mut planned_ops = Vec::new();
        let mut transferred_shards: BTreeSet<ShardId> = BTreeSet::new();

        // Greedily rebalance up to shards.len() steps until no hot node can improve skew via a follower promotion.
        for _ in 0..working_shards.len() {
            // Find the hottest alive node above threshold_qps.
            let Some((hot_node, hot_qps)) = node_primary_qps
                .iter()
                .filter(|(id, _)| alive_nodes.contains(id))
                .map(|(&id, &q)| (id, q))
                .max_by(|a, b| a.1.total_cmp(&b.1))
            else {
                break;
            };

            if hot_qps <= threshold_qps {
                break;
            }

            // Consider hot shards currently primary on `hot_node` that haven't been transferred yet.
            let mut best_candidate: Option<(usize, NodeId, f64, f64)> = None;
            for (idx, shard) in working_shards.iter().enumerate() {
                if shard.primary != hot_node || transferred_shards.contains(&shard.shard_id) {
                    continue;
                }
                let shard_qps = tracker
                    .shard_telemetry(shard.shard_id)
                    .map_or(1.0, |t| t.ewma_qps);
                if shard_qps < mean_shard_qps {
                    continue;
                }

                for &follower in &shard.followers {
                    if !alive_nodes.contains(&follower) {
                        continue;
                    }
                    let follower_qps = node_primary_qps.get(&follower).copied().unwrap_or(0.0);
                    let new_hot_qps = hot_qps - shard_qps;
                    let new_follower_qps = follower_qps + shard_qps;
                    let current_peak = hot_qps.max(follower_qps);
                    let new_peak = new_hot_qps.max(new_follower_qps);
                    let peak_reduction = current_peak - new_peak;

                    if peak_reduction > 1e-3 {
                        let is_better = best_candidate
                            .is_none_or(|(_, _, _, best_red)| peak_reduction > best_red);
                        if is_better {
                            best_candidate = Some((idx, follower, shard_qps, peak_reduction));
                        }
                    }
                }
            }

            let Some((shard_idx, target_follower, shard_qps, _)) = best_candidate else {
                break;
            };

            let shard_id = working_shards[shard_idx].shard_id;
            let old_primary = working_shards[shard_idx].primary;
            working_shards[shard_idx]
                .followers
                .retain(|&id| id != target_follower);
            working_shards[shard_idx].followers.push(old_primary);
            working_shards[shard_idx].primary = target_follower;
            working_shards[shard_idx].epoch += 1;

            *node_primary_qps.entry(old_primary).or_insert(0.0) -= shard_qps;
            *node_primary_qps.entry(target_follower).or_insert(0.0) += shard_qps;
            transferred_shards.insert(shard_id);

            planned_ops.push(PrimaryTransferOp {
                shard_id,
                from_node: old_primary,
                to_follower: target_follower,
                estimated_qps_shifted: shard_qps,
            });
        }

        planned_ops
    }

    /// Execute planned zero-copy primary transfers against a `TopologyStore`, advancing shard epochs.
    pub fn apply_transfers(
        &self,
        store: &mut dyn TopologyStore,
        transfers: &[PrimaryTransferOp],
    ) -> Result<Vec<Epoch>, VektorError> {
        let mut epochs = Vec::with_capacity(transfers.len());
        for op in transfers {
            let next_epoch = store.promote_primary(op.shard_id, op.to_follower)?;
            epochs.push(next_epoch);
        }
        Ok(epochs)
    }
}
