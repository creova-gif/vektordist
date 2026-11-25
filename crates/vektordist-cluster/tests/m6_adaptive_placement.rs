use vektordist_cluster::placement::{ShardLoadTracker, WorkloadAwarePlacer};
use vektordist_cluster::{ShardAssignment, ShardState, TopologySnapshot, TopologyStore};
use vektordist_core::adaptive::{AdaptiveSearchConfig, AdaptiveSearchController};
use vektordist_core::exact::{generate_seeded_unit_vectors, ExactIndex};
use vektordist_core::hnsw::{recall_at_k, HnswIndex, HnswParams};
use vektordist_core::{ClusterConfig, DistanceMetric, Epoch, NodeId, ShardId, VektorError};

struct InMemoryTopologyStore {
    snapshot: TopologySnapshot,
}

impl TopologyStore for InMemoryTopologyStore {
    fn snapshot(&self) -> TopologySnapshot {
        self.snapshot.clone()
    }

    fn promote_primary(
        &mut self,
        shard_id: ShardId,
        new_primary: NodeId,
    ) -> Result<Epoch, VektorError> {
        let shard = self
            .snapshot
            .shards
            .iter_mut()
            .find(|s| s.shard_id == shard_id)
            .ok_or(VektorError::InvalidConfig("unknown shard_id"))?;
        shard.followers.retain(|&id| id != new_primary);
        if shard.primary != new_primary {
            let old = shard.primary;
            shard.primary = new_primary;
            shard.followers.push(old);
        }
        shard.epoch += 1;
        Ok(shard.epoch)
    }
}

#[test]
fn m6_adaptive_ef_controller_maintains_recall_95_and_adapts_to_difficulty_and_budget() {
    let dims = 64;
    let num_vectors = 1_500;
    let k = 10;

    let params = HnswParams {
        m: 16,
        m_max0: 32,
        ef_construction: 128,
        ef_search: 64,
        ..HnswParams::default()
    };
    let mut hnsw =
        HnswIndex::with_capacity(dims, num_vectors, DistanceMetric::Cosine, params).unwrap();
    let mut exact = ExactIndex::with_capacity(dims, num_vectors, DistanceMetric::Cosine).unwrap();

    let base_vectors = generate_seeded_unit_vectors(num_vectors, dims, 2026);
    for (id, vec) in base_vectors.iter().enumerate() {
        hnsw.upsert(id as u64, vec).unwrap();
        exact.upsert(id as u64, vec).unwrap();
    }

    let config = AdaptiveSearchConfig {
        min_ef: 20,
        base_ef: 56,
        max_ef: 144,
        target_p95_ms: 50.0,
        window_size: 128,
    };
    let controller = AdaptiveSearchController::new(config).unwrap();

    // 1. Easy high-contrast queries (near-exact perturbations of existing vectors) must choose smaller ef.
    let mut easy_ef_sum = 0_usize;
    let mut easy_recall_sum = 0.0_f64;
    let easy_count = 30;
    for i in 0..easy_count {
        let mut q = base_vectors[i * 10].clone();
        q[0] += 0.01;
        let exact_hits = exact.search(&q, k).unwrap();
        let (approx_hits, ef_used) = controller.search_adaptive(&hnsw, &q, k).unwrap();
        easy_ef_sum += ef_used;
        easy_recall_sum += recall_at_k(&exact_hits, &approx_hits, k);
    }
    let mean_easy_ef = (easy_ef_sum as f64) / (easy_count as f64);
    let mean_easy_recall = easy_recall_sum / (easy_count as f64);

    // 2. Hard boundary queries (uniform random unit vectors in 64D have flat distance contrast near 1.0)
    // must automatically expand ef toward max_ef while keeping recall@10 >= 0.95.
    let hard_queries = generate_seeded_unit_vectors(50, dims, 9999);
    let mut hard_ef_sum = 0_usize;
    let mut hard_recall_sum = 0.0_f64;
    for q in &hard_queries {
        let exact_hits = exact.search(q, k).unwrap();
        let (approx_hits, ef_used) = controller.search_adaptive(&hnsw, q, k).unwrap();
        hard_ef_sum += ef_used;
        hard_recall_sum += recall_at_k(&exact_hits, &approx_hits, k);
    }
    let mean_hard_ef = (hard_ef_sum as f64) / (hard_queries.len() as f64);
    let mean_hard_recall = hard_recall_sum / (hard_queries.len() as f64);

    assert!(
        mean_easy_ef < mean_hard_ef,
        "expected mean_easy_ef ({mean_easy_ef:.1}) < mean_hard_ef ({mean_hard_ef:.1})"
    );
    assert!(
        mean_easy_recall >= 0.98,
        "expected easy query recall@10 >= 0.98, got {mean_easy_recall:.4}"
    );
    assert!(
        mean_hard_recall >= 0.95,
        "expected hard boundary query recall@10 >= 0.95, got {mean_hard_recall:.4}"
    );

    // 3. Closed-loop latency-budget throttling: when observed p95 exceeds target_p95_ms (50ms),
    // the controller must scale ef down toward min_ef.
    for _ in 0..128 {
        controller.observe_latency_ms(60.0);
    }
    assert!(controller.current_p95_ms() >= 55.0);
    let (_, throttled_ef) = controller
        .search_adaptive(&hnsw, &hard_queries[0], k)
        .unwrap();
    assert_eq!(
        throttled_ef, config.min_ef,
        "expected throttled ef == min_ef ({}) under latency budget overload",
        config.min_ef
    );
}

#[test]
fn m6_power_of_two_choices_routes_around_congested_replica() {
    let shard = ShardAssignment {
        shard_id: 0,
        epoch: 1,
        primary: 1,
        followers: vec![2, 3],
        state: ShardState::Healthy,
    };

    let mut tracker = ShardLoadTracker::new(0.4);
    // Node 1 (primary) is experiencing a tail-latency spike and high in-flight queue depth.
    tracker.set_node_load(1, 12, 42.0, 900.0);
    // Node 2 and Node 3 (followers) are cool and lightly loaded.
    tracker.set_node_load(2, 1, 2.5, 120.0);
    tracker.set_node_load(3, 0, 1.8, 100.0);

    let placer = WorkloadAwarePlacer::default();
    let mut hot_node_picks = 0_usize;
    let total_routes = 100_usize;
    for _ in 0..total_routes {
        let chosen = placer.select_read_replica(&shard, &mut tracker).unwrap();
        if chosen == 1 {
            hot_node_picks += 1;
        }
    }

    assert_eq!(
        hot_node_picks, 0,
        "expected Power-of-Two-Choices router to avoid congested primary node 1 completely"
    );
}

#[test]
fn m6_workload_aware_placer_rebalances_zipfian_hot_primaries_via_zero_copy_promotions() {
    // 4 nodes, 12 shards, RF=3.
    // Suppose shards 0, 4, 8 all land with primary on Node 1 and receive Zipfian hot traffic (500 QPS each),
    // while all other shards receive cold traffic (20 QPS each).
    let mut shards = Vec::new();
    let node_ids = [1_u32, 2, 3, 4];
    for shard_id in 0..12_u32 {
        let p_idx = (shard_id as usize) % node_ids.len();
        let primary = node_ids[p_idx];
        let followers = vec![node_ids[(p_idx + 1) % 4], node_ids[(p_idx + 2) % 4]];
        shards.push(ShardAssignment {
            shard_id,
            epoch: 1,
            primary,
            followers,
            state: ShardState::Healthy,
        });
    }

    let mut tracker = ShardLoadTracker::new(0.5);
    for &node_id in &node_ids {
        tracker.update_node_state(node_id, true, 1, 64 * 1024 * 1024);
    }

    for shard in &shards {
        // Shards 0, 4, 8 (all primary on Node 1) are hot (500 QPS each = 1,500 QPS on Node 1).
        // Remaining 9 shards are cold (20 QPS each = 60 QPS on Nodes 2, 3, 4).
        let qps = if shard.primary == 1 { 500.0 } else { 20.0 };
        tracker.set_shard_qps(shard.shard_id, qps, 4.0);
    }

    let placer = WorkloadAwarePlacer::new(1.25);
    let before_qps = placer.primary_qps_by_node(&shards, &tracker);
    let before_max = before_qps.values().copied().fold(0.0_f64, f64::max);
    let mean_qps: f64 = before_qps.values().sum::<f64>() / (before_qps.len() as f64);
    assert!(
        before_max / mean_qps > 3.0,
        "expected initial Zipfian primary skew > 3.0x mean, got {:.2}x",
        before_max / mean_qps
    );

    let transfers = placer.plan_hot_shard_transfers(&shards, &tracker);
    assert_eq!(
        transfers.len(),
        2,
        "expected 2 hot primaries on Node 1 to be promoted to cooler existing followers"
    );

    let mut store = InMemoryTopologyStore {
        snapshot: TopologySnapshot {
            config: ClusterConfig {
                num_shards: 12,
                replication_factor: 3,
                write_quorum: 2,
                dimensions: 64,
                metric: DistanceMetric::Cosine,
            },
            shards: shards.clone(),
        },
    };

    // Verify every transfer promotes an existing follower of that shard (zero vector data movement)
    // and advances the shard epoch.
    for op in &transfers {
        let orig = shards.iter().find(|s| s.shard_id == op.shard_id).unwrap();
        assert_eq!(op.from_node, 1);
        assert!(
            orig.followers.contains(&op.to_follower),
            "target node {} must already be a warm follower of shard {}",
            op.to_follower,
            op.shard_id
        );
    }

    let epochs = placer.apply_transfers(&mut store, &transfers).unwrap();
    assert!(epochs.iter().all(|&e| e == 2));

    let after_shards = store.snapshot().shards;
    let after_qps = placer.primary_qps_by_node(&after_shards, &tracker);
    let after_max = after_qps.values().copied().fold(0.0_f64, f64::max);

    assert!(
        after_max < before_max * 0.40,
        "expected peak node primary QPS to drop by >60% (before={before_max}, after={after_max})"
    );
}
