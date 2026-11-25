use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tonic::transport::Server;
use vektordist_cluster::client::ClusterClient;
use vektordist_cluster::node::NodeService;
use vektordist_cluster::raft::{
    compute_rebalance_plan, execute_online_rebalance, RaftCluster, RaftRole, RaftTopologyService,
    TopologyCommand,
};
use vektordist_cluster::{ShardAssignment, ShardState, TopologySnapshot, TopologyStore};
use vektordist_core::exact::SeededRng;
use vektordist_core::hnsw::HnswParams;
use vektordist_core::{ClusterConfig, DistanceMetric, NodeId, VektorError};
use vektordist_proto::v1::shard_replica_rpc_server::ShardReplicaRpcServer;
use vektordist_proto::v1::topology_rpc_server::TopologyRpcServer;

static M2_TEST_ID: AtomicU64 = AtomicU64::new(1);

fn temp_node_dir(tag: &str, node_id: NodeId) -> PathBuf {
    let seq = M2_TEST_ID.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("vektordist-m2-test-{tag}-{pid}-{seq}-n{node_id}"))
}

async fn spawn_node_server(
    node_id: NodeId,
    config: ClusterConfig,
    tag: &str,
) -> (String, PathBuf, JoinHandle<()>) {
    let dir = temp_node_dir(tag, node_id);
    let _ = std::fs::remove_dir_all(&dir);
    let svc = NodeService::open(node_id, &dir, config, HnswParams::default()).unwrap();

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);

    let handle = tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(ShardReplicaRpcServer::new(svc))
            .serve_with_incoming(incoming)
            .await;
    });

    (addr, dir, handle)
}

async fn spawn_raft_topology_server(raft: Arc<RwLock<RaftCluster>>) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
    let svc = RaftTopologyService::new(raft);

    let handle = tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(TopologyRpcServer::new(svc))
            .serve_with_incoming(incoming)
            .await;
    });

    (addr, handle)
}

fn sample_topology(num_shards: u32, nodes: &[NodeId], rf: u8) -> TopologySnapshot {
    let config = ClusterConfig {
        num_shards,
        replication_factor: rf,
        write_quorum: 2,
        dimensions: 16,
        metric: DistanceMetric::Cosine,
    };
    let eff_rf = (rf as usize).min(nodes.len());
    let mut shards = Vec::with_capacity(num_shards as usize);
    for shard_id in 0..num_shards {
        let primary_idx = (shard_id as usize) % nodes.len();
        let primary = nodes[primary_idx];
        let mut followers = Vec::with_capacity(eff_rf.saturating_sub(1));
        for offset in 1..eff_rf {
            followers.push(nodes[(primary_idx + offset) % nodes.len()]);
        }
        shards.push(ShardAssignment {
            shard_id,
            epoch: 1,
            primary,
            followers,
            state: ShardState::Healthy,
        });
    }
    TopologySnapshot { config, shards }
}

#[test]
fn raft_3_to_5_node_election_log_replication_and_leader_failover() {
    let initial_snap = sample_topology(6, &[1, 2, 3], 3);
    let mut cluster = RaftCluster::new(vec![1, 2, 3], initial_snap).unwrap();

    // 1. Elect initial leader (node 1) in 3-node cluster.
    let term1 = cluster.start_election(1).unwrap();
    assert_eq!(term1, 1);
    assert_eq!(cluster.leader_id(), Some(1));
    assert_eq!(cluster.node(1).unwrap().role, RaftRole::Leader);
    assert_eq!(cluster.node(2).unwrap().role, RaftRole::Follower);
    assert_eq!(cluster.node(3).unwrap().role, RaftRole::Follower);

    // 2. Propose topology command via TopologyStore::promote_primary and verify replication.
    let new_epoch = cluster.promote_primary(0, 2).unwrap();
    assert_eq!(new_epoch, 2);
    for id in [1, 2, 3] {
        let node = cluster.node(id).unwrap();
        assert_eq!(node.commit_index, 1);
        assert_eq!(node.last_applied, 1);
        assert_eq!(node.snapshot.shards[0].primary, 2);
        assert_eq!(node.snapshot.shards[0].epoch, 2);
    }

    // 3. Scale Raft cluster from 3 -> 5 voting nodes and replicate another topology command.
    cluster
        .add_node(4, Some("127.0.0.1:9004".to_string()))
        .unwrap();
    cluster
        .add_node(5, Some("127.0.0.1:9005".to_string()))
        .unwrap();
    assert_eq!(cluster.member_ids(), vec![1, 2, 3, 4, 5]);

    let epoch_shard_1 = cluster.promote_primary(1, 3).unwrap();
    assert_eq!(epoch_shard_1, 2);
    for id in 1..=5 {
        let node = cluster.node(id).unwrap();
        assert_eq!(node.snapshot.shards[1].primary, 3);
        assert_eq!(node.snapshot.shards[1].epoch, 2);
    }

    // 4. Isolate/kill current leader (node 1) and elect a new leader among surviving 4 nodes.
    cluster.isolate_node(1);
    let (new_leader, term2) = cluster.elect_any_leader().unwrap();
    assert_ne!(new_leader, 1);
    assert_eq!(term2, 2);
    assert_eq!(cluster.leader_id(), Some(new_leader));

    // Replicate a command under the new leader in 5-node cluster (4/5 nodes reachable >= quorum 3).
    let epoch_after_failover = cluster.promote_primary(2, 1).unwrap();
    assert_eq!(epoch_after_failover, 2);
    for id in [2, 3, 4, 5] {
        let node = cluster.node(id).unwrap();
        assert_eq!(node.current_term, 2);
        assert_eq!(node.snapshot.shards[2].primary, 1);
        assert_eq!(node.snapshot.shards[2].epoch, 2);
    }
}

#[test]
fn raft_split_brain_fencing_rejects_stale_leader_in_old_term() {
    let initial_snap = sample_topology(4, &[1, 2, 3, 4, 5], 3);
    let mut cluster = RaftCluster::new(vec![1, 2, 3, 4, 5], initial_snap).unwrap();

    // Node 1 becomes leader in term 1.
    let term_t = cluster.start_election(1).unwrap();
    assert_eq!(term_t, 1);
    assert_eq!(cluster.validate_leader_lease(1, term_t), Ok(()));

    // Partition node 1 away from the majority {2, 3, 4, 5}.
    cluster.isolate_node(1);

    // Majority elects node 2 in term T+1 (term 2).
    let term_t_plus_1 = cluster.start_election(2).unwrap();
    assert_eq!(term_t_plus_1, 2);
    assert_eq!(cluster.leader_id(), Some(2));

    // 1. Lease validation on isolated old leader (node 1, term T) MUST fail with WrongEpoch.
    let lease_err = cluster.validate_leader_lease(1, term_t).unwrap_err();
    assert_eq!(
        lease_err,
        VektorError::WrongEpoch {
            current: term_t_plus_1,
            received: term_t,
        }
    );

    // 2. Isolated old leader in term T cannot commit any topology changes.
    let stale_propose_err = cluster
        .propose_on(
            1,
            TopologyCommand::PromotePrimary {
                shard_id: 0,
                new_primary: 5,
            },
        )
        .unwrap_err();
    assert!(matches!(
        stale_propose_err,
        VektorError::QuorumUnavailable { .. }
    ));
    // Old leader's commit_index remains 0 and shard 0 primary is still 1.
    assert_eq!(cluster.node(1).unwrap().commit_index, 0);
    assert_eq!(cluster.node(1).unwrap().snapshot.shards[0].primary, 1);

    // 3. New leader in term T+1 commits a conflicting topology change for shard 0 -> primary 3.
    cluster
        .propose_on(
            2,
            TopologyCommand::PromotePrimary {
                shard_id: 0,
                new_primary: 3,
            },
        )
        .unwrap();
    assert_eq!(cluster.snapshot().shards[0].primary, 3);
    assert_eq!(cluster.snapshot().shards[0].epoch, 2);

    // 4. Heal partition: old leader (node 1) steps down to Follower, adopts term T+1, truncates its
    //    uncommitted term-T entry, and converges to shard 0 primary == 3.
    cluster.heal_node(1);
    let healed_node_1 = cluster.node(1).unwrap();
    assert_eq!(healed_node_1.role, RaftRole::Follower);
    assert_eq!(healed_node_1.current_term, term_t_plus_1);
    assert_eq!(healed_node_1.commit_index, 1);
    assert_eq!(healed_node_1.snapshot.shards[0].primary, 3);
    assert_eq!(healed_node_1.snapshot.shards[0].epoch, 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dynamic_online_shard_rebalance_3_to_5_nodes_preserves_all_wal_writes() {
    let config = ClusterConfig {
        num_shards: 10,
        replication_factor: 3,
        write_quorum: 2,
        dimensions: 16,
        metric: DistanceMetric::Cosine,
    };

    // 1. Launch initial 3-node cluster.
    let (addr1, dir1, h1) = spawn_node_server(1, config.clone(), "rebal").await;
    let (addr2, dir2, h2) = spawn_node_server(2, config.clone(), "rebal").await;
    let (addr3, dir3, h3) = spawn_node_server(3, config.clone(), "rebal").await;

    let mut initial_raft = RaftCluster::from_initial_nodes(
        config.clone(),
        vec![(1, addr1.clone()), (2, addr2.clone()), (3, addr3.clone())],
    )
    .unwrap();
    initial_raft.start_election(1).unwrap();
    initial_raft.push_all_shard_configs().await.unwrap();

    let raft = Arc::new(RwLock::new(initial_raft));
    let (coord_addr, h_coord) = spawn_raft_topology_server(Arc::clone(&raft)).await;

    let client = ClusterClient::connect(&coord_addr).await.unwrap();
    let mut rng = SeededRng::new(2026);
    let mut vec_buf = vec![0.0_f32; config.dimensions];

    // 2. Insert 60 vectors across the 10 shards on the 3-node cluster.
    let mut acked_ledger = Vec::new();
    for i in 1..=60_u64 {
        rng.fill_unit_vector(&mut vec_buf);
        let write_id = 5000 + i;
        let (shard_id, seq_no) = client.upsert(i, write_id, &vec_buf).await.unwrap();
        assert!(seq_no >= 1);
        acked_ledger.push((write_id, i, shard_id));
    }

    // 3. Verify pure rebalance planner properties (3 -> 5 nodes, 10 shards, RF=3 => 30 replicas -> 6 per node).
    let pre_shards = raft.read().unwrap().snapshot().shards;
    let preview_plan = compute_rebalance_plan(&pre_shards, &[1, 2, 3, 4, 5], 3).unwrap();
    let mut replica_counts: BTreeMap<NodeId, usize> = BTreeMap::new();
    let mut primary_counts: BTreeMap<NodeId, usize> = BTreeMap::new();
    for s in &preview_plan.target_shards {
        *primary_counts.entry(s.primary).or_insert(0) += 1;
        *replica_counts.entry(s.primary).or_insert(0) += 1;
        for &f in &s.followers {
            *replica_counts.entry(f).or_insert(0) += 1;
        }
    }
    for node_id in 1..=5 {
        // 10 shards * RF 3 = 30 total replicas / 5 nodes = exactly 6 replicas per node
        assert_eq!(replica_counts.get(&node_id).copied(), Some(6));
        // 10 primaries / 5 nodes = exactly 2 primaries per node
        assert_eq!(primary_counts.get(&node_id).copied(), Some(2));
    }

    // 4. Spawn 2 new nodes (node 4 and node 5) and execute online shard rebalance!
    let (addr4, dir4, h4) = spawn_node_server(4, config.clone(), "rebal").await;
    let (addr5, dir5, h5) = spawn_node_server(5, config.clone(), "rebal").await;

    let executed_plan = execute_online_rebalance(
        &raft,
        vec![(4, addr4.clone()), (5, addr5.clone())],
        vec![1, 2, 3, 4, 5],
    )
    .await
    .unwrap();
    assert!(!executed_plan.moves.is_empty());

    // Refresh client routing table from Raft topology server.
    client.refresh_topology().await.unwrap();

    // 5. Insert 20 additional vectors after rebalancing to verify new primaries accept quorum writes.
    for i in 61..=80_u64 {
        rng.fill_unit_vector(&mut vec_buf);
        let write_id = 5000 + i;
        let (shard_id, seq_no) = client.upsert(i, write_id, &vec_buf).await.unwrap();
        assert!(seq_no >= 1);
        acked_ledger.push((write_id, i, shard_id));
    }

    // 6. Verify 0 acknowledged writes lost across all 10 shards after 3 -> 5 online rebalance.
    let mut durable_write_ids = HashSet::new();
    for shard_id in 0..config.num_shards {
        let recs = client.verify_shard_wal_write_ids(shard_id).await.unwrap();
        for r in recs {
            durable_write_ids.insert(r.write_id);
        }
    }
    assert_eq!(acked_ledger.len(), 80);
    for &(write_id, vec_id, shard_id) in &acked_ledger {
        assert!(
            durable_write_ids.contains(&write_id),
            "acknowledged write_id={write_id} (vec={vec_id}, shard={shard_id}) lost after 3->5 rebalance!"
        );
    }

    // Verify scatter-gather search still finds top hits across the 5-node rebalanced topology.
    let hits = client.search(&vec_buf, 5, 32).await.unwrap();
    assert_eq!(hits.len(), 5);
    assert_eq!(hits[0].id, 80);

    h1.abort();
    h2.abort();
    h3.abort();
    h4.abort();
    h5.abort();
    h_coord.abort();
    let _ = std::fs::remove_dir_all(&dir1);
    let _ = std::fs::remove_dir_all(&dir2);
    let _ = std::fs::remove_dir_all(&dir3);
    let _ = std::fs::remove_dir_all(&dir4);
    let _ = std::fs::remove_dir_all(&dir5);
}
