use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tonic::transport::Server;
use tonic::Request;
use vektordist_cluster::client::ClusterClient;
use vektordist_cluster::coord::CoordinatorService;
use vektordist_cluster::node::{format_endpoint, NodeService};
use vektordist_core::exact::SeededRng;
use vektordist_core::hnsw::HnswParams;
use vektordist_core::{ClusterConfig, DistanceMetric, NodeId, VektorError};
use vektordist_proto::v1::shard_replica_rpc_client::ShardReplicaRpcClient;
use vektordist_proto::v1::shard_replica_rpc_server::ShardReplicaRpcServer;
use vektordist_proto::v1::topology_rpc_server::TopologyRpcServer;
use vektordist_proto::v1::ReplicateWalRequest;
use vektordist_proto::STATUS_WRONG_EPOCH;

static CLUSTER_TEST_ID: AtomicU64 = AtomicU64::new(1);

fn temp_node_dir(tag: &str, node_id: NodeId) -> PathBuf {
    let seq = CLUSTER_TEST_ID.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    std::env::temp_dir().join(format!(
        "vektordist-cluster-test-{tag}-{pid}-{seq}-n{node_id}"
    ))
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

async fn spawn_coord_server(coord: CoordinatorService) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);

    let handle = tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(TopologyRpcServer::new(coord))
            .serve_with_incoming(incoming)
            .await;
    });

    (addr, handle)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cluster_quorum_failover_under_quorum_and_replacement_recovery() {
    let config = ClusterConfig {
        num_shards: 4,
        replication_factor: 3,
        write_quorum: 2,
        dimensions: 16,
        metric: DistanceMetric::Cosine,
    };

    let (addr1, dir1, h1) = spawn_node_server(1, config.clone(), "e2e").await;
    let (addr2, dir2, h2) = spawn_node_server(2, config.clone(), "e2e").await;
    let (addr3, dir3, h3) = spawn_node_server(3, config.clone(), "e2e").await;

    let coord = CoordinatorService::new(
        config.clone(),
        vec![(1, addr1.clone()), (2, addr2.clone()), (3, addr3.clone())],
    )
    .unwrap();
    coord.push_all_shard_configs().await.unwrap();
    let (coord_addr, h_coord) = spawn_coord_server(coord.clone()).await;

    let client = ClusterClient::connect(&coord_addr).await.unwrap();
    let mut rng = SeededRng::new(42);
    let mut vec_buf = vec![0.0_f32; config.dimensions];

    // 1. Insert 40 vectors with W=2 quorum and record acknowledged write_ids.
    let mut acked_ledger = Vec::new();
    for i in 1..=40_u64 {
        rng.fill_unit_vector(&mut vec_buf);
        let write_id = 1000 + i;
        let (shard_id, seq_no) = client.upsert(i, write_id, &vec_buf).await.unwrap();
        assert!(seq_no >= 1);
        acked_ledger.push((write_id, i, shard_id));
    }

    // Verify distributed scatter-gather search works across all shards.
    let hits = client.search(&vec_buf, 5, 32).await.unwrap();
    assert_eq!(hits.len(), 5);
    assert_eq!(hits[0].id, 40);

    // 2. Kill node 1 (simulating SIGKILL) and run coordinator failover + union WAL reconciliation.
    h1.abort();
    coord.handle_node_failure(1).await.unwrap();
    client.refresh_topology().await.unwrap();

    // 3. Stale-epoch fencing check: node 2 must reject ReplicateWal with old epoch 1 (now at epoch 2).
    {
        let mut raw_client = ShardReplicaRpcClient::connect(format_endpoint(&addr2))
            .await
            .unwrap();
        let resp = raw_client
            .replicate_wal(Request::new(ReplicateWalRequest {
                shard_id: 0,
                epoch: 1, // stale!
                batched_fsync: false,
                records: Vec::new(),
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(resp.status_code, STATUS_WRONG_EPOCH);
        assert!(resp.current_epoch >= 2);
    }

    // 4. Insert 20 more vectors while degraded (2/3 nodes alive, W=2 still satisfied).
    for i in 41..=60_u64 {
        rng.fill_unit_vector(&mut vec_buf);
        let write_id = 1000 + i;
        let (shard_id, _) = client.upsert(i, write_id, &vec_buf).await.unwrap();
        acked_ledger.push((write_id, i, shard_id));
    }

    // Verify 0 acknowledged writes lost across all 4 shards!
    let mut durable_write_ids = std::collections::HashSet::new();
    for shard_id in 0..config.num_shards {
        let recs = client.verify_shard_wal_write_ids(shard_id).await.unwrap();
        for r in recs {
            durable_write_ids.insert(r.write_id);
        }
    }
    for &(write_id, vec_id, shard_id) in &acked_ledger {
        assert!(
            durable_write_ids.contains(&write_id),
            "acknowledged write_id={write_id} (vec={vec_id}, shard={shard_id}) lost after failover!"
        );
    }

    // 5. Double failure (< W=2): kill node 2 so only node 3 survives.
    h2.abort();
    coord.handle_node_failure(2).await.unwrap();
    client.refresh_topology().await.unwrap();

    // Reads continue to work on the surviving replica, while writes return QuorumUnavailable!
    let degraded_hits = client.search(&vec_buf, 5, 32).await.unwrap();
    assert_eq!(degraded_hits.len(), 5);

    let under_quorum_err = client.upsert(999, 9999, &vec_buf).await.unwrap_err();
    assert!(matches!(
        under_quorum_err,
        VektorError::QuorumUnavailable { .. }
    ));

    // 6. Spawn replacement node 4 -> RECOVERING catch-up restores quorum (2 alive replicas: node 3 & node 4)!
    let (addr4, dir4, h4) = spawn_node_server(4, config.clone(), "e2e").await;
    let recovered = coord.add_replacement_node(4, addr4).await.unwrap();
    assert_eq!(recovered.len(), config.num_shards as usize);
    client.refresh_topology().await.unwrap();

    // Writes succeed again now that W=2 quorum is restored!
    let (_, seq_after) = client.upsert(61, 1061, &vec_buf).await.unwrap();
    assert!(seq_after >= 1);

    h3.abort();
    h4.abort();
    h_coord.abort();
    let _ = std::fs::remove_dir_all(&dir1);
    let _ = std::fs::remove_dir_all(&dir2);
    let _ = std::fs::remove_dir_all(&dir3);
    let _ = std::fs::remove_dir_all(&dir4);
}
