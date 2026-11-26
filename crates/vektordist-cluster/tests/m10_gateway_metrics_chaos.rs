use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tonic::transport::Server;
use vektordist_cluster::client::ClusterClient;
use vektordist_cluster::coord::CoordinatorService;
use vektordist_cluster::gateway::{
    send_http_request, spawn_http_gateway, ChaosCampaign, ChaosStep, HttpGatewayServer, TraceSpan,
};
use vektordist_cluster::node::NodeService;
use vektordist_core::hnsw::HnswParams;
use vektordist_core::{ClusterConfig, DistanceMetric, NodeId};
use vektordist_proto::v1::shard_replica_rpc_server::ShardReplicaRpcServer;
use vektordist_proto::v1::topology_rpc_server::TopologyRpcServer;

static TEST_SEQ: AtomicU64 = AtomicU64::new(1);

fn temp_dir_for_node(tag: &str, node_id: NodeId) -> PathBuf {
    let seq = TEST_SEQ.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("vektordist-m10-test-{tag}-{pid}-{seq}-n{node_id}"))
}

async fn spawn_node(
    node_id: NodeId,
    config: ClusterConfig,
    tag: &str,
) -> (String, PathBuf, JoinHandle<()>) {
    let dir = temp_dir_for_node(tag, node_id);
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

async fn spawn_coord(coord: CoordinatorService) -> (String, JoinHandle<()>) {
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
async fn http_gateway_serves_healthz_prometheus_metrics_and_traced_upsert_search() {
    let config = ClusterConfig {
        num_shards: 4,
        replication_factor: 3,
        write_quorum: 2,
        dimensions: 8,
        metric: DistanceMetric::Cosine,
    };

    let (addr1, dir1, h1) = spawn_node(1, config.clone(), "http").await;
    let (addr2, dir2, h2) = spawn_node(2, config.clone(), "http").await;
    let (addr3, dir3, h3) = spawn_node(3, config.clone(), "http").await;

    let coord =
        CoordinatorService::new(config.clone(), vec![(1, addr1), (2, addr2), (3, addr3)]).unwrap();
    coord.push_all_shard_configs().await.unwrap();
    let (coord_addr, h_coord) = spawn_coord(coord.clone()).await;

    let client = ClusterClient::connect(&coord_addr).await.unwrap();
    let gateway = HttpGatewayServer::new(client);
    let (gw_addr, h_gw) = spawn_http_gateway("127.0.0.1:0", gateway.clone())
        .await
        .unwrap();

    // 1. Verify GET /healthz over real TCP socket.
    let health = send_http_request(&gw_addr, "GET", "/healthz", &[], "")
        .await
        .unwrap();
    assert_eq!(health.status_code, 200);
    assert!(health.body.contains("\"status\":\"ok\""));
    assert!(health.body.contains("\"active_nodes\":3"));
    assert!(health.body.contains("\"acked_writes_lost\":0"));
    assert!(health.headers.contains_key("x-trace-id"));
    assert!(health.headers.contains_key("x-latency-ms"));

    // 2. Verify POST /v1/upsert with W3C traceparent header propagation.
    let traceparent_in = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    let upsert_body_1 = r#"{"id":101,"write_id":5101,"vector":[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}"#;
    let upsert_resp_1 = send_http_request(
        &gw_addr,
        "POST",
        "/v1/upsert",
        &[("traceparent", traceparent_in)],
        upsert_body_1,
    )
    .await
    .unwrap();
    assert_eq!(upsert_resp_1.status_code, 200);
    assert_eq!(
        upsert_resp_1.headers.get("x-trace-id").map(String::as_str),
        Some("4bf92f3577b34da6a3ce929d0e0e4736")
    );
    let returned_tp = upsert_resp_1.headers.get("traceparent").unwrap();
    let parsed_tp = TraceSpan::parse_traceparent(returned_tp).unwrap();
    assert_eq!(parsed_tp.trace_id, "4bf92f3577b34da6a3ce929d0e0e4736");
    assert_ne!(parsed_tp.span_id, "00f067aa0ba902b7");
    assert!(upsert_resp_1.body.contains("\"status\":\"ok\""));

    let upsert_body_2 = r#"{"id":102,"write_id":5102,"vector":[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]}"#;
    let upsert_resp_2 = send_http_request(&gw_addr, "POST", "/v1/upsert", &[], upsert_body_2)
        .await
        .unwrap();
    assert_eq!(upsert_resp_2.status_code, 200);

    // Refresh coordinator heartbeat so live_vectors is populated across nodes.
    let _ = coord.heartbeat_tick().await;

    // 3. Verify POST /v1/search with W3C traceparent propagation.
    let search_body = r#"{"query":[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"top_k":2,"ef_search":32}"#;
    let search_resp = send_http_request(
        &gw_addr,
        "POST",
        "/v1/search",
        &[("traceparent", traceparent_in)],
        search_body,
    )
    .await
    .unwrap();
    assert_eq!(search_resp.status_code, 200);
    assert_eq!(
        search_resp.headers.get("x-trace-id").map(String::as_str),
        Some("4bf92f3577b34da6a3ce929d0e0e4736")
    );
    assert!(search_resp.body.contains("\"id\":101"));

    // 4. Verify GET /metrics returns Prometheus text exposition with updated counters and latency gauges.
    let metrics_resp = send_http_request(&gw_addr, "GET", "/metrics", &[], "")
        .await
        .unwrap();
    assert_eq!(metrics_resp.status_code, 200);
    assert!(metrics_resp
        .headers
        .get("content-type")
        .is_some_and(|ct| ct.contains("text/plain; version=0.0.4")));
    assert!(metrics_resp.body.contains("vektordist_active_nodes 3"));
    assert!(metrics_resp.body.contains("vektordist_configured_nodes 3"));
    assert!(metrics_resp
        .body
        .contains("vektordist_acked_writes_total 2"));
    assert!(metrics_resp
        .body
        .contains("vektordist_acked_writes_lost_total 0"));
    assert!(metrics_resp
        .body
        .contains("# TYPE vektordist_search_latency_p95_ms gauge"));

    h_gw.abort();
    h_coord.abort();
    h1.abort();
    h2.abort();
    h3.abort();
    let _ = std::fs::remove_dir_all(&dir1);
    let _ = std::fs::remove_dir_all(&dir2);
    let _ = std::fs::remove_dir_all(&dir3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn automated_chaos_campaign_executes_multi_step_failover_with_zero_write_loss() {
    let config = ClusterConfig {
        num_shards: 4,
        replication_factor: 3,
        write_quorum: 2,
        dimensions: 16,
        metric: DistanceMetric::Cosine,
    };

    let p95_sla_ms = if cfg!(debug_assertions) { 250.0 } else { 50.0 };
    let campaign = ChaosCampaign::new(
        config,
        vec![1, 2, 3],
        vec![
            ChaosStep::SeedVectors {
                count: 60,
                batch_size: 20,
                seed: 101,
            },
            ChaosStep::ConcurrentWorkload {
                writes: 25,
                reads: 20,
                top_k: 10,
                ef_search: 48,
                seed: 202,
            },
            ChaosStep::KillNode { node_id: 1 },
            ChaosStep::VerifyZeroWriteLoss,
            ChaosStep::ConcurrentWorkload {
                writes: 15,
                reads: 15,
                top_k: 10,
                ef_search: 48,
                seed: 303,
            },
            ChaosStep::AddReplacementNode { node_id: 4 },
            ChaosStep::VerifyZeroWriteLoss,
            ChaosStep::AssertRecallAndLatencySla {
                queries: 25,
                top_k: 10,
                ef_search: 64,
                min_recall_at_10: 0.95,
                max_p95_ms: p95_sla_ms,
                seed: 404,
            },
        ],
    );

    let report = campaign.run().await.unwrap();
    assert_eq!(report.steps_executed, 8);
    assert_eq!(report.acked_writes_verified, 100);
    assert_eq!(report.acked_writes_lost, 0);
    assert!(
        report.measured_recall_at_10 >= 0.95,
        "expected recall@10 >= 0.95, got {}",
        report.measured_recall_at_10
    );
    assert!(
        report.measured_p95_ms <= p95_sla_ms,
        "expected p95 <= {p95_sla_ms} ms, got {}",
        report.measured_p95_ms
    );
    assert!(report.sla_passed);
}
