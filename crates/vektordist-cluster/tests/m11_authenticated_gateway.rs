//! Milestone 11 integration tests: Zero-trust authenticated HTTP/1.1 REST gateway, quorum-replicated
//! `POST /v1/delete`, `GET /v1/topology`, and tamper-evident `GET /v1/audit` verification across a
//! live 3-node cluster.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tonic::transport::Server;
use vektordist_cluster::client::ClusterClient;
use vektordist_cluster::coord::CoordinatorService;
use vektordist_cluster::gateway::{send_http_request, spawn_http_gateway, HttpGatewayServer};
use vektordist_cluster::node::NodeService;
use vektordist_cluster::security::{OrgId, SecurityGate, TenantQuotaPolicy, TenantRole};
use vektordist_core::hnsw::HnswParams;
use vektordist_core::{ClusterConfig, DistanceMetric, NodeId};
use vektordist_proto::v1::shard_replica_rpc_server::ShardReplicaRpcServer;
use vektordist_proto::v1::topology_rpc_server::TopologyRpcServer;

static TEST_SEQ: AtomicU64 = AtomicU64::new(1);

fn temp_dir_for_node(tag: &str, node_id: NodeId) -> PathBuf {
    let seq = TEST_SEQ.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("vektordist-m11-test-{tag}-{pid}-{seq}-n{node_id}"))
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
async fn unauthenticated_gateway_supports_quorum_upsert_search_delete_topology_and_audit() {
    let config = ClusterConfig {
        num_shards: 4,
        replication_factor: 3,
        write_quorum: 2,
        dimensions: 8,
        metric: DistanceMetric::Cosine,
    };

    let (addr1, dir1, h1) = spawn_node(1, config.clone(), "unauth").await;
    let (addr2, dir2, h2) = spawn_node(2, config.clone(), "unauth").await;
    let (addr3, dir3, h3) = spawn_node(3, config.clone(), "unauth").await;

    let coord =
        CoordinatorService::new(config.clone(), vec![(1, addr1), (2, addr2), (3, addr3)]).unwrap();
    coord.push_all_shard_configs().await.unwrap();
    let (coord_addr, h_coord) = spawn_coord(coord.clone()).await;

    let client = ClusterClient::connect(&coord_addr).await.unwrap();
    let gateway = HttpGatewayServer::new(client);
    let (gw_addr, h_gw) = spawn_http_gateway("127.0.0.1:0", gateway.clone())
        .await
        .unwrap();

    // 1. Upsert two vectors without any Authorization header (backward-compatible unauthenticated mode).
    let upsert_1 = r#"{"id":42,"write_id":1042,"vector":[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}"#;
    let resp_u1 = send_http_request(&gw_addr, "POST", "/v1/upsert", &[], upsert_1)
        .await
        .unwrap();
    assert_eq!(resp_u1.status_code, 200);
    assert!(resp_u1.body.contains("\"status\":\"ok\""));

    let upsert_2 = r#"{"id":43,"write_id":1043,"vector":[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]}"#;
    let resp_u2 = send_http_request(&gw_addr, "POST", "/v1/upsert", &[], upsert_2)
        .await
        .unwrap();
    assert_eq!(resp_u2.status_code, 200);

    // 2. Search confirms both id=42 and id=43 are returned.
    let search_q = r#"{"query":[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"top_k":5,"ef_search":32}"#;
    let resp_s1 = send_http_request(&gw_addr, "POST", "/v1/search", &[], search_q)
        .await
        .unwrap();
    assert_eq!(resp_s1.status_code, 200);
    assert!(resp_s1.body.contains("\"id\":42"));
    assert!(resp_s1.body.contains("\"id\":43"));

    // 3. Quorum delete id=42 via POST /v1/delete (also test omitted write_id fallback on id=999 later).
    let delete_body = r#"{"id":42,"write_id":2042}"#;
    let resp_del = send_http_request(&gw_addr, "POST", "/v1/delete", &[], delete_body)
        .await
        .unwrap();
    assert_eq!(resp_del.status_code, 200);
    assert!(resp_del.body.contains("\"status\":\"ok\""));
    assert!(resp_del.body.contains("\"shard_id\":"));
    assert!(resp_del.body.contains("\"seq_no\":"));
    assert!(resp_del.body.contains("\"trace_id\":"));

    // Verify gateway metrics updated: 2 upserts + 1 delete = 3 acked writes, 2 - 1 = 1 indexed vector.
    let snap = gateway.metrics_snapshot();
    assert_eq!(snap.acked_writes_total, 3);
    assert_eq!(snap.indexed_vectors, 1);

    // 4. Subsequent search proves deleted vector id=42 disappeared across the 3-node cluster while id=43 remains.
    let resp_s2 = send_http_request(&gw_addr, "POST", "/v1/search", &[], search_q)
        .await
        .unwrap();
    assert_eq!(resp_s2.status_code, 200);
    assert!(
        !resp_s2.body.contains("\"id\":42"),
        "deleted vector 42 must not appear in search hits: {}",
        resp_s2.body
    );
    assert!(resp_s2.body.contains("\"id\":43"));

    // 5. GET /v1/topology returns cluster configuration & active node count.
    let _ = coord.heartbeat_tick().await;
    let resp_topo = send_http_request(&gw_addr, "GET", "/v1/topology", &[], "")
        .await
        .unwrap();
    assert_eq!(resp_topo.status_code, 200);
    assert!(resp_topo.body.contains("\"status\":\"ok\""));
    assert!(resp_topo.body.contains("\"epoch\":1"));
    assert!(resp_topo.body.contains("\"num_shards\":4"));
    assert!(resp_topo.body.contains("\"replication_factor\":3"));
    assert!(resp_topo.body.contains("\"write_quorum\":2"));
    assert!(resp_topo.body.contains("\"dimensions\":8"));
    assert!(resp_topo.body.contains("\"configured_nodes\":3"));
    assert!(resp_topo.body.contains("\"active_nodes\":3"));

    // 6. GET /v1/audit in unauthenticated mode reports security_enabled: false.
    let resp_audit = send_http_request(&gw_addr, "GET", "/v1/audit", &[], "")
        .await
        .unwrap();
    assert_eq!(resp_audit.status_code, 200);
    assert_eq!(
        resp_audit.body,
        "{\"status\":\"ok\",\"security_enabled\":false,\"entries_count\":0,\"head_hash\":0,\"chain_valid\":true}"
    );

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
async fn zero_trust_security_gate_enforces_auth_rbac_tenant_isolation_rate_limits_and_audit() {
    let config = ClusterConfig {
        num_shards: 4,
        replication_factor: 3,
        write_quorum: 2,
        dimensions: 8,
        metric: DistanceMetric::Cosine,
    };

    let (addr1, dir1, h1) = spawn_node(1, config.clone(), "ztrust").await;
    let (addr2, dir2, h2) = spawn_node(2, config.clone(), "ztrust").await;
    let (addr3, dir3, h3) = spawn_node(3, config.clone(), "ztrust").await;

    let coord =
        CoordinatorService::new(config.clone(), vec![(1, addr1), (2, addr2), (3, addr3)]).unwrap();
    coord.push_all_shard_configs().await.unwrap();
    let (coord_addr, h_coord) = spawn_coord(coord.clone()).await;

    let mut gate = SecurityGate::new([0x77_u8; 32], TenantQuotaPolicy::default());
    let org_acme = OrgId::new("acme").unwrap();
    let org_throttled = OrgId::new("throttled-org").unwrap();

    // Configure strict rate limit for throttled-org: burst of 2 tokens, tiny refill rate (0.2 QPS -> 5s per token).
    gate.quotas.set_policy(
        org_throttled.clone(),
        TenantQuotaPolicy {
            max_qps: 0.2,
            burst_capacity: 2.0,
            max_vectors: 10_000,
            max_memory_bytes: 64 * 1024 * 1024,
        },
        0,
    );

    let (reader_token, _) = gate.issue_api_key("k-reader", org_acme.clone(), TenantRole::Reader, 1);
    let (writer_token, _) = gate.issue_api_key("k-writer", org_acme.clone(), TenantRole::Writer, 2);
    let (admin_token, _) = gate.issue_api_key("k-admin", org_acme.clone(), TenantRole::Admin, 3);
    let (throttled_token, _) =
        gate.issue_api_key("k-throttled", org_throttled.clone(), TenantRole::Reader, 4);

    let shared_gate = Arc::new(Mutex::new(gate));
    let client = ClusterClient::connect(&coord_addr).await.unwrap();
    let gateway = HttpGatewayServer::new(client).with_security_gate(Arc::clone(&shared_gate));
    let (gw_addr, h_gw) = spawn_http_gateway("127.0.0.1:0", gateway).await.unwrap();

    let reader_auth = format!("Bearer {reader_token}");
    let writer_auth = format!("bearer {writer_token}"); // lowercase 'bearer ' verifies case-insensitive parsing
    let admin_auth = format!("Bearer {admin_token}");
    let throttled_auth = format!("Bearer {throttled_token}");

    // 1. Kubernetes /healthz and Prometheus /metrics remain open without credentials.
    let health = send_http_request(&gw_addr, "GET", "/healthz", &[], "")
        .await
        .unwrap();
    assert_eq!(health.status_code, 200);
    let metrics = send_http_request(&gw_addr, "GET", "/metrics", &[], "")
        .await
        .unwrap();
    assert_eq!(metrics.status_code, 200);

    // 2. Missing or invalid token on protected routes returns 401 Unauthorized.
    let missing_auth = send_http_request(&gw_addr, "GET", "/v1/topology", &[], "")
        .await
        .unwrap();
    assert_eq!(missing_auth.status_code, 401);

    let invalid_auth = send_http_request(
        &gw_addr,
        "GET",
        "/v1/topology",
        &[
            ("Authorization", "Bearer vdk.forged.badtag"),
            ("X-Org-Id", "acme"),
        ],
        "",
    )
    .await
    .unwrap();
    assert_eq!(invalid_auth.status_code, 401);

    // Whitespace-only X-Org-Id returns 400 Bad Request.
    let bad_org = send_http_request(
        &gw_addr,
        "GET",
        "/v1/topology",
        &[("Authorization", &reader_auth), ("X-Org-Id", "   ")],
        "",
    )
    .await
    .unwrap();
    assert_eq!(bad_org.status_code, 400);

    // 3. Cross-tenant access (acme key with X-Org-Id: other-org) returns 403 Forbidden.
    let cross_tenant = send_http_request(
        &gw_addr,
        "GET",
        "/v1/topology",
        &[("Authorization", &admin_auth), ("X-Org-Id", "other-org")],
        "",
    )
    .await
    .unwrap();
    assert_eq!(cross_tenant.status_code, 403);

    // 4. Reader key succeeds on /v1/topology, but receives 403 Forbidden on /v1/upsert, /v1/delete, and /v1/audit.
    let reader_topo = send_http_request(
        &gw_addr,
        "GET",
        "/v1/topology",
        &[("Authorization", &reader_auth), ("X-Org-Id", "acme")],
        "",
    )
    .await
    .unwrap();
    assert_eq!(reader_topo.status_code, 200);

    let upsert_body = r#"{"id":77,"vector":[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]}"#;
    let reader_upsert = send_http_request(
        &gw_addr,
        "POST",
        "/v1/upsert",
        &[("Authorization", &reader_auth), ("X-Org-Id", "acme")],
        upsert_body,
    )
    .await
    .unwrap();
    assert_eq!(reader_upsert.status_code, 403);

    let delete_body = r#"{"id":77}"#;
    let reader_delete = send_http_request(
        &gw_addr,
        "POST",
        "/v1/delete",
        &[("Authorization", &reader_auth), ("X-Org-Id", "acme")],
        delete_body,
    )
    .await
    .unwrap();
    assert_eq!(reader_delete.status_code, 403);

    let reader_audit = send_http_request(
        &gw_addr,
        "GET",
        "/v1/audit",
        &[("Authorization", &reader_auth), ("X-Org-Id", "acme")],
        "",
    )
    .await
    .unwrap();
    assert_eq!(reader_audit.status_code, 403);

    // 5. Writer key succeeds on /v1/upsert, /v1/search, and /v1/delete, and is denied /v1/audit (403).
    let writer_upsert = send_http_request(
        &gw_addr,
        "POST",
        "/v1/upsert",
        &[("Authorization", &writer_auth), ("X-Org-Id", "acme")],
        upsert_body,
    )
    .await
    .unwrap();
    assert_eq!(writer_upsert.status_code, 200);

    let search_body = r#"{"query":[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],"top_k":5,"ef_search":32}"#;
    let writer_search_before = send_http_request(
        &gw_addr,
        "POST",
        "/v1/search",
        &[("Authorization", &writer_auth), ("X-Org-Id", "acme")],
        search_body,
    )
    .await
    .unwrap();
    assert_eq!(writer_search_before.status_code, 200);
    assert!(writer_search_before.body.contains("\"id\":77"));

    let writer_delete = send_http_request(
        &gw_addr,
        "POST",
        "/v1/delete",
        &[("Authorization", &writer_auth), ("X-Org-Id", "acme")],
        delete_body,
    )
    .await
    .unwrap();
    assert_eq!(writer_delete.status_code, 200);

    let writer_search_after = send_http_request(
        &gw_addr,
        "POST",
        "/v1/search",
        &[("Authorization", &writer_auth), ("X-Org-Id", "acme")],
        search_body,
    )
    .await
    .unwrap();
    assert_eq!(writer_search_after.status_code, 200);
    assert!(!writer_search_after.body.contains("\"id\":77"));

    let writer_audit = send_http_request(
        &gw_addr,
        "GET",
        "/v1/audit",
        &[("Authorization", &writer_auth), ("X-Org-Id", "acme")],
        "",
    )
    .await
    .unwrap();
    assert_eq!(writer_audit.status_code, 403);

    // 6. Rate-limited tenant exhausts its 2-token burst and receives 429 Too Many Requests with Retry-After >= 1.
    for _ in 0..2 {
        let ok_resp = send_http_request(
            &gw_addr,
            "GET",
            "/v1/topology",
            &[
                ("Authorization", &throttled_auth),
                ("X-Org-Id", "throttled-org"),
            ],
            "",
        )
        .await
        .unwrap();
        assert_eq!(ok_resp.status_code, 200);
    }
    let throttled_resp = send_http_request(
        &gw_addr,
        "GET",
        "/v1/topology",
        &[
            ("Authorization", &throttled_auth),
            ("X-Org-Id", "throttled-org"),
        ],
        "",
    )
    .await
    .unwrap();
    assert_eq!(throttled_resp.status_code, 429);
    let retry_after_secs: u64 = throttled_resp
        .headers
        .get("retry-after")
        .expect("429 response must include Retry-After header")
        .parse()
        .unwrap();
    assert!(retry_after_secs >= 1);

    // 7. Admin key calling GET /v1/audit verifies the hash chain and reports non-zero entries_count & head_hash.
    let admin_audit = send_http_request(
        &gw_addr,
        "GET",
        "/v1/audit",
        &[("Authorization", &admin_auth), ("X-Org-Id", "acme")],
        "",
    )
    .await
    .unwrap();
    assert_eq!(admin_audit.status_code, 200);
    assert!(admin_audit.body.contains("\"status\":\"ok\""));
    assert!(admin_audit.body.contains("\"security_enabled\":true"));
    assert!(admin_audit.body.contains("\"chain_valid\":true"));
    assert!(!admin_audit.body.contains("\"entries_count\":0"));
    assert!(!admin_audit.body.contains("\"head_hash\":0"));

    h_gw.abort();
    h_coord.abort();
    h1.abort();
    h2.abort();
    h3.abort();
    let _ = std::fs::remove_dir_all(&dir1);
    let _ = std::fs::remove_dir_all(&dir2);
    let _ = std::fs::remove_dir_all(&dir3);
}
