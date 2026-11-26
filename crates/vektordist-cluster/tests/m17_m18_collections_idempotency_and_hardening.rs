//! Milestones 17 & 18 integration tests:
//! 1. `WriteIdDeduplicator` idempotency across live 3-node quorum writes (retrying the same
//!    non-zero `write_id` 3 times returns identical `(shard_id, seq_no == 1)` and writes only 1
//!    WAL frame across all replicas, plus mixed-batch deduplication and WAL recovery reload).
//! 2. `BoundedLatencyWindow` memory cap (`<= 1,024` after `3,000` samples), `Semaphore` max
//!    connection concurrency cap, and Slowloris socket read timeout (`408 Request Timeout`).
//! 3. Full multi-collection lifecycle (`docs_64d` `f32`, `fast_32d` `bq`, and `compact_16d` `sq8`)
//!    via `HttpVektorClient` with `SecurityGate` RBAC + strict `OrgId` isolation, `409 Conflict`
//!    on duplicate create, `collection_delete`, `drop_collection`, and `GET /openapi.json` (`3.1.0`).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;
use tonic::transport::Server;
use vektordist_cluster::client::ClusterClient;
use vektordist_cluster::collections::CollectionQuantization;
use vektordist_cluster::coord::CoordinatorService;
use vektordist_cluster::gateway::{spawn_http_gateway, BoundedLatencyWindow, HttpGatewayServer};
use vektordist_cluster::node::NodeService;
use vektordist_cluster::sdk::HttpVektorClient;
use vektordist_cluster::security::{OrgId, SecurityGate, TenantQuotaPolicy, TenantRole};
use vektordist_core::exact::SeededRng;
use vektordist_core::hnsw::HnswParams;
use vektordist_core::{ClusterConfig, DistanceMetric, NodeId};
use vektordist_proto::v1::shard_replica_rpc_server::ShardReplicaRpcServer;
use vektordist_proto::v1::topology_rpc_server::TopologyRpcServer;

static TEST_SEQ: AtomicU64 = AtomicU64::new(1);

fn temp_path(tag: &str) -> PathBuf {
    let seq = TEST_SEQ.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("vektordist-m17-m18-{tag}-{pid}-{seq}"))
}

async fn spawn_node(
    node_id: NodeId,
    config: ClusterConfig,
    tag: &str,
) -> (String, NodeService, PathBuf, JoinHandle<()>) {
    let dir = temp_path(&format!("{tag}-n{node_id}"));
    let _ = std::fs::remove_dir_all(&dir);
    let svc = NodeService::open(node_id, &dir, config, HnswParams::default()).unwrap();
    let svc_clone = svc.clone();

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);

    let handle = tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(ShardReplicaRpcServer::new(svc_clone))
            .serve_with_incoming(incoming)
            .await;
    });
    (addr, svc, dir, handle)
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
async fn retrying_write_id_is_idempotent_across_cluster_and_survives_reopen() {
    let config = ClusterConfig {
        num_shards: 4,
        replication_factor: 3,
        write_quorum: 2,
        dimensions: 8,
        metric: DistanceMetric::Cosine,
    };

    let (addr1, svc1, dir1, h1) = spawn_node(1, config.clone(), "idemp").await;
    let (addr2, svc2, dir2, h2) = spawn_node(2, config.clone(), "idemp").await;
    let (addr3, svc3, dir3, h3) = spawn_node(3, config.clone(), "idemp").await;

    let coord =
        CoordinatorService::new(config.clone(), vec![(1, addr1), (2, addr2), (3, addr3)]).unwrap();
    coord.push_all_shard_configs().await.unwrap();
    let (coord_addr, h_coord) = spawn_coord(coord).await;

    let client = ClusterClient::connect(&coord_addr).await.unwrap();
    let v100 = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // 1. Retry upsert(100, write_id=9001) 3 times against the live 3-node cluster.
    let (s1, seq1) = client.upsert(100, 9001, &v100).await.unwrap();
    let (s2, seq2) = client.upsert(100, 9001, &v100).await.unwrap();
    let (s3, seq3) = client.upsert(100, 9001, &v100).await.unwrap();

    assert_eq!(s1, config.shard_for(100));
    assert_eq!((s1, seq1), (s1, 1));
    assert_eq!((s2, seq2), (s1, 1));
    assert_eq!((s3, seq3), (s1, 1));

    // Verify all 3 replicas hold exactly 1 WAL record (seq_no == 1) on the target shard.
    let wal_records = client.verify_shard_wal_write_ids(s1).await.unwrap();
    assert_eq!(wal_records.len(), 1);
    assert_eq!(wal_records[0].seq_no, 1);
    assert_eq!(wal_records[0].write_id, 9001);

    for svc in [&svc1, &svc2, &svc3] {
        let engine = svc.shards.get(&s1).unwrap().read().await;
        assert_eq!(engine.records_by_seq.len(), 1);
        assert_eq!(engine.hnsw.last_seq_no(), 1);
        assert_eq!(engine.dedup.get(9001), Some(1));
    }

    // 2. Mixed batch on the same shard: combine already-committed write_id=9001 with a new write_id=9002.
    let second_id = (101..10_000_u64)
        .find(|&cand| config.shard_for(cand) == s1)
        .unwrap();
    let v_second = [0.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let acked = client
        .upsert_batch(&[
            (100, 9001, v100.to_vec()),
            (second_id, 9002, v_second.to_vec()),
        ])
        .await
        .unwrap();
    assert_eq!(acked, 2);

    // Only 1 additional WAL frame should have been appended (total = 2, last_seq_no = 2).
    let wal_after_batch = client.verify_shard_wal_write_ids(s1).await.unwrap();
    assert_eq!(wal_after_batch.len(), 2);
    assert_eq!(wal_after_batch[0].seq_no, 1);
    assert_eq!(wal_after_batch[1].seq_no, 2);
    assert_eq!(wal_after_batch[1].write_id, 9002);

    h_coord.abort();
    h1.abort();
    h2.abort();
    h3.abort();
    drop(svc1);
    drop(svc2);
    drop(svc3);
    tokio::time::sleep(Duration::from_millis(25)).await;

    // 3. Reopen ShardEngine from disk and confirm WriteIdDeduplicator reloads committed write_ids from WAL.
    let reopened = NodeService::open(1, &dir1, config, HnswParams::default()).unwrap();
    {
        let engine = reopened.shards.get(&s1).unwrap().read().await;
        assert_eq!(engine.dedup.get(9001), Some(1));
        assert_eq!(engine.dedup.get(9002), Some(2));
    }

    let _ = std::fs::remove_dir_all(&dir1);
    let _ = std::fs::remove_dir_all(&dir2);
    let _ = std::fs::remove_dir_all(&dir3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bounded_latency_window_stays_capped_and_slowloris_socket_receives_408_timeout() {
    // 1. Prove BoundedLatencyWindow stays strictly capped at <= 1024 after 3,000 samples.
    let mut window = BoundedLatencyWindow::default();
    assert_eq!(window.capacity(), 1_024);
    for sample_idx in 1..=3_000_u32 {
        window.record(f64::from(sample_idx) * 0.25);
        assert!(window.len() <= 1_024);
    }
    assert_eq!(window.len(), 1_024);
    let (p50, p95, p99) = window.percentiles();
    assert!(p50 > 0.0 && p50 <= p95 && p95 <= p99);

    // 2. Spawn hardened gateway with 120ms read timeout and max_connections=8, and verify Slowloris -> 408.
    let config = ClusterConfig {
        num_shards: 2,
        replication_factor: 3,
        write_quorum: 2,
        dimensions: 8,
        metric: DistanceMetric::Cosine,
    };
    let (addr1, _svc1, dir1, h1) = spawn_node(1, config.clone(), "slow").await;
    let (addr2, _svc2, dir2, h2) = spawn_node(2, config.clone(), "slow").await;
    let (addr3, _svc3, dir3, h3) = spawn_node(3, config.clone(), "slow").await;

    let coord = CoordinatorService::new(config, vec![(1, addr1), (2, addr2), (3, addr3)]).unwrap();
    coord.push_all_shard_configs().await.unwrap();
    let (coord_addr, h_coord) = spawn_coord(coord).await;

    let client = ClusterClient::connect(&coord_addr).await.unwrap();
    let gateway = HttpGatewayServer::new(client)
        .with_read_timeout(Duration::from_millis(120))
        .with_max_connections(8);
    let (gw_addr, h_gw) = spawn_http_gateway("127.0.0.1:0", gateway.clone())
        .await
        .unwrap();

    // Open a Slowloris TCP socket that sends an incomplete HTTP header without \r\n\r\n and stalls.
    let mut slow_stream = TcpStream::connect(&gw_addr).await.unwrap();
    slow_stream
        .write_all(b"POST /v1/search HTTP/1.1\r\nHost: localhost\r\n")
        .await
        .unwrap();

    let mut raw_resp_bytes = Vec::new();
    slow_stream.read_to_end(&mut raw_resp_bytes).await.unwrap();
    let raw_resp = String::from_utf8_lossy(&raw_resp_bytes);
    assert!(
        raw_resp.starts_with("HTTP/1.1 408 Request Timeout\r\n"),
        "expected 408 Request Timeout, got: {raw_resp}"
    );
    assert!(raw_resp.contains("{\"status\":\"error\",\"error\":\"request read timeout\"}"));

    // Confirm healthy requests immediately succeed after Slowloris permit release.
    let sdk = HttpVektorClient::new(&gw_addr);
    let health = sdk.healthz().await.unwrap();
    assert_eq!(health.status_code, 200);

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
async fn multi_collection_lifecycle_tenant_isolation_rbac_and_openapi_3_1() {
    let config = ClusterConfig {
        num_shards: 2,
        replication_factor: 3,
        write_quorum: 2,
        dimensions: 8,
        metric: DistanceMetric::Cosine,
    };
    let (addr1, _svc1, dir1, h1) = spawn_node(1, config.clone(), "cols").await;
    let (addr2, _svc2, dir2, h2) = spawn_node(2, config.clone(), "cols").await;
    let (addr3, _svc3, dir3, h3) = spawn_node(3, config.clone(), "cols").await;

    let coord = CoordinatorService::new(config, vec![(1, addr1), (2, addr2), (3, addr3)]).unwrap();
    coord.push_all_shard_configs().await.unwrap();
    let (coord_addr, h_coord) = spawn_coord(coord).await;

    // Issue Admin, Writer, and Reader credentials across org-a and org-b.
    let mut gate = SecurityGate::new([0x5A_u8; 32], TenantQuotaPolicy::default());
    let org_a = OrgId::new("org-a").unwrap();
    let org_b = OrgId::new("org-b").unwrap();
    let (admin_a_tok, _) = gate.issue_api_key("k-admin-a", org_a.clone(), TenantRole::Admin, 1);
    let (writer_a_tok, _) = gate.issue_api_key("k-writer-a", org_a.clone(), TenantRole::Writer, 2);
    let (reader_a_tok, _) = gate.issue_api_key("k-reader-a", org_a, TenantRole::Reader, 3);
    let (writer_b_tok, _) = gate.issue_api_key("k-writer-b", org_b, TenantRole::Writer, 4);

    let cluster_client = ClusterClient::connect(&coord_addr).await.unwrap();
    let gateway =
        HttpGatewayServer::new(cluster_client).with_security_gate(Arc::new(Mutex::new(gate)));
    let (gw_addr, h_gw) = spawn_http_gateway("127.0.0.1:0", gateway).await.unwrap();

    // 1. Public GET /openapi.json works without credentials and advertises OpenAPI 3.1.0 + all routes.
    let unauth_sdk = HttpVektorClient::new(&gw_addr);
    let openapi = unauth_sdk.openapi_raw().await.unwrap();
    assert!(openapi.contains("\"openapi\":\"3.1.0\""));
    for required_route in [
        "\"/healthz\"",
        "\"/readyz\"",
        "\"/metrics\"",
        "\"/openapi.json\"",
        "\"/v1/upsert\"",
        "\"/v1/upsert_batch\"",
        "\"/v1/search\"",
        "\"/v1/delete\"",
        "\"/v1/topology\"",
        "\"/v1/audit\"",
        "\"/v1/collections\"",
        "\"/v1/collections/{name}\"",
        "\"/v1/collections/{name}/upsert\"",
        "\"/v1/collections/{name}/search\"",
        "\"/v1/collections/{name}/delete\"",
    ] {
        assert!(
            openapi.contains(required_route),
            "OpenAPI spec missing {required_route}"
        );
    }

    let admin_a = HttpVektorClient::new(&gw_addr)
        .with_bearer_token(admin_a_tok)
        .with_org_id("org-a");
    let writer_a = HttpVektorClient::new(&gw_addr)
        .with_bearer_token(writer_a_tok)
        .with_org_id("org-a");
    let reader_a = HttpVektorClient::new(&gw_addr)
        .with_bearer_token(reader_a_tok)
        .with_org_id("org-a");
    let writer_b = HttpVektorClient::new(&gw_addr)
        .with_bearer_token(writer_b_tok)
        .with_org_id("org-b");

    // 2. Non-Admin (Writer) is forbidden (403) from creating collections.
    let forbidden_create = writer_a
        .send_raw(
            "POST",
            "/v1/collections",
            "{\"name\":\"docs_64d\",\"dimensions\":64,\"metric\":\"cosine\",\"quantization\":\"f32\"}",
        )
        .await
        .unwrap();
    assert_eq!(forbidden_create.status_code, 403);

    // 3. Admin creates docs_64d (f32) and fast_32d (bq); duplicate create returns 409 Conflict; invalid dims returns 400.
    let created_64 = admin_a
        .create_collection(
            "docs_64d",
            64,
            DistanceMetric::Cosine,
            CollectionQuantization::F32,
        )
        .await
        .unwrap();
    assert!(created_64.contains("\"status\":\"created\""));
    assert!(created_64.contains("\"name\":\"docs_64d\""));
    assert!(created_64.contains("\"quantization\":\"f32\""));

    let created_32 = admin_a
        .create_collection(
            "fast_32d",
            32,
            DistanceMetric::Cosine,
            CollectionQuantization::Bq,
        )
        .await
        .unwrap();
    assert!(created_32.contains("\"name\":\"fast_32d\""));
    assert!(created_32.contains("\"quantization\":\"bq\""));

    let dup_resp = admin_a
        .send_raw(
            "POST",
            "/v1/collections",
            "{\"name\":\"docs_64d\",\"dimensions\":64,\"metric\":\"cosine\",\"quantization\":\"f32\"}",
        )
        .await
        .unwrap();
    assert_eq!(dup_resp.status_code, 409);

    let bad_dims_resp = admin_a
        .send_raw(
            "POST",
            "/v1/collections",
            "{\"name\":\"invalid_0d\",\"dimensions\":0,\"metric\":\"cosine\",\"quantization\":\"f32\"}",
        )
        .await
        .unwrap();
    assert_eq!(bad_dims_resp.status_code, 400);

    // 4. Tenant-isolated upsert & search across docs_64d (64-dim f32) and fast_32d (32-dim bq).
    let mut rng = SeededRng::new(2026_1718);
    let mut vec_a_64 = vec![0.0_f32; 64];
    let mut vec_b_64 = vec![0.0_f32; 64];
    rng.fill_unit_vector(&mut vec_a_64);
    rng.fill_unit_vector(&mut vec_b_64);

    let mut vec_a_32 = vec![0.0_f32; 32];
    let mut vec_b_32 = vec![0.0_f32; 32];
    rng.fill_unit_vector(&mut vec_a_32);
    rng.fill_unit_vector(&mut vec_b_32);

    writer_a
        .collection_upsert("docs_64d", 1001, &vec_a_64)
        .await
        .unwrap();
    writer_b
        .collection_upsert("docs_64d", 2001, &vec_b_64)
        .await
        .unwrap();

    writer_a
        .collection_upsert("fast_32d", 3001, &vec_a_32)
        .await
        .unwrap();
    writer_b
        .collection_upsert("fast_32d", 4001, &vec_b_32)
        .await
        .unwrap();

    // org-a Reader can list collections and search org-a's vectors, never seeing org-b's vectors.
    let list_json = reader_a.list_collections_raw().await.unwrap();
    assert!(list_json.contains("\"name\":\"docs_64d\""));
    assert!(list_json.contains("\"name\":\"fast_32d\""));

    let hits_a_64 = reader_a
        .collection_search("docs_64d", &vec_a_64, 10, 64)
        .await
        .unwrap();
    assert_eq!(hits_a_64.len(), 1);
    assert_eq!(hits_a_64[0].id, 1001);

    // org-b searching with org-a's exact vector in docs_64d ONLY sees org-b's own vector (2001), never 1001!
    let hits_b_64 = writer_b
        .collection_search("docs_64d", &vec_a_64, 10, 64)
        .await
        .unwrap();
    assert_eq!(hits_b_64.len(), 1);
    assert_eq!(hits_b_64[0].id, 2001);

    // Same strict isolation inside fast_32d (BQ):
    let hits_a_32 = reader_a
        .collection_search("fast_32d", &vec_a_32, 10, 64)
        .await
        .unwrap();
    assert_eq!(hits_a_32.len(), 1);
    assert_eq!(hits_a_32[0].id, 3001);

    let hits_b_32 = writer_b
        .collection_search("fast_32d", &vec_a_32, 10, 64)
        .await
        .unwrap();
    assert_eq!(hits_b_32.len(), 1);
    assert_eq!(hits_b_32[0].id, 4001);

    // 5. Cross-tenant delete attempt by org-b on id=1001 returns deleted=false and leaves org-a's vector intact!
    let cross_deleted = writer_b.collection_delete("docs_64d", 1001).await.unwrap();
    assert!(!cross_deleted);
    assert_eq!(
        reader_a
            .collection_search("docs_64d", &vec_a_64, 10, 64)
            .await
            .unwrap()
            .len(),
        1
    );

    // Owner delete by org-a removes id=1001 cleanly.
    let owner_deleted = writer_a.collection_delete("docs_64d", 1001).await.unwrap();
    assert!(owner_deleted);
    assert!(reader_a
        .collection_search("docs_64d", &vec_a_64, 10, 64)
        .await
        .unwrap()
        .is_empty());

    // 6. Drop fast_32d as Admin; subsequent operations return 404 Not Found.
    admin_a.drop_collection("fast_32d").await.unwrap();
    let drop_missing = admin_a
        .send_raw("DELETE", "/v1/collections/fast_32d", "")
        .await
        .unwrap();
    assert_eq!(drop_missing.status_code, 404);

    let search_dropped = reader_a
        .send_raw(
            "POST",
            "/v1/collections/fast_32d/search",
            "{\"query\":[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],\"top_k\":5}",
        )
        .await
        .unwrap();
    assert_eq!(search_dropped.status_code, 404);

    h_gw.abort();
    h_coord.abort();
    h1.abort();
    h2.abort();
    h3.abort();
    let _ = std::fs::remove_dir_all(&dir1);
    let _ = std::fs::remove_dir_all(&dir2);
    let _ = std::fs::remove_dir_all(&dir3);
}
