//! Milestone 20 adversarial security penetration & durable `FP16` multi-collection tests:
//! 1. RFC 9112 HTTP request smuggling (`Transfer-Encoding: chunked`, duplicate conflicting
//!    `Content-Length`, negative/signed `Content-Length`, malformed request lines) & percent-encoded
//!    path traversal (`/v1/collections/%2e%2e%2fadmin`, `%00` null bytes) -> `400 Bad Request`.
//! 2. Adversarial JSON string-key shadowing (`{"note": "\"id\": 999999", "id": 42, ...}`) &
//!    non-finite float rejection (`NaN`, `Infinity`, `1e999` -> `400`, scientific `1.25e-2` -> `200`).
//! 3. Defense-in-depth HTTP security headers (`nosniff`, `DENY`, `no-referrer`, `no-store`),
//!    CORS `OPTIONS` preflight (`204 No Content`), and OpenAPI `0.4.0` `/` + `/console` paths.
//! 4. Crash-safe durable `FP16` + `SQ8` multi-collection persistence across gateway restarts
//!    (`200` vectors across `org-alpha` and `org-beta`, `recall@10 >= 0.98`) & CRC32 bit-rot detection.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;
use tonic::transport::Server;
use vektordist_cluster::client::ClusterClient;
use vektordist_cluster::collections::{
    CollectionQuantization, CollectionRegistry, COLLECTIONS_SNAPSHOT_FILENAME,
};
use vektordist_cluster::coord::CoordinatorService;
use vektordist_cluster::gateway::{send_http_request, spawn_http_gateway, HttpGatewayServer};
use vektordist_cluster::node::NodeService;
use vektordist_cluster::sdk::HttpVektorClient;
use vektordist_core::exact::{ExactIndex, ScoredId, SeededRng};
use vektordist_core::hnsw::HnswParams;
use vektordist_core::{ClusterConfig, DistanceMetric, NodeId, VektorError};
use vektordist_proto::v1::shard_replica_rpc_server::ShardReplicaRpcServer;
use vektordist_proto::v1::topology_rpc_server::TopologyRpcServer;

static TEST_SEQ: AtomicU64 = AtomicU64::new(1);

fn temp_path(tag: &str) -> PathBuf {
    let seq = TEST_SEQ.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("vektordist-m20-{tag}-{pid}-{seq}"))
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

async fn send_raw_tcp_payload(addr: &str, raw_bytes: &[u8]) -> String {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let _ = stream.write_all(raw_bytes).await;
    let _ = stream.shutdown().await;
    let mut resp = Vec::new();
    stream.read_to_end(&mut resp).await.unwrap();
    String::from_utf8_lossy(&resp).into_owned()
}

fn assert_security_headers(headers: &std::collections::BTreeMap<String, String>) {
    assert_eq!(
        headers.get("x-content-type-options").map(String::as_str),
        Some("nosniff")
    );
    assert_eq!(
        headers.get("x-frame-options").map(String::as_str),
        Some("DENY")
    );
    assert_eq!(
        headers.get("referrer-policy").map(String::as_str),
        Some("no-referrer")
    );
    assert_eq!(
        headers.get("cache-control").map(String::as_str),
        Some("no-store")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn http_request_smuggling_percent_traversal_and_malformed_frames_return_400() {
    let config = ClusterConfig {
        num_shards: 2,
        replication_factor: 3,
        write_quorum: 2,
        dimensions: 8,
        metric: DistanceMetric::Cosine,
    };
    let (addr1, _svc1, dir1, h1) = spawn_node(1, config.clone(), "smug").await;
    let (addr2, _svc2, dir2, h2) = spawn_node(2, config.clone(), "smug").await;
    let (addr3, _svc3, dir3, h3) = spawn_node(3, config.clone(), "smug").await;

    let coord = CoordinatorService::new(config, vec![(1, addr1), (2, addr2), (3, addr3)]).unwrap();
    coord.push_all_shard_configs().await.unwrap();
    let (coord_addr, h_coord) = spawn_coord(coord).await;

    let client = ClusterClient::connect(&coord_addr).await.unwrap();
    let gateway = HttpGatewayServer::new(client).with_read_timeout(Duration::from_millis(500));
    let (gw_addr, h_gw) = spawn_http_gateway("127.0.0.1:0", gateway).await.unwrap();

    // 1. Transfer-Encoding: chunked CL.TE smuggling attempt -> 400 Bad Request
    let te_resp = send_raw_tcp_payload(
        &gw_addr,
        b"POST /v1/upsert HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
    )
    .await;
    assert!(
        te_resp.starts_with("HTTP/1.1 400 Bad Request\r\n"),
        "expected 400 on Transfer-Encoding, got: {te_resp}"
    );
    assert!(te_resp.contains("Transfer-Encoding is not supported; use Content-Length"));

    // 2. Duplicate conflicting Content-Length headers -> 400 Bad Request
    let dup_cl_resp = send_raw_tcp_payload(
        &gw_addr,
        b"POST /v1/upsert HTTP/1.1\r\nHost: localhost\r\nContent-Length: 10\r\nContent-Length: 20\r\n\r\n{\"id\":100}",
    )
    .await;
    assert!(
        dup_cl_resp.starts_with("HTTP/1.1 400 Bad Request\r\n"),
        "expected 400 on conflicting Content-Length, got: {dup_cl_resp}"
    );
    assert!(dup_cl_resp.contains("invalid or conflicting Content-Length header"));

    // 3. Negative and signed Content-Length headers -> 400 Bad Request
    for bad_cl in ["-1", "+10", "12abc", "999999999999999999999999999999"] {
        let raw = format!(
            "POST /v1/upsert HTTP/1.1\r\nHost: localhost\r\nContent-Length: {bad_cl}\r\n\r\n"
        );
        let resp = send_raw_tcp_payload(&gw_addr, raw.as_bytes()).await;
        assert!(
            resp.starts_with("HTTP/1.1 400 Bad Request\r\n"),
            "expected 400 on Content-Length: {bad_cl}, got: {resp}"
        );
        assert!(resp.contains("invalid or conflicting Content-Length header"));
    }

    // 4. Malformed HTTP request line (missing HTTP version) -> 400 Bad Request
    let malformed_line =
        send_raw_tcp_payload(&gw_addr, b"GET /healthz\r\nHost: localhost\r\n\r\n").await;
    assert!(malformed_line.starts_with("HTTP/1.1 400 Bad Request\r\n"));

    // 5. Percent-encoded path traversal & null-byte injection on /v1/collections/* -> 400 Bad Request
    for bad_path in [
        "/v1/collections/%2e%2e%2fadmin",
        "/v1/collections/..%2fsecret",
        "/v1/collections/%2e%2e/upsert",
        "/v1/collections/col%00admin",
        "/v1/collections/col%5cadmin",
    ] {
        let resp = send_http_request(&gw_addr, "DELETE", bad_path, &[], "")
            .await
            .unwrap();
        assert_eq!(
            resp.status_code, 400,
            "expected 400 for adversarial path {bad_path}, got {}: {}",
            resp.status_code, resp.body
        );
        assert_security_headers(&resp.headers);
    }

    // Confirm gateway remains completely healthy and unpoisoned after penetration barrage.
    let sdk = HttpVektorClient::new(&gw_addr);
    let health = sdk.healthz().await.unwrap();
    assert_eq!(health.status_code, 200);
    assert!(health.ok);

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
async fn adversarial_json_key_shadowing_non_finite_floats_security_headers_and_cors() {
    let config = ClusterConfig {
        num_shards: 2,
        replication_factor: 3,
        write_quorum: 2,
        dimensions: 8,
        metric: DistanceMetric::Cosine,
    };
    let (addr1, _svc1, dir1, h1) = spawn_node(1, config.clone(), "jsonsec").await;
    let (addr2, _svc2, dir2, h2) = spawn_node(2, config.clone(), "jsonsec").await;
    let (addr3, _svc3, dir3, h3) = spawn_node(3, config.clone(), "jsonsec").await;

    let coord = CoordinatorService::new(config, vec![(1, addr1), (2, addr2), (3, addr3)]).unwrap();
    coord.push_all_shard_configs().await.unwrap();
    let (coord_addr, h_coord) = spawn_coord(coord).await;

    let client = ClusterClient::connect(&coord_addr).await.unwrap();
    let gateway = HttpGatewayServer::new(client).with_cors_origin("https://console.example.com");
    let (gw_addr, h_gw) = spawn_http_gateway("127.0.0.1:0", gateway).await.unwrap();

    // 1. CORS OPTIONS preflight returns 204 No Content with CORS + security headers.
    let preflight = send_http_request(&gw_addr, "OPTIONS", "/v1/search", &[], "")
        .await
        .unwrap();
    assert_eq!(preflight.status_code, 204);
    assert_security_headers(&preflight.headers);
    assert_eq!(
        preflight
            .headers
            .get("access-control-allow-origin")
            .map(String::as_str),
        Some("https://console.example.com")
    );
    assert_eq!(
        preflight
            .headers
            .get("access-control-allow-methods")
            .map(String::as_str),
        Some("GET, POST, DELETE, OPTIONS")
    );
    assert_eq!(
        preflight
            .headers
            .get("access-control-allow-headers")
            .map(String::as_str),
        Some("authorization, content-type, x-org-id, traceparent")
    );

    // 2. OpenAPI spec bumped to 0.4.0 and exposes "/" and "/console".
    let openapi_resp = send_http_request(&gw_addr, "GET", "/openapi.json", &[], "")
        .await
        .unwrap();
    assert_eq!(openapi_resp.status_code, 200);
    assert_security_headers(&openapi_resp.headers);
    assert!(openapi_resp.body.contains("\"version\":\"0.4.0\""));
    assert!(openapi_resp.body.contains("\"/\":"));
    assert!(openapi_resp.body.contains("\"/console\":"));

    // 3. Adversarial JSON string shadowing: `"note": "\"id\": 999999"` MUST NOT shadow `"id": 42`!
    let shadowed_payload =
        "{\"note\": \"\\\"id\\\": 999999\", \"id\": 42, \"vector\": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}";
    let upsert_resp = send_http_request(&gw_addr, "POST", "/v1/upsert", &[], shadowed_payload)
        .await
        .unwrap();
    assert_eq!(upsert_resp.status_code, 200);
    assert_security_headers(&upsert_resp.headers);
    assert_eq!(
        upsert_resp
            .headers
            .get("access-control-allow-origin")
            .map(String::as_str),
        Some("https://console.example.com")
    );

    let sdk = HttpVektorClient::new(&gw_addr);
    let hits = sdk
        .search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10, 32)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0].id, 42,
        "must upsert real key id=42, never shadowed 999999"
    );

    // 4. Non-finite float payloads (`NaN`, `Infinity`, `-Infinity`, `1e999`) return 400 Bad Request.
    for bad_vec_body in [
        "{\"id\": 50, \"vector\": [NaN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}",
        "{\"id\": 51, \"vector\": [Infinity, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}",
        "{\"id\": 52, \"vector\": [-Infinity, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}",
        "{\"id\": 53, \"vector\": [1e999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}",
    ] {
        let bad_resp = send_http_request(&gw_addr, "POST", "/v1/upsert", &[], bad_vec_body)
            .await
            .unwrap();
        assert_eq!(
            bad_resp.status_code, 400,
            "expected 400 for non-finite float body {bad_vec_body}, got {}",
            bad_resp.status_code
        );
        assert_security_headers(&bad_resp.headers);
    }

    // 5. Valid scientific notation (`1.25e-2`, `-2.5E+1`) succeeds cleanly.
    let sci_body = "{\"id\": 77, \"vector\": [1.25e-2, -2.5E+1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}";
    let sci_resp = send_http_request(&gw_addr, "POST", "/v1/upsert", &[], sci_body)
        .await
        .unwrap();
    assert_eq!(sci_resp.status_code, 200);

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
async fn durable_fp16_and_sq8_collections_survive_gateway_restart_and_detect_bit_rot() {
    let config = ClusterConfig {
        num_shards: 2,
        replication_factor: 3,
        write_quorum: 2,
        dimensions: 8,
        metric: DistanceMetric::Cosine,
    };
    let (addr1, _svc1, dir1, h1) = spawn_node(1, config.clone(), "dur").await;
    let (addr2, _svc2, dir2, h2) = spawn_node(2, config.clone(), "dur").await;
    let (addr3, _svc3, dir3, h3) = spawn_node(3, config.clone(), "dur").await;

    let coord = CoordinatorService::new(config, vec![(1, addr1), (2, addr2), (3, addr3)]).unwrap();
    coord.push_all_shard_configs().await.unwrap();
    let (coord_addr, h_coord) = spawn_coord(coord).await;

    let col_dir = temp_path("collections-durable");
    let _ = std::fs::remove_dir_all(&col_dir);

    let dims = 32_usize;
    let mut rng = SeededRng::new(2026_0020);
    let mut alpha_vecs: Vec<(u64, Vec<f32>)> = Vec::with_capacity(100);
    let mut beta_vecs: Vec<(u64, Vec<f32>)> = Vec::with_capacity(100);

    for i in 0..100_u64 {
        let mut va = vec![0.0_f32; dims];
        rng.fill_unit_vector(&mut va);
        alpha_vecs.push((1_000 + i, va));

        let mut vb = vec![0.0_f32; dims];
        rng.fill_unit_vector(&mut vb);
        beta_vecs.push((2_000 + i, vb));
    }

    let mut queries: Vec<Vec<f32>> = Vec::with_capacity(20);
    for _ in 0..20 {
        let mut q = vec![0.0_f32; dims];
        rng.fill_unit_vector(&mut q);
        queries.push(q);
    }

    // Phase 1: Start first gateway backed by `col_dir`, create `fp16` and `sq8` collections,
    // upsert 200 vectors across `org-alpha` and `org-beta`, and measure recall@10 >= 0.98.
    let pre_restart_alpha_hits: Vec<Vec<u64>> = {
        let cluster_client = ClusterClient::connect(&coord_addr).await.unwrap();
        let gw = HttpGatewayServer::new(cluster_client)
            .with_collections_dir(&col_dir)
            .unwrap();
        let (gw_addr, h_gw) = spawn_http_gateway("127.0.0.1:0", gw).await.unwrap();

        let alpha_sdk = HttpVektorClient::new(&gw_addr).with_org_id("org-alpha");
        let beta_sdk = HttpVektorClient::new(&gw_addr).with_org_id("org-beta");

        let created_fp16 = alpha_sdk
            .create_collection(
                "emb_fp16_32d",
                dims,
                DistanceMetric::Cosine,
                CollectionQuantization::Fp16,
            )
            .await
            .unwrap();
        assert!(created_fp16.contains("\"quantization\":\"fp16\""));

        let created_sq8 = alpha_sdk
            .create_collection(
                "emb_sq8_32d",
                dims,
                DistanceMetric::Cosine,
                CollectionQuantization::Sq8,
            )
            .await
            .unwrap();
        assert!(created_sq8.contains("\"quantization\":\"sq8\""));

        for (id, vec) in &alpha_vecs {
            alpha_sdk
                .collection_upsert("emb_fp16_32d", *id, vec)
                .await
                .unwrap();
            alpha_sdk
                .collection_upsert("emb_sq8_32d", *id, vec)
                .await
                .unwrap();
        }
        for (id, vec) in &beta_vecs {
            beta_sdk
                .collection_upsert("emb_fp16_32d", *id, vec)
                .await
                .unwrap();
            beta_sdk
                .collection_upsert("emb_sq8_32d", *id, vec)
                .await
                .unwrap();
        }

        let mut exact_alpha = ExactIndex::new(dims, DistanceMetric::Cosine).unwrap();
        for (id, vec) in &alpha_vecs {
            exact_alpha.upsert(*id, vec).unwrap();
        }

        let mut total_recall = 0.0_f64;
        let mut recorded_hits = Vec::with_capacity(queries.len());
        for q in &queries {
            let exact: Vec<ScoredId> = exact_alpha.search(q, 10).unwrap();
            let hits = alpha_sdk
                .collection_search("emb_fp16_32d", q, 10, 96)
                .await
                .unwrap();
            assert_eq!(hits.len(), 10);
            // Strict tenant isolation: org-alpha never sees org-beta IDs (2000..2100).
            assert!(hits.iter().all(|h| (1_000..1_100).contains(&h.id)));
            let overlap = hits
                .iter()
                .filter(|h| exact.iter().any(|gt| gt.id == h.id))
                .count();
            total_recall += (overlap as f64) / (exact.len().max(1) as f64);
            recorded_hits.push(hits.iter().map(|h| h.id).collect());
        }
        let mean_recall = total_recall / (queries.len() as f64);
        assert!(
            mean_recall >= 0.98,
            "expected FP16 recall@10 >= 0.98, got {mean_recall:.4}"
        );

        h_gw.abort();
        tokio::time::sleep(Duration::from_millis(25)).await;
        recorded_hits
    };

    // Verify `collections.vdcl` exists on disk.
    let vdcl_path = col_dir.join(COLLECTIONS_SNAPSHOT_FILENAME);
    assert!(vdcl_path.exists(), "collections.vdcl snapshot must exist");

    // Phase 2: Start a brand-new `HttpGatewayServer::with_collections_dir(&col_dir)` and verify
    // both collections and all 200 tenant vectors are immediately queryable with identical results!
    {
        let cluster_client2 = ClusterClient::connect(&coord_addr).await.unwrap();
        let gw2 = HttpGatewayServer::new(cluster_client2)
            .with_collections_dir(&col_dir)
            .unwrap();
        let (gw2_addr, h_gw2) = spawn_http_gateway("127.0.0.1:0", gw2).await.unwrap();

        let alpha_sdk2 = HttpVektorClient::new(&gw2_addr).with_org_id("org-alpha");
        let beta_sdk2 = HttpVektorClient::new(&gw2_addr).with_org_id("org-beta");

        let list_raw = alpha_sdk2.list_collections_raw().await.unwrap();
        assert!(list_raw.contains("\"name\":\"emb_fp16_32d\""));
        assert!(list_raw.contains("\"name\":\"emb_sq8_32d\""));
        assert!(list_raw.contains("\"live_vectors\":200"));

        for (q, expected_ids) in queries.iter().zip(pre_restart_alpha_hits.iter()) {
            let hits_after = alpha_sdk2
                .collection_search("emb_fp16_32d", q, 10, 96)
                .await
                .unwrap();
            let actual_ids: Vec<u64> = hits_after.iter().map(|h| h.id).collect();
            assert_eq!(
                &actual_ids, expected_ids,
                "restored FP16 HNSW graph must return identical top-10 IDs after gateway restart"
            );
        }

        // Verify org-beta vectors in both emb_fp16_32d and emb_sq8_32d are intact and isolated.
        let beta_probe = &beta_vecs[0].1;
        let beta_fp16_hits = beta_sdk2
            .collection_search("emb_fp16_32d", beta_probe, 5, 64)
            .await
            .unwrap();
        assert_eq!(beta_fp16_hits[0].id, beta_vecs[0].0);
        assert!(beta_fp16_hits
            .iter()
            .all(|h| (2_000..2_100).contains(&h.id)));

        let beta_sq8_hits = beta_sdk2
            .collection_search("emb_sq8_32d", beta_probe, 5, 64)
            .await
            .unwrap();
        assert_eq!(beta_sq8_hits[0].id, beta_vecs[0].0);
        assert!(beta_sq8_hits.iter().all(|h| (2_000..2_100).contains(&h.id)));

        // Verify GET / and GET /console serve the embedded zero-trust Web Control Console with CSP!
        for console_path in ["/", "/console"] {
            let console_resp = alpha_sdk2.send_raw("GET", console_path, "").await.unwrap();
            assert_eq!(console_resp.status_code, 200);
            assert!(console_resp
                .headers
                .get("content-type")
                .is_some_and(|ct| ct.contains("text/html")));
            assert!(console_resp
                .headers
                .get("content-security-policy")
                .is_some_and(|csp| csp.contains("default-src 'none'")
                    && csp.contains("frame-ancestors 'none'")));
            assert_eq!(
                console_resp
                    .headers
                    .get("x-frame-options")
                    .map(String::as_str),
                Some("DENY")
            );
            assert!(console_resp.body.contains("VektorDist Control Console"));
            assert!(console_resp.body.contains("role=\"tablist\""));
        }

        h_gw2.abort();
    }

    // Phase 3: Flip 1 byte in `collections.vdcl` and verify `load_from_dir` rejects bit-rot!
    let mut raw_vdcl = std::fs::read(&vdcl_path).unwrap();
    let mid = raw_vdcl.len() / 2;
    raw_vdcl[mid] ^= 0x40;
    std::fs::write(&vdcl_path, &raw_vdcl).unwrap();

    let corrupt_res = CollectionRegistry::load_from_dir(&col_dir);
    assert!(
        matches!(corrupt_res, Err(VektorError::WalCorruption { .. })),
        "expected VektorError::WalCorruption on 1-byte bit flip in collections.vdcl"
    );

    h_coord.abort();
    h1.abort();
    h2.abort();
    h3.abort();
    let _ = std::fs::remove_dir_all(&col_dir);
    let _ = std::fs::remove_dir_all(&dir1);
    let _ = std::fs::remove_dir_all(&dir2);
    let _ = std::fs::remove_dir_all(&dir3);
}
