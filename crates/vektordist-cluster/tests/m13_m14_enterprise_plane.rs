//! Milestones 13 & 14 enterprise integration tests:
//! 1. `HttpVektorClient` + `POST /v1/upsert_batch` high-throughput ingestion across 3 nodes over
//!    pooled gRPC channels (`GrpcChannelPool`), high-recall scatter-gather search, quorum delete,
//!    `ShardEngine::compact_history`, Kubernetes `GET /readyz` graceful drain (`200 -> 503 -> 200`),
//!    and `413 Payload Too Large` HTTP DoS body protection.
//! 2. `RaftCluster` replication of `UpsertApiKey`, `SetTenantQuota`, and `RevokeApiKey` across a
//!    3-node Raft quorum synchronized into independent stateless `SecurityGate` instances.
//! 3. `PersistentAuditLog` `fsync` durability across reopen and deterministic `WalCorruption`
//!    detection on any disk byte tampering or truncation.

use std::collections::HashSet;
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tonic::transport::Server;
use vektordist_cluster::client::ClusterClient;
use vektordist_cluster::coord::CoordinatorService;
use vektordist_cluster::gateway::{spawn_http_gateway, HttpGatewayServer};
use vektordist_cluster::node::NodeService;
use vektordist_cluster::raft::{RaftCluster, TopologyCommand};
use vektordist_cluster::sdk::HttpVektorClient;
use vektordist_cluster::security::{
    AuditOutcome, OrgId, PersistentAuditLog, SecurityError, SecurityGate, TenantQuotaPolicy,
    TenantRole,
};
use vektordist_core::exact::{ExactIndex, SeededRng};
use vektordist_core::hnsw::HnswParams;
use vektordist_core::{ClusterConfig, DistanceMetric, NodeId, VectorId, VektorError, WriteId};
use vektordist_proto::v1::shard_replica_rpc_server::ShardReplicaRpcServer;
use vektordist_proto::v1::topology_rpc_server::TopologyRpcServer;

static TEST_SEQ: AtomicU64 = AtomicU64::new(1);

fn temp_path(tag: &str) -> PathBuf {
    let seq = TEST_SEQ.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("vektordist-m13-m14-{tag}-{pid}-{seq}"))
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
async fn sdk_batch_ingestion_channel_pooling_readyz_drain_and_413_dos_guard() {
    let config = ClusterConfig {
        num_shards: 4,
        replication_factor: 3,
        write_quorum: 2,
        dimensions: 16,
        metric: DistanceMetric::Cosine,
    };

    let (addr1, svc1, dir1, h1) = spawn_node(1, config.clone(), "batch").await;
    let (addr2, svc2, dir2, h2) = spawn_node(2, config.clone(), "batch").await;
    let (addr3, svc3, dir3, h3) = spawn_node(3, config.clone(), "batch").await;

    let coord =
        CoordinatorService::new(config.clone(), vec![(1, addr1), (2, addr2), (3, addr3)]).unwrap();
    coord.push_all_shard_configs().await.unwrap();
    let _ = coord.heartbeat_tick().await;
    let (coord_addr, h_coord) = spawn_coord(coord.clone()).await;

    let mut gate = SecurityGate::new([0x42_u8; 32], TenantQuotaPolicy::default());
    let org_ent = OrgId::new("enterprise-prod").unwrap();
    let (admin_token, _) = gate.issue_api_key("k-ent-admin", org_ent, TenantRole::Admin, 101);
    let shared_gate = Arc::new(Mutex::new(gate));

    let cluster_client = ClusterClient::connect(&coord_addr).await.unwrap();
    // Cap body size at 64 KiB so 120 vectors (~24 KiB) succeed while oversized >64 KiB DoS bodies return 413.
    let gateway = HttpGatewayServer::new(cluster_client.clone())
        .with_security_gate(Arc::clone(&shared_gate))
        .with_max_body_bytes(64 * 1024);
    let (gw_addr, h_gw) = spawn_http_gateway("127.0.0.1:0", gateway.clone())
        .await
        .unwrap();

    let sdk = HttpVektorClient::new(format!("http://{gw_addr}"))
        .with_bearer_token(admin_token)
        .with_org_id("enterprise-prod")
        .with_max_retries(2)
        .with_base_backoff(Duration::from_millis(10));

    // 1. Verify /healthz and /readyz (200 -> 503 -> 200 on set_draining toggle).
    let health = sdk.healthz().await.unwrap();
    assert_eq!(health.status_code, 200);
    assert!(health.ok);
    assert_eq!(health.active_nodes, 3);

    let ready_initial = sdk.readyz().await.unwrap();
    assert_eq!(ready_initial.status_code, 200);
    assert!(ready_initial.ready);
    assert!(!ready_initial.draining);
    assert_eq!(ready_initial.under_quorum_shards, 0);

    gateway.set_draining(true);
    assert!(gateway.is_draining());
    let ready_draining = sdk.readyz().await.unwrap();
    assert_eq!(ready_draining.status_code, 503);
    assert!(!ready_draining.ready);
    assert!(ready_draining.draining);

    gateway.set_draining(false);
    assert!(!gateway.is_draining());
    let ready_restored = sdk.readyz().await.unwrap();
    assert_eq!(ready_restored.status_code, 200);
    assert!(ready_restored.ready);
    assert!(!ready_restored.draining);

    // 2. Batch-ingest 120 vectors across the 3-node cluster via POST /v1/upsert_batch.
    let mut rng = SeededRng::new(2026_1314);
    let mut exact = ExactIndex::new(config.dimensions, config.metric).unwrap();
    let mut batch: Vec<(VectorId, WriteId, Vec<f32>)> = Vec::with_capacity(120);
    for idx in 1..=120_u64 {
        let mut vec_buf = vec![0.0_f32; config.dimensions];
        rng.fill_unit_vector(&mut vec_buf);
        exact.upsert(idx, &vec_buf).unwrap();
        batch.push((idx, 10_000 + idx, vec_buf));
    }

    let upserted = sdk.upsert_batch(&batch).await.unwrap();
    assert_eq!(upserted, 120);
    assert_eq!(gateway.metrics_snapshot().acked_writes_total, 120);

    // 3. Measure distributed scatter-gather search recall@10 across 20 queries via SDK.
    let mut recall_sum = 0.0_f64;
    for _ in 0..20 {
        let mut q = vec![0.0_f32; config.dimensions];
        rng.fill_unit_vector(&mut q);
        let expected: HashSet<VectorId> = exact
            .search(&q, 10)
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        let hits = sdk.search(&q, 10, 64).await.unwrap();
        assert_eq!(hits.len(), 10);
        let overlap = hits.iter().filter(|h| expected.contains(&h.id)).count();
        recall_sum += (overlap as f64) / 10.0;
    }
    let avg_recall = recall_sum / 20.0;
    assert!(
        avg_recall >= 0.95,
        "expected recall@10 >= 0.95 across 120 batch-ingested vectors, got {avg_recall:.4}"
    );

    // Verify GrpcChannelPool reused channels across coordinator + 3 nodes (at most 4 cached endpoints)
    // and follower replication pools on NodeService cached <= 2 peer channels per node.
    let client_pool_len = cluster_client.channel_pool().len();
    assert!(
        (2..=4).contains(&client_pool_len),
        "expected pooled channels in 2..=4, got {client_pool_len}"
    );
    assert!(svc1.channel_pool.len() <= 2);
    assert!(svc2.channel_pool.len() <= 2);
    assert!(svc3.channel_pool.len() <= 2);

    // 4. Delete vector id=1 via SDK and verify it disappears from search hits.
    let target_vec = batch[0].2.clone();
    let (_del_shard, del_seq) = sdk.delete(1, 99_001).await.unwrap();
    assert!(del_seq >= 1);
    let post_del_hits = sdk.search(&target_vec, 10, 64).await.unwrap();
    assert!(
        post_del_hits.iter().all(|h| h.id != 1),
        "deleted vector id=1 must not appear in search results"
    );

    // 5. Overwrite id=2 five times and verify ShardEngine::compact_history deduplicates records_by_seq.
    let shard_for_2 = config.shard_for(2);
    for round in 0..5_u64 {
        let mut v = vec![0.0_f32; config.dimensions];
        rng.fill_unit_vector(&mut v);
        sdk.upsert(2, 50_000 + round, &v).await.unwrap();
    }
    for svc in [&svc1, &svc2, &svc3] {
        let shard_lock = svc.shards.get(&shard_for_2).unwrap();
        let mut engine = shard_lock.write().await;
        let before_len = engine.records_by_seq.len();
        engine.compact_history();
        let after_len = engine.records_by_seq.len();
        assert!(
            after_len < before_len,
            "compact_history must collapse 5 overwrites of id=2 (before={before_len}, after={after_len})"
        );
        let count_id_2 = engine
            .records_by_seq
            .iter()
            .filter(|r| match &r.op {
                vektordist_core::wal::WalOp::Upsert { id, .. }
                | vektordist_core::wal::WalOp::Delete { id } => *id == 2,
            })
            .count();
        assert_eq!(count_id_2, 1);
    }

    // 6. Verify topology_raw() and audit_raw() via SDK.
    let topo_json = sdk.topology_raw().await.unwrap();
    assert!(topo_json.contains("\"num_shards\":4"));
    let audit_json = sdk.audit_raw().await.unwrap();
    assert!(audit_json.contains("\"chain_valid\":true"));

    // 7. Oversized body (> 64 KiB) is immediately rejected with 413 Payload Too Large.
    let oversized_body = format!("{{\"records\":[{}]}}", " ".repeat(70 * 1024));
    let raw_413 = sdk
        .send_raw("POST", "/v1/upsert_batch", &oversized_body)
        .await
        .unwrap();
    assert_eq!(raw_413.status_code, 413);
    assert_eq!(
        raw_413.body,
        "{\"status\":\"error\",\"error\":\"payload exceeds max_body_bytes\"}"
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

#[test]
fn raft_replicates_api_keys_quotas_and_revocations_across_gateways() {
    let config = ClusterConfig {
        num_shards: 4,
        replication_factor: 3,
        write_quorum: 2,
        dimensions: 8,
        metric: DistanceMetric::Cosine,
    };

    let mut raft = RaftCluster::from_initial_nodes(
        config,
        vec![
            (1, "127.0.0.1:19001".to_string()),
            (2, "127.0.0.1:19002".to_string()),
            (3, "127.0.0.1:19003".to_string()),
        ],
    )
    .unwrap();
    let (leader_id, _term) = raft.elect_any_leader().unwrap();

    // Mint scoped keys on a control-plane key issuer sharing the cluster master secret.
    let master_secret = [0x99_u8; 32];
    let mut issuer = SecurityGate::new(master_secret, TenantQuotaPolicy::default());
    let org_fin = OrgId::new("fintech-corp").unwrap();
    let (writer_token, writer_record) =
        issuer.issue_api_key("k-fin-writer", org_fin.clone(), TenantRole::Writer, 1);
    let (reader_token, reader_record) =
        issuer.issue_api_key("k-fin-reader", org_fin.clone(), TenantRole::Reader, 2);

    // Propose API key records and strict tenant quota (0.5 QPS = 500 milli, burst 2.0 = 2000 milli) via Raft.
    raft.propose(TopologyCommand::UpsertApiKey {
        record: writer_record,
    })
    .unwrap();
    raft.propose(TopologyCommand::UpsertApiKey {
        record: reader_record,
    })
    .unwrap();
    raft.propose(TopologyCommand::SetTenantQuota {
        org_id: org_fin.clone(),
        max_qps_milli: 500,
        burst_milli: 2_000,
        max_vectors: 5_000,
        max_memory_bytes: 32 * 1024 * 1024,
    })
    .unwrap();

    // Pick leader and a distinct follower to prove full state-machine replication across Raft quorum.
    let follower_id = raft
        .member_ids()
        .into_iter()
        .find(|&id| id != leader_id)
        .unwrap();
    let leader_node = raft.node(leader_id).unwrap();
    let follower_node = raft.node(follower_id).unwrap();
    assert_eq!(leader_node.api_keys.len(), 2);
    assert_eq!(follower_node.api_keys, leader_node.api_keys);
    assert_eq!(follower_node.tenant_quotas, leader_node.tenant_quotas);

    // Two independent stateless gateways sync from different Raft nodes (leader vs follower).
    let mut gate_a = SecurityGate::new(master_secret, TenantQuotaPolicy::default());
    let mut gate_b = SecurityGate::new(master_secret, TenantQuotaPolicy::default());
    gate_a.sync_from_raft(leader_node, 1_000);
    gate_b.sync_from_raft(follower_node, 1_000);

    assert!(gate_a.get_key_record("k-fin-writer").is_some());
    assert!(gate_b.get_key_record("k-fin-reader").is_some());

    // Both gateways enforce identical RBAC & burst=2 rate limits at t=1_000ms.
    for gate in [&mut gate_a, &mut gate_b] {
        // Reader denied Writer action (does not consume token).
        let rbac_err = gate
            .authorize_request(
                1_000,
                &reader_token,
                &org_fin,
                TenantRole::Writer,
                "http.upsert",
            )
            .unwrap_err();
        assert!(matches!(rbac_err, SecurityError::InsufficientRole { .. }));

        // Writer consumes 2 burst tokens, then 3rd request at t=1_000ms is rate-limited!
        gate.authorize_request(
            1_000,
            &writer_token,
            &org_fin,
            TenantRole::Writer,
            "http.upsert",
        )
        .unwrap();
        gate.authorize_request(
            1_000,
            &writer_token,
            &org_fin,
            TenantRole::Reader,
            "http.search",
        )
        .unwrap();
        let rate_err = gate
            .authorize_request(
                1_000,
                &writer_token,
                &org_fin,
                TenantRole::Reader,
                "http.search",
            )
            .unwrap_err();
        assert!(matches!(rate_err, SecurityError::RateLimited { .. }));
    }

    // Propose cluster-wide key revocation through Raft and re-sync both gateways at t=5_000ms (tokens refilled).
    raft.propose(TopologyCommand::RevokeApiKey {
        key_id: "k-fin-writer".to_string(),
    })
    .unwrap();
    let leader_after = raft.node(leader_id).unwrap();
    let follower_after = raft.node(follower_id).unwrap();
    gate_a.sync_from_raft(leader_after, 5_000);
    gate_b.sync_from_raft(follower_after, 5_000);

    for gate in [&mut gate_a, &mut gate_b] {
        let rev_err = gate
            .authorize_request(
                5_000,
                &writer_token,
                &org_fin,
                TenantRole::Reader,
                "http.search",
            )
            .unwrap_err();
        assert_eq!(rev_err, SecurityError::Unauthorized);

        // Non-revoked reader key continues to work on Reader endpoints.
        gate.authorize_request(
            5_000,
            &reader_token,
            &org_fin,
            TenantRole::Reader,
            "http.search",
        )
        .unwrap();
        assert!(gate.audit_log.verify_integrity().is_ok());
    }
}

#[test]
fn persistent_audit_log_survives_reopen_and_detects_disk_tampering() {
    let dir = temp_path("persistent-audit");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let log_path = dir.join("security-audit.log");

    let org_a = OrgId::new("org-alpha").unwrap();
    let org_b = OrgId::new("org-beta").unwrap();

    let expected_head_hash = {
        let mut audit = PersistentAuditLog::open(&log_path).unwrap();
        assert_eq!(audit.entries().len(), 0);
        let e1 = audit
            .append_and_sync(
                1_001,
                "k-alpha-1",
                org_a.clone(),
                "http.upsert",
                AuditOutcome::Allowed,
            )
            .unwrap();
        assert_eq!(e1.seq_no, 1);

        let e2 = audit
            .append_and_sync(
                1_005,
                "k-beta-9",
                org_b.clone(),
                "http.delete",
                AuditOutcome::Denied {
                    reason: "insufficient RBAC role".to_string(),
                },
            )
            .unwrap();
        assert_eq!(e2.seq_no, 2);
        assert_eq!(e2.prev_hash, e1.entry_hash);

        let e3 = audit
            .append_and_sync(
                1_010,
                "k-alpha-1",
                org_a.clone(),
                "http.upsert_batch",
                AuditOutcome::Allowed,
            )
            .unwrap();
        assert_eq!(e3.seq_no, 3);
        assert_eq!(audit.verify_integrity(), Ok(3));
        audit.head_hash()
    };

    // 1. Reopen from disk and verify intact hash chain + ability to continue appending.
    {
        let mut reopened = PersistentAuditLog::reopen_and_verify(&log_path).unwrap();
        assert_eq!(reopened.entries().len(), 3);
        assert_eq!(reopened.head_hash(), expected_head_hash);
        assert_eq!(reopened.verify_integrity(), Ok(3));

        let e4 = reopened
            .append_and_sync(
                1_020,
                "k-alpha-1",
                org_a,
                "http.search",
                AuditOutcome::Allowed,
            )
            .unwrap();
        assert_eq!(e4.seq_no, 4);
        assert_eq!(e4.prev_hash, expected_head_hash);
    }

    let verified_four = PersistentAuditLog::reopen_and_verify(&log_path).unwrap();
    assert_eq!(verified_four.entries().len(), 4);
    drop(verified_four);

    // 2. Flip a single byte mid-file -> deterministically rejected with VektorError::WalCorruption.
    let tampered_path = dir.join("security-audit-tampered.log");
    std::fs::copy(&log_path, &tampered_path).unwrap();
    {
        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&tampered_path)
            .unwrap();
        f.seek(SeekFrom::Start(24)).unwrap();
        f.write_all(&[0xFF]).unwrap();
        f.sync_all().unwrap();
    }
    let tamper_err = PersistentAuditLog::reopen_and_verify(&tampered_path).unwrap_err();
    assert!(
        matches!(tamper_err, VektorError::WalCorruption { .. }),
        "expected WalCorruption on tampered byte, got {tamper_err:?}"
    );

    // 3. Truncate trailing 3 bytes -> deterministically rejected with VektorError::WalCorruption.
    let truncated_path = dir.join("security-audit-truncated.log");
    std::fs::copy(&log_path, &truncated_path).unwrap();
    {
        let meta = std::fs::metadata(&truncated_path).unwrap();
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(&truncated_path)
            .unwrap();
        f.set_len(meta.len() - 3).unwrap();
        f.sync_all().unwrap();
    }
    let trunc_err = PersistentAuditLog::reopen_and_verify(&truncated_path).unwrap_err();
    assert!(
        matches!(trunc_err, VektorError::WalCorruption { .. }),
        "expected WalCorruption on truncated tail, got {trunc_err:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
