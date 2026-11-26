use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use vektordist::cli::DemoClusterSupervisor;
use vektordist::config::VektorNodeConfig;
use vektordist_cluster::gateway::send_http_request;
use vektordist_core::DistanceMetric;

#[test]
fn node_config_round_trip_from_temp_file_and_cli_keygen_bench() {
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_vektordist"));
    let pid = std::process::id();
    let cfg_path = std::env::temp_dir().join(format!("vektordist-cfg-test-{pid}.toml"));

    let toml_body = r#"
# Declarative node config test
[node]
node_id = 7
listen_addr = "127.0.0.1:50107"
coord_addr = "127.0.0.1:50051"
data_dir = "/tmp/vektordist-node-7"

[cluster]
dimensions = 256
num_shards = 8
replication_factor = 3
write_quorum = 2
metric = "cosine"
"#;
    std::fs::write(&cfg_path, toml_body).unwrap();
    let parsed = VektorNodeConfig::from_file(&cfg_path).unwrap();
    let _ = std::fs::remove_file(&cfg_path);

    assert_eq!(parsed.node_id, 7);
    assert_eq!(parsed.dimensions, 256);
    assert_eq!(parsed.num_shards, 8);
    assert_eq!(parsed.metric, DistanceMetric::Cosine);

    // Verify `vektordist keygen` issues a canonical `vdk.<key_id>.<hex_tag>` bearer token.
    let keygen_out = Command::new(&bin)
        .args([
            "keygen",
            "--secret",
            "integration-test-secret",
            "--org",
            "acme-prod",
            "--role",
            "writer",
            "--key-id",
            "ci-writer-01",
        ])
        .output()
        .unwrap();
    assert!(keygen_out.status.success());
    let stdout = String::from_utf8_lossy(&keygen_out.stdout);
    assert!(stdout.contains("token=vdk.ci-writer-01."));
    assert!(stdout.contains("org=acme-prod role=writer"));

    // Verify `vektordist bench` executes cleanly end-to-end across all M1-M10 tiers.
    let bench_out = Command::new(&bin)
        .args([
            "bench",
            "--vectors",
            "180",
            "--dimensions",
            "64",
            "--queries",
            "5",
        ])
        .output()
        .unwrap();
    assert!(bench_out.status.success());
    let bench_stdout = String::from_utf8_lossy(&bench_out.stdout);
    assert!(bench_stdout.contains("VektorDist M1–M10 Benchmark Suite"));
    assert!(bench_stdout.contains("bq   (1b"));

    // Verify `vektordist doctor` preflight diagnostics pass end-to-end.
    let doctor_out = Command::new(&bin).args(["doctor"]).output().unwrap();
    assert!(doctor_out.status.success());
    let doctor_stdout = String::from_utf8_lossy(&doctor_out.stdout);
    assert!(doctor_stdout.contains("status: HEALTHY"));

    // Verify `vektordist bench-scale` streaming BQ 32x benchmark executes cleanly.
    let scale_out = Command::new(&bin)
        .args([
            "bench-scale",
            "--vectors",
            "1000",
            "--dimensions",
            "128",
            "--queries",
            "4",
        ])
        .output()
        .unwrap();
    assert!(scale_out.status.success());
    let scale_stdout = String::from_utf8_lossy(&scale_out.stdout);
    assert!(scale_stdout.contains("32.0x compression"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn demo_supervisor_exposes_live_http_prometheus_gateway_and_pause_toggle() {
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_vektordist"));
    let sup = DemoClusterSupervisor::start(bin, 3, 240, 32, 3, 77, false)
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(250)).await;
    let snap = sup.metrics_snapshot();
    assert_eq!(snap.active_nodes, 3);
    assert_eq!(snap.healthy_shards, 16);
    assert!(!snap.gateway_addr.is_empty());

    // Scrape `/healthz` and `/metrics` from the supervisor's embedded HTTP gateway.
    let health = send_http_request(&snap.gateway_addr, "GET", "/healthz", &[], "")
        .await
        .unwrap();
    assert_eq!(health.status_code, 200);
    assert!(health.body.contains("\"status\":\"ok\""));
    assert!(health.body.contains("\"active_nodes\":3"));

    let metrics = send_http_request(&snap.gateway_addr, "GET", "/metrics", &[], "")
        .await
        .unwrap();
    assert_eq!(metrics.status_code, 200);
    assert!(metrics.body.contains("vektordist_active_nodes 3"));
    assert!(metrics
        .body
        .contains("vektordist_acked_writes_lost_total 0"));

    // Verify pause/resume toggle updates supervisor state and cluster status.
    assert!(!sup.is_paused());
    assert!(sup.toggle_pause());
    assert!(sup.is_paused());
    let paused_snap = sup.metrics_snapshot();
    assert!(paused_snap.paused);
    assert!(paused_snap.cluster_status.contains("PAUSED"));
    assert!(!sup.toggle_pause());
    assert!(!sup.is_paused());

    sup.shutdown();
}
