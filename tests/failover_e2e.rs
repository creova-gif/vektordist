use std::path::PathBuf;
use std::time::Duration;

use vektordist::cli::DemoClusterSupervisor;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn multi_process_sigkill_failover_and_replacement_zero_data_loss() {
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_vektordist"));

    // Launch a real 5-process cluster on localhost, seed 800 x 32d vectors across 16 shards (RF=3, W=2),
    // and start continuous 10-client search + continuous quorum writes.
    let supervisor = DemoClusterSupervisor::start(
        bin, 5,   // 5 real OS child processes
        800, // initial vectors
        32,  // dimensions
        3,   // RF=3 (W=2)
        42,  // deterministic seed
        false,
    )
    .await
    .unwrap();

    // Let continuous search + write traffic run for 300ms.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let snap_initial = supervisor.metrics_snapshot();
    assert_eq!(snap_initial.active_nodes, 5);
    assert!(snap_initial.indexed_vectors >= 800);
    assert_eq!(snap_initial.acked_writes_lost, 0);
    assert!(
        snap_initial.recall_at_10 >= 0.95,
        "expected recall@10 >= 0.95, got {}",
        snap_initial.recall_at_10
    );

    // 1. Send real SIGKILL to one live OS node process (`k` key equivalent) while traffic is active.
    let killed_node = supervisor.kill_one_node().await.unwrap();
    assert!(killed_node.is_some());

    tokio::time::sleep(Duration::from_millis(300)).await;
    let snap_after_kill = supervisor.metrics_snapshot();
    assert_eq!(snap_after_kill.active_nodes, 4);
    assert_eq!(snap_after_kill.acked_writes_lost, 0);
    assert!(snap_after_kill.acked_writes > snap_initial.acked_writes);
    assert!(
        snap_after_kill.last_failover_ms.unwrap_or(9999.0) <= 5_000.0,
        "expected failover <= 5000ms, got {:?}",
        snap_after_kill.last_failover_ms
    );

    // 2. Spawn a replacement OS node process (`a` key equivalent) and verify RECOVERING -> HEALTHY catch-up.
    let new_node_id = supervisor.add_replacement_node().await.unwrap();
    assert_eq!(new_node_id, 6);

    tokio::time::sleep(Duration::from_millis(300)).await;
    let snap_after_recover = supervisor.metrics_snapshot();
    assert_eq!(snap_after_recover.active_nodes, 5);
    assert_eq!(snap_after_recover.acked_writes_lost, 0);
    assert!(!snap_after_recover.degraded_under_quorum);

    // Final independent verification across all 16 shards' WALs.
    let final_lost = supervisor.verify_all_acked_writes().await.unwrap();
    assert_eq!(final_lost, 0);

    supervisor.shutdown();
}
