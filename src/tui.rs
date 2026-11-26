use std::io::{self, IsTerminal};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Sparkline};
use ratatui::{Frame, Terminal};

use crate::cli::{ClusterMetricsSnapshot, DemoClusterSupervisor};

/// Run the interactive `ratatui` cluster dashboard (or headless console walkthrough if non-TTY).
#[allow(clippy::too_many_arguments)]
pub async fn run_demo_ui(
    node_bin: PathBuf,
    nodes: u32,
    vectors: usize,
    dimensions: usize,
    replication_factor: u8,
    seed: u64,
    keep_data: bool,
    headless: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if headless || !io::stdout().is_terminal() {
        return run_headless_demo(
            node_bin,
            nodes,
            vectors,
            dimensions,
            replication_factor,
            seed,
            keep_data,
        )
        .await;
    }

    println!(
        "Starting VektorDist cluster ({nodes} OS processes, seeding {vectors} x {dimensions}d vectors, RF={replication_factor}, seed={seed})..."
    );
    let mut supervisor = DemoClusterSupervisor::start(
        node_bin.clone(),
        nodes,
        vectors,
        dimensions,
        replication_factor,
        seed,
        keep_data,
    )
    .await?;

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let loop_res = run_tui_event_loop(
        &mut terminal,
        &mut supervisor,
        node_bin,
        nodes,
        vectors,
        dimensions,
        replication_factor,
        seed,
        keep_data,
    )
    .await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    supervisor.shutdown();

    loop_res
}

#[allow(clippy::too_many_arguments)]
async fn run_tui_event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    supervisor: &mut Arc<DemoClusterSupervisor>,
    node_bin: PathBuf,
    nodes: u32,
    vectors: usize,
    dimensions: usize,
    replication_factor: u8,
    seed: u64,
    keep_data: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        let snap = supervisor.metrics_snapshot();
        terminal.draw(|f| draw_dashboard(f, &snap))?;

        if event::poll(Duration::from_millis(120))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                    KeyCode::Char('k') => {
                        let _ = supervisor.kill_one_node().await;
                    }
                    KeyCode::Char('a') => {
                        let _ = supervisor.add_replacement_node().await;
                    }
                    KeyCode::Char('p') => {
                        let _ = supervisor.toggle_pause();
                    }
                    KeyCode::Char('r') => {
                        supervisor.shutdown();
                        *supervisor = DemoClusterSupervisor::start(
                            node_bin.clone(),
                            nodes,
                            vectors,
                            dimensions,
                            replication_factor,
                            seed,
                            keep_data,
                        )
                        .await?;
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

fn draw_dashboard(f: &mut Frame<'_>, snap: &ClusterMetricsSnapshot) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(9),
            Constraint::Length(8),
            Constraint::Length(3),
        ])
        .split(f.area());

    let status_color = if snap.degraded_under_quorum {
        Color::Red
    } else if snap.active_nodes < snap.configured_nodes {
        Color::Yellow
    } else {
        Color::Green
    };

    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            " VEKTORDIST v0.1 ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(
            " | Nodes: {}/{} | Vectors: {} ({}d) | RF={} W={} | Status: ",
            snap.active_nodes,
            snap.configured_nodes,
            snap.indexed_vectors,
            snap.dimensions,
            snap.replication_factor,
            snap.write_quorum
        )),
        Span::styled(
            &snap.cluster_status,
            Style::default()
                .fg(status_color)
                .add_modifier(Modifier::BOLD),
        ),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Cluster Summary"),
    );
    f.render_widget(header, chunks[0]);

    let mid_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(52), Constraint::Percentage(48)])
        .split(chunks[1]);

    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(7), Constraint::Length(4)])
        .split(mid_chunks[1]);

    let mut node_items: Vec<ListItem<'_>> = snap
        .nodes
        .iter()
        .map(|n| {
            let state_str = if n.alive { "[HEALTHY]" } else { "[DEAD]   " };
            let color = if n.alive { Color::Green } else { Color::Red };
            let mem_mb = (n.memory_bytes as f64) / (1024.0 * 1024.0);
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!(" node-{} {state_str} ", n.node_id),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!(
                    "shards: {:>2} | vecs: {:>7} | mem: {:>5.1} MiB ({})",
                    n.owned_shards, n.live_vectors, mem_mb, n.addr
                )),
            ]))
        })
        .collect();
    node_items.push(ListItem::new(Line::from(format!(
        " Shards: {} Healthy | {} Recovering | {} UnderQuorum",
        snap.healthy_shards, snap.recovering_shards, snap.under_quorum_shards
    ))));
    let nodes_list = List::new(node_items).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Cluster Topology & Shards"),
    );
    f.render_widget(nodes_list, mid_chunks[0]);

    let failover_str = snap.last_failover_ms.map_or_else(
        || "none (steady)".to_string(),
        |ms| format!("{ms:.1} ms (<=5000ms)"),
    );
    let lost_color = if snap.acked_writes_lost == 0 {
        Color::Green
    } else {
        Color::Red
    };
    let metrics_lines = vec![
        Line::from(format!(
            " Recall@10:          {:>6.2}%  (target >= 95.0%)",
            snap.recall_at_10 * 100.0
        )),
        Line::from(format!(
            " Search Latency p50: {:>6.2} ms | p95: {:>6.2} ms (<=50ms)",
            snap.latency_p50_ms, snap.latency_p95_ms
        )),
        Line::from(format!(
            " Search Latency p99: {:>6.2} ms | Throughput: {:>5.1} QPS",
            snap.latency_p99_ms, snap.search_qps
        )),
        Line::from(vec![
            Span::raw(format!(
                " Acknowledged Writes: {:>6} | Lost: ",
                snap.acked_writes
            )),
            Span::styled(
                format!("{}", snap.acked_writes_lost),
                Style::default().fg(lost_color).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(format!(" Last Failover Time: {failover_str}")),
        Line::from(format!(" Gateway: http://{}/metrics", snap.gateway_addr)),
    ];
    let metrics_panel = Paragraph::new(metrics_lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Live Telemetry"),
    );
    f.render_widget(metrics_panel, right_chunks[0]);

    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Search Latency Sparkline (us)"),
        )
        .data(&snap.latency_history_us)
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(sparkline, right_chunks[1]);

    let event_items: Vec<ListItem<'_>> = snap
        .events
        .iter()
        .rev()
        .take(5)
        .map(|e| ListItem::new(format!(" {e}")))
        .collect();
    let events_list = List::new(event_items).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Cluster Event Log (newest first)"),
    );
    f.render_widget(events_list, chunks[2]);

    let footer = Paragraph::new(
        " Keys: [k] Kill node (SIGKILL)   [a] Add replacement node   [p] Pause/Resume   [r] Reset   [q] Quit",
    )
    .style(Style::default().fg(Color::Cyan))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(footer, chunks[3]);
}

async fn run_headless_demo(
    node_bin: PathBuf,
    nodes: u32,
    vectors: usize,
    dimensions: usize,
    replication_factor: u8,
    seed: u64,
    keep_data: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "=== VEKTORDIST Interactive Cluster Demo (nodes={nodes}, vectors={vectors}, dims={dimensions}, RF={replication_factor}, seed={seed}) ==="
    );
    let supervisor = DemoClusterSupervisor::start(
        node_bin,
        nodes,
        vectors,
        dimensions,
        replication_factor,
        seed,
        keep_data,
    )
    .await?;

    tokio::time::sleep(Duration::from_millis(600)).await;
    print_snapshot("STEADY STATE", &supervisor.metrics_snapshot());

    if let Some(dead_id) = supervisor.kill_one_node().await? {
        tokio::time::sleep(Duration::from_millis(400)).await;
        print_snapshot(
            &format!("AFTER KILLING NODE-{dead_id} (SIGKILL)"),
            &supervisor.metrics_snapshot(),
        );
    }

    let new_id = supervisor.add_replacement_node().await?;
    tokio::time::sleep(Duration::from_millis(400)).await;
    print_snapshot(
        &format!("AFTER ADDING REPLACEMENT NODE-{new_id}"),
        &supervisor.metrics_snapshot(),
    );

    supervisor.shutdown();
    Ok(())
}

fn print_snapshot(stage: &str, snap: &ClusterMetricsSnapshot) {
    println!(
        "[{stage}] status=\"{}\" active_nodes={}/{} vectors={} recall@10={:.2}% p50={:.2}ms p95={:.2}ms acked_writes={} acked_writes_lost={}",
        snap.cluster_status,
        snap.active_nodes,
        snap.configured_nodes,
        snap.indexed_vectors,
        snap.recall_at_10 * 100.0,
        snap.latency_p50_ms,
        snap.latency_p95_ms,
        snap.acked_writes,
        snap.acked_writes_lost
    );
}
