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
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Sparkline};
use ratatui::{Frame, Terminal};

use crate::cli::{ClusterMetricsSnapshot, DemoClusterSupervisor};

const HEADER_VERSION_BADGE: &str = concat!(" VEKTORDIST v", env!("CARGO_PKG_VERSION"), " ");

/// Pure presentation state for the accessible `ratatui` terminal dashboard, decoupling
/// interactive modal/pause UI state from OS process supervision so `TestBackend` unit tests
/// exercise the exact same [`render_dashboard`] code path as `vektordist demo`.
#[derive(Debug, Clone)]
pub struct AppState {
    pub snapshot: ClusterMetricsSnapshot,
    pub show_help: bool,
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

impl AppState {
    /// Construct an empty initial `AppState` awaiting first cluster node heartbeats.
    #[must_use]
    pub fn new() -> Self {
        Self {
            snapshot: ClusterMetricsSnapshot {
                configured_nodes: 0,
                active_nodes: 0,
                indexed_vectors: 0,
                dimensions: 768,
                replication_factor: 3,
                write_quorum: 2,
                cluster_status: "Initializing".to_string(),
                degraded_under_quorum: false,
                paused: false,
                gateway_addr: "127.0.0.1:8080".to_string(),
                healthy_shards: 0,
                recovering_shards: 0,
                under_quorum_shards: 0,
                latency_history_us: Vec::new(),
                recall_at_10: 1.0,
                latency_p50_ms: 0.0,
                latency_p95_ms: 0.0,
                latency_p99_ms: 0.0,
                search_qps: 0.0,
                acked_writes: 0,
                acked_writes_lost: 0,
                last_failover_ms: None,
                nodes: Vec::new(),
                events: Vec::new(),
            },
            show_help: false,
        }
    }

    /// Wrap a live [`ClusterMetricsSnapshot`] together with the current help-overlay state.
    #[must_use]
    pub fn from_snapshot(snapshot: ClusterMetricsSnapshot, show_help: bool) -> Self {
        Self {
            snapshot,
            show_help,
        }
    }

    /// Toggle visibility of the keyboard shortcuts & accessibility legend modal (`[?]` / `[h]`).
    pub fn toggle_help(&mut self) {
        self.show_help = !self.show_help;
    }

    /// Toggle continuous workload pause state on this UI state snapshot (`[p]` / `[Space]`).
    pub fn toggle_paused(&mut self) {
        self.snapshot.paused = !self.snapshot.paused;
        self.snapshot.cluster_status = if self.snapshot.paused {
            "PAUSED (load suspended)".to_string()
        } else {
            "Online".to_string()
        };
    }
}

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
    let mut show_help = false;

    loop {
        let state = AppState::from_snapshot(supervisor.metrics_snapshot(), show_help);
        terminal.draw(|f| render_dashboard(f, &state))?;

        if event::poll(Duration::from_millis(120))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char('?') | KeyCode::Char('h') | KeyCode::Char('H') => {
                        show_help = !show_help;
                    }
                    KeyCode::Esc if show_help => {
                        show_help = false;
                    }
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                    KeyCode::Char('k') => {
                        let _ = supervisor.kill_one_node().await;
                    }
                    KeyCode::Char('a') => {
                        let _ = supervisor.add_replacement_node().await;
                    }
                    KeyCode::Char('p') | KeyCode::Char(' ') => {
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

/// Render the accessible, responsive VektorDist cluster dashboard into `frame`.
///
/// Automatically switches to a compact 3-row layout when `area.height < 20` or `area.width < 72`
/// so split-pane terminals never clip or garble status indicators or keyboard hints.
pub fn render_dashboard(frame: &mut Frame<'_>, state: &AppState) {
    let area = frame.area();
    let snap = &state.snapshot;
    let is_compact = area.height < 20 || area.width < 72;

    if is_compact {
        render_compact_layout(frame, area, snap);
    } else {
        render_standard_layout(frame, area, snap);
    }

    if state.show_help {
        render_help_modal(frame, area);
    }
}

fn cluster_status_badge(snap: &ClusterMetricsSnapshot) -> (&'static str, Color) {
    if snap.degraded_under_quorum || snap.under_quorum_shards > 0 {
        (" ▲ DEGRADED ", Color::Red)
    } else if snap.paused {
        (" ● PAUSED ", Color::Yellow)
    } else if snap.configured_nodes > 0 && snap.active_nodes < snap.configured_nodes {
        (" ▲ DEGRADED ", Color::Yellow)
    } else {
        (" ● HEALTHY ", Color::Green)
    }
}

fn build_header_widget(area: Rect, snap: &ClusterMetricsSnapshot) -> Paragraph<'static> {
    let (badge_label, badge_color) = cluster_status_badge(snap);
    let summary = if area.width < 88 {
        format!(
            " N:{}/{} | V:{} | RF={} W={}",
            snap.active_nodes,
            snap.configured_nodes,
            snap.indexed_vectors,
            snap.replication_factor,
            snap.write_quorum
        )
    } else {
        format!(
            " | Nodes: {}/{} | Vectors: {} ({}d) | RF={} W={} | {}",
            snap.active_nodes,
            snap.configured_nodes,
            snap.indexed_vectors,
            snap.dimensions,
            snap.replication_factor,
            snap.write_quorum,
            snap.cluster_status
        )
    };

    Paragraph::new(Line::from(vec![
        Span::styled(
            HEADER_VERSION_BADGE,
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled(
            badge_label,
            Style::default()
                .fg(Color::Black)
                .bg(badge_color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(summary),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Cluster Summary"),
    )
}

fn build_nodes_widget(snap: &ClusterMetricsSnapshot, compact: bool) -> List<'static> {
    let mut node_items: Vec<ListItem<'static>> = Vec::new();
    if snap.nodes.is_empty() {
        node_items.push(ListItem::new(Line::from(Span::styled(
            " Waiting for cluster node heartbeats...",
            Style::default().fg(Color::Gray),
        ))));
    } else {
        for n in &snap.nodes {
            let state_str = if n.alive {
                "● HEALTHY"
            } else {
                "✕ DEAD   "
            };
            let color = if n.alive { Color::Green } else { Color::Red };
            let mem_mb = (n.memory_bytes as f64) / (1024.0 * 1024.0);
            let detail = if compact {
                format!(" s:{:>2} v:{:>6}", n.owned_shards, n.live_vectors)
            } else {
                format!(
                    "shards: {:>2} | vecs: {:>7} | mem: {:>5.1} MiB ({})",
                    n.owned_shards, n.live_vectors, mem_mb, n.addr
                )
            };
            node_items.push(ListItem::new(Line::from(vec![
                Span::styled(
                    format!(" node-{} {state_str} ", n.node_id),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::raw(detail),
            ])));
        }
    }

    node_items.push(ListItem::new(Line::from(format!(
        " Shards: {} Healthy | {} Recovering | {} UnderQuorum",
        snap.healthy_shards, snap.recovering_shards, snap.under_quorum_shards
    ))));

    List::new(node_items).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Cluster Topology & Shards"),
    )
}

fn build_metrics_widget(snap: &ClusterMetricsSnapshot, compact: bool) -> Paragraph<'static> {
    let failover_str = snap.last_failover_ms.map_or_else(
        || "none (steady)".to_string(),
        |ms| format!("{ms:.1} ms (<=5000ms)"),
    );
    let (durability_text, lost_color) = if snap.acked_writes_lost == 0 {
        ("● 0 LOST".to_string(), Color::Green)
    } else {
        (format!("✕ {} LOST", snap.acked_writes_lost), Color::Red)
    };

    let mut metrics_lines = vec![
        Line::from(format!(
            " Recall@10: {:>6.2}% (target >=95%)",
            snap.recall_at_10 * 100.0
        )),
        Line::from(format!(
            " p50: {:>5.2} ms | p95: {:>5.2} ms",
            snap.latency_p50_ms, snap.latency_p95_ms
        )),
        Line::from(format!(
            " p99: {:>5.2} ms | QPS: {:>6.1}",
            snap.latency_p99_ms, snap.search_qps
        )),
        Line::from(vec![
            Span::raw(format!(" Acked: {:>6} | ", snap.acked_writes)),
            Span::styled(
                durability_text,
                Style::default().fg(lost_color).add_modifier(Modifier::BOLD),
            ),
        ]),
    ];

    if !compact {
        metrics_lines.push(Line::from(format!(" Failover: {failover_str}")));
        metrics_lines.push(Line::from(format!(
            " Gateway:  http://{}/metrics",
            snap.gateway_addr
        )));
    }

    Paragraph::new(metrics_lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Live Telemetry"),
    )
}

fn build_footer_widget(area: Rect) -> Paragraph<'static> {
    let label = if area.width < 96 {
        " [k] Kill  [a] Add  [p] Pause  [r] Reset  [?] Help  [q] Quit"
    } else {
        " Keys: [k] Kill node (SIGKILL)  [a] Add node  [p] Pause/Resume  [r] Reset  [?] Help  [q] Quit"
    };
    Paragraph::new(label)
        .style(Style::default().fg(Color::Cyan))
        .block(Block::default().borders(Borders::ALL))
}

fn render_standard_layout(frame: &mut Frame<'_>, area: Rect, snap: &ClusterMetricsSnapshot) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(9),
            Constraint::Length(7),
            Constraint::Length(3),
        ])
        .split(area);

    frame.render_widget(build_header_widget(area, snap), chunks[0]);

    let mid_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(52), Constraint::Percentage(48)])
        .split(chunks[1]);

    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(7), Constraint::Length(4)])
        .split(mid_chunks[1]);

    frame.render_widget(build_nodes_widget(snap, false), mid_chunks[0]);
    frame.render_widget(build_metrics_widget(snap, false), right_chunks[0]);

    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Search Latency Sparkline (us)"),
        )
        .data(&snap.latency_history_us)
        .style(Style::default().fg(Color::Cyan));
    frame.render_widget(sparkline, right_chunks[1]);

    let event_items: Vec<ListItem<'static>> = if snap.events.is_empty() {
        vec![ListItem::new(Line::from(Span::styled(
            " No cluster events recorded yet. Press [?] for keyboard controls.",
            Style::default().fg(Color::Gray),
        )))]
    } else {
        snap.events
            .iter()
            .rev()
            .take(5)
            .map(|e| ListItem::new(format!(" {e}")))
            .collect()
    };

    let events_list = List::new(event_items).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Cluster Event Log (newest first)"),
    );
    frame.render_widget(events_list, chunks[2]);
    frame.render_widget(build_footer_widget(area), chunks[3]);
}

fn render_compact_layout(frame: &mut Frame<'_>, area: Rect, snap: &ClusterMetricsSnapshot) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(3),
        ])
        .split(area);

    frame.render_widget(build_header_widget(area, snap), chunks[0]);

    let body_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(48), Constraint::Percentage(52)])
        .split(chunks[1]);

    frame.render_widget(build_nodes_widget(snap, true), body_chunks[0]);
    frame.render_widget(build_metrics_widget(snap, true), body_chunks[1]);
    frame.render_widget(build_footer_widget(area), chunks[2]);
}

fn render_help_modal(frame: &mut Frame<'_>, area: Rect) {
    let modal_w = area.width.saturating_sub(6).clamp(24, 72);
    let modal_h = area.height.saturating_sub(2).clamp(8, 15);
    let modal_x = area.x + area.width.saturating_sub(modal_w) / 2;
    let modal_y = area.y + area.height.saturating_sub(modal_h) / 2;
    let modal_rect = Rect::new(modal_x, modal_y, modal_w, modal_h);

    let lines = vec![
        Line::from(Span::styled(
            " Operator Chaos & Cluster Controls",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from("  [k]           Kill one live node (SIGKILL) & reconcile WAL"),
        Line::from("  [a]           Add replacement node & catch up shards"),
        Line::from("  [p] / [Space] Pause / Resume continuous search & writes"),
        Line::from("  [r]           Reset & re-seed local cluster processes"),
        Line::from("  [?] / [h]     Toggle Keyboard Shortcuts & Legend modal"),
        Line::from("  [q] / [Esc]   Close modal or quit interactive dashboard"),
        Line::from(""),
        Line::from(Span::styled(
            " Accessible Non-Color Status Glyphs",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from("  ● HEALTHY   Quorum active, all replicas online"),
        Line::from("  ● PAUSED    Background workload suspended by operator"),
        Line::from("  ▲ DEGRADED  Replica lost or under write quorum"),
        Line::from("  ✕ DEAD      Node process terminated (SIGKILL)"),
    ];

    frame.render_widget(Clear, modal_rect);
    let modal = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(" Keyboard Shortcuts & Legend ([Esc] to close) "),
    );
    frame.render_widget(modal, modal_rect);
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

#[cfg(test)]
mod tests {
    use ratatui::backend::TestBackend;
    use vektordist_proto::v1::WireNodeStatus;

    use super::*;

    fn buffer_to_text(terminal: &Terminal<TestBackend>) -> String {
        let buf = terminal.backend().buffer();
        let area = buf.area;
        let mut out = String::new();
        for y in 0..area.height {
            for x in 0..area.width {
                if let Some(cell) = buf.cell((x, y)) {
                    out.push_str(cell.symbol());
                }
            }
            out.push('\n');
        }
        out
    }

    fn sample_populated_state() -> AppState {
        AppState {
            snapshot: ClusterMetricsSnapshot {
                configured_nodes: 3,
                active_nodes: 2,
                indexed_vectors: 25_000,
                dimensions: 768,
                replication_factor: 3,
                write_quorum: 2,
                cluster_status: "Online (degraded replica set, W=2 active)".to_string(),
                degraded_under_quorum: false,
                paused: false,
                gateway_addr: "127.0.0.1:8080".to_string(),
                healthy_shards: 14,
                recovering_shards: 2,
                under_quorum_shards: 0,
                latency_history_us: vec![950, 1100, 1250, 1400, 980],
                recall_at_10: 0.984,
                latency_p50_ms: 1.12,
                latency_p95_ms: 2.84,
                latency_p99_ms: 4.15,
                search_qps: 312.5,
                acked_writes: 25_400,
                acked_writes_lost: 0,
                last_failover_ms: Some(142.3),
                nodes: vec![
                    WireNodeStatus {
                        node_id: 1,
                        addr: "127.0.0.1:50052".to_string(),
                        alive: true,
                        owned_shards: 16,
                        live_vectors: 25_000,
                        memory_bytes: 78 * 1024 * 1024,
                    },
                    WireNodeStatus {
                        node_id: 2,
                        addr: "127.0.0.1:50053".to_string(),
                        alive: false,
                        owned_shards: 0,
                        live_vectors: 0,
                        memory_bytes: 0,
                    },
                ],
                events: vec![
                    "[OK] 3 nodes healthy; 25000 vectors indexed.".to_string(),
                    "[FAILOVER] Killed node-2 (SIGKILL); recovered in 142.3ms.".to_string(),
                ],
            },
            show_help: false,
        }
    }

    #[test]
    fn standard_dashboard_renders_dynamic_version_accessible_glyphs_and_durability() {
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        let mut state = sample_populated_state();
        state.snapshot.active_nodes = 3;
        terminal.draw(|f| render_dashboard(f, &state)).unwrap();

        let rendered = buffer_to_text(&terminal);
        let expected_ver = format!("VEKTORDIST v{}", env!("CARGO_PKG_VERSION"));
        assert!(rendered.contains(&expected_ver));
        assert!(rendered.contains("● HEALTHY"));
        assert!(rendered.contains("✕ DEAD"));
        assert!(rendered.contains("1.12 ms"));
        assert!(rendered.contains("2.84 ms"));
        assert!(rendered.contains("4.15 ms"));
        assert!(rendered.contains("0 LOST"));
        assert!(rendered.contains("[?] Help"));
    }

    #[test]
    fn empty_state_dashboard_renders_guidance_without_panicking() {
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        let state = AppState::new();
        terminal.draw(|f| render_dashboard(f, &state)).unwrap();

        let rendered = buffer_to_text(&terminal);
        assert!(rendered.contains("Waiting for cluster node heartbeats..."));
        assert!(
            rendered.contains("No cluster events recorded yet. Press [?] for keyboard controls.")
        );
    }

    #[test]
    fn compact_viewport_fallback_preserves_header_and_footer_without_clipping() {
        let mut terminal = Terminal::new(TestBackend::new(68, 14)).unwrap();
        let state = sample_populated_state();
        terminal.draw(|f| render_dashboard(f, &state)).unwrap();

        let rendered = buffer_to_text(&terminal);
        let expected_ver = format!("VEKTORDIST v{}", env!("CARGO_PKG_VERSION"));
        assert!(rendered.contains(&expected_ver));
        assert!(rendered.contains("▲ DEGRADED"));
        assert!(rendered.contains("[?] Help"));
        assert!(rendered.contains("[q] Quit"));
        assert!(!rendered.contains("Search Latency Sparkline"));
    }

    #[test]
    fn help_modal_overlay_and_paused_badge_toggle_cleanly() {
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        let mut state = sample_populated_state();
        state.snapshot.active_nodes = 3;

        state.toggle_paused();
        assert!(state.snapshot.paused);
        terminal.draw(|f| render_dashboard(f, &state)).unwrap();
        let paused_view = buffer_to_text(&terminal);
        assert!(paused_view.contains("● PAUSED"));

        state.toggle_help();
        assert!(state.show_help);
        terminal.draw(|f| render_dashboard(f, &state)).unwrap();
        let modal_view = buffer_to_text(&terminal);
        assert!(modal_view.contains("Keyboard Shortcuts & Legend"));
        assert!(modal_view.contains("Accessible Non-Color Status Glyphs"));
    }
}
