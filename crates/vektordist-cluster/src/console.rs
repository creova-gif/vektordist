//! Embedded Zero-Dependency, Air-Gapped Web Control Console (`VektorDist Control Console`).
//!
//! Serves a self-contained HTML5/CSS3/ES2022 operational control plane directly from static binary
//! memory with zero external CDN requests, strict Content-Security-Policy headers, WCAG 2.1 AA
//! contrast and keyboard/screen-reader semantics, and ephemeral `sessionStorage`-only credentials.

/// Strict Content-Security-Policy header enforced when serving [`render_control_console_html`].
///
/// Blocks all external scripts, stylesheets, fonts, frames, form submissions, and network origins
/// beyond same-origin gateway API calls (`connect-src 'self'`).
pub const CONSOLE_CSP_HEADER: &str = "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

/// Return the self-contained HTML5/CSS3/ES2022 **VektorDist Control Console** document.
#[must_use]
pub fn render_control_console_html() -> &'static str {
    CONSOLE_DOCUMENT_HTML
}

const CONSOLE_DOCUMENT_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>VektorDist Control Console</title>
<style>
:root {
  --bg-canvas: #090D16;
  --bg-surface: #111827;
  --bg-elevated: #172033;
  --bg-input: #0B111E;
  --border-subtle: #1E293B;
  --border-strong: #334155;
  --text-primary: #F8FAFC;
  --text-muted: #94A3B8;
  --accent-cyan: #22D3EE;
  --accent-cyan-dim: rgba(34, 211, 238, 0.14);
  --status-emerald: #34D399;
  --status-emerald-bg: rgba(52, 211, 153, 0.13);
  --status-amber: #FBBF24;
  --status-amber-bg: rgba(251, 191, 36, 0.14);
  --status-rose: #FB7185;
  --status-rose-bg: rgba(251, 113, 133, 0.14);
  --font-sans: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  --radius-sm: 6px;
  --radius-md: 10px;
}

*, *::before, *::after {
  box-sizing: border-box;
}

html, body {
  margin: 0;
  padding: 0;
  background-color: var(--bg-canvas);
  color: var(--text-primary);
  font-family: var(--font-sans);
  font-size: 14px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

.tabular, code, pre, .metric-value, .mono {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
}

.skip-link {
  position: absolute;
  top: -120px;
  left: 16px;
  z-index: 1000;
  padding: 10px 16px;
  background: var(--accent-cyan);
  color: var(--bg-canvas);
  font-weight: 700;
  border-radius: var(--radius-sm);
  text-decoration: none;
}

.skip-link:focus,
.skip-link:focus-visible {
  top: 12px;
  outline: 3px solid var(--text-primary);
  outline-offset: 2px;
}

:focus-visible {
  outline: 2px solid var(--accent-cyan);
  outline-offset: 2px;
}

.app-shell {
  max-width: 1380px;
  margin: 0 auto;
  padding: 0 20px 40px;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.site-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 16px 0;
  border-bottom: 1px solid var(--border-subtle);
  flex-wrap: wrap;
}

.brand-group {
  display: flex;
  align-items: center;
  gap: 12px;
}

.brand-mark {
  width: 34px;
  height: 34px;
  flex-shrink: 0;
}

.brand-title {
  margin: 0;
  font-size: 17px;
  font-weight: 700;
  letter-spacing: -0.01em;
  display: flex;
  align-items: center;
  gap: 8px;
}

.brand-pill {
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 600;
  color: var(--accent-cyan);
  background: var(--accent-cyan-dim);
  border: 1px solid rgba(34, 211, 238, 0.35);
  padding: 2px 7px;
  border-radius: 999px;
}

.brand-subtitle {
  margin: 2px 0 0;
  font-size: 12px;
  color: var(--text-muted);
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  font-size: 12px;
  font-weight: 700;
  border: 1px solid var(--border-strong);
  background: var(--bg-surface);
  color: var(--text-primary);
}

.status-ready {
  color: var(--status-emerald);
  background: var(--status-emerald-bg);
  border-color: rgba(52, 211, 153, 0.4);
}

.status-degraded {
  color: var(--status-amber);
  background: var(--status-amber-bg);
  border-color: rgba(251, 191, 36, 0.4);
}

.status-offline {
  color: var(--status-rose);
  background: var(--status-rose-bg);
  border-color: rgba(251, 113, 133, 0.4);
}

.pulse-dot {
  display: inline-block;
  animation: pulse-opacity 2.4s ease-in-out infinite;
}

@keyframes pulse-opacity {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.55; }
}

.session-bar {
  margin-top: 14px;
  padding: 12px 14px;
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  display: grid;
  grid-template-columns: minmax(220px, 2fr) minmax(130px, 1fr) minmax(120px, auto) auto;
  gap: 12px;
  align-items: end;
}

.field-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.field-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-muted);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.input-row {
  display: flex;
  gap: 6px;
  align-items: center;
}

input[type="text"],
input[type="password"],
input[type="number"],
select,
textarea {
  width: 100%;
  padding: 8px 10px;
  background: var(--bg-input);
  color: var(--text-primary);
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  font-size: 13px;
  transition: border-color 0.15s ease;
}

textarea {
  min-height: 76px;
  resize: vertical;
  line-height: 1.45;
}

input::placeholder,
textarea::placeholder {
  color: var(--text-muted);
  opacity: 0.75;
}

button,
.btn-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 8px 13px;
  font-family: var(--font-sans);
  font-size: 13px;
  font-weight: 600;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-strong);
  background: var(--bg-elevated);
  color: var(--text-primary);
  cursor: pointer;
  text-decoration: none;
  white-space: nowrap;
  transition: background-color 0.15s ease, border-color 0.15s ease;
}

button:hover,
.btn-link:hover {
  border-color: var(--accent-cyan);
  background: #1E293B;
}

.btn-primary {
  background: var(--accent-cyan);
  color: var(--bg-canvas);
  border-color: var(--accent-cyan);
  font-weight: 700;
}

.btn-primary:hover {
  background: #67E8F9;
  border-color: #67E8F9;
}

.btn-danger {
  color: var(--status-rose);
  border-color: rgba(251, 113, 133, 0.45);
  background: rgba(251, 113, 133, 0.08);
}

.btn-danger:hover,
.btn-danger[data-confirming="true"] {
  color: var(--bg-canvas);
  background: var(--status-rose);
  border-color: var(--status-rose);
}

.btn-sm {
  padding: 5px 9px;
  font-size: 12px;
}

.console-nav {
  margin-top: 14px;
  border-bottom: 1px solid var(--border-subtle);
}

.tablist {
  display: flex;
  gap: 6px;
  overflow-x: auto;
  padding-bottom: 1px;
}

.tab-btn {
  border: 1px solid transparent;
  border-bottom: 2px solid transparent;
  border-radius: var(--radius-sm) var(--radius-sm) 0 0;
  background: transparent;
  color: var(--text-muted);
  padding: 10px 16px;
  font-weight: 600;
  font-size: 13px;
}

.tab-btn:hover {
  color: var(--text-primary);
  background: var(--bg-surface);
  border-color: var(--border-subtle);
}

.tab-btn[aria-selected="true"] {
  color: var(--accent-cyan);
  background: var(--bg-surface);
  border-color: var(--border-subtle);
  border-bottom: 2px solid var(--accent-cyan);
}

.kbd-hint {
  font-family: var(--font-mono);
  font-size: 11px;
  padding: 1px 5px;
  border-radius: 4px;
  background: var(--bg-canvas);
  border: 1px solid var(--border-strong);
  color: var(--text-muted);
}

.status-strip {
  margin-top: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 8px 12px;
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  font-size: 12px;
  color: var(--text-muted);
  flex-wrap: wrap;
}

.error-banner {
  margin-top: 12px;
  padding: 12px 14px;
  border-radius: var(--radius-sm);
  border: 1px solid rgba(251, 113, 133, 0.55);
  background: var(--status-rose-bg);
  color: var(--text-primary);
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.error-banner[hidden],
.tabpanel[hidden],
.modal-backdrop[hidden] {
  display: none !important;
}

main {
  flex: 1;
  margin-top: 16px;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}

.card {
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 16px;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
}

.card-title {
  margin: 0;
  font-size: 13px;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.metric-value {
  font-size: 24px;
  font-weight: 700;
  color: var(--text-primary);
  margin: 4px 0;
}

.metric-sub {
  font-size: 12px;
  color: var(--text-muted);
}

.section-stack {
  margin-top: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.split-grid {
  display: grid;
  grid-template-columns: minmax(300px, 1fr) minmax(360px, 1.45fr);
  gap: 16px;
  align-items: start;
}

.equal-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
  align-items: start;
}

.table-wrap {
  width: 100%;
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

th, td {
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border-subtle);
}

th {
  color: var(--text-muted);
  font-weight: 600;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  background: rgba(9, 13, 22, 0.45);
}

tbody tr:hover {
  background: rgba(30, 41, 59, 0.35);
}

.quant-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  border: 1px solid var(--border-strong);
  background: var(--bg-elevated);
  color: var(--accent-cyan);
}

.empty-state {
  padding: 24px 16px;
  text-align: center;
  color: var(--text-muted);
  border: 1px dashed var(--border-strong);
  border-radius: var(--radius-sm);
  margin-top: 8px;
}

.form-stack {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.inline-fields {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.trace-meta {
  margin-top: 10px;
  padding: 8px 10px;
  background: var(--bg-canvas);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  font-size: 12px;
  color: var(--text-muted);
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
}

.site-footer {
  margin-top: 28px;
  padding-top: 16px;
  border-top: 1px solid var(--border-subtle);
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  font-size: 12px;
  color: var(--text-muted);
}

.footer-links {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
}

.footer-links a {
  color: var(--accent-cyan);
  text-decoration: none;
}

.footer-links a:hover {
  text-decoration: underline;
}

.modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(9, 13, 22, 0.82);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
  z-index: 900;
}

.modal-card {
  width: 100%;
  max-width: 540px;
  background: var(--bg-surface);
  border: 1px solid var(--accent-cyan);
  border-radius: var(--radius-md);
  padding: 20px;
}

.shortcut-list {
  margin: 12px 0 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.shortcut-list li {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 0;
  border-bottom: 1px solid var(--border-subtle);
}

@media (max-width: 960px) {
  .session-bar {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .kpi-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .split-grid,
  .equal-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 640px) {
  .app-shell {
    padding: 0 12px 28px;
  }
  .session-bar,
  .kpi-grid,
  .inline-fields {
    grid-template-columns: 1fr;
  }
  .tab-btn {
    padding: 8px 11px;
    font-size: 12px;
  }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.001ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.001ms !important;
    scroll-behavior: auto !important;
  }
  .pulse-dot {
    animation: none !important;
  }
}
</style>
</head>
<body>
<a class="skip-link" href="#main-content">Skip to main content</a>

<div class="app-shell">
  <header class="site-header" role="banner">
    <div class="brand-group">
      <svg class="brand-mark" viewBox="0 0 36 36" aria-hidden="true" focusable="false">
        <rect x="1.5" y="1.5" width="33" height="33" rx="8" fill="#111827" stroke="#22D3EE" stroke-width="2"/>
        <path d="M9 11 L18 26 L27 11" fill="none" stroke="#22D3EE" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
        <line x1="9" y1="11" x2="27" y2="11" stroke="#334155" stroke-width="1.5" stroke-dasharray="2 2"/>
        <circle cx="9" cy="11" r="2.8" fill="#34D399"/>
        <circle cx="27" cy="11" r="2.8" fill="#34D399"/>
        <circle cx="18" cy="26" r="3.1" fill="#22D3EE"/>
      </svg>
      <div>
        <h1 class="brand-title">
          VektorDist Control Console
          <span class="brand-pill">ZERO-TRUST AIR-GAPPED</span>
        </h1>
        <p class="brand-subtitle">Distributed Quorum-Durable HNSW Vector Engine &amp; Multi-Collection Gateway</p>
      </div>
    </div>
    <div class="header-actions">
      <span id="global-health-badge" class="status-badge status-ready">
        <span class="pulse-dot" aria-hidden="true">●</span>
        <span id="global-health-text">● READY</span>
      </span>
      <span id="last-refresh-ts" class="mono" style="font-size:12px;color:var(--text-muted);">Syncing...</span>
      <button type="button" id="btn-refresh-now" title="Refresh telemetry and topology (Shortcut: r)">
        Refresh <span class="kbd-hint">r</span>
      </button>
      <button type="button" id="btn-open-shortcuts" aria-haspopup="dialog" aria-controls="shortcuts-modal" title="Keyboard shortcuts (Shortcut: ?)">
        Shortcuts <span class="kbd-hint">?</span>
      </button>
    </div>
  </header>

  <section class="session-bar" aria-label="Session Credentials and Auto-Refresh">
    <div class="field-group">
      <label class="field-label" for="session-token">
        <span>Bearer Token (Ephemeral Session Only)</span>
        <span id="session-auth-indicator" class="mono">Anonymous</span>
      </label>
      <div class="input-row">
        <input id="session-token" type="password" autocomplete="off" spellcheck="false" placeholder="vdk.key-01.hex_tag (optional when unauthenticated)">
        <button type="button" id="btn-toggle-token" class="btn-sm" aria-pressed="false" aria-controls="session-token">Show</button>
      </div>
    </div>
    <div class="field-group">
      <label class="field-label" for="session-org">X-Org-Id Tenant</label>
      <input id="session-org" type="text" value="default" autocomplete="off" spellcheck="false" placeholder="default">
    </div>
    <div class="field-group">
      <label class="field-label" for="auto-refresh-select">Auto-Refresh</label>
      <select id="auto-refresh-select">
        <option value="0">Off</option>
        <option value="2000">2s</option>
        <option value="5000" selected>5s</option>
        <option value="15000">15s</option>
      </select>
    </div>
    <div class="input-row">
      <button type="button" id="btn-save-session" class="btn-primary">Apply</button>
      <button type="button" id="btn-clear-session" class="btn-danger">Clear Session</button>
    </div>
  </section>

  <nav class="console-nav" aria-label="Primary Console Navigation">
    <div class="tablist" role="tablist" aria-label="Console Operational Panels">
      <button type="button" role="tab" class="tab-btn" id="tab-overview" aria-selected="true" aria-controls="panel-overview" tabindex="0" data-panel="panel-overview">
        <span class="kbd-hint">1</span> Cluster &amp; Telemetry
      </button>
      <button type="button" role="tab" class="tab-btn" id="tab-collections" aria-selected="false" aria-controls="panel-collections" tabindex="-1" data-panel="panel-collections">
        <span class="kbd-hint">2</span> Collections Studio
      </button>
      <button type="button" role="tab" class="tab-btn" id="tab-playground" aria-selected="false" aria-controls="panel-playground" tabindex="-1" data-panel="panel-playground">
        <span class="kbd-hint">3</span> Vector Playground
      </button>
      <button type="button" role="tab" class="tab-btn" id="tab-security" aria-selected="false" aria-controls="panel-security" tabindex="-1" data-panel="panel-security">
        <span class="kbd-hint">4</span> Security &amp; Audit
      </button>
    </div>
  </nav>

  <div class="status-strip">
    <div id="live-announcer" role="status" aria-live="polite" aria-atomic="true">
      Initializing VektorDist Control Console telemetry...
    </div>
    <div class="tabular">
      Last Trace: <span id="global-trace-id">—</span> · Gateway Latency: <span id="global-latency-ms">—</span>
    </div>
  </div>

  <div id="error-banner" class="error-banner" role="alert" hidden>
    <div>
      <strong id="error-banner-title">✕ Gateway Request Error</strong>
      <div id="error-banner-detail" style="margin-top:4px;font-size:13px;color:var(--text-muted);"></div>
    </div>
    <button type="button" id="btn-dismiss-error" class="btn-sm">Dismiss</button>
  </div>

  <main id="main-content" tabindex="-1">
    <section id="panel-overview" class="tabpanel" role="tabpanel" aria-labelledby="tab-overview">
      <div class="kpi-grid">
        <article class="card">
          <div class="card-header">
            <h2 class="card-title">Readiness &amp; Quorum</h2>
            <span class="mono" style="font-size:11px;color:var(--text-muted);">/healthz · /readyz</span>
          </div>
          <div id="kpi-readiness" class="metric-value">● READY</div>
          <div id="kpi-readiness-sub" class="metric-sub tabular">Probing gateway readiness...</div>
        </article>

        <article class="card">
          <div class="card-header">
            <h2 class="card-title">Cluster Nodes</h2>
            <span class="mono" style="font-size:11px;color:var(--text-muted);">vektordist_active_nodes</span>
          </div>
          <div id="kpi-nodes" class="metric-value">— / —</div>
          <div id="kpi-nodes-sub" class="metric-sub tabular">Active / Configured physical replicas</div>
        </article>

        <article class="card">
          <div class="card-header">
            <h2 class="card-title">Indexed Vectors</h2>
            <span class="mono" style="font-size:11px;color:var(--text-muted);">vektordist_indexed_vectors</span>
          </div>
          <div id="kpi-vectors" class="metric-value">0</div>
          <div id="kpi-vectors-sub" class="metric-sub tabular">Live vectors across active shards</div>
        </article>

        <article class="card">
          <div class="card-header">
            <h2 class="card-title">Search Latency (ms)</h2>
            <span class="mono" style="font-size:11px;color:var(--text-muted);">p50 · p95 · p99</span>
          </div>
          <div id="kpi-latency" class="metric-value">0.00 ms</div>
          <div id="kpi-latency-sub" class="metric-sub tabular">p50: 0.00 ms · p95: 0.00 ms · p99: 0.00 ms</div>
        </article>

        <article class="card">
          <div class="card-header">
            <h2 class="card-title">Recall@10 Quality</h2>
            <span class="mono" style="font-size:11px;color:var(--text-muted);">vektordist_recall_at_10</span>
          </div>
          <div id="kpi-recall" class="metric-value">100.00%</div>
          <div id="kpi-recall-sub" class="metric-sub tabular">SLA Target &gt;= 95.00% vs exact ground truth</div>
        </article>

        <article class="card">
          <div class="card-header">
            <h2 class="card-title">Quorum Write Durability</h2>
            <span class="mono" style="font-size:11px;color:var(--text-muted);">acked_writes_total</span>
          </div>
          <div id="kpi-durability" class="metric-value">● 0 LOST</div>
          <div id="kpi-durability-sub" class="metric-sub tabular">0 quorum-acknowledged writes verified</div>
        </article>
      </div>

      <div class="section-stack">
        <article class="card">
          <div class="card-header">
            <div>
              <h2 class="card-title" style="font-size:14px;color:var(--text-primary);">Shard Ring &amp; Epoch Topology</h2>
              <div id="topo-summary-line" class="metric-sub tabular" style="margin-top:3px;">
                Polling GET /v1/topology...
              </div>
            </div>
            <span id="topo-status-pill" class="status-badge status-ready">● READY</span>
          </div>
          <div class="table-wrap">
            <table aria-label="Cluster Shard Ring and Epoch Topology">
              <thead>
                <tr>
                  <th scope="col">Shard ID</th>
                  <th scope="col">Epoch</th>
                  <th scope="col">Primary Replica</th>
                  <th scope="col">Follower Replicas</th>
                  <th scope="col">Quorum State</th>
                </tr>
              </thead>
              <tbody id="topology-tbody" class="tabular">
                <tr>
                  <td colspan="5" style="color:var(--text-muted);">Loading cluster shard assignments...</td>
                </tr>
              </tbody>
            </table>
          </div>
        </article>
      </div>
    </section>

    <section id="panel-collections" class="tabpanel" role="tabpanel" aria-labelledby="tab-collections" hidden>
      <div class="split-grid">
        <article class="card">
          <div class="card-header">
            <h2 class="card-title" style="font-size:14px;color:var(--text-primary);">Create Isolated Collection</h2>
            <span class="mono" style="font-size:11px;color:var(--text-muted);">POST /v1/collections</span>
          </div>
          <form id="form-create-collection" class="form-stack">
            <div class="field-group">
              <label class="field-label" for="col-name">Collection Namespace</label>
              <input id="col-name" type="text" required maxlength="128" pattern="[A-Za-z0-9_-]+" placeholder="docs_embeddings_v1">
            </div>
            <div class="inline-fields">
              <div class="field-group">
                <label class="field-label" for="col-dims">Dimensions</label>
                <input id="col-dims" type="number" required min="1" max="65536" value="768">
              </div>
              <div class="field-group">
                <label class="field-label" for="col-metric">Distance Metric</label>
                <select id="col-metric">
                  <option value="cosine" selected>cosine (Normalized Dot)</option>
                  <option value="l2">l2 (Squared Euclidean)</option>
                </select>
              </div>
            </div>
            <div class="field-group">
              <label class="field-label" for="col-quant">Quantization Profile</label>
              <select id="col-quant">
                <option value="f32" selected>f32 — Full Precision IEEE-754 (1x)</option>
                <option value="fp16">fp16 — Half Precision IEEE-754 (2x)</option>
                <option value="sq8">sq8 — 8-Bit Scalar Quantized (4x)</option>
                <option value="bq">bq — 1-Bit Binary POPCNT + Rerank (32x)</option>
              </select>
            </div>
            <button type="submit" class="btn-primary">Create Collection</button>
          </form>
          <div id="col-create-meta" class="trace-meta">
            <span>Requires <code>admin</code> role when zero-trust SecurityGate is active.</span>
          </div>
        </article>

        <article class="card">
          <div class="card-header">
            <div>
              <h2 class="card-title" style="font-size:14px;color:var(--text-primary);">Multi-Collection Namespaces</h2>
              <div class="metric-sub">Strict per-tenant <code>X-Org-Id</code> vector &amp; memory isolation</div>
            </div>
            <button type="button" id="btn-refresh-collections" class="btn-sm">Reload List</button>
          </div>

          <div id="collections-empty" class="empty-state" hidden>
            <strong style="color:var(--text-primary);display:block;margin-bottom:6px;">No vector collections registered yet</strong>
            Create your first namespace using the form on the left (for example <code>docs_embeddings_v1</code> with <code>768</code> dimensions and <code>f32</code> or <code>sq8</code> quantization) to start tenant-isolated indexing.
          </div>

          <div id="collections-table-wrap" class="table-wrap">
            <table aria-label="Registered Vector Collections">
              <thead>
                <tr>
                  <th scope="col">Namespace</th>
                  <th scope="col">Dims</th>
                  <th scope="col">Metric</th>
                  <th scope="col">Quantization</th>
                  <th scope="col">Live Vectors</th>
                  <th scope="col">Memory</th>
                  <th scope="col">Actions</th>
                </tr>
              </thead>
              <tbody id="collections-tbody" class="tabular">
                <tr>
                  <td colspan="7" style="color:var(--text-muted);">Loading collections from GET /v1/collections...</td>
                </tr>
              </tbody>
            </table>
          </div>
        </article>
      </div>
    </section>

    <section id="panel-playground" class="tabpanel" role="tabpanel" aria-labelledby="tab-playground" hidden>
      <article class="card" style="margin-bottom:16px;">
        <div class="card-header">
          <div>
            <h2 class="card-title" style="font-size:14px;color:var(--text-primary);">Target Route &amp; Deterministic Unit Vector Generator</h2>
            <div class="metric-sub">Test scatter-gather cluster endpoints or named collection namespaces in seconds without hand-typing floats.</div>
          </div>
          <button type="button" id="btn-generate-vector" class="btn-primary">Generate Normalized Vector</button>
        </div>
        <div class="equal-grid" style="grid-template-columns: 2fr 1fr 1fr;">
          <div class="field-group">
            <label class="field-label" for="pg-target">Routing Target</label>
            <select id="pg-target">
              <option value="__cluster__">Cluster Scatter-Gather (/v1/upsert &amp; /v1/search)</option>
            </select>
          </div>
          <div class="field-group">
            <label class="field-label" for="pg-dims">Vector Dimensions</label>
            <input id="pg-dims" type="number" min="1" max="65536" value="768">
          </div>
          <div class="field-group">
            <label class="field-label" for="pg-seed">Generator Seed</label>
            <input id="pg-seed" type="number" value="42">
          </div>
        </div>
      </article>

      <div class="equal-grid">
        <article class="card">
          <div class="card-header">
            <h2 class="card-title" style="font-size:14px;color:var(--text-primary);">Quorum / Namespace Upsert</h2>
            <span id="pg-upsert-route" class="mono" style="font-size:11px;color:var(--accent-cyan);">POST /v1/upsert</span>
          </div>
          <form id="form-playground-upsert" class="form-stack">
            <div class="inline-fields">
              <div class="field-group">
                <label class="field-label" for="pg-upsert-id">Vector ID (u64)</label>
                <input id="pg-upsert-id" type="number" min="0" step="1" required value="1001">
              </div>
              <div class="field-group">
                <label class="field-label" for="pg-write-id">Idempotency Write ID (u64)</label>
                <input id="pg-write-id" type="number" min="0" step="1" required value="5001">
              </div>
            </div>
            <div class="field-group">
              <label class="field-label" for="pg-upsert-vec">Vector Payload (JSON f32 Array)</label>
              <textarea id="pg-upsert-vec" required placeholder="[0.0361, -0.0412, ...]"></textarea>
            </div>
            <button type="submit" class="btn-primary">Upsert Vector</button>
          </form>
          <div id="pg-upsert-result" class="trace-meta">
            <span>Ready to upsert. Click "Generate Normalized Vector" to populate dimensions automatically.</span>
          </div>
        </article>

        <article class="card">
          <div class="card-header">
            <h2 class="card-title" style="font-size:14px;color:var(--text-primary);">Top-K Nearest Neighbor Search</h2>
            <span id="pg-search-route" class="mono" style="font-size:11px;color:var(--accent-cyan);">POST /v1/search</span>
          </div>
          <form id="form-playground-search" class="form-stack">
            <div class="inline-fields">
              <div class="field-group">
                <label class="field-label" for="pg-top-k">Top-K Neighbors</label>
                <input id="pg-top-k" type="number" min="1" max="1000" required value="10">
              </div>
              <div class="field-group">
                <label class="field-label" for="pg-ef-search">HNSW ef_search Beam</label>
                <input id="pg-ef-search" type="number" min="1" max="4096" required value="64">
              </div>
            </div>
            <div class="field-group">
              <label class="field-label" for="pg-search-vec">Query Vector (JSON f32 Array)</label>
              <textarea id="pg-search-vec" required placeholder="[0.0361, -0.0412, ...]"></textarea>
            </div>
            <button type="submit" class="btn-primary">Execute Top-K Search</button>
          </form>
          <div id="pg-search-meta" class="trace-meta">
            <span>x-trace-id: —</span>
            <span>x-latency-ms: —</span>
          </div>
          <div class="table-wrap" style="margin-top:10px;">
            <table aria-label="Vector Search Nearest Neighbor Hits">
              <thead>
                <tr>
                  <th scope="col">Rank</th>
                  <th scope="col">Vector ID</th>
                  <th scope="col">Score</th>
                </tr>
              </thead>
              <tbody id="pg-hits-tbody" class="tabular">
                <tr>
                  <td colspan="3" style="color:var(--text-muted);">No search executed yet.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </article>
      </div>
    </section>

    <section id="panel-security" class="tabpanel" role="tabpanel" aria-labelledby="tab-security" hidden>
      <div class="equal-grid">
        <article class="card">
          <div class="card-header">
            <div>
              <h2 class="card-title" style="font-size:14px;color:var(--text-primary);">Cryptographic Audit Chain Inspector</h2>
              <div class="metric-sub">Tamper-evident hash-chained security ledger (GET /v1/audit)</div>
            </div>
            <button type="button" id="btn-verify-audit" class="btn-primary btn-sm">Verify Chain</button>
          </div>
          <div class="kpi-grid" style="grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top:8px;">
            <div class="card" style="padding:12px;background:var(--bg-canvas);">
              <div class="card-title">Integrity Status</div>
              <div id="audit-chain-status" class="metric-value" style="font-size:18px;">● READY</div>
              <div id="audit-security-mode" class="metric-sub">Checking SecurityGate...</div>
            </div>
            <div class="card" style="padding:12px;background:var(--bg-canvas);">
              <div class="card-title">Verified Entries</div>
              <div id="audit-entries-count" class="metric-value" style="font-size:18px;">0</div>
              <div class="metric-sub">Monotonic sequence records</div>
            </div>
          </div>
          <div class="trace-meta" style="margin-top:12px;flex-direction:column;gap:6px;">
            <div>Chain Head Digest: <strong id="audit-head-hash" style="color:var(--accent-cyan);">0x0000000000000000</strong></div>
            <div>Last Audit Trace: <span id="audit-trace-id">—</span> (<span id="audit-latency-ms">—</span>)</div>
          </div>
        </article>

        <article class="card">
          <div class="card-header">
            <h2 class="card-title" style="font-size:14px;color:var(--text-primary);">Zero-Trust Security &amp; API Contracts</h2>
            <span class="quant-badge">CSP HARDENED</span>
          </div>
          <ul style="margin:0;padding-left:18px;color:var(--text-muted);display:flex;flex-direction:column;gap:8px;">
            <li><strong style="color:var(--text-primary);">Ephemeral Credentials:</strong> Bearer tokens are held strictly in tab-scoped <code>sessionStorage</code> and never written to URL query parameters or persistent disk storage.</li>
            <li><strong style="color:var(--text-primary);">Air-Gapped CSP:</strong> Served with <code>default-src 'none'; connect-src 'self'</code> and zero external scripts, fonts, or telemetry beacons.</li>
            <li><strong style="color:var(--text-primary);">RBAC Privilege Tiers:</strong> <code>reader</code> (search, topology, list collections) &lt; <code>writer</code> (upsert, delete) &lt; <code>admin</code> (create/drop collections, verify <code>/v1/audit</code>).</li>
          </ul>
          <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:16px;">
            <a class="btn-link" href="/openapi.json" target="_blank" rel="noopener">OpenAPI 3.1 Spec (/openapi.json)</a>
            <a class="btn-link" href="/metrics" target="_blank" rel="noopener">Prometheus Metrics (/metrics)</a>
            <a class="btn-link" href="/healthz" target="_blank" rel="noopener">Health Probe (/healthz)</a>
            <a class="btn-link" href="/readyz" target="_blank" rel="noopener">Readiness Probe (/readyz)</a>
          </div>
        </article>
      </div>
    </section>
  </main>

  <footer class="site-footer" role="contentinfo">
    <div>
      <strong>VektorDist Control Console</strong> · Zero External Dependencies · W3C <code>traceparent</code> Propagated
    </div>
    <div class="footer-links">
      <a href="/openapi.json" target="_blank" rel="noopener">/openapi.json</a>
      <a href="/metrics" target="_blank" rel="noopener">/metrics</a>
      <a href="/healthz" target="_blank" rel="noopener">/healthz</a>
      <a href="/readyz" target="_blank" rel="noopener">/readyz</a>
    </div>
  </footer>
</div>

<div id="shortcuts-modal" class="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="shortcuts-title" hidden>
  <div class="modal-card">
    <div class="card-header">
      <h2 id="shortcuts-title" class="card-title" style="font-size:15px;color:var(--text-primary);">Keyboard Navigation &amp; Accessibility Legend</h2>
      <button type="button" id="btn-close-shortcuts" class="btn-sm">Close (Esc)</button>
    </div>
    <ul class="shortcut-list">
      <li><span>Switch Operational Views</span><span><span class="kbd-hint">1</span> <span class="kbd-hint">2</span> <span class="kbd-hint">3</span> <span class="kbd-hint">4</span></span></li>
      <li><span>Cycle Focused Navigation Tab</span><span><span class="kbd-hint">←</span> <span class="kbd-hint">→</span> <span class="kbd-hint">Home</span> <span class="kbd-hint">End</span></span></li>
      <li><span>Refresh Live Telemetry &amp; Topology</span><span class="kbd-hint">r</span></li>
      <li><span>Toggle Keyboard Shortcuts Dialog</span><span class="kbd-hint">?</span></li>
      <li><span>Accessible Non-Color Status Glyphs</span><span class="mono">● READY · ▲ DRAINING / DEGRADED · ✕ OFFLINE / UNDER_QUORUM</span></li>
    </ul>
  </div>
</div>

<script>
(() => {
  "use strict";

  const SESSION_TOKEN_KEY = "vektordist.session.bearer_token";
  const SESSION_ORG_KEY = "vektordist.session.org_id";
  const SESSION_REFRESH_KEY = "vektordist.session.refresh_ms";

  const state = {
    clusterDimensions: 768,
    collections: [],
    refreshTimer: null,
    dropConfirmTimers: new Map()
  };

  const el = (id) => document.getElementById(id);

  function randomHex(byteCount) {
    const bytes = new Uint8Array(byteCount);
    if (window.crypto && typeof window.crypto.getRandomValues === "function") {
      window.crypto.getRandomValues(bytes);
    } else {
      for (let i = 0; i < byteCount; i += 1) {
        bytes[i] = Math.floor(Math.random() * 256);
      }
    }
    if (bytes.every((b) => b === 0)) {
      bytes[byteCount - 1] = 1;
    }
    return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  }

  function makeTraceparent() {
    return `00-${randomHex(16)}-${randomHex(8)}-01`;
  }

  function getSessionToken() {
    try {
      return (window.sessionStorage.getItem(SESSION_TOKEN_KEY) || "").trim();
    } catch (_) {
      return (el("session-token").value || "").trim();
    }
  }

  function getSessionOrg() {
    try {
      return (window.sessionStorage.getItem(SESSION_ORG_KEY) || el("session-org").value || "default").trim() || "default";
    } catch (_) {
      return (el("session-org").value || "default").trim() || "default";
    }
  }

  function announce(msg) {
    const node = el("live-announcer");
    if (node) {
      node.textContent = msg;
    }
  }

  function showErrorBanner(status, serverMsg, path) {
    const banner = el("error-banner");
    const title = el("error-banner-title");
    const detail = el("error-banner-detail");
    let guidance = serverMsg || `HTTP ${status} on ${path}`;
    if (status === 401) {
      title.textContent = "✕ 401 Unauthorized — Bearer Token Required or Invalid";
      guidance = `${serverMsg || "Authentication failed."} Configure a valid vdk.<key_id>.<hex_tag> Bearer Token in the Session Credentials bar above and click Apply.`;
    } else if (status === 403) {
      title.textContent = "▲ 403 Forbidden — RBAC Role or Tenant Mismatch";
      guidance = `${serverMsg || "Access denied."} Verify your API key holds sufficient privilege (reader / writer / admin) and X-Org-Id matches the key's bound tenant.`;
    } else if (status === 429) {
      title.textContent = "▲ 429 Too Many Requests — Tenant Rate Limit Reached";
      guidance = `${serverMsg || "Token-bucket quota exhausted."} Reduce auto-refresh frequency or wait for the Retry-After window.`;
    } else if (status === 503) {
      title.textContent = "✕ 503 Service Unavailable — Draining or Under Quorum";
      guidance = `${serverMsg || "Cluster unavailable."} Check /readyz and shard quorum status in Cluster & Telemetry.`;
    } else {
      title.textContent = `✕ Gateway Error (HTTP ${status}) on ${path}`;
    }
    detail.textContent = guidance;
    banner.hidden = false;
    announce(`${title.textContent}. ${guidance}`);
  }

  function clearErrorBanner() {
    el("error-banner").hidden = true;
  }

  async function apiRequest(method, path, jsonBody, opts = {}) {
    const headers = {
      "traceparent": makeTraceparent(),
      "x-org-id": getSessionOrg()
    };
    const token = getSessionToken();
    if (token) {
      headers["authorization"] = `Bearer ${token}`;
    }
    const fetchInit = { method, headers };
    if (jsonBody !== undefined && jsonBody !== null) {
      headers["content-type"] = "application/json";
      fetchInit.body = JSON.stringify(jsonBody);
    }

    const resp = await fetch(path, fetchInit);
    const traceId = resp.headers.get("x-trace-id") || "—";
    const latencyRaw = resp.headers.get("x-latency-ms");
    const latencyMs = latencyRaw ? `${Number(latencyRaw).toFixed(2)} ms` : "—";
    el("global-trace-id").textContent = traceId;
    el("global-latency-ms").textContent = latencyMs;

    const text = await resp.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch (_) {
      data = null;
    }

    if (!resp.ok && !opts.allowNon2xx) {
      const errMsg = data && data.error ? data.error : `HTTP ${resp.status}`;
      showErrorBanner(resp.status, errMsg, path);
    }

    return { ok: resp.ok, status: resp.status, text, data, traceId, latencyMs };
  }

  function parsePrometheusMetrics(rawText) {
    const map = Object.create(null);
    const lines = String(rawText || "").split("\n");
    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line || line.startsWith("#")) {
        continue;
      }
      const spaceIdx = line.search(/\s+/);
      if (spaceIdx <= 0) {
        continue;
      }
      const metricKey = line.slice(0, spaceIdx).split("{")[0];
      const val = Number(line.slice(spaceIdx).trim().split(/\s+/)[0]);
      if (Number.isFinite(val)) {
        map[metricKey] = val;
      }
    }
    return map;
  }

  function formatBytes(bytes) {
    const b = Number(bytes) || 0;
    if (b < 1024) return `${b} B`;
    if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KiB`;
    return `${(b / (1024 * 1024)).toFixed(2)} MiB`;
  }

  function setBadgeState(badgeEl, textEl, tier, label) {
    badgeEl.classList.remove("status-ready", "status-degraded", "status-offline");
    if (tier === "ready") {
      badgeEl.classList.add("status-ready");
    } else if (tier === "degraded") {
      badgeEl.classList.add("status-degraded");
    } else {
      badgeEl.classList.add("status-offline");
    }
    if (textEl) {
      textEl.textContent = label;
    } else {
      badgeEl.textContent = label;
    }
  }

  async function refreshOverview() {
    try {
      const [healthRes, readyRes, metricsRes, topoRes] = await Promise.all([
        apiRequest("GET", "/healthz", null, { allowNon2xx: true }),
        apiRequest("GET", "/readyz", null, { allowNon2xx: true }),
        apiRequest("GET", "/metrics", null, { allowNon2xx: true }),
        apiRequest("GET", "/v1/topology", null, { allowNon2xx: true })
      ]);

      const prom = parsePrometheusMetrics(metricsRes.text);
      const activeNodes = prom.vektordist_active_nodes ?? (healthRes.data?.active_nodes ?? 0);
      const configuredNodes = prom.vektordist_configured_nodes ?? (healthRes.data?.configured_nodes ?? activeNodes);
      const indexedVecs = prom.vektordist_indexed_vectors ?? 0;
      const p50 = prom.vektordist_search_latency_p50_ms ?? 0;
      const p95 = prom.vektordist_search_latency_p95_ms ?? 0;
      const p99 = prom.vektordist_search_latency_p99_ms ?? 0;
      const recall = (prom.vektordist_recall_at_10 ?? 1) * 100;
      const ackedTotal = prom.vektordist_acked_writes_total ?? 0;
      const ackedLost = prom.vektordist_acked_writes_lost_total ?? (healthRes.data?.acked_writes_lost ?? 0);

      const draining = Boolean(readyRes.data?.draining);
      const underQuorumShards = Number(readyRes.data?.under_quorum_shards ?? 0);
      let tier = "ready";
      let statusLabel = "● READY";
      if (!healthRes.ok || activeNodes === 0 || underQuorumShards > 0) {
        tier = "offline";
        statusLabel = "✕ OFFLINE / UNDER_QUORUM";
      } else if (!readyRes.ok || draining || activeNodes < configuredNodes) {
        tier = "degraded";
        statusLabel = "▲ DRAINING / DEGRADED";
      }

      setBadgeState(el("global-health-badge"), el("global-health-text"), tier, statusLabel);
      el("kpi-readiness").textContent = statusLabel;
      el("kpi-readiness-sub").textContent = `draining=${draining} · under_quorum_shards=${underQuorumShards}`;
      el("kpi-nodes").textContent = `${activeNodes} / ${configuredNodes}`;
      el("kpi-vectors").textContent = indexedVecs.toLocaleString("en-US");
      el("kpi-latency").textContent = `${p95.toFixed(2)} ms (p95)`;
      el("kpi-latency-sub").textContent = `p50: ${p50.toFixed(2)} ms · p95: ${p95.toFixed(2)} ms · p99: ${p99.toFixed(2)} ms`;
      el("kpi-recall").textContent = `${recall.toFixed(2)}%`;
      el("kpi-durability").textContent = ackedLost === 0 ? "● 0 LOST" : `✕ ${ackedLost} LOST`;
      el("kpi-durability").style.color = ackedLost === 0 ? "var(--status-emerald)" : "var(--status-rose)";
      el("kpi-durability-sub").textContent = `${ackedTotal.toLocaleString("en-US")} quorum-acknowledged writes`;

      renderTopologyTable(topoRes, tier, statusLabel);
      el("last-refresh-ts").textContent = `Updated ${new Date().toLocaleTimeString()}`;
    } catch (_) {
      setBadgeState(el("global-health-badge"), el("global-health-text"), "offline", "✕ OFFLINE / UNDER_QUORUM");
      el("kpi-readiness").textContent = "✕ OFFLINE / UNDER_QUORUM";
      el("kpi-readiness-sub").textContent = "Unable to reach gateway endpoints";
    }
  }

  function renderTopologyTable(topoRes, fallbackTier, fallbackLabel) {
    const tbody = el("topology-tbody");
    tbody.replaceChildren();

    if (!topoRes.ok || !topoRes.data) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 5;
      td.style.color = "var(--status-amber)";
      td.textContent = topoRes.status === 401
        ? "▲ Topology protected by zero-trust SecurityGate — enter a valid Bearer Token above to inspect shards."
        : `✕ Topology unavailable (HTTP ${topoRes.status}).`;
      tr.appendChild(td);
      tbody.appendChild(tr);
      setBadgeState(el("topo-status-pill"), null, topoRes.status === 401 ? "degraded" : "offline", topoRes.status === 401 ? "▲ AUTH REQUIRED" : "✕ UNAVAILABLE");
      return;
    }

    const d = topoRes.data;
    if (Number(d.dimensions) > 0) {
      state.clusterDimensions = Number(d.dimensions);
      if (el("pg-target").value === "__cluster__") {
        el("pg-dims").value = String(state.clusterDimensions);
      }
    }

    const numShards = Number(d.num_shards ?? 0);
    const rf = Number(d.replication_factor ?? 1);
    const wq = Number(d.write_quorum ?? 1);
    const epoch = Number(d.epoch ?? 1);
    const cfgNodes = Math.max(Number(d.configured_nodes ?? 1), 1);
    const actNodes = Number(d.active_nodes ?? 0);

    el("topo-summary-line").textContent =
      `epoch=${epoch} · shards=${numShards} · RF=${rf} · W=${wq} · dims=${d.dimensions ?? "—"} · active_nodes=${actNodes}/${cfgNodes}`;
    setBadgeState(el("topo-status-pill"), null, fallbackTier, fallbackLabel);

    const shardRows = Array.isArray(d.shards) && d.shards.length > 0
      ? d.shards
      : Array.from({ length: Math.min(numShards, 64) }, (_, s) => {
          const primary = (s % cfgNodes) + 1;
          const followers = [];
          for (let r = 1; r < Math.min(rf, cfgNodes); r += 1) {
            followers.push(((s + r) % cfgNodes) + 1);
          }
          const stateStr = actNodes >= wq ? (actNodes === cfgNodes ? "healthy" : "degraded") : "under_quorum";
          return { shard_id: s, epoch, primary, followers, state: stateStr };
        });

    if (shardRows.length === 0) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 5;
      td.textContent = "No shards reported by coordinator.";
      tr.appendChild(td);
      tbody.appendChild(tr);
      return;
    }

    for (const shard of shardRows) {
      const tr = document.createElement("tr");
      const tdId = document.createElement("td");
      tdId.textContent = `shard-${String(shard.shard_id).padStart(2, "0")}`;

      const tdEpoch = document.createElement("td");
      tdEpoch.textContent = String(shard.epoch ?? epoch);

      const tdPrimary = document.createElement("td");
      tdPrimary.textContent = `node-${shard.primary}`;

      const tdFollowers = document.createElement("td");
      const fol = Array.isArray(shard.followers) ? shard.followers : [];
      tdFollowers.textContent = fol.length > 0 ? fol.map((n) => `node-${n}`).join(", ") : "—";

      const tdState = document.createElement("td");
      const rawState = String(shard.state ?? "healthy").toLowerCase();
      if (rawState.includes("under") || rawState === "4") {
        tdState.textContent = "✕ OFFLINE / UNDER_QUORUM";
        tdState.style.color = "var(--status-rose)";
      } else if (rawState.includes("degrad") || rawState.includes("recover") || rawState.includes("reconcil")) {
        tdState.textContent = "▲ DRAINING / DEGRADED";
        tdState.style.color = "var(--status-amber)";
      } else {
        tdState.textContent = "● READY";
        tdState.style.color = "var(--status-emerald)";
      }

      tr.append(tdId, tdEpoch, tdPrimary, tdFollowers, tdState);
      tbody.appendChild(tr);
    }
  }

  async function refreshCollections() {
    const res = await apiRequest("GET", "/v1/collections", null, { allowNon2xx: true });
    const tbody = el("collections-tbody");
    const emptyBox = el("collections-empty");
    const tableWrap = el("collections-table-wrap");
    tbody.replaceChildren();

    if (!res.ok || !res.data) {
      emptyBox.hidden = true;
      tableWrap.hidden = false;
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 7;
      td.style.color = "var(--status-amber)";
      td.textContent = res.status === 401
        ? "▲ Enter a valid Bearer Token in the Session Credentials bar above to list collections."
        : `✕ Could not load collections (HTTP ${res.status}).`;
      tr.appendChild(td);
      tbody.appendChild(tr);
      return;
    }

    const list = Array.isArray(res.data.collections) ? res.data.collections : [];
    state.collections = list;
    syncPlaygroundTargets();

    if (list.length === 0) {
      emptyBox.hidden = false;
      tableWrap.hidden = true;
      return;
    }

    emptyBox.hidden = true;
    tableWrap.hidden = false;

    for (const col of list) {
      const tr = document.createElement("tr");

      const tdName = document.createElement("td");
      const strong = document.createElement("strong");
      strong.textContent = col.name;
      tdName.appendChild(strong);

      const tdDims = document.createElement("td");
      tdDims.textContent = String(col.dimensions);

      const tdMetric = document.createElement("td");
      tdMetric.textContent = String(col.metric);

      const tdQuant = document.createElement("td");
      const qBadge = document.createElement("span");
      qBadge.className = "quant-badge";
      qBadge.textContent = String(col.quantization);
      tdQuant.appendChild(qBadge);

      const tdVecs = document.createElement("td");
      tdVecs.textContent = Number(col.live_vectors || 0).toLocaleString("en-US");

      const tdMem = document.createElement("td");
      tdMem.textContent = formatBytes(col.memory_bytes);

      const tdAct = document.createElement("td");
      const actRow = document.createElement("div");
      actRow.className = "input-row";

      const useBtn = document.createElement("button");
      useBtn.type = "button";
      useBtn.className = "btn-sm";
      useBtn.textContent = "Playground";
      useBtn.addEventListener("click", () => {
        el("pg-target").value = col.name;
        el("pg-dims").value = String(col.dimensions);
        updatePlaygroundRouteLabels();
        activateTab("tab-playground", true);
        announce(`Opened collection ${col.name} in Vector Playground.`);
      });

      const dropBtn = document.createElement("button");
      dropBtn.type = "button";
      dropBtn.className = "btn-sm btn-danger";
      dropBtn.textContent = "Drop";
      dropBtn.addEventListener("click", () => handleTwoClickDrop(col.name, dropBtn));

      actRow.append(useBtn, dropBtn);
      tdAct.appendChild(actRow);

      tr.append(tdName, tdDims, tdMetric, tdQuant, tdVecs, tdMem, tdAct);
      tbody.appendChild(tr);
    }
  }

  async function handleTwoClickDrop(name, btn) {
    if (btn.getAttribute("data-confirming") !== "true") {
      btn.setAttribute("data-confirming", "true");
      btn.textContent = "Confirm Drop?";
      announce(`Press Confirm Drop again within 4 seconds to permanently delete collection ${name}.`);
      const prevTimer = state.dropConfirmTimers.get(name);
      if (prevTimer) clearTimeout(prevTimer);
      const timer = setTimeout(() => {
        btn.removeAttribute("data-confirming");
        btn.textContent = "Drop";
        state.dropConfirmTimers.delete(name);
      }, 4000);
      state.dropConfirmTimers.set(name, timer);
      return;
    }

    const prevTimer = state.dropConfirmTimers.get(name);
    if (prevTimer) clearTimeout(prevTimer);
    state.dropConfirmTimers.delete(name);
    btn.disabled = true;
    btn.textContent = "Dropping...";

    const path = `/v1/collections/${encodeURIComponent(name)}`;
    const res = await apiRequest("DELETE", path, null);
    if (res.ok) {
      clearErrorBanner();
      announce(`Dropped collection ${name} (trace ${res.traceId}).`);
      await refreshCollections();
    } else {
      btn.disabled = false;
      btn.removeAttribute("data-confirming");
      btn.textContent = "Drop";
    }
  }

  function syncPlaygroundTargets() {
    const select = el("pg-target");
    const current = select.value;
    select.replaceChildren();

    const clusterOpt = document.createElement("option");
    clusterOpt.value = "__cluster__";
    clusterOpt.textContent = `Cluster Scatter-Gather (/v1/upsert & /v1/search · ${state.clusterDimensions}d)`;
    select.appendChild(clusterOpt);

    for (const col of state.collections) {
      const opt = document.createElement("option");
      opt.value = col.name;
      opt.textContent = `Collection: ${col.name} (${col.dimensions}d · ${col.metric} · ${col.quantization})`;
      select.appendChild(opt);
    }

    if (Array.from(select.options).some((o) => o.value === current)) {
      select.value = current;
    }
    updatePlaygroundRouteLabels();
  }

  function updatePlaygroundRouteLabels() {
    const target = el("pg-target").value;
    if (target === "__cluster__") {
      el("pg-upsert-route").textContent = "POST /v1/upsert";
      el("pg-search-route").textContent = "POST /v1/search";
      el("pg-dims").value = String(state.clusterDimensions);
    } else {
      const encoded = encodeURIComponent(target);
      el("pg-upsert-route").textContent = `POST /v1/collections/${encoded}/upsert`;
      el("pg-search-route").textContent = `POST /v1/collections/${encoded}/search`;
      const found = state.collections.find((c) => c.name === target);
      if (found) {
        el("pg-dims").value = String(found.dimensions);
      }
    }
  }

  function generateNormalizedVector(dims, seedInput) {
    let s = (Number(seedInput) || 42) >>> 0;
    const raw = new Array(dims);
    let normSq = 0;
    for (let i = 0; i < dims; i += 1) {
      s = (s + 0x6D2B79F5) >>> 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      const unit = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      const signed = unit * 2 - 1;
      raw[i] = signed;
      normSq += signed * signed;
    }
    const invNorm = normSq > 0 ? 1 / Math.sqrt(normSq) : 1;
    return raw.map((v) => Number((v * invNorm).toFixed(6)));
  }

  function parseJsonFloatVector(rawStr) {
    const parsed = JSON.parse(rawStr);
    if (!Array.isArray(parsed) || parsed.length === 0 || !parsed.every((n) => typeof n === "number" && Number.isFinite(n))) {
      throw new Error("Vector must be a non-empty JSON array of finite numbers.");
    }
    return parsed;
  }

  async function refreshAuditChain() {
    const res = await apiRequest("GET", "/v1/audit", null, { allowNon2xx: true });
    el("audit-trace-id").textContent = res.traceId;
    el("audit-latency-ms").textContent = res.latencyMs;

    if (!res.ok || !res.data) {
      el("audit-chain-status").textContent = res.status === 401 || res.status === 403
        ? "▲ ADMIN TOKEN REQUIRED"
        : "✕ AUDIT ERROR";
      el("audit-chain-status").style.color = res.status === 500 ? "var(--status-rose)" : "var(--status-amber)";
      el("audit-security-mode").textContent = res.data?.error || `HTTP ${res.status} from /v1/audit`;
      return;
    }

    const d = res.data;
    const valid = Boolean(d.chain_valid);
    const enabled = Boolean(d.security_enabled);
    el("audit-entries-count").textContent = Number(d.entries_count ?? 0).toLocaleString("en-US");
    const headBig = BigInt(d.head_hash ?? 0);
    el("audit-head-hash").textContent = `0x${headBig.toString(16).padStart(16, "0")} (${headBig.toString(10)})`;

    if (!valid) {
      el("audit-chain-status").textContent = "✕ INTEGRITY VIOLATION";
      el("audit-chain-status").style.color = "var(--status-rose)";
      el("audit-security-mode").textContent = "Hash chain verification failed";
    } else if (enabled) {
      el("audit-chain-status").textContent = "● READY (VERIFIED)";
      el("audit-chain-status").style.color = "var(--status-emerald)";
      el("audit-security-mode").textContent = "SecurityGate active · HMAC + RBAC enforced";
    } else {
      el("audit-chain-status").textContent = "▲ UNPROTECTED DEV MODE";
      el("audit-chain-status").style.color = "var(--status-amber)";
      el("audit-security-mode").textContent = "Start gateway with --master-secret for zero-trust enforcement";
    }
  }

  async function refreshAll() {
    announce("Refreshing cluster telemetry, shard topology, collections, and audit chain...");
    await Promise.all([refreshOverview(), refreshCollections(), refreshAuditChain()]);
    announce("Console telemetry synchronized.");
  }

  function scheduleAutoRefresh(ms) {
    if (state.refreshTimer) {
      clearInterval(state.refreshTimer);
      state.refreshTimer = null;
    }
    if (ms > 0) {
      state.refreshTimer = setInterval(() => {
        refreshOverview();
      }, ms);
    }
  }

  const tabIds = ["tab-overview", "tab-collections", "tab-playground", "tab-security"];

  function activateTab(selectedTabId, focusTab) {
    for (const tId of tabIds) {
      const btn = el(tId);
      const panel = el(btn.getAttribute("aria-controls"));
      const isSelected = tId === selectedTabId;
      btn.setAttribute("aria-selected", isSelected ? "true" : "false");
      btn.tabIndex = isSelected ? 0 : -1;
      panel.hidden = !isSelected;
      if (isSelected && focusTab) {
        btn.focus();
      }
    }
  }

  function wireTabsAndKeyboard() {
    for (let idx = 0; idx < tabIds.length; idx += 1) {
      const btn = el(tabIds[idx]);
      btn.addEventListener("click", () => activateTab(tabIds[idx], false));
      btn.addEventListener("keydown", (ev) => {
        let nextIdx = null;
        if (ev.key === "ArrowRight") nextIdx = (idx + 1) % tabIds.length;
        else if (ev.key === "ArrowLeft") nextIdx = (idx - 1 + tabIds.length) % tabIds.length;
        else if (ev.key === "Home") nextIdx = 0;
        else if (ev.key === "End") nextIdx = tabIds.length - 1;
        if (nextIdx !== null) {
          ev.preventDefault();
          activateTab(tabIds[nextIdx], true);
        }
      });
    }

    document.addEventListener("keydown", (ev) => {
      const modal = el("shortcuts-modal");
      if (ev.key === "Escape" && !modal.hidden) {
        ev.preventDefault();
        modal.hidden = true;
        el("btn-open-shortcuts").focus();
        return;
      }

      const activeTag = document.activeElement ? document.activeElement.tagName : "";
      const isEditing = activeTag === "INPUT" || activeTag === "TEXTAREA" || activeTag === "SELECT" || document.activeElement?.isContentEditable;
      if (isEditing || ev.metaKey || ev.ctrlKey || ev.altKey) {
        return;
      }

      if (ev.key >= "1" && ev.key <= "4") {
        ev.preventDefault();
        activateTab(tabIds[Number(ev.key) - 1], true);
      } else if (ev.key === "r" || ev.key === "R") {
        ev.preventDefault();
        refreshAll();
      } else if (ev.key === "?") {
        ev.preventDefault();
        modal.hidden = !modal.hidden;
        if (!modal.hidden) {
          el("btn-close-shortcuts").focus();
        }
      }
    });
  }

  function initSessionControls() {
    try {
      const savedToken = window.sessionStorage.getItem(SESSION_TOKEN_KEY) || "";
      const savedOrg = window.sessionStorage.getItem(SESSION_ORG_KEY) || "default";
      const savedRefresh = window.sessionStorage.getItem(SESSION_REFRESH_KEY) || "5000";
      el("session-token").value = savedToken;
      el("session-org").value = savedOrg;
      el("auto-refresh-select").value = savedRefresh;
      el("session-auth-indicator").textContent = savedToken ? "● Bearer Active" : "Anonymous";
      scheduleAutoRefresh(Number(savedRefresh));
    } catch (_) {
      scheduleAutoRefresh(5000);
    }

    el("btn-toggle-token").addEventListener("click", () => {
      const input = el("session-token");
      const btn = el("btn-toggle-token");
      const reveal = input.type === "password";
      input.type = reveal ? "text" : "password";
      btn.textContent = reveal ? "Hide" : "Show";
      btn.setAttribute("aria-pressed", reveal ? "true" : "false");
    });

    el("btn-save-session").addEventListener("click", () => {
      const token = el("session-token").value.trim();
      const org = el("session-org").value.trim() || "default";
      const refreshMs = Number(el("auto-refresh-select").value);
      try {
        if (token) window.sessionStorage.setItem(SESSION_TOKEN_KEY, token);
        else window.sessionStorage.removeItem(SESSION_TOKEN_KEY);
        window.sessionStorage.setItem(SESSION_ORG_KEY, org);
        window.sessionStorage.setItem(SESSION_REFRESH_KEY, String(refreshMs));
      } catch (_) {}
      el("session-auth-indicator").textContent = token ? "● Bearer Active" : "Anonymous";
      clearErrorBanner();
      scheduleAutoRefresh(refreshMs);
      refreshAll();
    });

    el("btn-clear-session").addEventListener("click", () => {
      el("session-token").value = "";
      el("session-org").value = "default";
      try {
        window.sessionStorage.removeItem(SESSION_TOKEN_KEY);
        window.sessionStorage.removeItem(SESSION_ORG_KEY);
      } catch (_) {}
      el("session-auth-indicator").textContent = "Anonymous";
      announce("Cleared ephemeral session credentials from sessionStorage.");
      refreshAll();
    });

    el("auto-refresh-select").addEventListener("change", () => {
      const ms = Number(el("auto-refresh-select").value);
      try {
        window.sessionStorage.setItem(SESSION_REFRESH_KEY, String(ms));
      } catch (_) {}
      scheduleAutoRefresh(ms);
      announce(ms > 0 ? `Auto-refresh set to ${ms / 1000} seconds.` : "Auto-refresh paused.");
    });
  }

  function initFormsAndActions() {
    el("btn-refresh-now").addEventListener("click", () => refreshAll());
    el("btn-refresh-collections").addEventListener("click", () => refreshCollections());
    el("btn-verify-audit").addEventListener("click", () => refreshAuditChain());
    el("btn-dismiss-error").addEventListener("click", () => clearErrorBanner());

    el("btn-open-shortcuts").addEventListener("click", () => {
      el("shortcuts-modal").hidden = false;
      el("btn-close-shortcuts").focus();
    });
    el("btn-close-shortcuts").addEventListener("click", () => {
      el("shortcuts-modal").hidden = true;
      el("btn-open-shortcuts").focus();
    });

    el("form-create-collection").addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const payload = {
        name: el("col-name").value.trim(),
        dimensions: Number(el("col-dims").value),
        metric: el("col-metric").value,
        quantization: el("col-quant").value
      };
      const res = await apiRequest("POST", "/v1/collections", payload);
      el("col-create-meta").textContent = res.ok
        ? `● Created '${payload.name}' (${payload.dimensions}d, ${payload.metric}, ${payload.quantization}) · x-trace-id=${res.traceId} · x-latency-ms=${res.latencyMs}`
        : `✕ Failed (${res.status}): ${res.data?.error || "error"} · x-trace-id=${res.traceId}`;
      if (res.ok) {
        clearErrorBanner();
        el("col-name").value = "";
        announce(`Created collection ${payload.name}.`);
        await refreshCollections();
      }
    });

    el("pg-target").addEventListener("change", () => updatePlaygroundRouteLabels());

    el("btn-generate-vector").addEventListener("click", () => {
      const dims = Math.max(1, Math.min(65536, Number(el("pg-dims").value) || 768));
      const seed = Number(el("pg-seed").value) || 42;
      const vec = generateNormalizedVector(dims, seed);
      const serialized = `[${vec.join(", ")}]`;
      el("pg-upsert-vec").value = serialized;
      el("pg-search-vec").value = serialized;
      el("pg-seed").value = String(seed + 1);
      announce(`Generated normalized ${dims}-dimensional unit vector (seed ${seed}).`);
    });

    el("form-playground-upsert").addEventListener("submit", async (ev) => {
      ev.preventDefault();
      try {
        const vector = parseJsonFloatVector(el("pg-upsert-vec").value);
        const id = Number(el("pg-upsert-id").value);
        const writeId = Number(el("pg-write-id").value);
        const target = el("pg-target").value;
        const path = target === "__cluster__"
          ? "/v1/upsert"
          : `/v1/collections/${encodeURIComponent(target)}/upsert`;
        const res = await apiRequest("POST", path, { id, write_id: writeId, vector });
        if (res.ok) {
          clearErrorBanner();
          const detail = target === "__cluster__"
            ? `shard_id=${res.data?.shard_id} seq_no=${res.data?.seq_no}`
            : `collection=${target} id=${id}`;
          el("pg-upsert-result").textContent =
            `● Upserted (${detail}) · x-trace-id=${res.traceId} · x-latency-ms=${res.latencyMs}`;
          el("pg-write-id").value = String(writeId + 1);
          announce(`Upserted vector ${id} in ${res.latencyMs}.`);
        } else {
          el("pg-upsert-result").textContent =
            `✕ HTTP ${res.status}: ${res.data?.error || "upsert failed"} · x-trace-id=${res.traceId}`;
        }
      } catch (err) {
        showErrorBanner(400, err.message, "Vector Playground Upsert");
      }
    });

    el("form-playground-search").addEventListener("submit", async (ev) => {
      ev.preventDefault();
      try {
        const query = parseJsonFloatVector(el("pg-search-vec").value);
        const topK = Number(el("pg-top-k").value);
        const efSearch = Number(el("pg-ef-search").value);
        const target = el("pg-target").value;
        const path = target === "__cluster__"
          ? "/v1/search"
          : `/v1/collections/${encodeURIComponent(target)}/search`;
        const res = await apiRequest("POST", path, { query, top_k: topK, ef_search: efSearch });
        el("pg-search-meta").textContent = `x-trace-id: ${res.traceId} · x-latency-ms: ${res.latencyMs}`;
        const tbody = el("pg-hits-tbody");
        tbody.replaceChildren();

        if (!res.ok || !res.data) {
          const tr = document.createElement("tr");
          const td = document.createElement("td");
          td.colSpan = 3;
          td.style.color = "var(--status-rose)";
          td.textContent = `✕ Search failed (HTTP ${res.status}): ${res.data?.error || "error"}`;
          tr.appendChild(td);
          tbody.appendChild(tr);
          return;
        }

        clearErrorBanner();
        const hits = Array.isArray(res.data.hits) ? res.data.hits : [];
        if (hits.length === 0) {
          const tr = document.createElement("tr");
          const td = document.createElement("td");
          td.colSpan = 3;
          td.textContent = "0 nearest neighbors returned (namespace or cluster index is currently empty).";
          tr.appendChild(td);
          tbody.appendChild(tr);
          announce(`Search completed in ${res.latencyMs} with 0 hits.`);
          return;
        }

        hits.forEach((hit, idx) => {
          const tr = document.createElement("tr");
          const tdRank = document.createElement("td");
          tdRank.textContent = `#${idx + 1}`;
          const tdId = document.createElement("td");
          tdId.textContent = String(hit.id);
          const tdScore = document.createElement("td");
          tdScore.textContent = Number(hit.score).toFixed(6);
          tr.append(tdRank, tdId, tdScore);
          tbody.appendChild(tr);
        });
        announce(`Search returned ${hits.length} hits in ${res.latencyMs}.`);
      } catch (err) {
        showErrorBanner(400, err.message, "Vector Playground Search");
      }
    });
  }

  wireTabsAndKeyboard();
  initSessionControls();
  initFormsAndActions();
  refreshAll();
})();
</script>
</body>
</html>
"##;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn console_csp_header_enforces_zero_trust_airgapped_policy() {
        assert!(CONSOLE_CSP_HEADER.contains("default-src 'none'"));
        assert!(CONSOLE_CSP_HEADER.contains("style-src 'unsafe-inline'"));
        assert!(CONSOLE_CSP_HEADER.contains("script-src 'unsafe-inline'"));
        assert!(CONSOLE_CSP_HEADER.contains("connect-src 'self'"));
        assert!(CONSOLE_CSP_HEADER.contains("img-src 'self' data:"));
        assert!(CONSOLE_CSP_HEADER.contains("base-uri 'none'"));
        assert!(CONSOLE_CSP_HEADER.contains("form-action 'none'"));
        assert!(CONSOLE_CSP_HEADER.contains("frame-ancestors 'none'"));
    }

    #[test]
    fn console_html_has_valid_doctype_landmarks_aria_tabs_and_wcag_features() {
        let html = render_control_console_html();
        assert!(html.starts_with("<!DOCTYPE html>\n<html lang=\"en\">"));
        assert!(
            html.contains("<a class=\"skip-link\" href=\"#main-content\">Skip to main content</a>")
        );
        assert!(html.contains("<header class=\"site-header\" role=\"banner\">"));
        assert!(
            html.contains("<nav class=\"console-nav\" aria-label=\"Primary Console Navigation\">")
        );
        assert!(html.contains("<main id=\"main-content\""));
        assert!(html.contains("<footer class=\"site-footer\" role=\"contentinfo\">"));
        assert!(html.contains("aria-live=\"polite\""));
        assert!(html.contains("role=\"tablist\""));
        assert!(html.contains("role=\"tab\""));
        assert!(html.contains("role=\"tabpanel\""));
        assert!(html.contains("font-variant-numeric: tabular-nums"));
        assert!(html.contains("@media (prefers-reduced-motion: reduce)"));
        assert!(html.contains("@media (max-width: 960px)"));
        assert!(html.contains("@media (max-width: 640px)"));
        assert!(html.contains("● READY"));
        assert!(html.contains("▲ DRAINING / DEGRADED"));
        assert!(html.contains("✕ OFFLINE / UNDER_QUORUM"));
    }

    #[test]
    fn console_html_uses_session_storage_only_and_zero_external_network_urls() {
        let html = render_control_console_html();
        assert!(html.contains("sessionStorage"));
        assert!(!html.contains("localStorage"));
        assert!(!html.contains("http://"));
        assert!(!html.contains("https://"));
        assert!(!html.contains("<script src="));
        assert!(!html.contains("<link rel=\"stylesheet\" href="));
    }

    #[test]
    fn console_html_wires_all_quantization_profiles_and_gateway_endpoints() {
        let html = render_control_console_html();
        for quant in [
            "value=\"f32\"",
            "value=\"fp16\"",
            "value=\"sq8\"",
            "value=\"bq\"",
        ] {
            assert!(html.contains(quant), "missing quantization option {quant}");
        }
        for metric in ["value=\"cosine\"", "value=\"l2\""] {
            assert!(
                html.contains(metric),
                "missing distance metric option {metric}"
            );
        }
        for endpoint in [
            "/healthz",
            "/readyz",
            "/metrics",
            "/v1/topology",
            "/v1/collections",
            "/v1/upsert",
            "/v1/search",
            "/v1/audit",
            "/openapi.json",
        ] {
            assert!(
                html.contains(endpoint),
                "missing gateway endpoint {endpoint}"
            );
        }
        for prom_metric in [
            "vektordist_active_nodes",
            "vektordist_indexed_vectors",
            "vektordist_search_latency_p50_ms",
            "vektordist_search_latency_p95_ms",
            "vektordist_search_latency_p99_ms",
            "vektordist_recall_at_10",
            "vektordist_acked_writes_total",
            "vektordist_acked_writes_lost_total",
        ] {
            assert!(
                html.contains(prom_metric),
                "missing Prometheus metric parser key {prom_metric}"
            );
        }
        assert!(html.contains("encodeURIComponent"));
    }
}
