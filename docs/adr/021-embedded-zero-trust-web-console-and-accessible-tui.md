# ADR-021: Embedded Zero-Trust Air-Gapped Web Control Console and Accessible Responsive TUI
- **Status:** Accepted
- **Owner:** Justin Mafie

## Context
As VektorDist operates inside enterprise VPCs, Kubernetes clusters (`deploy/k8s/statefulset.yaml`), and air-gapped environments alongside multi-collection namespaces (`/v1/collections/*`), W3C distributed tracing (`traceparent`), and HMAC-authenticated RBAC (`SecurityGate`), platform SREs and AI engineers interact with the system across two primary operator surfaces:

1. **Browser-Based Cluster & Namespace Operations:** Operators inspecting quorum shard topology, Prometheus latency/recall SLAs, multi-collection quantization schemas (`f32`, `fp16`, `sq8`, `bq`), vector search quality, and tamper-evident audit chains previously had to assemble manual `curl` pipelines or external dashboards that require Node.js build toolchains or outbound CDN access.
2. **Terminal Chaos & Failover Dashboard (`vektordist demo`):** Operators running split-pane terminal multiplexers (`tmux`, IDE terminals, SSH sessions) or monochrome / color-vision-deficient setups need non-color status semantics, dynamic binary versioning, empty-state onboarding guidance, discoverable keyboard controls, and graceful layout degradation below 20 terminal rows.

## Decision
1. **Zero-Dependency In-Memory Web Control Console (`crates/vektordist-cluster/src/console.rs`):**
   - Embed a complete, self-contained HTML5/CSS3/ES2022 **VektorDist Control Console** (`render_control_console_html() -> &'static str`) directly in compiled binary memory with zero external JavaScript, font, or stylesheet requests.
   - Pair the console with `CONSOLE_CSP_HEADER` (`default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'`), preventing clickjacking, cross-origin exfiltration, and supply-chain script injection in air-gapped VPCs.
   - Provide four operational panels wired to live gateway REST endpoints with automatic W3C `traceparent` generation:
     - **Cluster & Telemetry:** Polls `/healthz`, `/readyz`, `/metrics` (client-side Prometheus `0.0.4` parser), and `/v1/topology`.
     - **Collections Studio:** Manages isolated namespaces via `GET /v1/collections`, `POST /v1/collections` (`cosine`/`l2`, `f32`/`fp16`/`sq8`/`bq`), and safe two-click `DELETE /v1/collections/{encodeURIComponent(name)}`.
     - **Vector Playground:** Supports both cluster scatter-gather (`/v1/upsert`, `/v1/search`) and named collections with a one-click deterministic L2-normalized unit vector generator (`||v||_2 = 1.0`), surfacing `x-trace-id` and `x-latency-ms`.
     - **Security & Audit:** Verifies the hash-chained audit ledger (`GET /v1/audit`) and links directly to `/openapi.json` and `/metrics`.
2. **Ephemeral `sessionStorage` Credential Scoping:**
   - Store operator `Bearer` tokens (`type="password"` with explicit Show/Hide toggle and `Clear Session` control) and `X-Org-Id` headers strictly in tab-scoped `sessionStorage`, never in persistent browser storage or URL query parameters where tokens could leak via access logs or shared links.
3. **WCAG 2.1 AA Accessibility & Motion Ergonomics Across Web and TUI (`src/tui.rs`):**
   - Enforce `>= 4.8:1` contrast ratios on obsidian slate surfaces (`#090D16` / `#111827` with `#F8FAFC` primary and `#94A3B8` secondary text), `font-variant-numeric: tabular-nums` to eliminate layout jitter during live polling, and `@media (prefers-reduced-motion: reduce)` to disable animations for vestibular accessibility.
   - Implement skip-link navigation, semantic landmarks (`banner`, `nav`, `main`, `contentinfo`), full WAI-ARIA `tablist`/`tab`/`tabpanel` roving `tabindex` keyboard support (`ArrowLeft`/`ArrowRight`/`Home`/`End` plus `1`–`4`, `r`, `?`), `aria-live="polite"` announcements, and actionable `401`/`403`/`429`/`503` guidance banners.
   - Upgrade `src/tui.rs` with compile-time `env!("CARGO_PKG_VERSION")` branding, non-color status glyphs (`● HEALTHY`, `● PAUSED`, `▲ DEGRADED`, `✕ DEAD`, `● 0 LOST`), empty-state guidance, an interactive `[?]`/`[h]` help modal, a compact 3-row layout fallback (`height < 20` or `width < 72`), and deterministic `ratatui::backend::TestBackend` unit tests via pure `render_dashboard`.

## Alternatives Considered
- **Separate React/Vite Single-Page Application Bundle:** Rejected because introducing Node.js/`npm` build steps, hashed static asset directories, or external CDN imports breaks single-binary deployment (`vektordist gateway`) and fails strict air-gapped enterprise VPC requirements.
- **Persisting API Keys in Long-Lived Browser Storage or URL Fragments:** Rejected because shared SRE workstations or copied diagnostic URLs could retain or leak privileged `admin` tokens across browser restarts.
- **Fixed-Height Terminal Layout Only:** Rejected because `tmux` horizontal splits routinely drop terminal height to 12–18 lines, which previously starved the node topology panel when the sparkline and event log consumed fixed vertical rows.

## Consequences & Failure Modes
- **`unsafe-inline` Script/Style Trade-Off Under Single-Document Delivery:** Serving a single static `&'static str` without external asset routes requires `style-src 'unsafe-inline'` and `script-src 'unsafe-inline'` (or per-response cryptographic nonces); all DOM updates use safe `textContent` / `createElement` construction (`replaceChildren`) rather than `innerHTML` to prevent DOM XSS from untrusted collection names or error strings.
- **Client-Side Vector Payload Size:** Browser JSON serialization in the Vector Playground is designed for interactive debugging up to `65,536` dimensions; bulk million-vector ingestion continues to use the Rust `HttpVektorClient` or Python `VektorDistClient.upsert_batch` APIs.
