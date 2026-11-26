# ADR-015: Production Operations Hardening, Quorum PodDisruptionBudgets, and SRE Runbook
- **Status:** Accepted
- **Owner:** Justin Mafie

## Context
As VektorDist advances to `v0.2.0` (incorporating `ShardCheckpointManager` atomic WAL compaction, `GrpcChannelPool` & `/v1/upsert_batch`, `/readyz` readiness draining, `HttpVektorClient`, Raft-replicated `SecurityGate` credentials & `PersistentAuditLog`, and streaming `1M`-vector scale validation), production Kubernetes operators require structural safeguards against accidental quorum loss during voluntary node maintenance, strict least-privilege container isolation, automated Prometheus scraping, and an authoritative SRE runbook (`docs/OPERATIONS.md`).

Without a `PodDisruptionBudget` coupled to VektorDist's `RF=3, W=2` quorum math, concurrent Kubernetes node drains (`kubectl drain`) or aggressive rolling updates could simultaneously evict 2 of the 5 `vektordist-node` pods, pushing `RF=3` shards into `UnderQuorum` write unavailability. Similarly, conflating liveness (`/healthz`) and readiness (`/readyz`) on `vektordist-gateway` risks either black-holing traffic during pod shutdown or triggering spurious kubelet container restarts during transient upstream failovers.

## Decision
1. **Quorum-Aware Kubernetes Manifests (`deploy/k8s/statefulset.yaml`):**
   - Harden `vektordist-node` (`5` replicas, `podManagementPolicy: Parallel`, `RollingUpdate` with `maxUnavailable: 1`, `terminationGracePeriodSeconds: 30`) with `PodAntiAffinity` (`requiredDuringSchedulingIgnoredDuringExecution` across `kubernetes.io/hostname` and `preferredDuringSchedulingIgnoredDuringExecution` across `topology.kubernetes.io/zone`) and SSD `volumeClaimTemplates` (`premium-rwo`).
   - Add `PodDisruptionBudget` `vektordist-node-pdb` (`minAvailable: 4`), guaranteeing at most 1 replica node is voluntarily disrupted at a time so all 16 `RF=3` shards retain `>= 2` healthy replicas (`W=2` write quorum).
   - Deploy `vektordist-gateway` (`2` stateless replicas, `maxUnavailable: 0`, `PodDisruptionBudget` `minAvailable: 1`) behind a `ClusterIP` `Service` (`8080`) with `livenessProbe` on `GET /healthz`, `readinessProbe` on `GET /readyz`, `Secret` injection for `VEKTORDIST_MASTER_SECRET` (`vektordist-auth-secret`), and standard Prometheus scrape annotations (`prometheus.io/scrape: "true"`, `prometheus.io/port: "8080"`, `prometheus.io/path: "/metrics"`).
2. **Zero-Trust Container Runtime (`deploy/Dockerfile` & Pod Security Contexts):**
   - Enforce `runAsNonRoot: true`, `runAsUser: 65532`, `runAsGroup: 65532`, `fsGroup: 65532`, `seccompProfile: RuntimeDefault`, `automountServiceAccountToken: false`, `allowPrivilegeEscalation: false`, `readOnlyRootFilesystem: true`, and `capabilities: drop: ["ALL"]` across every workload.
   - Build `--release --locked` in `deploy/Dockerfile`, bundle `/usr/share/licenses/vektordist/LICENSE` inside `gcr.io/distroless/cc-debian12:nonroot`, and expose ports `50051`, `50052`, and `8080`.
3. **Enterprise SRE Runbook (`docs/OPERATIONS.md`):**
   - Document exact capacity formulas (`f32`, `SQ8` 4x, `PQ` 16x, `BQ` 32x, and HNSW link overhead), rolling-upgrade drain mechanics, `ShardCheckpointManager` (`b"VDCK"` + `b"VDSG"`) + `vektordist snapshot` disaster recovery, Raft-replicated key rotation & `/v1/audit` forensics, and `vektordist doctor` + Prometheus `PrometheusRule` alerts.

## Alternatives Considered
- **`minAvailable: 3` on a 5-Node Cluster:** Allowing 2 nodes down simultaneously can knock out 2 of the 3 replicas assigned to specific shards, forcing those shards into `ShardState::UnderQuorum` (`QUORUM_UNAVAILABLE` for writes). Setting `minAvailable: 4` preserves 100% write availability during planned maintenance.
- **Single `/healthz` Probe for Both Liveness and Readiness:** Returning `503` on `/healthz` during cluster topology convergence causes kubelet to restart healthy gateway processes, severing active connections; separating `/readyz` (endpoint traffic gating) from `/healthz` (process liveness) eliminates cascading restarts.

## Consequences & Failure Modes
- **Blocked Node Drain During Unplanned Node Failure:** If 1 of 5 nodes has already crashed (`active_nodes == 4`), `vektordist-node-pdb` (`minAvailable: 4`) intentionally blocks `kubectl drain` on the remaining 4 nodes until the failed pod recovers or a replacement node joins — protecting `W=2` quorum durability over automated node eviction speed.
- **Read-Only Root Filesystem Discipline:** Because `readOnlyRootFilesystem: true` is enforced on all containers, `vektordist-node` writes exclusively to its mounted SSD volume (`/var/lib/vektordist/data`) and bounded memory-backed `/tmp` `emptyDir`.
