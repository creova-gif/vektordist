# VektorDist Production Operations & SRE Runbook

This runbook defines the authoritative operational procedures, capacity sizing formulas, zero-downtime upgrade invariants, disaster recovery playbooks, zero-trust security workflows, and Prometheus alerting rules for running **VektorDist** in production.

---

## 1. Capacity Planning & Memory Sizing Formulas

Every VektorDist shard stores dense vectors in a contiguous cache-aligned arena alongside a multi-layer HNSW graph (`HnswIndex`), optional compressed quantization payloads (`Sq8HnswIndex`, `PqIndex`, `BqIndex`), and an append-only IEEE CRC32 write-ahead log (`WalLog`) backed by zero-copy binary checkpoints (`PersistentVectorSegment`).

### 1.1 Per-Vector Payload Byte Formulas

For vectors of dimensionality $D$:

| Storage Tier | Precision / Encoding | Bytes per Vector | Compression vs `f32` | Notes & Auxiliary Overhead |
| :--- | :--- | :--- | :--- | :--- |
| **`f32` (`VectorArena`)** | IEEE 754 32-bit float | $4 \times D\text{ B}$ | **1.0x** (baseline) | Exact L2 / unit-normalized Cosine SIMD (`aarch64` NEON / portable SIMD) |
| **`SQ8` (`Sq8HnswIndex`)** | Affine 8-bit unsigned (`u8`) | $1 \times D\text{ B}$ | **4.0x** | Asymmetric `f32 x u8` dot product + fixed $8 \times D\text{ B}$ calibration table |
| **`PQ` (`PqIndex`)** | $M$ subspaces $\times$ 256 centroids (`u8` codes) | $M\text{ B}$ | **16.0x** (at $M = D / 4$) | Asymmetric Distance Computation (`ADC`) + fixed $256 \times D \times 4\text{ B}$ codebook |
| **`BQ` (`BqIndex`)** | 1-bit sign quantization (packed `u64` words) | $\lceil D / 64 \rceil \times 8 = D / 8\text{ B}$ | **32.0x** | `XOR + POPCNT` Hamming scan; combine with `3x`–`10x` exact reranking |

**Worked Example ($N = 10{,}000{,}000$ vectors, $D = 768$ dimensions, per replica):**

- **`f32` raw payload:** $10{,}000{,}000 \times (4 \times 768)\text{ B} = 30{,}720{,}000{,}000\text{ B} = \mathbf{28.61\text{ GiB}}$ (`29,296.9 MiB`)
- **`SQ8` raw payload (4x):** $10{,}000{,}000 \times 768\text{ B} = 7{,}680{,}000{,}000\text{ B} = \mathbf{7.15\text{ GiB}}$ (`7,324.2 MiB`)
- **`PQ` raw payload (16x, $M = 192$):** $10{,}000{,}000 \times 192\text{ B} = 1{,}920{,}000{,}000\text{ B} = \mathbf{1.79\text{ GiB}}$ (`1,831.1 MiB`, plus `768 KiB` codebook)
- **`BQ` raw bitvector payload (32x):** $10{,}000{,}000 \times (768 / 8)\text{ B} = 960{,}000{,}000\text{ B} = \mathbf{915.5\text{ MiB}}$ (`96 B/vec` — fits a 10M-vector first-stage candidate index under `1 GiB` of RAM!)

### 1.2 HNSW Graph Link & Slot Metadata Overhead

On 64-bit architectures (`usize` = 8 bytes), each indexed vector holds up to $M_{\max0}$ bidirectional neighbor links at layer `0` (default $M_{\max0} = 2M$) and $M$ links at layers $l \ge 1$ with exponential level decay $p = 1/M$:

$$\text{Bytes}_{\text{HNSW}}(M) \approx \left(M_{\max0} + \frac{M}{M - 1}\right) \times 8\text{ bytes/vec}$$

With default `HnswParams` ($M = 16$, $M_{\max0} = 32$):

- **Graph neighbor edges:** $(32 + 16 / 15) \times 8 = 33.067 \times 8 \approx \mathbf{264.5\text{ B/vec}}$
- **Slot ID, layer-vec headers & tombstone bitset:** $\approx \mathbf{32\text{ B/vec}}$
- **Total HNSW structural overhead ($10\text{M}$ vectors):** $\approx 296.5\text{ B/vec} \times 10{,}000{,}000 \approx \mathbf{2.76\text{ GiB}}$

### 1.3 Physical Node RAM & Disk Sizing Across Shards and Replicas

Across a cluster of $N_{\text{nodes}}$ physical nodes with replication factor $\text{RF}$ (default $N_{\text{nodes}} = 5$, $\text{RF} = 3$, $S = 16$ shards), each node hosts $N_{\text{local}} = N_{\text{logical}} \times \frac{\text{RF}}{N_{\text{nodes}}}$ replica vectors. Provision container `resources.limits.memory` with **25% headroom** for Tokio worker stacks, `GrpcChannelPool` buffers, `TieredCompactor` scratch arenas, and OS page cache:

$$\text{RAM}_{\text{node}} = 1.25 \times N_{\text{logical}} \times \frac{\text{RF}}{N_{\text{nodes}}} \times \left(\text{Bytes}_{\text{payload}} + \text{Bytes}_{\text{HNSW}}(M)\right)$$

| Logical Dataset ($D=768$, $M=16$) | Vectors / Node ($N_{\text{nodes}}=5, \text{RF}=3$) | `f32` + HNSW Node RAM (`1.25x`) | `SQ8` + HNSW Node RAM (`1.25x`) | `BQ` + Disk Rerank Node RAM (`1.25x`) | Recommended PVC (`pd-ssd` / `premium-rwo`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`1,000,000`** | `600,000` | `2.38 GiB` -> **`4 GiB`** | `800 MiB` -> **`2 GiB`** | `300 MiB` -> **`1 GiB`** | `50 GiB` |
| **`10,000,000`** | `6,000,000` | `23.80 GiB` -> **`32 GiB`** | `7.71 GiB` -> **`12 GiB`** | `2.76 GiB` -> **`6 GiB`** | `250 GiB` |
| **`100,000,000`** | `60,000,000` | `238.0 GiB` -> **`256 GiB`** | `77.1 GiB` -> **`96 GiB`** | `27.6 GiB` -> **`40 GiB`** | `2 TiB` |

---

## 2. Zero-Downtime Rolling Upgrades

VektorDist guarantees **zero dropped reads** and **zero lost acknowledged writes (`vektordist_acked_writes_lost_total == 0`)** during single-node rolling restarts, OS kernel patching, and `kubectl drain` operations through four coordinated mechanisms:

1. **Liveness (`/healthz`) vs. Readiness (`/readyz`) Separation on `vektordist-gateway`:**
   - `GET /healthz` (`livenessProbe`) returns `200 OK` whenever the Tokio HTTP/1.1 listener loop is alive and responsive, preventing kubelet from `SIGKILL`-restarting a gateway pod simply because an upstream coordinator or shard is transiently reconciling.
   - `GET /readyz` (`readinessProbe`) verifies upstream cluster connectivity, active quorum availability across shards, and non-draining gateway state. As soon as a gateway pod begins draining or loses quorum visibility, `/readyz` returns `503 Service Unavailable`, causing `kube-proxy` / Envoy / Ingress controllers to remove the pod from active load-balancing endpoints before terminating open keep-alive sockets.
2. **Graceful Connection Drain (`terminationGracePeriodSeconds: 30`):**
   - Both `vektordist-node` and `vektordist-gateway` configure `terminationGracePeriodSeconds: 30` — more than an order of magnitude above scatter-gather replica read timeouts (`200ms`) and `W=2` quorum write timeouts. In-flight RPCs complete cleanly before container exit.
3. **`PodDisruptionBudget` (`minAvailable: 4` on `5` Replica Nodes):**
   - `vektordist-node-pdb` enforces `minAvailable: 4` across the 5-replica `StatefulSet` (`RollingUpdate` with `maxUnavailable: 1`). Kubernetes refuses voluntary evictions (`kubectl drain`) whenever 1 node is already down or unready, preventing simultaneous double-node eviction that could drop any `RF=3` shard below its `W=2` write quorum.
4. **Quorum Overlap (`W=2` of `RF=3`) & Anti-Entropy Catch-Up:**
   - While `vektordist-node-i` restarts, every shard hosted on `node-i` retains **2 surviving online replicas**.
   - If `node-i` held shard primaries, the coordinator/Raft state machine increments shard `epoch`, fences stale RPCs via `STATUS_WRONG_EPOCH`, runs union WAL reconciliation across the 2 survivors, and promotes a surviving replica to primary in `< 2s`.
   - Live writes continue succeeding at `W=2` (`primary + 1 follower`), and `R=1` reads route exclusively to healthy replicas via `AdaptiveRouter` (`P2C`) and `GrpcChannelPool`.
   - When `vektordist-node-i` comes back online with its `wal-data` PVC intact, it replays its local checkpoint + WAL delta, catches up any missed `seq_no` records (`ShardState::Recovering -> ShardState::Healthy`), and restores full `RF=3` durability before the controller rolls the next pod.

### Rolling Upgrade Checklist

```bash
# 1. Verify all 5 replica nodes and 2 gateways are Ready and active_nodes == configured_nodes
kubectl get pods,pdb -l app.kubernetes.io/name=vektordist
curl -sf http://vektordist-gateway:8080/readyz | jq .

# 2. Apply updated image tag to StatefulSet and Gateway Deployment
kubectl apply -f deploy/k8s/statefulset.yaml

# 3. Watch sequential rollout and verify zero write loss throughout
kubectl rollout status statefulset/vektordist-node --timeout=10m
kubectl rollout status deployment/vektordist-gateway --timeout=5m
curl -sf http://vektordist-gateway:8080/metrics | grep -E "vektordist_(active_nodes|acked_writes_lost_total)"
```

---

## 3. WAL Checkpointing, Compaction & Snapshot Disaster Recovery

### 3.1 Operating `ShardCheckpointManager` (`b"VDCK"` + `b"VDSG"`) & Atomic WAL Compaction

Unbounded append-only WAL growth increases cold-start replay time and SSD usage. VektorDist bounds WAL size per shard via `ShardCheckpointManager` and `TieredCompactor`:

- **Binary Segment (`b"VDSG"`):** `PersistentVectorSegment` writes an 8-byte-aligned snapshot of the shard's `VectorArena` (`32`-byte header, `slot_ids`, `tombstones`, contiguous `f32` vector payload, and trailing 4-byte IEEE CRC32 checksum over `0..len - 4`), allowing zero-copy `&[f32]` reads without per-query heap allocations.
- **Checkpoint Manifest (`b"VDCK"`):** `ShardCheckpointManager` pairs the `b"VDSG"` segment with a CRC32-verified `b"VDCK"` checkpoint metadata header recording `(shard_id, epoch, checkpoint_seq_no, live_vectors, segment_crc32)`.
- **Crash-Safe Atomic Compaction:**
  1. Write compacted arena (`TieredCompactor` strips 100% of tombstoned slots) to temporary `shard-XXXX.vdsg.tmp` and `shard-XXXX.vdck.tmp` files and `fsync` both files.
  2. Write any post-checkpoint WAL tail (`seq_no > checkpoint_seq_no`) to `shard-XXXX.wal.tmp` and `fsync`.
  3. Atomically `rename` `.vdsg.tmp -> .vdsg`, `.vdck.tmp -> .vdck`, and `.wal.tmp -> .wal`, then `fsync` the parent directory.
  - Because `rename(2)` on POSIX filesystems (`ext4` / `xfs`) is atomic, a `SIGKILL` or power loss at any stage leaves either the complete pre-compaction WAL or the verified `VDCK`/`VDSG` checkpoint + truncated WAL delta intact.

### 3.2 Point-in-Time Cluster Snapshots (`b"VDSN"`)

Capture, verify, and restore point-in-time node backups using the built-in `vektordist snapshot` CLI:

```bash
# 1. Capture a coordinated multi-shard snapshot (ordered shard read-locks + fsync + manifest.bin)
vektordist snapshot create \
  --node-id 1 \
  --node-dir /var/lib/vektordist/data \
  --snapshot-dir /backups/vektordist/node-1-20261126T1200Z \
  --dimensions 768 \
  --num-shards 16

# 2. Cryptographically/CRC32 verify manifest.bin (b"VDSN") and every shard-XXXX.wal file
vektordist snapshot verify \
  --snapshot-dir /backups/vektordist/node-1-20261126T1200Z

# 3. Restore into a clean replacement volume and verify reconstructed seq_no & vector counts
vektordist snapshot restore \
  --snapshot-dir /backups/vektordist/node-1-20261126T1200Z \
  --target-dir /var/lib/vektordist/data-restored
```

> **Operational Rule:** Always execute `vektordist snapshot verify` immediately after uploading/downloading backups from object storage (`GCS` / `S3`) and **before** running `snapshot restore`. Any 1-byte bit-rot in `manifest.bin` or `shard-XXXX.wal` aborts pre-restore verification before touching `--target-dir`.

### 3.3 Handling `VektorError::WalCorruption` vs. Torn-EOF Recovery

VektorDist distinguishes cleanly between expected crash tails and silent storage media corruption:

| Condition | Detection Mechanism | Automatic Behavior | Operator Action Required |
| :--- | :--- | :--- | :--- |
| **Torn EOF Frame** (power loss / `SIGKILL` mid-`write`) | Incomplete frame header or truncated trailing payload at EOF | `WalLog::open` safely truncates the unacknowledged tail frame to the last durable `fsync` offset | **None.** All quorum-acked writes remain intact. |
| **Mid-File Bit-Rot** (`VektorError::WalCorruption { offset }`) | IEEE CRC32 mismatch on an interior WAL record, `VDCK` header, or `VDSG` segment | **Fail-closed startup refusal.** Node refuses to open corrupt shard to prevent serving poisoned vectors or advancing epochs on bad state | **Follow Remediation Procedure below.** |

#### `VektorError::WalCorruption` Remediation Playbook

1. **Do NOT lower `write_quorum` or force-edit the corrupt `.wal` file.** Confirm the remaining 4 cluster nodes are healthy (`vektordist_active_nodes >= 4`), ensuring the other 2 replicas of affected shards continue serving `W=2` writes and `R=1` reads.
2. **Preserve Forensic Evidence:** Move `/var/lib/vektordist/data` on the affected host/PVC to `/var/lib/vektordist/corrupt-<timestamp>` and inspect underlying NVMe/`pd-ssd` SMART / kernel `dmesg` I/O errors. Provision a fresh PVC if hardware media errors are present.
3. **Recover Replica State (Two Safe Paths):**
   - **Path A — Online Quorum Catch-Up (Preferred when $\ge 2$ peer replicas are Healthy):** Start `vektordist node` with an empty `--data-dir`. The coordinator places local shards into `ShardState::Recovering`, dual-streams live writes from shard primaries, and replays historical records until `lag == 0` (`ShardState::Healthy`).
   - **Path B — Full Cluster Disaster Recovery (Catastrophic multi-node loss):** Verify the latest off-site snapshot (`vektordist snapshot verify --snapshot-dir ...`), run `vektordist snapshot restore` into each node's `/var/lib/vektordist/data`, and bring up the cluster.

---

## 4. Zero-Trust Key Rotation & Audit Forensics

### 4.1 Issuing Scoped Tenant API Keys (`vektordist keygen`)

When `VEKTORDIST_MASTER_SECRET` (recommended: 64-character CSPRNG hex string stored in Kubernetes `Secret` `vektordist-auth-secret`) is configured on `vektordist-gateway`, all `/v1/*` data and control endpoints enforce constant-time (`constant_time_eq_32`) MAC verification of `Authorization: Bearer vdk.<key_id>.<hex_tag>` tokens bound to `(OrgId, TenantRole, nonce)`:

```bash
# Generate a 256-bit (64-hex) master secret once per environment
export VEKTORDIST_MASTER_SECRET="$(openssl rand -hex 32)"

# Issue least-privilege tenant keys (roles: reader < writer < admin)
vektordist keygen --org acme-prod --role reader --key-id acme-search-2026q4 --nonce 1
vektordist keygen --org acme-prod --role writer --key-id acme-ingest-2026q4 --nonce 1
vektordist keygen --org secops    --role admin  --key-id secops-audit-2026q4 --nonce 1
```

### 4.2 Zero-Downtime Raft-Replicated Key Rotation & Instant Revocation

Dynamic key issuance (`IssueApiKey`), quota updates (`SetTenantQuota`), and revocations (`RevokeApiKey`) replicate deterministically through the Raft security state machine so every stateless `vektordist-gateway` enforces identical credentials:

1. **Dual-Key Overlap:** Provision the replacement key (`--key-id acme-ingest-2027q1 --nonce 2`) and commit `IssueApiKey` to the Raft security log. Both `acme-ingest-2026q4` and `acme-ingest-2027q1` are simultaneously valid for `X-Org-Id: acme-prod`.
2. **Client Cutover:** Roll application secrets (`HttpVektorClient` / client deployments) to the new bearer token.
3. **Instant Cluster-Wide Revocation:** Commit `RevokeApiKey { key_id: "acme-ingest-2026q4" }`. Every gateway replica immediately rejects the retired token with `401 Unauthorized` and records the denied attempt in `PersistentAuditLog`.

### 4.3 Noisy-Neighbor Rate Limiting (`429 Retry-After`) & Quota Tuning

`SecurityGate` isolates multi-tenant workloads via per-`OrgId` token-bucket rate limiters and atomic vector/byte storage ceilings (`TenantQuotaPolicy`):

- **QPS / Burst Exhaustion (`429 Too Many Requests`):** Returns RFC 9110 `Retry-After: <seconds>` (integer seconds $\ge 1$). `HttpVektorClient` and compliant SDKs honor `Retry-After` backoff without amplifying retry storms. Increase `qps_limit` / `burst_capacity` via Raft-replicated `SetTenantQuota` for legitimate traffic spikes, or leave capped to contain runaway clients.
- **Payload DoS Guard (`413 Content Too Large`):** Requests exceeding the gateway's maximum body ceiling are rejected immediately before JSON allocation or shard RPC dispatch; callers ingesting large corpora should chunk `POST /v1/upsert_batch` into batches of `256`–`1,000` vectors.
- **Storage Ceiling (`507 Insufficient Storage`):** Triggered when `current_vectors > max_vectors` or `current_bytes > max_bytes`; remediate by deleting obsolete vectors (`POST /v1/delete` + compaction) or raising the tenant's quota policy.

### 4.4 Tamper-Evident Audit Forensics (`GET /v1/audit` & `PersistentAuditLog`)

Every allowed and denied request appends a chained `AuditEntry` (`seq_no`, `timestamp_ms`, `org_id`, `key_id`, `action`, `allowed`, `prev_hash`, `entry_hash`) to `PersistentAuditLog`. Verify live cryptographic hash-chain continuity from genesis at any time using an `Admin`-scoped token:

```bash
curl -sf http://vektordist-gateway:8080/v1/audit \
  -H "Authorization: Bearer vdk.secops-audit-2026q4.<hex_tag>" \
  -H "X-Org-Id: secops" | jq .
# Expected healthy output:
# {
#   "status": "ok",
#   "security_enabled": true,
#   "entries_count": 48291,
#   "head_hash": 14982736451928374651,
#   "chain_valid": true
# }
```

- If `GET /v1/audit` returns `500 Internal Server Error` (`tamper-evident audit log integrity violation at seq_no <N>`), **escalate immediately as a SEV-1 security incident**: snapshot the gateway/node audit log directory for forensic preservation, isolate the host, and rotate `VEKTORDIST_MASTER_SECRET`.

---

## 5. Pre-Flight Diagnostics & Prometheus Alerting Rules

### 5.1 Pre-Flight Cluster Qualification (`vektordist doctor` & `bench-scale`)

Run `vektordist doctor` during CI/CD image qualification or host provisioning before joining a node to production traffic, and use `vektordist bench-scale` to validate streaming `1M`-vector memory footprint and tail latency on target hardware:

```bash
# 1. Pre-flight environment, config, fsync durability, SIMD ISA, and secret hygiene checks
vektordist doctor --data-dir /var/lib/vektordist/data

# 2. Streaming 1,000,000-vector scale & memory benchmark (bounded RAM via BQ/SQ8)
vektordist bench-scale --vectors 1000000 --dimensions 768 --seed 42
```

### 5.2 Production Prometheus Alerting Rules

`vektordist-gateway` pods carry standard scrape annotations (`prometheus.io/scrape: "true"`, `prometheus.io/port: "8080"`, `prometheus.io/path: "/metrics"`) and export Prometheus `0.0.4` metrics without requiring credentials. Deploy the following `PrometheusRule` / `alerting_rules.yml` in every production cluster:

```yaml
groups:
  - name: vektordist.production.sla
    interval: 15s
    rules:
      - alert: VektorDistAckedWriteLost
        expr: vektordist_acked_writes_lost_total > 0
        for: 0m
        labels:
          severity: critical
          service: vektordist
        annotations:
          summary: "CRITICAL: Quorum-acknowledged write loss detected in VektorDist"
          description: "vektordist_acked_writes_lost_total is {{ $value }} (> 0). Page on-call SRE immediately, halt topology mutations, and preserve shard WALs for forensic inspection."

      - alert: VektorDistNodeDegraded
        expr: vektordist_active_nodes < vektordist_configured_nodes
        for: 2m
        labels:
          severity: warning
          service: vektordist
        annotations:
          summary: "VektorDist cluster has offline replica node(s)"
          description: "Active nodes ({{ $value }}) is below configured nodes for > 2m. Shards remain available under W=2 of RF=3 quorum, but single-node fault tolerance margin is consumed."

      - alert: VektorDistQuorumAtRisk
        expr: (vektordist_configured_nodes - vektordist_active_nodes) >= 2
        for: 30s
        labels:
          severity: critical
          service: vektordist
        annotations:
          summary: "CRITICAL: Multiple VektorDist replica nodes offline (W=2 write quorum degraded)"
          description: "Two or more replica nodes are offline; affected RF=3 shards enter UnderQuorum read-only protection (QUORUM_UNAVAILABLE on writes). Restore offline nodes immediately."

      - alert: VektorDistHighSearchLatencyP95
        expr: vektordist_search_latency_p95_ms > 50
        for: 5m
        labels:
          severity: warning
          service: vektordist
        annotations:
          summary: "VektorDist p95 scatter-gather search latency above 50ms SLA"
          description: "95th percentile search latency is {{ $value }}ms (> 50ms) for 5m. Inspect hot-shard skew, CPU throttling, or AdaptiveSearchController ef_search bounds."

      - alert: VektorDistRecallDegradation
        expr: vektordist_recall_at_10 < 0.95
        for: 5m
        labels:
          severity: warning
          service: vektordist
        annotations:
          summary: "VektorDist live recall@10 dropped below 95% SLO"
          description: "Measured recall@10 is {{ $value }} (< 0.95). Check ef_search latency budget pressure, tombstone ratio awaiting TieredCompactor, or quantization rerank oversampling."
```
