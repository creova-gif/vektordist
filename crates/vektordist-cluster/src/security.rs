//! Enterprise multi-tenant security: constant-time scoped API-key authentication, RBAC
//! authorization (`Reader`, `Writer`, `Admin`), per-tenant token-bucket rate limiting and
//! storage quotas, and a tamper-evident hash-chained audit log.

use std::collections::HashMap;
use std::fmt;

pub use vektordist_core::hybrid::OrgId;

/// Role-based access control (RBAC) privilege tier assigned to a tenant-scoped API key.
///
/// Ordered monotonically by privilege: `Reader < Writer < Admin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TenantRole {
    /// Read-only search (`search_dense_filtered`, `search_lexical_filtered`, `hybrid_search_rrf`,
    /// `get_metadata`) within the key's bound `OrgId`.
    Reader = 1,
    /// Read + write mutations (`upsert`, `delete`) within the key's bound `OrgId`.
    Writer = 2,
    /// Full tenant administration (key rotation, quota inspection, snapshot verification) in
    /// addition to all `Writer` and `Reader` permissions within the bound `OrgId`.
    Admin = 3,
}

impl TenantRole {
    /// Returns `true` if `self` holds sufficient privilege to perform an operation requiring
    /// `required`.
    #[must_use]
    pub const fn allows(self, required: Self) -> bool {
        (self as u8) >= (required as u8)
    }

    /// Canonical string label for audit logging and digest binding.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reader => "reader",
            Self::Writer => "writer",
            Self::Admin => "admin",
        }
    }
}

impl fmt::Display for TenantRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Security, authorization, quota, and audit verification errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecurityError {
    /// Presented API key is malformed, unknown, revoked, or failed constant-time MAC verification.
    Unauthorized,
    /// Valid API key bound to `key_org` attempted to access `target_org`.
    CrossTenantDenied { key_org: OrgId, target_org: OrgId },
    /// Valid API key for the target tenant holds role `actual`, which is below `required`.
    InsufficientRole {
        actual: TenantRole,
        required: TenantRole,
    },
    /// Tenant exhausted its token-bucket QPS/burst allocation.
    RateLimited { org_id: OrgId, retry_after_ms: u64 },
    /// Tenant exceeded its configured vector count or memory byte ceiling.
    StorageQuotaExceeded {
        org_id: OrgId,
        current_vectors: u64,
        max_vectors: u64,
        current_bytes: u64,
        max_bytes: u64,
    },
    /// Audit log hash chain verification failed at `corrupted_seq_no`.
    AuditIntegrityViolation { corrupted_seq_no: u64 },
}

impl fmt::Display for SecurityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unauthorized => write!(f, "unauthorized: invalid or revoked API key"),
            Self::CrossTenantDenied {
                key_org,
                target_org,
            } => write!(
                f,
                "cross-tenant access denied: key scoped to '{}' cannot access '{}'",
                key_org.as_str(),
                target_org.as_str()
            ),
            Self::InsufficientRole { actual, required } => write!(
                f,
                "insufficient RBAC role: key has '{actual}', operation requires '{required}'"
            ),
            Self::RateLimited {
                org_id,
                retry_after_ms,
            } => write!(
                f,
                "rate limit exceeded for tenant '{}': retry after {retry_after_ms}ms",
                org_id.as_str()
            ),
            Self::StorageQuotaExceeded {
                org_id,
                current_vectors,
                max_vectors,
                current_bytes,
                max_bytes,
            } => write!(
                f,
                "storage quota exceeded for tenant '{}': vectors {current_vectors}/{max_vectors}, bytes {current_bytes}/{max_bytes}",
                org_id.as_str()
            ),
            Self::AuditIntegrityViolation { corrupted_seq_no } => write!(
                f,
                "tamper-evident audit log integrity violation at seq_no {corrupted_seq_no}"
            ),
        }
    }
}

impl std::error::Error for SecurityError {}

/// Constant-time 256-bit (32-byte) equality check.
///
/// Accumulates bitwise XOR differences across all 32 bytes without short-circuit branching so
/// timing side-channels cannot leak secret MAC bytes.
#[must_use]
pub fn constant_time_eq_32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0_u8;
    let mut idx = 0;
    while idx < 32 {
        diff |= a[idx] ^ b[idx];
        idx += 1;
    }
    diff == 0
}

/// Compute a deterministic 256-bit (32-byte) keyed authentication digest over `(secret, parts)`
/// using four domain-separated 64-bit SplitMix64/CRC32 lanes.
#[must_use]
fn compute_keyed_digest_32(secret: &[u8; 32], parts: &[&[u8]]) -> [u8; 32] {
    let mut out = [0_u8; 32];
    for lane in 0..4_usize {
        let off = lane * 8;
        let seed_lane = u64::from_le_bytes([
            secret[off],
            secret[off + 1],
            secret[off + 2],
            secret[off + 3],
            secret[off + 4],
            secret[off + 5],
            secret[off + 6],
            secret[off + 7],
        ]);
        let mut state =
            seed_lane ^ (0x9e37_79b9_7f4a_7c15_u64.wrapping_mul((lane as u64).wrapping_add(1)));
        for part in parts {
            let mut crc_hasher = crc32fast::Hasher::new_with_initial(state as u32);
            crc_hasher.update(part);
            let crc_lo = u64::from(crc_hasher.finalize());

            let mut crc_hasher_hi = crc32fast::Hasher::new_with_initial((state >> 32) as u32);
            crc_hasher_hi.update(&[lane as u8, 0xA5]);
            crc_hasher_hi.update(part);
            let crc_hi = u64::from(crc_hasher_hi.finalize());

            let combined =
                (crc_hi << 32) | crc_lo ^ (part.len() as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
            state = splitmix64(state ^ combined);
        }
        let bytes = state.to_le_bytes();
        out[off..off + 8].copy_from_slice(&bytes);
    }
    out
}

#[must_use]
const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[must_use]
fn hex_encode_32(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(64);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

/// Server-side record of an issued tenant-scoped API key.
///
/// The raw secret token is returned only once at issuance; `ScopedApiKey` stores only the
/// 32-byte keyed digest (`secret_digest`) verified via [`constant_time_eq_32`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopedApiKey {
    pub key_id: String,
    pub org_id: OrgId,
    pub role: TenantRole,
    pub secret_digest: [u8; 32],
    pub revoked: bool,
}

/// Per-tenant rate limit and storage quota policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TenantQuotaPolicy {
    /// Continuous token refill rate in queries/operations per second.
    pub max_qps: f64,
    /// Maximum instantaneous burst capacity in tokens.
    pub burst_capacity: f64,
    /// Maximum live vectors permitted for this tenant across the cluster/node.
    pub max_vectors: u64,
    /// Maximum memory bytes permitted for this tenant's vectors and metadata.
    pub max_memory_bytes: u64,
}

impl Default for TenantQuotaPolicy {
    fn default() -> Self {
        Self {
            max_qps: 1_000.0,
            burst_capacity: 200.0,
            max_vectors: 1_000_000,
            max_memory_bytes: 4 * 1024 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone)]
struct TenantQuotaState {
    policy: TenantQuotaPolicy,
    tokens: f64,
    last_refill_ms: u64,
    live_vectors: u64,
    live_memory_bytes: u64,
}

impl TenantQuotaState {
    fn new(policy: TenantQuotaPolicy, now_ms: u64) -> Self {
        Self {
            policy,
            tokens: policy.burst_capacity.max(0.0),
            last_refill_ms: now_ms,
            live_vectors: 0,
            live_memory_bytes: 0,
        }
    }

    fn refill(&mut self, now_ms: u64) {
        if now_ms > self.last_refill_ms {
            let elapsed_secs = (now_ms - self.last_refill_ms) as f64 / 1000.0;
            let added = elapsed_secs * self.policy.max_qps.max(0.0);
            self.tokens = (self.tokens + added).min(self.policy.burst_capacity.max(0.0));
            self.last_refill_ms = now_ms;
        }
    }
}

/// Per-tenant token-bucket rate limiter and vector/memory resource quota tracker.
///
/// Isolates noisy-neighbor tenants by enforcing independent token buckets and storage ceilings
/// keyed by [`OrgId`].
#[derive(Debug, Clone, Default)]
pub struct TenantQuotaTracker {
    default_policy: TenantQuotaPolicy,
    tenants: HashMap<OrgId, TenantQuotaState>,
}

impl TenantQuotaTracker {
    /// Create a quota tracker with the given default [`TenantQuotaPolicy`] for tenants without an
    /// explicit override.
    #[must_use]
    pub fn new(default_policy: TenantQuotaPolicy) -> Self {
        Self {
            default_policy,
            tenants: HashMap::new(),
        }
    }

    /// Configure or update the [`TenantQuotaPolicy`] for `org_id` at timestamp `now_ms`.
    pub fn set_policy(&mut self, org_id: OrgId, policy: TenantQuotaPolicy, now_ms: u64) {
        match self.tenants.get_mut(&org_id) {
            Some(state) => {
                state.refill(now_ms);
                state.policy = policy;
                state.tokens = state.tokens.min(policy.burst_capacity.max(0.0));
            }
            None => {
                self.tenants
                    .insert(org_id, TenantQuotaState::new(policy, now_ms));
            }
        }
    }

    /// Attempt to consume `cost` tokens from `org_id`'s token bucket at `now_ms`.
    ///
    /// Returns [`SecurityError::RateLimited`] with an estimated `retry_after_ms` when insufficient
    /// tokens remain.
    pub fn acquire_request_token(
        &mut self,
        org_id: &OrgId,
        now_ms: u64,
        cost: f64,
    ) -> Result<(), SecurityError> {
        let effective_cost = cost.max(1.0);
        let default_policy = self.default_policy;
        let state = self
            .tenants
            .entry(org_id.clone())
            .or_insert_with(|| TenantQuotaState::new(default_policy, now_ms));

        state.refill(now_ms);

        if state.tokens + 1e-9 >= effective_cost {
            state.tokens = (state.tokens - effective_cost).max(0.0);
            Ok(())
        } else {
            let deficit = (effective_cost - state.tokens).max(0.0);
            let retry_after_ms = if state.policy.max_qps > 0.0 {
                ((deficit / state.policy.max_qps) * 1000.0).ceil() as u64
            } else {
                u64::MAX
            };
            Err(SecurityError::RateLimited {
                org_id: org_id.clone(),
                retry_after_ms: retry_after_ms.max(1),
            })
        }
    }

    /// Reserve `add_vectors` and `add_bytes` against `org_id`'s storage quota.
    ///
    /// Atomically checks both vector count and byte limits before mutating usage counters.
    pub fn reserve_storage(
        &mut self,
        org_id: &OrgId,
        add_vectors: u64,
        add_bytes: u64,
    ) -> Result<(), SecurityError> {
        let default_policy = self.default_policy;
        let state = self
            .tenants
            .entry(org_id.clone())
            .or_insert_with(|| TenantQuotaState::new(default_policy, 0));

        let next_vectors = state.live_vectors.saturating_add(add_vectors);
        let next_bytes = state.live_memory_bytes.saturating_add(add_bytes);

        if next_vectors > state.policy.max_vectors || next_bytes > state.policy.max_memory_bytes {
            return Err(SecurityError::StorageQuotaExceeded {
                org_id: org_id.clone(),
                current_vectors: next_vectors,
                max_vectors: state.policy.max_vectors,
                current_bytes: next_bytes,
                max_bytes: state.policy.max_memory_bytes,
            });
        }

        state.live_vectors = next_vectors;
        state.live_memory_bytes = next_bytes;
        Ok(())
    }

    /// Release `remove_vectors` and `remove_bytes` from `org_id`'s tracked storage usage.
    pub fn release_storage(&mut self, org_id: &OrgId, remove_vectors: u64, remove_bytes: u64) {
        if let Some(state) = self.tenants.get_mut(org_id) {
            state.live_vectors = state.live_vectors.saturating_sub(remove_vectors);
            state.live_memory_bytes = state.live_memory_bytes.saturating_sub(remove_bytes);
        }
    }

    /// Return `(live_vectors, live_memory_bytes)` currently attributed to `org_id`.
    #[must_use]
    pub fn usage(&self, org_id: &OrgId) -> (u64, u64) {
        self.tenants
            .get(org_id)
            .map_or((0, 0), |s| (s.live_vectors, s.live_memory_bytes))
    }

    /// Return the remaining token balance for `org_id` at `now_ms` without consuming tokens.
    #[must_use]
    pub fn available_tokens(&self, org_id: &OrgId, now_ms: u64) -> f64 {
        let Some(state) = self.tenants.get(org_id) else {
            return self.default_policy.burst_capacity.max(0.0);
        };
        if now_ms > state.last_refill_ms {
            let elapsed_secs = (now_ms - state.last_refill_ms) as f64 / 1000.0;
            let added = elapsed_secs * state.policy.max_qps.max(0.0);
            (state.tokens + added).min(state.policy.burst_capacity.max(0.0))
        } else {
            state.tokens
        }
    }
}

/// Outcome recorded in an [`AuditEntry`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditOutcome {
    /// Operation was authenticated, authorized, within quota, and permitted.
    Allowed,
    /// Operation was rejected with a structured security or quota reason.
    Denied { reason: String },
}

impl AuditOutcome {
    #[must_use]
    fn tag_byte(&self) -> u8 {
        match self {
            Self::Allowed => 1,
            Self::Denied { .. } => 2,
        }
    }

    #[must_use]
    fn reason_bytes(&self) -> &[u8] {
        match self {
            Self::Allowed => b"",
            Self::Denied { reason } => reason.as_bytes(),
        }
    }
}

/// Single tamper-evident record in an [`AuditLog`].
///
/// `entry_hash` chains `prev_hash` and every field of this entry (`timestamp_ms`, `seq_no`,
/// `actor_key_id`, `org_id`, `action`, `outcome`). Mutating any historical field invalidates
/// `entry_hash` and all subsequent entries in the chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditEntry {
    pub timestamp_ms: u64,
    pub seq_no: u64,
    pub actor_key_id: String,
    pub org_id: OrgId,
    pub action: String,
    pub outcome: AuditOutcome,
    pub prev_hash: u64,
    pub entry_hash: u64,
}

impl AuditEntry {
    /// Deterministically compute the 64-bit chained digest over `prev_hash` and all entry fields.
    #[must_use]
    pub fn compute_entry_hash(&self) -> u64 {
        let ts_bytes = self.timestamp_ms.to_le_bytes();
        let seq_bytes = self.seq_no.to_le_bytes();
        let prev_bytes = self.prev_hash.to_le_bytes();
        let outcome_tag = [self.outcome.tag_byte()];

        let parts: [&[u8]; 8] = [
            &prev_bytes,
            &seq_bytes,
            &ts_bytes,
            self.actor_key_id.as_bytes(),
            self.org_id.as_str().as_bytes(),
            self.action.as_bytes(),
            &outcome_tag,
            self.outcome.reason_bytes(),
        ];

        let mut acc = splitmix64(self.prev_hash ^ 0x4155_4454_4348_4E31); // "AUDTCHN1"
        for (idx, part) in parts.iter().enumerate() {
            let mut hasher_lo = crc32fast::Hasher::new_with_initial(acc as u32);
            hasher_lo.update(&[idx as u8]);
            hasher_lo.update(&(part.len() as u32).to_le_bytes());
            hasher_lo.update(part);
            let lo = u64::from(hasher_lo.finalize());

            let mut hasher_hi =
                crc32fast::Hasher::new_with_initial((acc >> 32) as u32 ^ 0x5A5A_5A5A);
            hasher_hi.update(part);
            hasher_hi.update(&[idx as u8 ^ 0xFF]);
            let hi = u64::from(hasher_hi.finalize());

            acc = splitmix64(acc ^ ((hi << 32) | lo));
        }
        acc
    }
}

/// Append-only tamper-evident hash-chained security audit log.
#[derive(Debug, Clone, Default)]
pub struct AuditLog {
    pub entries: Vec<AuditEntry>,
    head_hash: u64,
}

impl AuditLog {
    /// Construct an empty `AuditLog` anchored at the genesis hash (`0`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            head_hash: 0,
        }
    }

    /// Append a new security-relevant event to the hash chain and return a reference to the
    /// committed [`AuditEntry`].
    pub fn append(
        &mut self,
        timestamp_ms: u64,
        actor_key_id: impl Into<String>,
        org_id: OrgId,
        action: impl Into<String>,
        outcome: AuditOutcome,
    ) -> &AuditEntry {
        let seq_no = (self.entries.len() as u64).saturating_add(1);
        let prev_hash = self.head_hash;
        let mut entry = AuditEntry {
            timestamp_ms,
            seq_no,
            actor_key_id: actor_key_id.into(),
            org_id,
            action: action.into(),
            outcome,
            prev_hash,
            entry_hash: 0,
        };
        entry.entry_hash = entry.compute_entry_hash();
        self.head_hash = entry.entry_hash;
        self.entries.push(entry);
        &self.entries[self.entries.len() - 1]
    }

    /// Borrow the slice of all recorded [`AuditEntry`] records.
    #[must_use]
    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }

    /// Return the latest `entry_hash` (`0` if the log is empty).
    #[must_use]
    pub const fn head_hash(&self) -> u64 {
        self.head_hash
    }

    /// Recompute the entire hash chain from genesis (`prev_hash = 0`).
    ///
    /// Returns `Ok(verified_entry_count)` if every entry's `seq_no`, `prev_hash`, and `entry_hash`
    /// are intact, or `Err(corrupted_seq_no)` identifying the first tampered, reordered, or
    /// truncated entry.
    pub fn verify_integrity(&self) -> Result<usize, u64> {
        let mut expected_prev_hash = 0_u64;
        for (idx, entry) in self.entries.iter().enumerate() {
            let expected_seq = (idx as u64).saturating_add(1);
            if entry.seq_no != expected_seq {
                return Err(expected_seq);
            }
            if entry.prev_hash != expected_prev_hash {
                return Err(entry.seq_no);
            }
            let recomputed = entry.compute_entry_hash();
            if entry.entry_hash != recomputed {
                return Err(entry.seq_no);
            }
            expected_prev_hash = entry.entry_hash;
        }
        if self.head_hash != expected_prev_hash {
            return Err((self.entries.len() as u64).saturating_add(1));
        }
        Ok(self.entries.len())
    }
}

/// Unified security gateway combining scoped API-key authentication, RBAC authorization,
/// per-tenant token-bucket rate limiting and quotas, and tamper-evident audit logging.
#[derive(Debug, Clone)]
pub struct SecurityGate {
    master_secret: [u8; 32],
    keys: HashMap<String, ScopedApiKey>,
    pub quotas: TenantQuotaTracker,
    pub audit_log: AuditLog,
}

impl SecurityGate {
    /// Create a new `SecurityGate` with the given 32-byte master secret and default quota policy.
    #[must_use]
    pub fn new(master_secret: [u8; 32], default_quota: TenantQuotaPolicy) -> Self {
        Self {
            master_secret,
            keys: HashMap::new(),
            quotas: TenantQuotaTracker::new(default_quota),
            audit_log: AuditLog::new(),
        }
    }

    /// Issue a new tenant-scoped API key bound to `org_id` and `role`.
    ///
    /// Returns `(raw_token, scoped_api_key)` where `raw_token` has the canonical format
    /// `vdk.<key_id>.<hex_tag>` and `scoped_api_key` stores only the 32-byte keyed digest of
    /// `raw_token`.
    pub fn issue_api_key(
        &mut self,
        key_id: impl Into<String>,
        org_id: OrgId,
        role: TenantRole,
        nonce: u64,
    ) -> (String, ScopedApiKey) {
        let key_id_str = key_id.into();
        let nonce_bytes = nonce.to_le_bytes();
        let tag_bytes = compute_keyed_digest_32(
            &self.master_secret,
            &[
                b"vdk-issue-v1",
                key_id_str.as_bytes(),
                org_id.as_str().as_bytes(),
                role.as_str().as_bytes(),
                &nonce_bytes,
            ],
        );
        let raw_token = format!("vdk.{}.{}", key_id_str, hex_encode_32(&tag_bytes));
        let secret_digest = compute_keyed_digest_32(
            &self.master_secret,
            &[b"vdk-verify-v1", raw_token.as_bytes()],
        );

        let record = ScopedApiKey {
            key_id: key_id_str.clone(),
            org_id,
            role,
            secret_digest,
            revoked: false,
        };
        self.keys.insert(key_id_str, record.clone());
        (raw_token, record)
    }

    /// Revoke an existing API key by `key_id`. Returns `true` if the key existed.
    pub fn revoke_api_key(&mut self, key_id: &str) -> bool {
        if let Some(entry) = self.keys.get_mut(key_id) {
            entry.revoked = true;
            true
        } else {
            false
        }
    }

    /// Authenticate `raw_key` in constant time without checking target tenant or role.
    pub fn authenticate(&self, raw_key: &str) -> Result<&ScopedApiKey, SecurityError> {
        let mut parts = raw_key.splitn(3, '.');
        let Some(prefix) = parts.next() else {
            return Err(SecurityError::Unauthorized);
        };
        let Some(key_id) = parts.next() else {
            return Err(SecurityError::Unauthorized);
        };
        let Some(tag_hex) = parts.next() else {
            return Err(SecurityError::Unauthorized);
        };
        if prefix != "vdk" || key_id.is_empty() || tag_hex.len() != 64 {
            return Err(SecurityError::Unauthorized);
        }

        let candidate_digest =
            compute_keyed_digest_32(&self.master_secret, &[b"vdk-verify-v1", raw_key.as_bytes()]);

        let Some(stored) = self.keys.get(key_id) else {
            // Execute a dummy constant-time comparison against the master secret so unknown key_ids
            // perform identical digest comparison work.
            let _ = constant_time_eq_32(&candidate_digest, &self.master_secret);
            return Err(SecurityError::Unauthorized);
        };

        if !constant_time_eq_32(&candidate_digest, &stored.secret_digest) || stored.revoked {
            return Err(SecurityError::Unauthorized);
        }

        Ok(stored)
    }

    /// Authenticate `raw_key`, verify that its bound `org_id` matches `target_org`, and verify
    /// that its `role` satisfies `required_role`.
    pub fn authorize(
        &self,
        raw_key: &str,
        target_org: &OrgId,
        required_role: TenantRole,
    ) -> Result<&ScopedApiKey, SecurityError> {
        let key = self.authenticate(raw_key)?;
        if &key.org_id != target_org {
            return Err(SecurityError::CrossTenantDenied {
                key_org: key.org_id.clone(),
                target_org: target_org.clone(),
            });
        }
        if !key.role.allows(required_role) {
            return Err(SecurityError::InsufficientRole {
                actual: key.role,
                required: required_role,
            });
        }
        Ok(key)
    }

    /// Authorize an operation, enforce the tenant's token-bucket QPS limit, and append a
    /// tamper-evident [`AuditEntry`] recording either `Allowed` or `Denied`.
    pub fn authorize_request(
        &mut self,
        timestamp_ms: u64,
        raw_key: &str,
        target_org: &OrgId,
        required_role: TenantRole,
        action: &str,
    ) -> Result<ScopedApiKey, SecurityError> {
        let auth_res = self.authorize(raw_key, target_org, required_role).cloned();
        match auth_res {
            Ok(scoped_key) => {
                if let Err(rate_err) =
                    self.quotas
                        .acquire_request_token(target_org, timestamp_ms, 1.0)
                {
                    self.audit_log.append(
                        timestamp_ms,
                        scoped_key.key_id.clone(),
                        target_org.clone(),
                        action,
                        AuditOutcome::Denied {
                            reason: rate_err.to_string(),
                        },
                    );
                    return Err(rate_err);
                }
                self.audit_log.append(
                    timestamp_ms,
                    scoped_key.key_id.clone(),
                    target_org.clone(),
                    action,
                    AuditOutcome::Allowed,
                );
                Ok(scoped_key)
            }
            Err(err) => {
                let actor_id = extract_key_id_hint(raw_key);
                self.audit_log.append(
                    timestamp_ms,
                    actor_id,
                    target_org.clone(),
                    action,
                    AuditOutcome::Denied {
                        reason: err.to_string(),
                    },
                );
                Err(err)
            }
        }
    }
}

#[must_use]
fn extract_key_id_hint(raw_key: &str) -> String {
    let mut parts = raw_key.splitn(3, '.');
    if parts.next() == Some("vdk") {
        if let Some(kid) = parts.next() {
            if !kid.is_empty() {
                return kid.to_owned();
            }
        }
    }
    "anonymous".to_owned()
}
