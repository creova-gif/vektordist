//! Milestone 8 integration tests: Scoped API-Key Authentication, RBAC Authorization,
//! Per-Tenant Token-Bucket Rate Limiting & Quotas, and Tamper-Evident Hash-Chained Audit Log.

use vektordist_cluster::security::{
    constant_time_eq_32, AuditLog, AuditOutcome, OrgId, SecurityError, SecurityGate,
    TenantQuotaPolicy, TenantQuotaTracker, TenantRole,
};

#[test]
fn rbac_enforces_reader_writer_admin_and_rejects_cross_tenant_and_invalid_keys() {
    let master_secret = [0x42_u8; 32];
    let mut gate = SecurityGate::new(master_secret, TenantQuotaPolicy::default());

    let org_a = OrgId::new("org-a").unwrap();
    let org_b = OrgId::new("org-b").unwrap();

    let (reader_key_a, _) =
        gate.issue_api_key("key-reader-a", org_a.clone(), TenantRole::Reader, 1);
    let (writer_key_a, _) =
        gate.issue_api_key("key-writer-a", org_a.clone(), TenantRole::Writer, 2);
    let (admin_key_a, _) = gate.issue_api_key("key-admin-a", org_a.clone(), TenantRole::Admin, 3);
    let (writer_key_b, _) =
        gate.issue_api_key("key-writer-b", org_b.clone(), TenantRole::Writer, 4);

    // 1. Reader can perform Reader operations on org-a, but is denied Writer and Admin operations.
    assert!(gate
        .authorize_request(
            1_000,
            &reader_key_a,
            &org_a,
            TenantRole::Reader,
            "hybrid_search"
        )
        .is_ok());
    assert_eq!(
        gate.authorize_request(
            1_001,
            &reader_key_a,
            &org_a,
            TenantRole::Writer,
            "upsert_vector"
        ),
        Err(SecurityError::InsufficientRole {
            actual: TenantRole::Reader,
            required: TenantRole::Writer,
        })
    );
    assert_eq!(
        gate.authorize_request(
            1_002,
            &reader_key_a,
            &org_a,
            TenantRole::Admin,
            "rotate_keys"
        ),
        Err(SecurityError::InsufficientRole {
            actual: TenantRole::Reader,
            required: TenantRole::Admin,
        })
    );

    // 2. Writer can perform Reader and Writer operations on org-a, but is denied Admin operations.
    assert!(gate
        .authorize_request(
            1_003,
            &writer_key_a,
            &org_a,
            TenantRole::Reader,
            "hybrid_search"
        )
        .is_ok());
    assert!(gate
        .authorize_request(
            1_004,
            &writer_key_a,
            &org_a,
            TenantRole::Writer,
            "upsert_vector"
        )
        .is_ok());
    assert_eq!(
        gate.authorize_request(
            1_005,
            &writer_key_a,
            &org_a,
            TenantRole::Admin,
            "rotate_keys"
        ),
        Err(SecurityError::InsufficientRole {
            actual: TenantRole::Writer,
            required: TenantRole::Admin,
        })
    );

    // 3. Admin can perform Reader, Writer, and Admin operations on org-a.
    for required in [TenantRole::Reader, TenantRole::Writer, TenantRole::Admin] {
        assert!(gate
            .authorize_request(1_006, &admin_key_a, &org_a, required, "admin_op")
            .is_ok());
    }

    // 4. Cross-tenant isolation: org-a Admin/Writer/Reader keys cannot access org-b, and vice versa.
    assert_eq!(
        gate.authorize_request(
            1_010,
            &admin_key_a,
            &org_b,
            TenantRole::Reader,
            "cross_tenant_read"
        ),
        Err(SecurityError::CrossTenantDenied {
            key_org: org_a.clone(),
            target_org: org_b.clone(),
        })
    );
    assert_eq!(
        gate.authorize_request(
            1_011,
            &writer_key_b,
            &org_a,
            TenantRole::Writer,
            "cross_tenant_write"
        ),
        Err(SecurityError::CrossTenantDenied {
            key_org: org_b.clone(),
            target_org: org_a.clone(),
        })
    );

    // 5. Forged, malformed, or 1-nibble tampered API keys are rejected with Unauthorized.
    let mut tampered = writer_key_a.clone();
    let last_ch = if tampered.ends_with('0') { '1' } else { '0' };
    tampered.pop();
    tampered.push(last_ch);
    assert_eq!(
        gate.authorize_request(1_012, &tampered, &org_a, TenantRole::Reader, "tampered_key"),
        Err(SecurityError::Unauthorized)
    );
    assert_eq!(
        gate.authorize_request(
            1_013,
            "invalid-bearer-token",
            &org_a,
            TenantRole::Reader,
            "malformed_key"
        ),
        Err(SecurityError::Unauthorized)
    );

    // 6. Revoked key is immediately rejected even if its digest matches.
    assert!(gate.revoke_api_key("key-writer-a"));
    assert_eq!(
        gate.authorize_request(
            1_014,
            &writer_key_a,
            &org_a,
            TenantRole::Reader,
            "revoked_use"
        ),
        Err(SecurityError::Unauthorized)
    );

    // 7. Constant-time 32-byte digest comparator sanity check.
    let d1 = [0xAA_u8; 32];
    let mut d2 = [0xAA_u8; 32];
    assert!(constant_time_eq_32(&d1, &d2));
    d2[31] ^= 0x01;
    assert!(!constant_time_eq_32(&d1, &d2));
}

#[test]
fn token_bucket_rate_limiter_and_storage_quotas_isolate_noisy_neighbor() {
    let policy = TenantQuotaPolicy {
        max_qps: 10.0,       // 1 token every 100ms
        burst_capacity: 3.0, // max burst of 3 requests
        max_vectors: 5,
        max_memory_bytes: 1_024,
    };
    let mut tracker = TenantQuotaTracker::new(policy);

    let noisy_org = OrgId::new("org-noisy").unwrap();
    let quiet_org = OrgId::new("org-quiet").unwrap();

    // At t = 1_000ms, noisy_org consumes its full burst of 3 tokens.
    for _ in 0..3 {
        assert_eq!(
            tracker.acquire_request_token(&noisy_org, 1_000, 1.0),
            Ok(())
        );
    }
    // 4th immediate request for noisy_org is rate-limited (100ms until next token).
    assert_eq!(
        tracker.acquire_request_token(&noisy_org, 1_000, 1.0),
        Err(SecurityError::RateLimited {
            org_id: noisy_org.clone(),
            retry_after_ms: 100,
        })
    );

    // Meanwhile, quiet_org is completely unaffected at t = 1_000ms!
    for _ in 0..3 {
        assert_eq!(
            tracker.acquire_request_token(&quiet_org, 1_000, 1.0),
            Ok(())
        );
    }

    // Advance clock by 50ms (t = 1_050ms): 0.5 tokens refilled for noisy_org, still < 1.0.
    assert_eq!(
        tracker.acquire_request_token(&noisy_org, 1_050, 1.0),
        Err(SecurityError::RateLimited {
            org_id: noisy_org.clone(),
            retry_after_ms: 50,
        })
    );

    // Advance clock to t = 1_100ms (100ms elapsed total): 1.0 token refilled -> succeeds.
    assert_eq!(
        tracker.acquire_request_token(&noisy_org, 1_100, 1.0),
        Ok(())
    );

    // Storage quotas: noisy_org reserves 5 vectors (800 bytes) -> within limits (5 vectors, 1024 bytes).
    assert_eq!(tracker.reserve_storage(&noisy_org, 5, 800), Ok(()));
    assert_eq!(tracker.usage(&noisy_org), (5, 800));

    // Attempting to add 1 more vector for noisy_org fails with StorageQuotaExceeded.
    assert_eq!(
        tracker.reserve_storage(&noisy_org, 1, 64),
        Err(SecurityError::StorageQuotaExceeded {
            org_id: noisy_org.clone(),
            current_vectors: 6,
            max_vectors: 5,
            current_bytes: 864,
            max_bytes: 1_024,
        })
    );
    // Usage remains unchanged after failed reservation.
    assert_eq!(tracker.usage(&noisy_org), (5, 800));

    // Releasing 1 vector (200 bytes) allows a 1-vector (200-byte) write, while 1-vector (500-byte)
    // write is rejected by the byte ceiling.
    tracker.release_storage(&noisy_org, 1, 200);
    assert_eq!(tracker.usage(&noisy_org), (4, 600));
    assert_eq!(
        tracker.reserve_storage(&noisy_org, 1, 500),
        Err(SecurityError::StorageQuotaExceeded {
            org_id: noisy_org.clone(),
            current_vectors: 5,
            max_vectors: 5,
            current_bytes: 1_100,
            max_bytes: 1_024,
        })
    );
    assert_eq!(tracker.reserve_storage(&noisy_org, 1, 200), Ok(()));

    // quiet_org can still reserve its own storage independently.
    assert_eq!(tracker.reserve_storage(&quiet_org, 4, 512), Ok(()));
    assert_eq!(tracker.usage(&quiet_org), (4, 512));
}

#[test]
fn tamper_evident_audit_log_verifies_valid_chain_and_detects_any_historical_mutation() {
    let mut log = AuditLog::new();
    let org_a = OrgId::new("org-a").unwrap();
    let org_b = OrgId::new("org-b").unwrap();

    log.append(
        10_000,
        "key-reader-a",
        org_a.clone(),
        "hybrid_search_rrf",
        AuditOutcome::Allowed,
    );
    log.append(
        10_050,
        "key-writer-a",
        org_a.clone(),
        "upsert_vector",
        AuditOutcome::Allowed,
    );
    log.append(
        10_100,
        "key-reader-a",
        org_b.clone(),
        "delete_vector",
        AuditOutcome::Denied {
            reason: "cross-tenant access denied".to_owned(),
        },
    );
    log.append(
        10_150,
        "key-admin-a",
        org_a.clone(),
        "verify_snapshot",
        AuditOutcome::Allowed,
    );

    // Intact 4-entry chain passes verification and links each prev_hash to the prior entry_hash.
    assert_eq!(log.verify_integrity(), Ok(4));
    assert_eq!(log.entries()[0].prev_hash, 0);
    for i in 1..log.entries().len() {
        assert_eq!(log.entries()[i].prev_hash, log.entries()[i - 1].entry_hash);
    }

    // Tamper case 1: Mutating the `action` field of historical entry #2 (`seq_no = 2`) is caught.
    let mut tampered_action = log.clone();
    tampered_action.entries[1].action = "drop_index".to_owned();
    assert_eq!(tampered_action.verify_integrity(), Err(2));

    // Tamper case 2: Mutating the `outcome` of historical entry #3 (`seq_no = 3`) from Denied to
    // Allowed is caught.
    let mut tampered_outcome = log.clone();
    tampered_outcome.entries[2].outcome = AuditOutcome::Allowed;
    assert_eq!(tampered_outcome.verify_integrity(), Err(3));

    // Tamper case 3: Even if an attacker recomputes entry #2's `entry_hash` after mutating it,
    // entry #3's `prev_hash` link breaks and `verify_integrity` catches the tamper at `seq_no = 3`.
    let mut tampered_rehashed = log.clone();
    tampered_rehashed.entries[1].actor_key_id = "forged-actor".to_owned();
    tampered_rehashed.entries[1].entry_hash = tampered_rehashed.entries[1].compute_entry_hash();
    assert_eq!(tampered_rehashed.verify_integrity(), Err(3));

    // Tamper case 4: Deleting entry #2 from the middle of the log breaks both seq_no and prev_hash.
    let mut tampered_deleted = log;
    tampered_deleted.entries.remove(1);
    assert_eq!(tampered_deleted.verify_integrity(), Err(2));
}
