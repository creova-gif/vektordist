# Threat Model
- **Assets:** API credentials, tenant data (`org_id`), execution logs, downstream tools.
- **Trust Boundaries:** Client -> Gateway -> Sandbox -> Secret Manager / Storage.
- **Mitigations:** Strict schema validation, `org_id` tenant scoping, GCP Secret Manager + OIDC, `gitleaks` in CI.
