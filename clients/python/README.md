# VektorDist Python Client SDK

Zero-dependency, strictly typed (`PEP 561` / `mypy --strict`) Python client for the **VektorDist** distributed sharded HNSW HTTP/1.1 REST gateway.

Built exclusively on the Python 3.11+ / 3.12+ standard library (`urllib.request`, `json`, `dataclasses`, `os`, `time`, `typing`) so AI, RAG, and agentic Python services can import `VektorDistClient` directly without C-extension compilation or third-party transitive dependencies.

## Features

- **Zero Runtime Dependencies:** Drop-in standard-library client with `py.typed` strict static typing.
- **Zero-Trust Auth & Multi-Tenancy:** Automatic `Authorization: Bearer vdk.<key_id>.<tag>` and `X-Org-Id` header propagation.
- **W3C Distributed Tracing:** Generates compliant 55-character W3C Trace Context v00 `traceparent` (`00-<32hex>-<16hex>-01`) spans on every request, or forwards caller-supplied parent spans.
- **Bounded Resilience:** Automatic bounded exponential backoff retries on `429 Too Many Requests` (honoring `Retry-After`) and `503 Service Unavailable`.
- **Multi-Collection & Sharded Cluster Coverage:** Full support for `/v1/upsert`, `/v1/upsert_batch`, `/v1/search`, `/v1/delete`, `/v1/topology`, `/v1/audit`, `/healthz`, `/readyz`, and `/v1/collections/*` (`f32`, `fp16`, `sq8`, `bq`).

## Quickstart

```python
from clients.python import VektorDistClient

client = VektorDistClient(
    base_url="http://127.0.0.1:8080",
    bearer_token="vdk.key-01.0123456789abcdef",
    org_id="acme-prod",
    timeout_secs=5.0,
    max_retries=2,
)

# 1. Check cluster liveness & readiness
print(client.healthz())
print(client.readyz())

# 2. Quorum-durable single & batch upserts
ack = client.upsert(id=1001, vector=[0.12, -0.34, 0.56, 0.78])
print(f"Committed to shard={ack.shard_id} seq_no={ack.seq_no} trace={ack.trace_id}")

upserted = client.upsert_batch([
    (1002, [0.21, -0.43, 0.65, 0.87]),
    (1003, [0.11, -0.22, 0.33, 0.44]),
])

# 3. Scatter-gather top-k search
hits = client.search(query=[0.12, -0.34, 0.56, 0.78], top_k=5, ef_search=64)
for hit in hits:
    print(f"id={hit.id} score={hit.score:.4f}")

# 4. Named multi-collection lifecycle (f32 / fp16 / sq8 / bq)
client.create_collection("kb_chunks", dimensions=4, metric="cosine", quantization="fp16")
client.collection_upsert("kb_chunks", id=1, vector=[0.12, -0.34, 0.56, 0.78])
col_hits = client.collection_search("kb_chunks", query=[0.12, -0.34, 0.56, 0.78], top_k=3)
```

## Typed Error Hierarchy

| Exception | HTTP Status | Trigger |
| --- | --- | --- |
| `VektorDistError` | Any / `0` | Base exception (`status_code`, `message`, `trace_id`) |
| `AuthenticationError` | `401` | Missing, malformed, or revoked `vdk.*` bearer token |
| `AuthorizationError` | `403` | Cross-tenant `X-Org-Id` mismatch or insufficient RBAC role |
| `PayloadTooLargeError` | `413` | Request body exceeds gateway `max_body_bytes` cap |
| `RateLimitError` | `429` | Per-tenant QPS quota exceeded after `max_retries` (`retry_after_secs`) |

## Verification

From the repository root:

```bash
mypy --strict clients/python
python3 -m unittest discover -s clients/python -p "test_*.py" -v
```
