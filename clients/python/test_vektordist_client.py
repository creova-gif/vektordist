"""Deterministic unit & integration tests for the zero-dependency VektorDist Python SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import threading
from typing import Any
import unittest

from clients.python.vektordist_client import (
    AuthenticationError,
    AuthorizationError,
    PayloadTooLargeError,
    RateLimitError,
    ScoredHit,
    UpsertAck,
    VektorDistClient,
    VektorDistError,
)

TRACEPARENT_RE = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-0[01]$")


@dataclass
class RecordedRequest:
    """Captured inbound HTTP request observed by the ephemeral test server."""

    method: str
    path: str
    headers: dict[str, str]
    body: dict[str, Any] | None


@dataclass
class MockGatewayState:
    """Thread-safe programmable state backing the ephemeral test HTTP gateway."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    requests: list[RecordedRequest] = field(default_factory=list)
    rate_limit_remaining: int = 0
    retry_after_header: str = "0"
    forced_status: int | None = None
    forced_error: str = "forced gateway error"
    draining: bool = False
    collections: dict[str, dict[str, Any]] = field(default_factory=dict)


class _GatewayTestServer(ThreadingHTTPServer):
    state: MockGatewayState

    def __init__(self, server_address: tuple[str, int], state: MockGatewayState) -> None:
        super().__init__(server_address, _GatewayRequestHandler)
        self.state = state


class _GatewayRequestHandler(BaseHTTPRequestHandler):
    server: _GatewayTestServer

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        content_len = int(self.headers.get("Content-Length", "0"))
        raw_bytes = self.rfile.read(content_len) if content_len > 0 else b""
        parsed_body: dict[str, Any] | None = None
        if raw_bytes:
            loaded: object = json.loads(raw_bytes.decode("utf-8"))
            if isinstance(loaded, dict):
                parsed_body = {str(k): v for k, v in loaded.items()}

        lower_headers = {k.lower(): str(v) for k, v in self.headers.items()}
        inbound_tp = lower_headers.get(
            "traceparent",
            "00-11111111111111111111111111111111-2222222222222222-01",
        )
        trace_id = inbound_tp.split("-")[1] if "-" in inbound_tp else "0" * 32

        state = self.server.state
        with state.lock:
            state.requests.append(
                RecordedRequest(
                    method=method,
                    path=self.path,
                    headers=lower_headers,
                    body=parsed_body,
                )
            )
            if state.forced_status is not None:
                status = state.forced_status
                err_msg = state.forced_error
                self._write_json(
                    status,
                    {"status": "error", "error": err_msg, "trace_id": trace_id},
                    trace_id=trace_id,
                    traceparent=inbound_tp,
                )
                return

            if state.rate_limit_remaining > 0:
                state.rate_limit_remaining -= 1
                self._write_json(
                    429,
                    {
                        "status": "error",
                        "error": "token bucket rate limit exceeded",
                        "retry_after_ms": 0,
                        "trace_id": trace_id,
                    },
                    trace_id=trace_id,
                    traceparent=inbound_tp,
                    extra_headers={"Retry-After": state.retry_after_header},
                )
                return

        if (method, self.path) == ("GET", "/healthz"):
            self._write_json(
                200,
                {
                    "status": "ok",
                    "active_nodes": 3,
                    "configured_nodes": 3,
                    "acked_writes_lost": 0,
                },
                trace_id=trace_id,
                traceparent=inbound_tp,
            )
            return

        if (method, self.path) == ("GET", "/readyz"):
            if state.draining:
                self._write_json(
                    503,
                    {
                        "status": "not_ready",
                        "draining": True,
                        "active_nodes": 3,
                        "under_quorum_shards": 0,
                    },
                    trace_id=trace_id,
                    traceparent=inbound_tp,
                )
            else:
                self._write_json(
                    200,
                    {
                        "status": "ready",
                        "draining": False,
                        "active_nodes": 3,
                        "under_quorum_shards": 0,
                    },
                    trace_id=trace_id,
                    traceparent=inbound_tp,
                )
            return

        if (method, self.path) == ("POST", "/v1/upsert"):
            self._write_json(
                200,
                {"status": "ok", "shard_id": 5, "seq_no": 12, "trace_id": trace_id},
                trace_id=trace_id,
                traceparent=inbound_tp,
            )
            return

        if (method, self.path) == ("POST", "/v1/upsert_batch"):
            recs = parsed_body.get("records", []) if parsed_body else []
            count = len(recs) if isinstance(recs, list) else 0
            self._write_json(
                200,
                {"status": "ok", "upserted": count, "trace_id": trace_id},
                trace_id=trace_id,
                traceparent=inbound_tp,
            )
            return

        if (method, self.path) == ("POST", "/v1/search"):
            self._write_json(
                200,
                {
                    "status": "ok",
                    "hits": [{"id": 101, "score": 0.9925}, {"id": 202, "score": 0.875}],
                    "trace_id": trace_id,
                },
                trace_id=trace_id,
                traceparent=inbound_tp,
            )
            return

        if (method, self.path) == ("POST", "/v1/delete"):
            self._write_json(
                200,
                {"status": "ok", "shard_id": 5, "seq_no": 13, "trace_id": trace_id},
                trace_id=trace_id,
                traceparent=inbound_tp,
            )
            return

        if (method, self.path) == ("GET", "/v1/topology"):
            self._write_json(
                200,
                {
                    "status": "ok",
                    "epoch": 4,
                    "num_shards": 16,
                    "replication_factor": 3,
                    "write_quorum": 2,
                    "dimensions": 768,
                    "configured_nodes": 3,
                    "active_nodes": 3,
                },
                trace_id=trace_id,
                traceparent=inbound_tp,
            )
            return

        if (method, self.path) == ("GET", "/v1/audit"):
            self._write_json(
                200,
                {
                    "status": "ok",
                    "security_enabled": True,
                    "entries_count": 7,
                    "head_hash": 987654321,
                    "chain_valid": True,
                },
                trace_id=trace_id,
                traceparent=inbound_tp,
            )
            return

        if (method, self.path) == ("POST", "/v1/collections"):
            body_dict = parsed_body or {}
            col_name = str(body_dict.get("name", ""))
            col_meta: dict[str, Any] = {
                "name": col_name,
                "dimensions": int(body_dict.get("dimensions", 4)),
                "metric": str(body_dict.get("metric", "cosine")),
                "quantization": str(body_dict.get("quantization", "f32")),
            }
            with state.lock:
                state.collections[col_name] = col_meta
            self._write_json(
                201,
                {"status": "ok", "collection": col_meta, "trace_id": trace_id},
                trace_id=trace_id,
                traceparent=inbound_tp,
            )
            return

        if (method, self.path) == ("GET", "/v1/collections"):
            with state.lock:
                cols = list(state.collections.values())
            self._write_json(
                200,
                {"status": "ok", "collections": cols, "trace_id": trace_id},
                trace_id=trace_id,
                traceparent=inbound_tp,
            )
            return

        if self.path.startswith("/v1/collections/"):
            suffix = self.path[len("/v1/collections/") :]
            if method == "DELETE" and "/" not in suffix:
                with state.lock:
                    state.collections.pop(suffix, None)
                self._write_json(
                    200,
                    {"status": "ok", "dropped": suffix, "trace_id": trace_id},
                    trace_id=trace_id,
                    traceparent=inbound_tp,
                )
                return
            if method == "POST" and suffix.endswith("/upsert"):
                col = suffix[: -len("/upsert")]
                self._write_json(
                    200,
                    {"status": "ok", "collection": col, "upserted": 1, "trace_id": trace_id},
                    trace_id=trace_id,
                    traceparent=inbound_tp,
                )
                return
            if method == "POST" and suffix.endswith("/search"):
                self._write_json(
                    200,
                    {
                        "status": "ok",
                        "hits": [{"id": 7, "score": 0.981}],
                        "trace_id": trace_id,
                    },
                    trace_id=trace_id,
                    traceparent=inbound_tp,
                )
                return
            if method == "POST" and suffix.endswith("/delete"):
                col = suffix[: -len("/delete")]
                self._write_json(
                    200,
                    {"status": "ok", "collection": col, "deleted": True, "trace_id": trace_id},
                    trace_id=trace_id,
                    traceparent=inbound_tp,
                )
                return

        self._write_json(
            404,
            {"status": "error", "error": "route not found", "trace_id": trace_id},
            trace_id=trace_id,
            traceparent=inbound_tp,
        )

    def _write_json(
        self,
        status_code: int,
        payload: dict[str, Any],
        *,
        trace_id: str,
        traceparent: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("x-trace-id", trace_id)
        self.send_header("traceparent", traceparent)
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(encoded)


class VektorDistClientTest(unittest.TestCase):
    """End-to-end verification of `VektorDistClient` over a live ephemeral HTTP server."""

    state: MockGatewayState
    httpd: _GatewayTestServer
    server_thread: threading.Thread
    base_url: str

    def setUp(self) -> None:
        self.state = MockGatewayState()
        self.httpd = _GatewayTestServer(("127.0.0.1", 0), self.state)
        host, port = self.httpd.server_address[0], self.httpd.server_address[1]
        self.base_url = f"http://{str(host)}:{port}"
        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.server_thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.server_thread.join(timeout=2.0)

    def _assert_request_headers(self, req: RecordedRequest, expected_token: str, expected_org: str) -> None:
        self.assertEqual(req.headers.get("authorization"), f"Bearer {expected_token}")
        self.assertEqual(req.headers.get("x-org-id"), expected_org)
        tp = req.headers.get("traceparent", "")
        self.assertEqual(len(tp), 55)
        self.assertRegex(tp, TRACEPARENT_RE)

    def test_header_propagation_across_core_and_collection_endpoints(self) -> None:
        token = "vdk.key-01.deadbeefcafebabe"
        org = "acme-rag"
        client = VektorDistClient(
            base_url=self.base_url,
            bearer_token=token,
            org_id=org,
            timeout_secs=2.0,
        )

        health = client.healthz()
        self.assertEqual(health["status"], "ok")

        ready = client.readyz()
        self.assertEqual(ready["status"], "ready")

        ack = client.upsert(101, [0.25, 0.5, 0.75, 1.0])
        self.assertIsInstance(ack, UpsertAck)
        self.assertEqual(ack.shard_id, 5)
        self.assertEqual(ack.seq_no, 12)
        self.assertEqual(len(ack.trace_id), 32)

        batch_count = client.upsert_batch(
            [
                (102, [0.1, 0.2, 0.3, 0.4]),
                (103, 9003, [0.5, 0.6, 0.7, 0.8]),
            ]
        )
        self.assertEqual(batch_count, 2)

        hits = client.search([0.25, 0.5, 0.75, 1.0], top_k=2, ef_search=64)
        self.assertEqual(
            hits,
            [ScoredHit(id=101, score=0.9925), ScoredHit(id=202, score=0.875)],
        )

        del_ack = client.delete(101)
        self.assertEqual(del_ack.seq_no, 13)

        topo = client.topology()
        self.assertEqual(topo["num_shards"], 16)

        audit = client.audit()
        self.assertTrue(audit["chain_valid"])

        created = client.create_collection(
            "docs_fp16",
            dimensions=4,
            metric="cosine",
            quantization="fp16",
        )
        self.assertEqual(created["status"], "ok")

        listed = client.list_collections()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["name"], "docs_fp16")

        col_up = client.collection_upsert("docs_fp16", 7, [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(col_up["upserted"], 1)

        col_hits = client.collection_search("docs_fp16", [0.1, 0.2, 0.3, 0.4], top_k=1)
        self.assertEqual(col_hits, [ScoredHit(id=7, score=0.981)])

        col_del = client.collection_delete("docs_fp16", 7)
        self.assertTrue(col_del["deleted"])

        dropped = client.drop_collection("docs_fp16")
        self.assertEqual(dropped["dropped"], "docs_fp16")

        with self.state.lock:
            recorded = list(self.state.requests)

        self.assertEqual(len(recorded), 14)
        for req in recorded:
            self._assert_request_headers(req, expected_token=token, expected_org=org)

        # Verify caller-supplied valid traceparent is preserved verbatim when passed.
        explicit_tp = "00-abcdef0123456789abcdef0123456789-0123456789abcdef-01"
        explicit_ack = client.upsert(500, [1.0, 0.0, 0.0, 0.0], traceparent=explicit_tp)
        self.assertEqual(explicit_ack.trace_id, "abcdef0123456789abcdef0123456789")
        with self.state.lock:
            last_req = self.state.requests[-1]
        self.assertEqual(last_req.headers.get("traceparent"), explicit_tp)

    def test_retry_on_429_succeeds_on_second_attempt_and_raises_when_exhausted(self) -> None:
        client = VektorDistClient(
            base_url=self.base_url,
            bearer_token="vdk.key-01.retry",
            max_retries=2,
            base_backoff_secs=0.01,
        )

        # Single 429 (`Retry-After: 0`) succeeds cleanly on attempt 2.
        with self.state.lock:
            self.state.rate_limit_remaining = 1
            self.state.retry_after_header = "0"

        hits = client.search([1.0, 0.0, 0.0, 0.0], top_k=2)
        self.assertEqual(len(hits), 2)
        with self.state.lock:
            self.assertEqual(len(self.state.requests), 2)

        # Persistent 429 exceeding `max_retries=2` (3 total attempts) raises `RateLimitError`.
        with self.state.lock:
            self.state.requests.clear()
            self.state.rate_limit_remaining = 5
            self.state.retry_after_header = "0"

        with self.assertRaises(RateLimitError) as ctx:
            client.upsert(99, [1.0, 0.0, 0.0, 0.0])

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(ctx.exception.retry_after_secs, 0.0)
        self.assertIsNotNone(ctx.exception.trace_id)
        with self.state.lock:
            self.assertEqual(len(self.state.requests), 3)

    def test_typed_security_and_payload_exceptions(self) -> None:
        client = VektorDistClient(
            base_url=self.base_url,
            bearer_token="vdk.bad.token",
            max_retries=2,
            base_backoff_secs=0.01,
        )

        for status_code, exc_cls, err_text in [
            (401, AuthenticationError, "invalid or revoked API token"),
            (403, AuthorizationError, "insufficient RBAC role: requires Admin"),
            (413, PayloadTooLargeError, "payload exceeds max_body_bytes"),
        ]:
            with self.subTest(status_code=status_code):
                with self.state.lock:
                    self.state.requests.clear()
                    self.state.forced_status = status_code
                    self.state.forced_error = err_text

                with self.assertRaises(exc_cls) as ctx:
                    client.upsert(1, [0.1, 0.2, 0.3, 0.4])

                err: VektorDistError = ctx.exception
                self.assertEqual(err.status_code, status_code)
                self.assertIn(err_text, err.message)
                self.assertIsNotNone(err.trace_id)
                # Non-transient 4xx errors must never retry.
                with self.state.lock:
                    self.assertEqual(len(self.state.requests), 1)

        # Verify `/readyz` returns structured `not_ready` dict on 503 without raising.
        with self.state.lock:
            self.state.forced_status = None
            self.state.draining = True
        ready_state = client.readyz()
        self.assertEqual(ready_state["status"], "not_ready")
        self.assertTrue(ready_state["draining"])


if __name__ == "__main__":
    unittest.main()
