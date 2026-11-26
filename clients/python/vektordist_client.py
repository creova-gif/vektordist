"""Zero-dependency typed Python client SDK for VektorDist.

Communicates with the VektorDist zero-trust HTTP/1.1 REST gateway using only the
Python standard library (`urllib.request`, `json`, `os`, `time`, `dataclasses`,
`typing`) so AI, RAG, and agentic workloads on Python 3.11+ / 3.12+ can import it
immediately without native C-extension build steps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from email.message import Message
import json
import os
import time
from typing import Any, TypeAlias
import urllib.error
import urllib.parse
import urllib.request

MAX_RETRY_BACKOFF_CAP_SECS: float = 2.0

BatchRecord: TypeAlias = (
    tuple[int, Sequence[float]]
    | tuple[int, int, Sequence[float]]
    | Mapping[str, Any]
)


@dataclass(frozen=True, slots=True)
class ScoredHit:
    """Nearest-neighbor vector match returned by distributed or collection search."""

    id: int
    score: float


@dataclass(frozen=True, slots=True)
class UpsertAck:
    """Quorum durability acknowledgment returned by upsert and delete mutations."""

    shard_id: int
    seq_no: int
    trace_id: str


class VektorDistError(Exception):
    """Base exception for VektorDist HTTP gateway and transport errors."""

    status_code: int
    message: str
    trace_id: str | None

    def __init__(
        self,
        status_code: int,
        message: str,
        trace_id: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.message = message
        self.trace_id = trace_id
        prefix = f"HTTP {status_code}: " if status_code > 0 else ""
        suffix = f" (trace_id={trace_id})" if trace_id else ""
        super().__init__(f"{prefix}{message}{suffix}")


class AuthenticationError(VektorDistError):
    """Raised on HTTP 401 Unauthorized (missing, malformed, or revoked bearer token)."""

    def __init__(
        self,
        message: str,
        trace_id: str | None = None,
        status_code: int = 401,
    ) -> None:
        super().__init__(status_code=status_code, message=message, trace_id=trace_id)


class AuthorizationError(VektorDistError):
    """Raised on HTTP 403 Forbidden (cross-tenant access or insufficient RBAC role)."""

    def __init__(
        self,
        message: str,
        trace_id: str | None = None,
        status_code: int = 403,
    ) -> None:
        super().__init__(status_code=status_code, message=message, trace_id=trace_id)


class RateLimitError(VektorDistError):
    """Raised on HTTP 429 Too Many Requests when bounded retries are exhausted."""

    retry_after_secs: float

    def __init__(
        self,
        message: str,
        retry_after_secs: float,
        trace_id: str | None = None,
        status_code: int = 429,
    ) -> None:
        self.retry_after_secs = retry_after_secs
        super().__init__(status_code=status_code, message=message, trace_id=trace_id)


class PayloadTooLargeError(VektorDistError):
    """Raised on HTTP 413 Payload Too Large when request exceeds gateway body limits."""

    def __init__(
        self,
        message: str,
        trace_id: str | None = None,
        status_code: int = 413,
    ) -> None:
        super().__init__(status_code=status_code, message=message, trace_id=trace_id)


def generate_traceparent(sampled: bool = True) -> str:
    """Generate a compliant 55-character W3C Trace Context v00 `traceparent` header.

    Format: `00-<32-hex-trace-id>-<16-hex-span-id>-<2-hex-flags>`.
    """
    trace_id = os.urandom(16).hex()
    if trace_id == "0" * 32:
        trace_id = "1" + ("0" * 31)
    span_id = os.urandom(8).hex()
    if span_id == "0" * 16:
        span_id = "1" + ("0" * 15)
    flags = "01" if sampled else "00"
    return f"00-{trace_id}-{span_id}-{flags}"


def _is_valid_traceparent(header: str) -> bool:
    parts = header.strip().split("-")
    if len(parts) != 4 or parts[0] != "00":
        return False
    trace_id, span_id, flags = parts[1], parts[2], parts[3]
    if len(trace_id) != 32 or trace_id == "0" * 32:
        return False
    if len(span_id) != 16 or span_id == "0" * 16:
        return False
    if len(flags) != 2:
        return False
    try:
        int(trace_id, 16)
        int(span_id, 16)
        int(flags, 16)
    except ValueError:
        return False
    return True


def _normalize_base_url(raw_url: str) -> str:
    trimmed = raw_url.strip()
    if not trimmed:
        raise ValueError("base_url must not be empty")
    if not trimmed.startswith(("http://", "https://")):
        trimmed = f"http://{trimmed}"
    return trimmed.rstrip("/")


def _extract_header_case_insensitive(headers: Message, target: str) -> str | None:
    target_lower = target.lower()
    for key, val in headers.items():
        if key.lower() == target_lower:
            return str(val).strip()
    return None


def _parse_json_object(raw_body: str) -> dict[str, Any]:
    if not raw_body.strip():
        return {}
    parsed: object = json.loads(raw_body)
    if not isinstance(parsed, dict):
        raise ValueError("expected JSON object from VektorDist gateway")
    return {str(k): v for k, v in parsed.items()}


def _try_parse_json_object(raw_body: str) -> dict[str, Any]:
    try:
        return _parse_json_object(raw_body)
    except (json.JSONDecodeError, ValueError):
        return {}


def _parse_hits(payload: Mapping[str, Any]) -> list[ScoredHit]:
    raw_hits: object = payload.get("hits", payload.get("results", []))
    if not isinstance(raw_hits, list):
        raise VektorDistError(0, "malformed 'hits' array in search response")
    out: list[ScoredHit] = []
    for item in raw_hits:
        if not isinstance(item, dict):
            raise VektorDistError(0, "malformed hit entry in search response")
        raw_id = item.get("id")
        raw_score = item.get("score")
        if not isinstance(raw_id, (int, float)) or isinstance(raw_id, bool):
            raise VektorDistError(0, "missing integer 'id' in search hit")
        if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool):
            raise VektorDistError(0, "missing numeric 'score' in search hit")
        out.append(ScoredHit(id=int(raw_id), score=float(raw_score)))
    return out


class VektorDistClient:
    """Synchronous zero-dependency HTTP/1.1 client for VektorDist."""

    def __init__(
        self,
        base_url: str,
        bearer_token: str | None = None,
        org_id: str = "default",
        timeout_secs: float = 5.0,
        max_retries: int = 2,
        base_backoff_secs: float = 0.05,
    ) -> None:
        if timeout_secs <= 0.0:
            raise ValueError("timeout_secs must be > 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if base_backoff_secs < 0.0:
            raise ValueError("base_backoff_secs must be >= 0")

        self._base_url: str = _normalize_base_url(base_url)
        self._bearer_token: str | None = (
            bearer_token.strip() if bearer_token is not None and bearer_token.strip() else None
        )
        self._org_id: str = org_id.strip()
        self._timeout_secs: float = timeout_secs
        self._max_retries: int = max_retries
        self._base_backoff_secs: float = base_backoff_secs

    @property
    def base_url(self) -> str:
        """Normalized gateway base URL without trailing slash."""
        return self._base_url

    @property
    def org_id(self) -> str:
        """Configured tenant organization identifier sent via `X-Org-Id`."""
        return self._org_id

    def healthz(self, *, traceparent: str | None = None) -> dict[str, Any]:
        """Query `GET /healthz` process liveness and node count telemetry."""
        data, _ = self._request_json("GET", "/healthz", None, traceparent=traceparent)
        return data

    def readyz(self, *, traceparent: str | None = None) -> dict[str, Any]:
        """Query `GET /readyz` quorum and graceful-drain readiness status.

        Returns the structured JSON status dictionary for both `200 OK` (`ready`)
        and `503 Service Unavailable` (`not_ready`) without retrying `503`.
        """
        status_code, headers, raw_body = self._send_once(
            "GET",
            "/readyz",
            None,
            traceparent=traceparent,
        )
        if status_code in (200, 503):
            try:
                return _parse_json_object(raw_body)
            except (json.JSONDecodeError, ValueError) as exc:
                trace_id = _extract_header_case_insensitive(headers, "x-trace-id")
                raise VektorDistError(status_code, f"invalid JSON in /readyz: {exc}", trace_id) from exc
        self._raise_http_error(status_code, headers, raw_body)
        return {}

    def upsert(
        self,
        id: int,
        vector: Sequence[float],
        write_id: int | None = None,
        *,
        traceparent: str | None = None,
    ) -> UpsertAck:
        """Perform a `W=2` quorum-durable vector upsert via `POST /v1/upsert`."""
        resolved_write_id = id if write_id is None else write_id
        payload: dict[str, Any] = {
            "id": int(id),
            "write_id": int(resolved_write_id),
            "vector": [float(v) for v in vector],
        }
        data, resp_trace_id = self._request_json(
            "POST",
            "/v1/upsert",
            payload,
            traceparent=traceparent,
        )
        return self._build_upsert_ack(data, resp_trace_id)

    def upsert_batch(
        self,
        records: Sequence[BatchRecord],
        *,
        traceparent: str | None = None,
    ) -> int:
        """Perform a batched quorum upsert via `POST /v1/upsert_batch`.

        Each element of `records` may be `(id, vector)`, `(id, write_id, vector)`,
        or a mapping `{"id": ..., "vector": [...], "write_id": ...}`.
        Returns the integer count of durably acknowledged vectors.
        """
        serialized: list[dict[str, Any]] = []
        for entry in records:
            if isinstance(entry, Mapping):
                rec_id = int(entry["id"])
                rec_write_id = int(entry.get("write_id", rec_id))
                raw_vec = entry["vector"]
                serialized.append(
                    {
                        "id": rec_id,
                        "write_id": rec_write_id,
                        "vector": [float(x) for x in raw_vec],
                    }
                )
            elif len(entry) == 2:
                rec_id_2, vec_2 = entry
                serialized.append(
                    {
                        "id": int(rec_id_2),
                        "write_id": int(rec_id_2),
                        "vector": [float(x) for x in vec_2],
                    }
                )
            elif len(entry) == 3:
                rec_id_3, rec_wid_3, vec_3 = entry
                serialized.append(
                    {
                        "id": int(rec_id_3),
                        "write_id": int(rec_wid_3),
                        "vector": [float(x) for x in vec_3],
                    }
                )
            else:
                raise ValueError("each batch record must be a 2-tuple, 3-tuple, or mapping")

        data, _ = self._request_json(
            "POST",
            "/v1/upsert_batch",
            {"records": serialized},
            traceparent=traceparent,
        )
        upserted = data.get("upserted")
        if not isinstance(upserted, int) or isinstance(upserted, bool):
            raise VektorDistError(200, "missing integer 'upserted' in upsert_batch response")
        return upserted

    def search(
        self,
        query: Sequence[float],
        top_k: int = 10,
        ef_search: int = 64,
        *,
        traceparent: str | None = None,
    ) -> list[ScoredHit]:
        """Execute a distributed scatter-gather search via `POST /v1/search`."""
        payload: dict[str, Any] = {
            "query": [float(v) for v in query],
            "top_k": int(top_k),
            "ef_search": int(ef_search),
        }
        data, _ = self._request_json("POST", "/v1/search", payload, traceparent=traceparent)
        return _parse_hits(data)

    def delete(
        self,
        id: int,
        write_id: int | None = None,
        *,
        traceparent: str | None = None,
    ) -> UpsertAck:
        """Perform a quorum-durable tombstone delete via `POST /v1/delete`."""
        resolved_write_id = id if write_id is None else write_id
        payload: dict[str, Any] = {
            "id": int(id),
            "write_id": int(resolved_write_id),
        }
        data, resp_trace_id = self._request_json(
            "POST",
            "/v1/delete",
            payload,
            traceparent=traceparent,
        )
        return self._build_upsert_ack(data, resp_trace_id)

    def topology(self, *, traceparent: str | None = None) -> dict[str, Any]:
        """Fetch live cluster shard/epoch topology summary via `GET /v1/topology`."""
        data, _ = self._request_json("GET", "/v1/topology", None, traceparent=traceparent)
        return data

    def audit(self, *, traceparent: str | None = None) -> dict[str, Any]:
        """Fetch tamper-evident hash-chained audit verification via `GET /v1/audit`."""
        data, _ = self._request_json("GET", "/v1/audit", None, traceparent=traceparent)
        return data

    def create_collection(
        self,
        name: str,
        dimensions: int,
        metric: str = "cosine",
        quantization: str = "f32",
        *,
        traceparent: str | None = None,
    ) -> dict[str, Any]:
        """Create an isolated named vector collection via `POST /v1/collections`."""
        if not name.strip():
            raise ValueError("collection name must not be empty")
        payload: dict[str, Any] = {
            "name": name,
            "dimensions": int(dimensions),
            "metric": metric,
            "quantization": quantization,
        }
        data, _ = self._request_json(
            "POST",
            "/v1/collections",
            payload,
            traceparent=traceparent,
        )
        return data

    def list_collections(
        self,
        *,
        traceparent: str | None = None,
    ) -> list[dict[str, Any]]:
        """List registered vector collections via `GET /v1/collections`."""
        data, _ = self._request_json("GET", "/v1/collections", None, traceparent=traceparent)
        raw_collections = data.get("collections", [])
        if not isinstance(raw_collections, list):
            raise VektorDistError(200, "malformed 'collections' list in response")
        out: list[dict[str, Any]] = []
        for item in raw_collections:
            if isinstance(item, dict):
                out.append({str(k): v for k, v in item.items()})
        return out

    def drop_collection(
        self,
        name: str,
        *,
        traceparent: str | None = None,
    ) -> dict[str, Any]:
        """Drop a named vector collection via `DELETE /v1/collections/{name}`."""
        encoded = self._encode_collection_name(name)
        data, _ = self._request_json(
            "DELETE",
            f"/v1/collections/{encoded}",
            None,
            traceparent=traceparent,
        )
        return data

    def collection_upsert(
        self,
        collection: str,
        id: int,
        vector: Sequence[float],
        *,
        write_id: int | None = None,
        traceparent: str | None = None,
    ) -> dict[str, Any]:
        """Upsert a vector into a named collection via `POST /v1/collections/{name}/upsert`."""
        encoded = self._encode_collection_name(collection)
        payload: dict[str, Any] = {
            "id": int(id),
            "vector": [float(v) for v in vector],
        }
        if write_id is not None:
            payload["write_id"] = int(write_id)
        data, _ = self._request_json(
            "POST",
            f"/v1/collections/{encoded}/upsert",
            payload,
            traceparent=traceparent,
        )
        return data

    def collection_search(
        self,
        collection: str,
        query: Sequence[float],
        top_k: int = 10,
        ef_search: int = 64,
        *,
        traceparent: str | None = None,
    ) -> list[ScoredHit]:
        """Search a named collection via `POST /v1/collections/{name}/search`."""
        encoded = self._encode_collection_name(collection)
        payload: dict[str, Any] = {
            "query": [float(v) for v in query],
            "top_k": int(top_k),
            "ef_search": int(ef_search),
        }
        data, _ = self._request_json(
            "POST",
            f"/v1/collections/{encoded}/search",
            payload,
            traceparent=traceparent,
        )
        return _parse_hits(data)

    def collection_delete(
        self,
        collection: str,
        id: int,
        *,
        traceparent: str | None = None,
    ) -> dict[str, Any]:
        """Delete a vector from a named collection via `POST /v1/collections/{name}/delete`."""
        encoded = self._encode_collection_name(collection)
        payload: dict[str, Any] = {"id": int(id)}
        data, _ = self._request_json(
            "POST",
            f"/v1/collections/{encoded}/delete",
            payload,
            traceparent=traceparent,
        )
        return data

    @staticmethod
    def _encode_collection_name(name: str) -> str:
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("collection name must not be empty")
        return urllib.parse.quote(cleaned, safe="")

    @staticmethod
    def _build_upsert_ack(data: Mapping[str, Any], header_trace_id: str | None) -> UpsertAck:
        shard_id_raw = data.get("shard_id", 0)
        seq_no_raw = data.get("seq_no", 0)
        trace_id_raw = data.get("trace_id", header_trace_id or "")
        if not isinstance(shard_id_raw, int) or isinstance(shard_id_raw, bool):
            raise VektorDistError(200, "invalid 'shard_id' in gateway response")
        if not isinstance(seq_no_raw, int) or isinstance(seq_no_raw, bool):
            raise VektorDistError(200, "invalid 'seq_no' in gateway response")
        return UpsertAck(
            shard_id=shard_id_raw,
            seq_no=seq_no_raw,
            trace_id=str(trace_id_raw),
        )

    def _build_headers(
        self,
        has_body: bool,
        traceparent: str | None,
    ) -> dict[str, str]:
        if traceparent is not None:
            if not _is_valid_traceparent(traceparent):
                raise ValueError(
                    "traceparent must match W3C v00 format '00-<32hex>-<16hex>-<2hex>'"
                )
            resolved_tp = traceparent.strip()
        else:
            resolved_tp = generate_traceparent()

        headers: dict[str, str] = {
            "Accept": "application/json",
            "traceparent": resolved_tp,
        }
        if self._org_id:
            headers["X-Org-Id"] = self._org_id
        if self._bearer_token is not None:
            if self._bearer_token.lower().startswith("bearer "):
                headers["Authorization"] = self._bearer_token
            else:
                headers["Authorization"] = f"Bearer {self._bearer_token}"
        if has_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _send_once(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None,
        *,
        traceparent: str | None = None,
    ) -> tuple[int, Message, str]:
        body_bytes: bytes | None = None
        if payload is not None:
            body_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")

        headers = self._build_headers(has_body=body_bytes is not None, traceparent=traceparent)
        req = urllib.request.Request(
            url=f"{self._base_url}{path}",
            data=body_bytes,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_secs) as resp:
                raw_text = resp.read().decode("utf-8", errors="replace")
                return int(resp.status), resp.headers, raw_text
        except urllib.error.HTTPError as http_err:
            err_body = http_err.read().decode("utf-8", errors="replace")
            return int(http_err.code), http_err.headers, err_body
        except urllib.error.URLError as url_err:
            raise VektorDistError(0, f"transport error: {url_err.reason}") from url_err
        except TimeoutError as timeout_err:
            raise VektorDistError(0, f"request timed out after {self._timeout_secs}s") from timeout_err
        except OSError as os_err:
            raise VektorDistError(0, f"socket I/O error: {os_err}") from os_err

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None,
        *,
        traceparent: str | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        attempt = 0
        while True:
            status_code, headers, raw_body = self._send_once(
                method,
                path,
                payload,
                traceparent=traceparent,
            )
            if 200 <= status_code < 300:
                trace_id = _extract_header_case_insensitive(headers, "x-trace-id")
                try:
                    parsed = _parse_json_object(raw_body)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise VektorDistError(
                        status_code,
                        f"invalid JSON response from {path}: {exc}",
                        trace_id,
                    ) from exc
                body_trace = parsed.get("trace_id")
                resolved_trace = str(body_trace) if isinstance(body_trace, str) else trace_id
                return parsed, resolved_trace

            if status_code in (429, 503) and attempt < self._max_retries:
                delay = self._compute_retry_delay(attempt, status_code, headers, raw_body)
                if delay > 0.0:
                    time.sleep(delay)
                attempt += 1
                continue

            self._raise_http_error(status_code, headers, raw_body)

    def _compute_retry_delay(
        self,
        attempt: int,
        status_code: int,
        headers: Message,
        raw_body: str,
    ) -> float:
        shift = min(attempt, 10)
        exp_delay = min(
            self._base_backoff_secs * float(1 << shift),
            MAX_RETRY_BACKOFF_CAP_SECS,
        )
        if status_code == 429:
            retry_after = self._parse_retry_after_secs(headers, raw_body, default_secs=exp_delay)
            return max(0.0, min(retry_after, MAX_RETRY_BACKOFF_CAP_SECS))
        return max(0.0, exp_delay)

    @staticmethod
    def _parse_retry_after_secs(
        headers: Message,
        raw_body: str,
        default_secs: float = 0.0,
    ) -> float:
        header_val = _extract_header_case_insensitive(headers, "Retry-After")
        if header_val is not None:
            try:
                return max(0.0, float(header_val))
            except ValueError:
                pass
        parsed = _try_parse_json_object(raw_body)
        retry_ms = parsed.get("retry_after_ms")
        if isinstance(retry_ms, (int, float)) and not isinstance(retry_ms, bool):
            return max(0.0, float(retry_ms) / 1000.0)
        return max(0.0, default_secs)

    def _raise_http_error(
        self,
        status_code: int,
        headers: Message,
        raw_body: str,
    ) -> None:
        parsed = _try_parse_json_object(raw_body)
        raw_err = parsed.get("error")
        message = (
            str(raw_err)
            if isinstance(raw_err, str) and raw_err
            else (raw_body.strip() or f"HTTP error {status_code}")
        )
        body_trace = parsed.get("trace_id")
        trace_id = (
            str(body_trace)
            if isinstance(body_trace, str) and body_trace
            else _extract_header_case_insensitive(headers, "x-trace-id")
        )

        if status_code == 401:
            raise AuthenticationError(message=message, trace_id=trace_id)
        if status_code == 403:
            raise AuthorizationError(message=message, trace_id=trace_id)
        if status_code == 413:
            raise PayloadTooLargeError(message=message, trace_id=trace_id)
        if status_code == 429:
            retry_secs = self._parse_retry_after_secs(headers, raw_body, default_secs=0.0)
            raise RateLimitError(
                message=message,
                retry_after_secs=retry_secs,
                trace_id=trace_id,
            )
        raise VektorDistError(status_code=status_code, message=message, trace_id=trace_id)
