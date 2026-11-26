"""Zero-dependency typed Python client SDK for the VektorDist HTTP/1.1 gateway."""

from .vektordist_client import (
    AuthenticationError,
    AuthorizationError,
    PayloadTooLargeError,
    RateLimitError,
    ScoredHit,
    UpsertAck,
    VektorDistClient,
    VektorDistError,
    generate_traceparent,
)

__all__ = [
    "AuthenticationError",
    "AuthorizationError",
    "PayloadTooLargeError",
    "RateLimitError",
    "ScoredHit",
    "UpsertAck",
    "VektorDistClient",
    "VektorDistError",
    "generate_traceparent",
]
