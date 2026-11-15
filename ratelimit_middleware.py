"""Per-route token-bucket rate limiting middleware (design review P5).

Protects inbound endpoints from traffic spikes and abuse. Mirrors the Go
services' middleware: limiters are built once at construction from a fixed
route table (so the map cannot grow with attacker-controlled paths), and
liveness/readiness/metrics endpoints are exempt because Kubernetes probes and
Prometheus scrapers must never be throttled.
"""

import time

import structlog
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

log = structlog.get_logger()

# Paths that bypass rate limiting entirely.
EXEMPT_PATHS = frozenset({"/healthz", "/ready", "/metrics"})

# Budget applied to any route not listed explicitly. Unknown paths 404 at the
# router anyway; one shared limiter keeps a path-scanning client from
# amplifying load without growing the route table.
DEFAULT_RPS = 20.0
DEFAULT_BURST = 40


class TokenBucket:
    """Classic token bucket: refills continuously at `rps`, caps at `burst`."""

    __slots__ = ("_burst", "_rps", "_tokens", "_updated")

    def __init__(self, rps: float, burst: int):
        self._rps = rps
        self._burst = float(burst)
        self._tokens = float(burst)
        self._updated = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        self._tokens = min(self._burst, self._tokens + (now - self._updated) * self._rps)
        self._updated = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def retry_after(self) -> float:
        """Seconds until the next token is available."""
        if self._rps <= 0:
            return 1.0
        deficit = 1.0 - self._tokens
        return max(1.0, deficit / self._rps)


class RateLimitMiddleware:
    """Pure-ASGI per-route rate limiter.

    `limits` maps a path to an (rps, burst) pair. Requests to listed paths
    draw from their own bucket; unlisted non-exempt paths share the default
    bucket. Over-budget requests are answered 429 with a Retry-After hint
    before they reach the endpoint.
    """

    def __init__(self, app: ASGIApp, service_name: str, limits: dict[str, tuple[float, int]]):
        self.app = app
        self.service_name = service_name
        self._buckets: dict[str, TokenBucket] = {}
        for path, (rps, burst) in limits.items():
            self._buckets[path] = TokenBucket(rps, max(1, burst))
        self._default = TokenBucket(DEFAULT_RPS, DEFAULT_BURST)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        bucket = self._buckets.get(path, self._default)
        if not bucket.allow():
            request = Request(scope)
            log.warning(
                "rate_limit_exceeded",
                service=self.service_name,
                path=path,
                remote_addr=request.client.host if request.client else None,
            )
            response = JSONResponse(
                {"detail": "Too Many Requests"},
                status_code=429,
                headers={"Retry-After": str(int(bucket.retry_after()) + 1)},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
