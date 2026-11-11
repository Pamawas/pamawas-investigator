"""Prometheus HTTP API adapter with safety bounds and validation."""

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from adapters.circuit_breaker import CircuitBreaker
from metrics import increment_tool_calls, observe_tool_call_duration

logger = logging.getLogger(__name__)


@dataclass
class PrometheusConfig:
    """Configuration for Prometheus adapter."""

    base_url: str
    timeout_seconds: float = 10.0
    max_response_bytes: int = 1_000_000  # 1MB
    max_series: int = 1000
    max_samples_per_series: int = 10000
    allowed_time_range_hours: int = 72  # Max 72 hours relative to incident
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: float = 30.0
    circuit_breaker_half_open_max_calls: int = 3

    def __post_init__(self):
        if not self.base_url:
            raise ValueError("Prometheus base_url is required")
        # Ensure URL has no trailing slash
        self.base_url = self.base_url.rstrip("/")


class PrometheusError(Exception):
    """Base exception for Prometheus adapter errors."""

    pass


class PrometheusTimeoutError(PrometheusError):
    """Prometheus query timed out."""

    pass


class PrometheusResponseTooLargeError(PrometheusError):
    """Prometheus response exceeded size limits."""

    pass


class PrometheusQueryValidationError(PrometheusError):
    """Prometheus query validation failed."""

    pass


class PrometheusAdapter:
    """Real Prometheus HTTP API adapter with bounded queries and validation."""

    def __init__(self, config: PrometheusConfig):
        self.config = config
        self._client: httpx.AsyncClient | None = None
        self._circuit_breaker = CircuitBreaker(
            name="prometheus",
            failure_threshold=config.circuit_breaker_failure_threshold,
            timeout_seconds=config.circuit_breaker_timeout_seconds,
            half_open_max_calls=config.circuit_breaker_half_open_max_calls,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(self.config.timeout_seconds),
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def circuit_breaker_state(self) -> dict:
        """Return circuit breaker state for health checks."""
        return self._circuit_breaker.snapshot()

    def validate_promql(self, promql: str) -> None:
        """Validate PromQL query for safety."""
        if not promql or not promql.strip():
            raise PrometheusQueryValidationError("PromQL query cannot be empty")

        # Length limit
        if len(promql) > 10000:
            raise PrometheusQueryValidationError(
                "PromQL query exceeds maximum length (10000 chars)"
            )

        # Basic injection prevention - reject suspicious patterns
        dangerous_patterns = [
            r";\s*\w",  # Multiple statements
            r"--\s",  # SQL comments
            r"/\*.*\*/",  # Block comments
            r"union\s+select",  # SQL injection (case insensitive)
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, promql, re.IGNORECASE):
                raise PrometheusQueryValidationError(
                    f"PromQL query contains prohibited pattern: {pattern}"
                )

    def validate_time_range(self, start: str, end: str) -> tuple[float, float]:
        """Parse and validate time range. Returns (start_ts, end_ts) as Unix timestamps."""
        from datetime import datetime

        try:
            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except ValueError as e:
            raise PrometheusQueryValidationError(f"Invalid timestamp format (use RFC3339): {e}")

        if start_dt.tzinfo is None or end_dt.tzinfo is None:
            raise PrometheusQueryValidationError("Timestamps must include timezone (RFC3339)")

        start_ts = start_dt.timestamp()
        end_ts = end_dt.timestamp()

        if end_ts <= start_ts:
            raise PrometheusQueryValidationError("End time must be after start time")

        # Check max range
        range_hours = (end_ts - start_ts) / 3600
        if range_hours > self.config.allowed_time_range_hours:
            raise PrometheusQueryValidationError(
                f"Time range {range_hours:.1f}h exceeds maximum "
                f"{self.config.allowed_time_range_hours}h"
            )

        return start_ts, end_ts

    def _validate_response_size(self, response_bytes: int) -> None:
        if response_bytes > self.config.max_response_bytes:
            raise PrometheusResponseTooLargeError(
                f"Response size {response_bytes} bytes exceeds limit "
                f"{self.config.max_response_bytes}"
            )

    def _validate_result_bounds(self, data: dict[str, Any]) -> None:
        """Validate Prometheus result bounds."""
        if data.get("resultType") == "matrix":
            results = data.get("result", [])
            if len(results) > self.config.max_series:
                raise PrometheusResponseTooLargeError(
                    f"Too many series: {len(results)} > {self.config.max_series}"
                )
            for series in results:
                values = series.get("values", [])
                if len(values) > self.config.max_samples_per_series:
                    raise PrometheusResponseTooLargeError(
                        f"Too many samples per series: {len(values)} > "
                        f"{self.config.max_samples_per_series}"
                    )

    def _unavailable_response(self, reason_code: str) -> dict[str, Any]:
        """Return standardized unavailable response (circuit open)."""
        return {
            "status": "unavailable",
            "reason_code": reason_code,
            "data": {},
        }

    async def query_range(
        self,
        promql: str,
        start: str,
        end: str,
        step_seconds: int = 60,
    ) -> dict[str, Any]:
        """
        Execute a Prometheus range query.

        Args:
            promql: PromQL query expression
            start: Start timestamp (RFC3339)
            end: End timestamp (RFC3339)
            step_seconds: Query resolution step in seconds

        Returns:
            Prometheus response dict with status, data, resultType, result

        Raises:
            PrometheusTimeoutError: Query timed out
            PrometheusResponseTooLargeError: Response exceeds size/sample limits
            PrometheusQueryValidationError: Invalid query or parameters
        """
        start_time = time.time()

        # Validate inputs
        self.validate_promql(promql)
        self.validate_time_range(start, end)

        if step_seconds <= 0 or step_seconds > 86400:
            raise PrometheusQueryValidationError("Step must be between 1 and 86400 seconds")

        client = await self._get_client()

        # Circuit breaker: return unavailable immediately if open
        if not self._circuit_breaker.allow_call():
            observe_tool_call_duration("prometheus", time.time() - start_time)
            increment_tool_calls("prometheus", "circuit_open")
            logger.warning("Prometheus circuit breaker open, rejecting query")
            return self._unavailable_response("circuit_breaker_open")

        try:
            params = {
                "query": promql,
                "start": start,
                "end": end,
                "step": str(step_seconds),
            }

            response = await client.get("/api/v1/query_range", params=params)
            response.raise_for_status()

            # Check response size
            self._validate_response_size(len(response.content))

            data = response.json()

            # Validate Prometheus response structure
            if data.get("status") != "success":
                error_msg = data.get("error", "Unknown Prometheus error")
                self._circuit_breaker.record_failure()
                raise PrometheusError(f"Prometheus error: {error_msg}")

            # Validate result bounds
            self._validate_result_bounds(data.get("data", {}))

            self._circuit_breaker.record_success()
            observe_tool_call_duration("prometheus", time.time() - start_time)
            increment_tool_calls("prometheus", "success")

            return {
                "status": "success",
                "data": data.get("data", {}),
            }

        except httpx.TimeoutException:
            self._circuit_breaker.record_failure()
            observe_tool_call_duration("prometheus", time.time() - start_time)
            increment_tool_calls("prometheus", "timeout")
            raise PrometheusTimeoutError(
                f"Prometheus query timed out after {self.config.timeout_seconds}s"
            )
        except httpx.HTTPStatusError as e:
            self._circuit_breaker.record_failure()
            observe_tool_call_duration("prometheus", time.time() - start_time)
            increment_tool_calls("prometheus", "error")
            raise PrometheusError(
                f"Prometheus HTTP error {e.response.status_code}: {e.response.text}"
            )
        except PrometheusError:
            # Validation errors are client-side; don't trip the circuit breaker.
            # Non-validation PrometheusError already recorded failure above.
            observe_tool_call_duration("prometheus", time.time() - start_time)
            increment_tool_calls("prometheus", "error")
            raise
        except (httpx.HTTPError, ValueError, KeyError, UnicodeDecodeError) as e:
            self._circuit_breaker.record_failure()
            observe_tool_call_duration("prometheus", time.time() - start_time)
            increment_tool_calls("prometheus", "error")
            logger.error("Prometheus query failed: %s", e)
            raise PrometheusError(f"Prometheus query failed: {e}")

    async def query_instant(
        self,
        promql: str,
        eval_time: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute a Prometheus instant query.

        Args:
            promql: PromQL query expression
            eval_time: Optional evaluation timestamp (RFC3339), defaults to now

        Returns:
            Prometheus response dict
        """
        start_time = time.time()

        self.validate_promql(promql)

        client = await self._get_client()

        # Circuit breaker: return unavailable immediately if open
        if not self._circuit_breaker.allow_call():
            observe_tool_call_duration("prometheus", time.time() - start_time)
            increment_tool_calls("prometheus", "circuit_open")
            logger.warning("Prometheus circuit breaker open, rejecting instant query")
            return self._unavailable_response("circuit_breaker_open")

        try:
            params = {"query": promql}
            if eval_time:
                self.validate_time_range(eval_time, eval_time)  # Validate format
                params["time"] = eval_time

            response = await client.get("/api/v1/query", params=params)
            response.raise_for_status()

            self._validate_response_size(len(response.content))

            data = response.json()

            if data.get("status") != "success":
                error_msg = data.get("error", "Unknown Prometheus error")
                self._circuit_breaker.record_failure()
                raise PrometheusError(f"Prometheus error: {error_msg}")

            self._circuit_breaker.record_success()
            observe_tool_call_duration("prometheus", time.time() - start_time)
            increment_tool_calls("prometheus", "success")

            return {
                "status": "success",
                "data": data.get("data", {}),
            }

        except httpx.TimeoutException:
            self._circuit_breaker.record_failure()
            observe_tool_call_duration("prometheus", time.time() - start_time)
            increment_tool_calls("prometheus", "timeout")
            raise PrometheusTimeoutError(
                f"Prometheus query timed out after {self.config.timeout_seconds}s"
            )
        except httpx.HTTPStatusError as e:
            self._circuit_breaker.record_failure()
            observe_tool_call_duration("prometheus", time.time() - start_time)
            increment_tool_calls("prometheus", "error")
            raise PrometheusError(
                f"Prometheus HTTP error {e.response.status_code}: {e.response.text}"
            )
        except PrometheusError:
            observe_tool_call_duration("prometheus", time.time() - start_time)
            increment_tool_calls("prometheus", "error")
            raise
        except (httpx.HTTPError, ValueError, KeyError, UnicodeDecodeError) as e:
            self._circuit_breaker.record_failure()
            observe_tool_call_duration("prometheus", time.time() - start_time)
            increment_tool_calls("prometheus", "error")
            logger.error("Prometheus instant query failed: %s", e)
            raise PrometheusError(f"Prometheus query failed: {e}")
