"""Loki HTTP API adapter with safety bounds and validation."""

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from metrics import increment_tool_calls, observe_tool_call_duration

logger = logging.getLogger(__name__)


@dataclass
class LokiConfig:
    """Configuration for Loki adapter."""
    base_url: str
    timeout_seconds: float = 10.0
    max_response_bytes: int = 1_000_000  # 1MB
    max_log_lines: int = 10000
    max_log_line_length: int = 10000
    allowed_time_range_hours: int = 72  # Max 72 hours relative to incident

    def __post_init__(self):
        if not self.base_url:
            raise ValueError("Loki base_url is required")
        # Ensure URL has no trailing slash
        self.base_url = self.base_url.rstrip("/")


class LokiError(Exception):
    """Base exception for Loki adapter errors."""
    pass


class LokiTimeoutError(LokiError):
    """Loki query timed out."""
    pass


class LokiResponseTooLargeError(LokiError):
    """Loki response exceeded size limits."""
    pass


class LokiQueryValidationError(LokiError):
    """Loki query validation failed."""
    pass


class LokiAdapter:
    """Real Loki HTTP API adapter with bounded queries and validation."""

    def __init__(self, config: LokiConfig):
        self.config = config
        self._client: httpx.AsyncClient | None = None

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

    def validate_logql(self, logql: str) -> None:
        """Validate LogQL query for safety."""
        if not logql or not logql.strip():
            raise LokiQueryValidationError("LogQL query cannot be empty")

        # Length limit
        if len(logql) > 10000:
            raise LokiQueryValidationError(
                "LogQL query exceeds maximum length (10000 chars)"
            )

        # Basic injection prevention
        dangerous_patterns = [
            r";\s*\w",  # Multiple statements
            r"--\s",    # SQL comments
            r"/\*.*\*/",  # Block comments
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, logql, re.IGNORECASE):
                raise LokiQueryValidationError(
                    f"LogQL query contains prohibited pattern: {pattern}"
                )

    def validate_time_range(self, start: str, end: str) -> tuple[float, float]:
        """Parse and validate time range. Returns (start_ts, end_ts) as Unix timestamps."""
        from datetime import datetime

        try:
            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except ValueError as e:
            raise LokiQueryValidationError(
                f"Invalid timestamp format (use RFC3339): {e}"
            )

        if start_dt.tzinfo is None or end_dt.tzinfo is None:
            raise LokiQueryValidationError("Timestamps must include timezone (RFC3339)")

        start_ts = start_dt.timestamp()
        end_ts = end_dt.timestamp()

        if end_ts <= start_ts:
            raise LokiQueryValidationError("End time must be after start time")

        # Check max range
        range_hours = (end_ts - start_ts) / 3600
        if range_hours > self.config.allowed_time_range_hours:
            raise LokiQueryValidationError(
                f"Time range {range_hours:.1f}h exceeds maximum "
                f"{self.config.allowed_time_range_hours}h"
            )

        return start_ts, end_ts

    def _validate_response_size(self, response_bytes: int) -> None:
        if response_bytes > self.config.max_response_bytes:
            raise LokiResponseTooLargeError(
                f"Response size {response_bytes} bytes exceeds limit "
                f"{self.config.max_response_bytes}"
            )

    def _validate_result_bounds(self, data: dict[str, Any]) -> None:
        """Validate Loki result bounds."""
        results = data.get("result", [])
        total_lines = 0
        for stream in results:
            values = stream.get("values", [])
            total_lines += len(values)
            for _, line in values:
                if len(line) > self.config.max_log_line_length:
                    raise LokiResponseTooLargeError(
                        f"Log line length {len(line)} exceeds limit "
                        f"{self.config.max_log_line_length}"
                    )
        if total_lines > self.config.max_log_lines:
            raise LokiResponseTooLargeError(
                f"Too many log lines: {total_lines} > {self.config.max_log_lines}"
            )

    async def query_range(
        self,
        logql: str,
        start: str,
        end: str,
        limit: int = 100,
        direction: str = "forward",
    ) -> dict[str, Any]:
        """
        Execute a Loki range query (query_range endpoint).

        Args:
            logql: LogQL query expression
            start: Start timestamp (RFC3339)
            end: End timestamp (RFC3339)
            limit: Maximum number of log entries to return (1-10000)
            direction: "forward" or "backward"

        Returns:
            Loki response dict with status, data, result

        Raises:
            LokiTimeoutError: Query timed out
            LokiResponseTooLargeError: Response exceeds size/line limits
            LokiQueryValidationError: Invalid query or parameters
        """
        start_time = time.time()

        # Validate inputs
        self.validate_logql(logql)
        self.validate_time_range(start, end)

        if limit <= 0 or limit > self.config.max_log_lines:
            raise LokiQueryValidationError(
                f"Limit must be between 1 and {self.config.max_log_lines}"
            )

        if direction not in ("forward", "backward"):
            raise LokiQueryValidationError(
                "Direction must be 'forward' or 'backward'"
            )

        client = await self._get_client()

        try:
            params = {
                "query": logql,
                "start": start,
                "end": end,
                "limit": str(min(limit, self.config.max_log_lines)),
                "direction": direction,
            }

            response = await client.get("/loki/api/v1/query_range", params=params)
            response.raise_for_status()

            # Check response size
            self._validate_response_size(len(response.content))

            data = response.json()

            # Validate Loki response structure
            if data.get("status") != "success":
                error_msg = data.get("error", "Unknown Loki error")
                raise LokiError(f"Loki error: {error_msg}")

            # Validate result bounds
            self._validate_result_bounds(data.get("data", {}))

            observe_tool_call_duration("loki", time.time() - start_time)
            increment_tool_calls("loki", "success")

            return {
                "status": "success",
                "data": data.get("data", {}),
            }

        except httpx.TimeoutException:
            observe_tool_call_duration("loki", time.time() - start_time)
            increment_tool_calls("loki", "timeout")
            raise LokiTimeoutError(
                f"Loki query timed out after {self.config.timeout_seconds}s"
            )
        except httpx.HTTPStatusError as e:
            observe_tool_call_duration("loki", time.time() - start_time)
            increment_tool_calls("loki", "error")
            raise LokiError(
                f"Loki HTTP error {e.response.status_code}: {e.response.text}"
            )
        except LokiError:
            # Re-raise our validation errors
            observe_tool_call_duration("loki", time.time() - start_time)
            increment_tool_calls("loki", "error")
            raise
        except Exception as e:  # noqa: BLE001
            observe_tool_call_duration("loki", time.time() - start_time)
            increment_tool_calls("loki", "error")
            logger.error("Loki query failed: %s", e)
            raise LokiError(f"Loki query failed: {e}")

    async def query_instant(
        self,
        logql: str,
        eval_time: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        Execute a Loki instant query (query endpoint).

        Args:
            logql: LogQL query expression
            eval_time: Optional evaluation timestamp (RFC3339), defaults to now
            limit: Maximum number of log entries to return

        Returns:
            Loki response dict
        """
        start_time = time.time()

        self.validate_logql(logql)

        if limit <= 0 or limit > self.config.max_log_lines:
            raise LokiQueryValidationError(
                f"Limit must be between 1 and {self.config.max_log_lines}"
            )

        if eval_time:
            self.validate_time_range(eval_time, eval_time)  # Validate format

        client = await self._get_client()

        try:
            params = {
                "query": logql,
                "limit": str(min(limit, self.config.max_log_lines)),
            }
            if eval_time:
                params["time"] = eval_time

            response = await client.get("/loki/api/v1/query", params=params)
            response.raise_for_status()

            self._validate_response_size(len(response.content))

            data = response.json()

            if data.get("status") != "success":
                error_msg = data.get("error", "Unknown Loki error")
                raise LokiError(f"Loki error: {error_msg}")

            observe_tool_call_duration("loki", time.time() - start_time)
            increment_tool_calls("loki", "success")

            return {
                "status": "success",
                "data": data.get("data", {}),
            }

        except httpx.TimeoutException:
            observe_tool_call_duration("loki", time.time() - start_time)
            increment_tool_calls("loki", "timeout")
            raise LokiTimeoutError(
                f"Loki query timed out after {self.config.timeout_seconds}s"
            )
        except httpx.HTTPStatusError as e:
            observe_tool_call_duration("loki", time.time() - start_time)
            increment_tool_calls("loki", "error")
            raise LokiError(
                f"Loki HTTP error {e.response.status_code}: {e.response.text}"
            )
        except LokiError:
            observe_tool_call_duration("loki", time.time() - start_time)
            increment_tool_calls("loki", "error")
            raise
        except Exception as e:  # noqa: BLE001
            observe_tool_call_duration("loki", time.time() - start_time)
            increment_tool_calls("loki", "error")
            logger.error("Loki instant query failed: %s", e)
            raise LokiError(f"Loki query failed: {e}")

    async def query_labels(self) -> dict[str, Any]:
        """Get all label names from Loki."""
        start_time = time.time()

        client = await self._get_client()

        try:
            response = await client.get("/loki/api/v1/labels")
            response.raise_for_status()

            data = response.json()

            if data.get("status") != "success":
                error_msg = data.get("error", "Unknown Loki error")
                raise LokiError(f"Loki error: {error_msg}")

            observe_tool_call_duration("loki", time.time() - start_time)
            increment_tool_calls("loki", "success")

            return {
                "status": "success",
                "data": data.get("data", []),
            }

        except httpx.TimeoutException:
            observe_tool_call_duration("loki", time.time() - start_time)
            increment_tool_calls("loki", "timeout")
            raise LokiTimeoutError(
                f"Loki query timed out after {self.config.timeout_seconds}s"
            )
        except httpx.HTTPStatusError as e:
            observe_tool_call_duration("loki", time.time() - start_time)
            increment_tool_calls("loki", "error")
            raise LokiError(
                f"Loki HTTP error {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:  # noqa: BLE001
            observe_tool_call_duration("loki", time.time() - start_time)
            increment_tool_calls("loki", "error")
            logger.error("Loki labels query failed: %s", e)
            raise LokiError(f"Loki labels query failed: {e}")
