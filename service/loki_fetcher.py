"""Loki fetcher for evidence enrichment using query_range endpoint."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LogsFetcher:
    """Interface for fetching Loki logs."""

    def fetch(
        self, incident_id: str, start: str, end: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Fetch logs for an incident. Returns list of log entry dicts."""
        raise NotImplementedError


class LokiFetcher(LogsFetcher):
    """Loki query_range fetcher for evidence enrichment."""

    def __init__(
        self,
        base_url: str,
        query_template: str = '{incident="%s"}',
        timeout_seconds: float = 10.0,
        max_lines: int = 10000,
        max_response_bytes: int = 1_000_000,
    ):
        self.base_url = base_url.rstrip("/")
        self.query_template = query_template
        self.timeout_seconds = timeout_seconds
        self.max_lines = max_lines
        self.max_response_bytes = max_response_bytes
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout_seconds),
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _validate_logql(self, logql: str) -> None:
        """Validate LogQL query for safety."""
        if not logql or not logql.strip():
            raise ValueError("LogQL query cannot be empty")

        if len(logql) > 10000:
            raise ValueError("LogQL query exceeds maximum length (10000 chars)")

        dangerous_patterns = [
            r";\s*\w",
            r"--\s",
            r"/\*.*\*/",
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, logql, re.IGNORECASE):
                raise ValueError(f"LogQL query contains prohibited pattern: {pattern}")

    def _parse_timestamp(self, ts: float) -> datetime:
        """Parse Unix timestamp to datetime."""
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    async def _query_range(self, logql: str, start: str, end: str, limit: int) -> dict[str, Any]:
        """Execute a Loki query_range request."""
        client = await self._get_client()

        # Parse timestamps to nanoseconds
        try:
            start_ts = float(start)
        except ValueError:
            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            start_ts = start_dt.timestamp()

        try:
            end_ts = float(end)
        except ValueError:
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
            end_ts = end_dt.timestamp()

        # Format as nanoseconds (Loki expects seconds with fractional)
        params = {
            "query": logql,
            "start": str(start_ts),
            "end": str(end_ts),
            "limit": str(min(limit, self.max_lines)),
        }

        response = await client.get("/loki/api/v1/query_range", params=params)
        response.raise_for_status()

        if len(response.content) > self.max_response_bytes:
            raise ValueError("Response size exceeds limit")

        return response.json()

    async def fetch_async(
        self, incident_id: str, start: str, end: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Fetch Loki logs for an incident.

        Args:
            incident_id: The incident ID to query for
            start: Start timestamp (RFC3339)
            end: End timestamp (RFC3339)
            limit: Maximum number of log lines to return

        Returns:
            List of log entry dictionaries with timestamp, stream, line
        """
        # Build query using template
        tmpl = self.query_template
        query = tmpl % incident_id if "%s" in tmpl else tmpl

        # Normalize the caller-provided window. A missing/invalid end falls
        # back to now; a missing/invalid start defaults to 5 minutes before
        # end. The caller's start is always honored when provided — never
        # silently replaced with a hardcoded window.
        try:
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except ValueError, AttributeError:
            end_dt = datetime.now(tz=timezone.utc)

        if start:
            try:
                start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            except ValueError, AttributeError:
                start_dt = datetime.fromtimestamp(end_dt.timestamp() - 300, tz=timezone.utc)
        else:
            start_dt = datetime.fromtimestamp(end_dt.timestamp() - 300, tz=timezone.utc)

        start = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        end = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        self._validate_logql(query)

        try:
            result = await self._query_range(query, start, end, limit)
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Loki query failed for incident {incident_id}: {e}")
            return []

        if result.get("status") != "success":
            logger.error(f"Loki returned error: {result.get('error', 'Unknown error')}")
            return []

        # Parse log entries
        logs = []
        data = result.get("data", {})
        results = data.get("result", [])

        for stream_result in results:
            stream = stream_result.get("stream", {})
            values = stream_result.get("values", [])

            for value in values:
                if not isinstance(value, list) or len(value) < 2:
                    continue

                try:
                    ts = float(value[0])
                    line = str(value[1])
                except ValueError, TypeError:
                    continue

                # Get stream name from labels
                stream_name = stream.get("__name__", stream.get("stream", ""))

                logs.append(
                    {
                        "timestamp": self._parse_timestamp(ts),
                        "stream": stream_name,
                        "line": line,
                    }
                )

        return logs

    def fetch(
        self, incident_id: str, start: str, end: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Synchronous wrapper for async fetch."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.fetch_async(incident_id, start, end, limit))
