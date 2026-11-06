"""Prometheus fetcher for evidence enrichment using instant queries."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class MetricsFetcher:
    """Interface for fetching Prometheus metrics."""

    def fetch(self, incident_id: str, start: str, end: str) -> list[dict[str, Any]]:
        """Fetch metrics for an incident. Returns list of metric dicts."""
        raise NotImplementedError


class PrometheusFetcher(MetricsFetcher):
    """Prometheus instant query fetcher for evidence enrichment."""

    def __init__(
        self,
        base_url: str,
        query_template: str = '{incident="%s"}',
        timeout_seconds: float = 10.0,
        max_response_bytes: int = 1_000_000,
    ):
        self.base_url = base_url.rstrip("/")
        self.query_template = query_template
        self.timeout_seconds = timeout_seconds
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

    def _validate_promql(self, promql: str) -> None:
        """Validate PromQL query for safety."""
        if not promql or not promql.strip():
            raise ValueError("PromQL query cannot be empty")

        if len(promql) > 10000:
            raise ValueError("PromQL query exceeds maximum length (10000 chars)")

        dangerous_patterns = [
            r";\s*\w",
            r"--\s",
            r"/\*.*\*/",
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, promql, re.IGNORECASE):
                raise ValueError(f"PromQL query contains prohibited pattern: {pattern}")

    async def _query_instant(self, promql: str, eval_time: str | None = None) -> dict[str, Any]:
        """Execute a Prometheus instant query."""
        client = await self._get_client()
        params = {"query": promql}
        if eval_time:
            params["time"] = eval_time

        response = await client.get("/api/v1/query", params=params)
        response.raise_for_status()

        if len(response.content) > self.max_response_bytes:
            raise ValueError(f"Response size exceeds limit")

        data = response.json()
        if data.get("status") != "success":
            raise ValueError(f"Prometheus error: {data.get('error', 'Unknown error')}")

        return data

    async def fetch_async(
        self, incident_id: str, start: str, end: str
    ) -> list[dict[str, Any]]:
        """Fetch Prometheus metrics for an incident using instant query.

        Args:
            incident_id: The incident ID to query for
            start: Start timestamp (RFC3339, used for context)
            end: End timestamp (RFC3339, used as evaluation time)

        Returns:
            List of metric dictionaries with name, value, labels, at
        """
        # Build query using template
        query = self.query_template % incident_id if "%s" in self.query_template else self.query_template

        # For instant query, we use the end time as evaluation time
        try:
            # Parse end time to ensure it's valid
            if end.endswith("Z"):
                eval_time = end
            else:
                # Assume UTC if no timezone
                dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
                eval_time = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            # Default to now if parsing fails
            eval_time = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        self._validate_promql(query)

        try:
            result = await self._query_instant(query, eval_time)
        except Exception as e:
            logger.error(f"Prometheus query failed for incident {incident_id}: {e}")
            return []

        # Parse instant query result
        metrics = []
        data = result.get("data", {})
        result_vec = data.get("result", [])

        for item in result_vec:
            if item.get("metric", {}).get("__name__") == "__name__":
                continue

            metric_name = item["metric"].get("__name__", "")
            if not metric_name:
                continue

            value = 0.0
            if "value" in item:
                # Instant query result format
                value_data = item.get("value", [0, 0])
                if isinstance(value_data, list) and len(value_data) >= 2:
                    try:
                        value = float(value_data[1])
                    except (ValueError, TypeError):
                        value = 0.0

            labels = {k: v for k, v in item.get("labels", {}).items()
                      if k != "__name__"}

            # For instant query, use the timestamp from the result or current time
            at = datetime.now(tz=timezone.utc)
            metrics.append({
                "name": metric_name,
                "value": value,
                "labels": labels,
                "at": at,
            })

        return metrics

    def fetch(self, incident_id: str, start: str, end: str) -> list[dict[str, Any]]:
        """Synchronous wrapper for async fetch."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.fetch_async(incident_id, start, end))