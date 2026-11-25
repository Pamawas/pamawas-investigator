"""Prometheus fetcher for evidence enrichment using instant queries."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from typing import Any

import httpx

from models.models import Metric

logger = logging.getLogger(__name__)


class MetricsFetcher:
    """Interface for fetching Prometheus metrics."""

    def fetch(self, incident_id: str, start: str, end: str) -> list[Metric]:
        """Fetch metrics for an incident. Returns list of Metric samples."""
        raise NotImplementedError


class PrometheusFetcher(MetricsFetcher):
    """Prometheus instant query fetcher for evidence enrichment."""

    def __init__(
        self,
        base_url: str,
        service_label: str = "",
        timeout_seconds: float = 10.0,
        max_response_bytes: int = 1_000_000,
    ):
        self.base_url = base_url.rstrip("/")
        # Optional explicit override for legacy callers. There is intentionally
        # no default: deployment label schemas must not be guessed.
        self.service_label = service_label
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
            raise ValueError("Response size exceeds limit")

        data = response.json()
        if data.get("status") != "success":
            raise ValueError(f"Prometheus error: {data.get('error', 'Unknown error')}")

        return data

    def _build_selector(self, service: str) -> str:
        """Build a label selector for a service name.

        Backslashes and quotes are escaped so a service name cannot break out
        of the string literal and inject LogQL/PromQL.
        """
        safe = service.replace("\\", "\\\\").replace('"', '\\"')
        return "{" + self.service_label + '="' + safe + '"}'

    def _eval_time(self, end: str) -> str:
        """Return the instant-query evaluation time for an end timestamp."""
        try:
            if end.endswith("Z"):
                return end
            dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    async def fetch_async(
        self,
        incident_id: str,
        start: str,
        end: str,
        service: str | None = None,
    ) -> list[Metric]:
        """Fetch Prometheus metrics for an incident using instant query.

        Args:
            incident_id: The incident ID (used for logging only)
            start: Start timestamp (RFC3339, used for context)
            end: End timestamp (RFC3339, used as evaluation time)
            service: The incident's affected service. Required to build a
                truthful selector — without it there is nothing to select on.

        Returns:
            List of Metric samples (the typed model, so `at` is serialized
            consistently with everything else the investigator persists).
        """
        if not service or not self.service_label:
            logger.info(
                "Skipping Prometheus enrichment for incident %s: no explicit selector mapping",
                incident_id,
            )
            return []

        query = self._build_selector(service)

        eval_time = self._eval_time(end)

        self._validate_promql(query)

        try:
            result = await self._query_instant(query, eval_time)
        except (OSError, ValueError, RuntimeError) as e:
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

            labels = {k: v for k, v in item.get("labels", {}).items() if k != "__name__"}

            # For instant query, use the timestamp from the result or current time
            at = datetime.now(tz=UTC)
            metrics.append(
                Metric(
                    name=metric_name,
                    value=value,
                    labels=labels,
                    at=at,
                )
            )

        return metrics

    def fetch(self, incident_id: str, start: str, end: str) -> list[Metric]:
        """Synchronous wrapper for async fetch."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.fetch_async(incident_id, start, end))
