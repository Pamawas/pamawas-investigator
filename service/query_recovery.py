"""Bounded query planning from each provider's discovered schema."""

import json
import logging
import re
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from adapters import LokiError, PrometheusError
from models import IncidentContext, ToolResult

logger = logging.getLogger(__name__)
_METRIC_SELECTOR = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^{}]*\})?$")


def investigation_window(context: IncidentContext) -> tuple[str, str]:
    """Use the incident's historical window, including a short baseline and tail."""
    start = datetime.fromisoformat(context.started_at.replace("Z", "+00:00"))
    end = (
        datetime.fromisoformat(context.resolved_at.replace("Z", "+00:00"))
        if context.resolved_at
        else datetime.now(UTC)
    )
    return (
        (start - timedelta(minutes=5)).isoformat(),
        (end + timedelta(minutes=5)).isoformat(),
    )


class QueryRecovery:
    """Record discovery and data calls so recovery evidence remains auditable."""

    def __init__(self, tools, context: IncidentContext, max_logs: int):
        self.tools = tools
        self.context = context
        self.max_logs = max_logs
        self.calls: list[ToolResult] = []
        self.counter_metric = False

    async def call(self, name: str, arguments: dict, method, *args) -> dict[str, Any]:
        started = time.monotonic()
        try:
            result = await method(*args)
        except (LokiError, PrometheusError, TimeoutError) as exc:
            result = {"status": "error", "error": str(exc)}
        self.calls.append(
            ToolResult(name, arguments, result, (time.monotonic() - started) * 1000)
        )
        return result

    async def selector(self, provider: str) -> str | None:
        """Match incident identities against returned keys and values for this provider."""
        adapter = getattr(self.tools, f"_{provider}_adapter")
        catalog = await self.call(
            f"get_{provider}_labels", {}, adapter.query_labels
        )
        if catalog.get("status") != "success" or not isinstance(catalog.get("data"), list):
            return None
        available = {str(key) for key in catalog["data"]}
        incident_labels: dict[str, str] = {}
        for event in self.context.events or []:
            labels = event.get("labels")
            if isinstance(labels, dict):
                incident_labels.update({str(key): str(value) for key, value in labels.items()})
        identities = list(dict.fromkeys(self.context.affected_services or []))
        keys = sorted(
            available,
            key=lambda key: (
                key not in incident_labels,
                incident_labels.get(key) not in identities,
                identities.index(incident_labels[key])
                if incident_labels.get(key) in identities else len(identities),
                key,
            ),
        )
        # Prefer label keys carried by the incident itself. If the provider
        # does not expose those keys, probe only a small set of identity-like
        # labels instead of enumerating unrelated dimensions such as boot IDs,
        # clusters, and filenames. A missing target is a valid source gap.
        direct_keys = [key for key in keys if key in incident_labels]
        identity_hints = (
            "instance", "host", "hostname", "service", "service_name",
            "job", "pod", "container", "unit", "node", "device", "scope",
        )
        candidate_keys = direct_keys or [
            key for key in keys
            if any(hint in key.lower() for hint in identity_hints)
        ]
        for key in candidate_keys[:4]:
            response = await self.call(
                f"get_{provider}_labels", {"label": key}, adapter.query_label_values, key
            )
            if response.get("status") != "success" or not isinstance(response.get("data"), list):
                continue
            values = {str(value) for value in response["data"]}
            candidates = [*identities]
            if key in incident_labels:
                candidates.append(incident_labels[key])
            matched = next((value for value in candidates if value in values), None)
            if matched is not None:
                return "{" + key + "=" + json.dumps(matched) + "}"
        return None

    async def recover_loki(self) -> None:
        selector = await self.selector("loki")
        if selector is None:
            return
        start, end = investigation_window(self.context)
        arguments = {"logql": selector, "start": start, "end": end, "limit": self.max_logs}
        await self.call(
            "query_loki", arguments, self.tools._loki_adapter.query_range,
            selector, start, end, self.max_logs,
        )

    async def recover_prometheus(self, candidates: list[dict]) -> None:
        selector = await self.selector("prometheus")
        if selector is None:
            return
        start, end = investigation_window(self.context)
        for descriptor in candidates[:3]:
            expression = descriptor["name"] + selector
            if descriptor.get("type") == "counter":
                expression = f"rate({expression}[5m])"
            result = await self.call(
                "query_prometheus", {"promql": expression, "start": start, "end": end},
                self.tools._prometheus_adapter.query_range, expression, start, end,
            )
            result["metric_metadata"] = descriptor
            if result.get("status") == "success" and result.get("data", {}).get("result"):
                return

    async def counter_rate(self, arguments: dict) -> None:
        match = _METRIC_SELECTOR.fullmatch(str(arguments.get("promql", "")))
        if match is None:
            return
        adapter = self.tools._prometheus_adapter
        catalog = await self.call("get_prometheus_metadata", {}, adapter.query_metric_catalog)
        metadata = next(
            (item for item in catalog.get("data", []) if item.get("name") == match.group(1)),
            None,
        )
        if not metadata or metadata.get("type") != "counter":
            return
        self.counter_metric = True
        start, end = investigation_window(self.context)
        expression = f"rate({arguments['promql']}[5m])"
        result = await self.call(
            "query_prometheus", {"promql": expression, "start": start, "end": end},
            adapter.query_range, expression, start, end,
        )
        result["metric_metadata"] = metadata
