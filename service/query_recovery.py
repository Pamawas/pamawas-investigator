"""Bounded query planning from each provider's discovered schema."""

import json
import logging
import math
import re
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from adapters import LokiError, PrometheusError
from models import IncidentContext, ToolResult

logger = logging.getLogger(__name__)
_METRIC_SELECTOR = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^{}]*\})?$")
_METRIC_REFERENCE = re.compile(r"([a-zA-Z_:][a-zA-Z0-9_:]*)\s*\{")
_TOKEN = re.compile(r"[a-z0-9]+")
# OpenMetrics-generated families that are not independent signals on their own.
_NOISY_METRIC_SUFFIXES = ("_bucket", "_created", "_info")
_RELATED_QUERY_ATTEMPT_LIMIT = 6
_MAX_RELATED_SIGNALS = 2
_MAX_SIGNATURE_SIMILARITY = 0.6


def _tokens(value: Any) -> set[str]:
    return set(_TOKEN.findall(str(value or "").lower().replace("_", " ")))


def _context_terms(context: IncidentContext) -> tuple[set[str], set[str]]:
    """Return primary-alert and target-scope terms without assuming a metric schema."""
    primary = _tokens(context.title)
    scope = set()
    for target in context.affected_services or []:
        scope.update(_tokens(target))
    for event in context.events or []:
        labels = event.get("labels")
        if not isinstance(labels, dict):
            continue
        for value in labels.values():
            value_tokens = _tokens(value)
            scope.update(value_tokens)
    return primary, scope


def _descriptor_tokens(descriptor: dict[str, Any]) -> set[str]:
    return _tokens(f"{descriptor.get('name', '')} {descriptor.get('help', '')}")


def _common_catalog_terms(catalog: list[dict[str, Any]]) -> set[str]:
    """Derive generic vocabulary from this catalog instead of a built-in taxonomy."""
    if not catalog:
        return set()
    counts: dict[str, int] = {}
    for descriptor in catalog:
        for token in _descriptor_tokens(descriptor):
            counts[token] = counts.get(token, 0) + 1
    threshold = max(3, (len(catalog) + 9) // 10)
    return {token for token, count in counts.items() if count >= threshold}


def _signature(
    descriptor: dict[str, Any], scope_terms: set[str], common_terms: set[str]
) -> set[str]:
    tokens = _descriptor_tokens(descriptor) - scope_terms - common_terms
    return tokens or (_tokens(descriptor.get("name")) - scope_terms)


def _similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / len(left.union(right))


def _related_metric_candidates(
    catalog: list[dict[str, Any]],
    context: IncidentContext,
    excluded_names: set[str],
) -> list[tuple[dict[str, Any], set[str]]]:
    """Rank target-scoped metrics by metadata quality and semantic distance."""
    primary_terms, scope_terms = _context_terms(context)
    common_terms = _common_catalog_terms(catalog)
    primary_signature = primary_terms | _tokens(" ".join(sorted(excluded_names)))
    ranked = []
    for descriptor in catalog:
        name = str(descriptor.get("name") or "")
        metric_type = str(descriptor.get("type") or "").lower()
        if (
            not name
            or name in excluded_names
            or metric_type not in {"counter", "gauge"}
            or any(name.endswith(suffix) for suffix in _NOISY_METRIC_SUFFIXES)
        ):
            continue
        signature = _signature(descriptor, scope_terms, common_terms)
        if primary_terms.intersection(signature):
            continue
        distance = 1.0 - _similarity(signature, primary_signature)
        name_tokens = _tokens(name)
        score = (
            len(name_tokens.intersection(scope_terms)),
            int(bool(descriptor.get("unit"))),
            int(bool(descriptor.get("help"))),
            int(metric_type == "counter"),
            int(distance * 1000),
        )
        ranked.append((score, name, descriptor, signature))
    ranked.sort(key=lambda item: (*tuple(-part for part in item[0]), item[1]))
    return [(descriptor, signature) for _, _, descriptor, signature in ranked]


def _metric_expression(descriptor: dict[str, Any], selector: str) -> str:
    expression = str(descriptor["name"]) + selector
    if str(descriptor.get("type") or "").lower() == "counter":
        return f"rate({expression}[5m])"
    return expression


def _has_series(result: dict[str, Any]) -> bool:
    return (
        result.get("status") == "success"
        and isinstance(result.get("data"), dict)
        and bool(result["data"].get("result"))
    )


def _has_time_variation(result: dict[str, Any]) -> bool:
    """Related evidence must change in-window, not merely exist as static metadata."""
    if not _has_series(result):
        return False
    for series in result["data"].get("result", []):
        points = series.get("values") if isinstance(series, dict) else None
        if not isinstance(points, list):
            continue
        values = []
        for point in points:
            if not isinstance(point, list) or len(point) < 2:
                continue
            try:
                value = float(point[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        if len(values) >= 2:
            spread = max(values) - min(values)
            tolerance = max(1e-12, max(abs(value) for value in values) * 1e-9)
            if spread > tolerance:
                return True
    return False


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
        self._selector_cache: dict[str, str | None] = {}
        self._metric_catalog: list[dict[str, Any]] | None = None

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
        if provider in self._selector_cache:
            return self._selector_cache[provider]
        adapter = getattr(self.tools, f"_{provider}_adapter")
        catalog = await self.call(
            f"get_{provider}_labels", {}, adapter.query_labels
        )
        if catalog.get("status") != "success" or not isinstance(catalog.get("data"), list):
            self._selector_cache[provider] = None
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
                selector = "{" + key + "=" + json.dumps(matched) + "}"
                self._selector_cache[provider] = selector
                return selector
        self._selector_cache[provider] = None
        return None

    async def metric_catalog(self) -> list[dict[str, Any]]:
        if self._metric_catalog is not None:
            return self._metric_catalog
        adapter = self.tools._prometheus_adapter
        response = await self.call(
            "get_prometheus_metadata", {}, adapter.query_metric_catalog
        )
        data = response.get("data")
        self._metric_catalog = data if isinstance(data, list) else []
        return self._metric_catalog

    async def _query_metric(
        self,
        descriptor: dict[str, Any],
        selector: str,
        start: str,
        end: str,
        role: str,
    ) -> dict[str, Any]:
        expression = _metric_expression(descriptor, selector)
        result = await self.call(
            "query_prometheus",
            {"promql": expression, "start": start, "end": end},
            self.tools._prometheus_adapter.query_range,
            expression,
            start,
            end,
        )
        result["metric_metadata"] = descriptor
        result["signal_role"] = role
        return result

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
        excluded_names = set()
        for descriptor in candidates[:3]:
            name = str(descriptor.get("name") or "")
            if not name:
                continue
            excluded_names.add(name)
            result = await self._query_metric(
                descriptor, selector, start, end, "recovered_primary"
            )
            if _has_series(result):
                break
        await self.recover_related_prometheus(selector, excluded_names)

    async def recover_related_prometheus(
        self,
        selector: str | None = None,
        excluded_names: set[str] | None = None,
        primary_expression: str = "",
    ) -> None:
        """Query up to two changing, semantically distinct target signals."""
        selector = selector or await self.selector("prometheus")
        if selector is None:
            return
        excluded = set(excluded_names or ())
        excluded.update(_METRIC_REFERENCE.findall(primary_expression))
        catalog = await self.metric_catalog()
        start, end = investigation_window(self.context)
        selected_signatures: list[set[str]] = []
        attempts = 0
        for descriptor, signature in _related_metric_candidates(
            catalog, self.context, excluded
        ):
            if attempts >= _RELATED_QUERY_ATTEMPT_LIMIT:
                return
            if any(
                _similarity(signature, selected) > _MAX_SIGNATURE_SIMILARITY
                for selected in selected_signatures
            ):
                continue
            attempts += 1
            result = await self._query_metric(
                descriptor, selector, start, end, "related_candidate"
            )
            if not _has_time_variation(result):
                continue
            result["signal_role"] = "related"
            selected_signatures.append(signature)
            if len(selected_signatures) >= _MAX_RELATED_SIGNALS:
                return

    async def counter_rate(self, arguments: dict) -> None:
        match = _METRIC_SELECTOR.fullmatch(str(arguments.get("promql", "")))
        if match is None:
            return
        adapter = self.tools._prometheus_adapter
        catalog = await self.metric_catalog()
        metadata = next(
            (item for item in catalog if item.get("name") == match.group(1)),
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
        result["signal_role"] = "recovered_primary"
