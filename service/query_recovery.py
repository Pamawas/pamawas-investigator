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
_RELATED_QUERY_ATTEMPT_LIMIT = 6
_MAX_RELATED_SIGNALS = 2
_MAX_SIGNATURE_SIMILARITY = 0.6
_SELECTOR_LABEL_PROBE_LIMIT = 32
_CANDIDATE_AUDIT_LIMIT = 12
_EXACT_LABEL_MATCHER = re.compile(
    r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:\\.|[^"\\])*)"'
)

# Provider-neutral words that describe alert severity, metric encoding, or
# measurement units rather than an incident domain. They must not be enough to
# make two otherwise unrelated metrics "relevant" to each other.
_SEMANTIC_STOP_TERMS = frozenset(
    {
        "alert",
        "bucket",
        "byte",
        "bytes",
        "count",
        "counter",
        "critical",
        "cluster",
        "gauge",
        "high",
        "histogram",
        "incident",
        "info",
        "instance",
        "job",
        "low",
        "namespace",
        "production",
        "second",
        "seconds",
        "service",
        "sum",
        "target",
        "total",
        "unit",
        "value",
        "values",
        "warning",
    }
)


def _tokens(value: Any) -> set[str]:
    return set(_TOKEN.findall(str(value or "").lower().replace("_", " ")))


def metric_semantic_terms(value: Any) -> list[str]:
    """Return ordered domain terms without metric-format or severity noise."""
    terms = []
    for token in _TOKEN.findall(str(value or "").lower().replace("_", " ")):
        normalized = token[:-1] if len(token) > 3 and token.endswith("s") else token
        if len(normalized) < 2 or normalized in _SEMANTIC_STOP_TERMS:
            continue
        terms.append(normalized)
    return terms


def _semantic_tokens(value: Any) -> set[str]:
    return set(metric_semantic_terms(value))


def _context_terms(context: IncidentContext) -> tuple[set[str], set[str]]:
    """Return primary-alert and target-scope terms without assuming a metric schema."""
    primary = _semantic_tokens(context.title)
    scope: set[str] = set()
    for target in context.affected_services or []:
        scope.update(_tokens(target))
    return primary, scope


def _descriptor_tokens(descriptor: dict[str, Any]) -> set[str]:
    return _semantic_tokens(
        f"{descriptor.get('name', '')} {descriptor.get('help', '')}"
    )


def _canonical_metric_catalog(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve duplicate metadata rows independently of provider response order."""
    selected: dict[str, dict[str, Any]] = {}
    for descriptor in catalog:
        if not isinstance(descriptor, dict):
            continue
        name = str(descriptor.get("name") or "")
        if not name:
            continue
        candidate_key = (
            int(bool(descriptor.get("unit"))),
            int(bool(descriptor.get("help"))),
            int(bool(descriptor.get("type"))),
            json.dumps(descriptor, sort_keys=True, default=str),
        )
        current = selected.get(name)
        current_key = (
            int(bool(current.get("unit"))),
            int(bool(current.get("help"))),
            int(bool(current.get("type"))),
            json.dumps(current, sort_keys=True, default=str),
        ) if current is not None else None
        if current_key is None or candidate_key > current_key:
            selected[name] = descriptor
    return [selected[name] for name in sorted(selected)]


def _common_catalog_terms(catalog: list[dict[str, Any]]) -> set[str]:
    """Derive generic vocabulary from this catalog instead of a built-in taxonomy."""
    if not catalog:
        return set()
    counts: dict[str, int] = {}
    for descriptor in catalog:
        for token in _descriptor_tokens(descriptor):
            counts[token] = counts.get(token, 0) + 1
    threshold = max(5, (len(catalog) + 9) // 10)
    return {token for token, count in counts.items() if count >= threshold}


def _signature(
    descriptor: dict[str, Any], scope_terms: set[str], common_terms: set[str]
) -> set[str]:
    return _descriptor_tokens(descriptor) - scope_terms - common_terms


def _similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / len(left.union(right))


def _related_metric_plan(
    catalog: list[dict[str, Any]],
    context: IncidentContext,
    excluded_names: set[str],
) -> tuple[list[tuple[dict[str, Any], set[str]]], dict[str, Any]]:
    """Require incident-domain overlap before ranking related metrics."""
    primary_terms, scope_terms = _context_terms(context)
    common_terms = _common_catalog_terms(catalog)
    primary_signature = (
        primary_terms | _semantic_tokens(" ".join(sorted(excluded_names)))
    ) - common_terms - scope_terms
    ranked = []
    rejected = []
    rejected_total = 0
    for descriptor in catalog:
        name = str(descriptor.get("name") or "")
        if not name or name in excluded_names:
            continue
        signature = _signature(descriptor, scope_terms, common_terms)
        overlap = signature.intersection(primary_signature)
        if not overlap:
            rejected_total += 1
            if len(rejected) < _CANDIDATE_AUDIT_LIMIT:
                rejected.append({
                    "name": name,
                    "reason": "no_incident_domain_overlap",
                })
            continue
        similarity = _similarity(signature, primary_signature)
        score = (
            len(overlap),
            int(similarity * 1000),
            int(bool(descriptor.get("unit"))),
            int(bool(descriptor.get("help"))),
        )
        ranked.append((score, name, descriptor, signature, sorted(overlap)))
    ranked.sort(key=lambda item: (*tuple(-part for part in item[0]), item[1]))
    candidates = [
        (descriptor, signature)
        for _, _, descriptor, signature, _ in ranked
    ]
    audit = {
        "status": "success",
        "_private": True,
        "policy_version": 1,
        "stage": "related_metric_recovery",
        "primary_terms": sorted(primary_signature),
        "accepted": [
            {
                "name": name,
                "domain_overlap": overlap,
                "score": {
                    "domain_overlap": score[0],
                    "semantic_similarity_milli": score[1],
                    "has_unit": score[2],
                    "has_help": score[3],
                },
            }
            for score, name, _, _, overlap in ranked[:_CANDIDATE_AUDIT_LIMIT]
        ],
        "accepted_total": len(ranked),
        "rejected": rejected,
        "rejected_total": rejected_total,
    }
    return candidates, audit


def _related_metric_candidates(
    catalog: list[dict[str, Any]],
    context: IncidentContext,
    excluded_names: set[str],
) -> list[tuple[dict[str, Any], set[str]]]:
    """Compatibility wrapper returning only admitted related candidates."""
    return _related_metric_plan(catalog, context, excluded_names)[0]


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


def _incident_label_values(context: IncidentContext) -> dict[str, list[str]]:
    """Return stable, de-duplicated label values from the incident lifecycle."""
    values_by_key: dict[str, list[str]] = {}
    events = sorted(
        context.events or [],
        key=lambda event: (
            str(event.get("timestamp") or ""),
            str(event.get("id") or ""),
            json.dumps(event.get("labels") or {}, sort_keys=True, default=str),
        ),
    )
    for event in events:
        reference = event.get("condition_ref")
        reference_labels = (
            reference.get("labels")
            if isinstance(reference, dict)
            else getattr(reference, "labels", None)
        )
        for labels in (event.get("labels"), reference_labels):
            if not isinstance(labels, dict):
                continue
            for key, value in labels.items():
                normalized_key = str(key)
                normalized_value = str(value)
                values = values_by_key.setdefault(normalized_key, [])
                if normalized_value not in values:
                    values.append(normalized_value)
    return values_by_key


def _selector_keys(
    available: set[str], values_by_key: dict[str, list[str]], identities: list[str]
) -> list[str]:
    keys = sorted(
        available,
        key=lambda key: (
            key not in values_by_key,
            not any(value in identities for value in values_by_key.get(key, [])),
            key,
        ),
    )
    direct = [key for key in keys if key in values_by_key]
    # When the provider exposes incident-carried keys, a mismatch means it has
    # no coverage for this target. Probing unrelated dimensions would turn a
    # source gap into an inferred mapping. Only inspect the bounded provider
    # schema when there are no shared keys at all.
    return direct or keys


def _matched_target_value(
    response: dict[str, Any],
    identities: list[str],
) -> str | None:
    if response.get("status") != "success" or not isinstance(response.get("data"), list):
        return None
    available_values = {str(value) for value in response["data"]}
    return next((value for value in identities if value in available_values), None)


def _expression_target_coverage(
    expression: str, context: IncidentContext
) -> dict[str, Any] | None:
    """Validate one exact PromQL matcher against an affected target identity."""
    identities = {
        str(value).strip()
        for value in context.affected_services or []
        if str(value).strip()
    }
    for key, encoded in _EXACT_LABEL_MATCHER.findall(expression):
        try:
            value = json.loads(f'"{encoded}"')
        except json.JSONDecodeError:
            continue
        if value in identities:
            return {
                "validated": True,
                "method": "exact_incident_identity",
                "label": key,
                "value": value,
            }
    return None


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
        self._target_coverage: dict[str, dict[str, Any]] = {}
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
            coverage = {
                "validated": False,
                "reason": "provider_label_catalog_unavailable",
            }
            self._target_coverage[provider] = coverage
            catalog["target_coverage"] = coverage
            self._selector_cache[provider] = None
            return None
        available = {str(key) for key in catalog["data"]}
        incident_label_values = _incident_label_values(self.context)
        identities = [
            value
            for value in dict.fromkeys(
                str(item).strip() for item in self.context.affected_services or []
            )
            if value
        ]
        if not identities:
            coverage = {
                "validated": False,
                "reason": "incident_has_no_target_identity",
            }
            self._target_coverage[provider] = coverage
            catalog["target_coverage"] = coverage
            self._selector_cache[provider] = None
            return None
        # Prefer incident-carried keys. Probe the bounded provider schema only
        # when it shares no keys with the incident; no label vocabulary is baked in.
        candidate_keys = _selector_keys(available, incident_label_values, identities)
        for key in candidate_keys[:_SELECTOR_LABEL_PROBE_LIMIT]:
            response = await self.call(
                f"get_{provider}_labels", {"label": key}, adapter.query_label_values, key
            )
            matched = _matched_target_value(response, identities)
            if matched is None:
                continue
            selector = "{" + key + "=" + json.dumps(matched) + "}"
            coverage = {
                "validated": True,
                "method": "exact_incident_identity",
                "label": key,
                "value": matched,
            }
            self._target_coverage[provider] = coverage
            catalog["target_coverage"] = coverage
            self._selector_cache[provider] = selector
            return selector
        coverage = {
            "validated": False,
            "reason": "no_exact_target_label_match",
        }
        self._target_coverage[provider] = coverage
        catalog["target_coverage"] = coverage
        self._selector_cache[provider] = None
        return None

    @property
    def target_coverage(self) -> dict[str, dict[str, Any]]:
        """Return the bounded coverage decision for recovery model context and audit."""
        return {
            provider: dict(coverage)
            for provider, coverage in self._target_coverage.items()
        }

    async def metric_catalog(self) -> list[dict[str, Any]]:
        if self._metric_catalog is not None:
            return self._metric_catalog
        adapter = self.tools._prometheus_adapter
        response = await self.call(
            "get_prometheus_metadata", {}, adapter.query_metric_catalog
        )
        data = response.get("data")
        self._metric_catalog = (
            _canonical_metric_catalog(data) if isinstance(data, list) else []
        )
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
        result["target_coverage"] = self._target_coverage.get(
            "prometheus", {"validated": False, "reason": "selector_not_validated"}
        )
        return result

    async def recover_loki(self) -> None:
        selector = await self.selector("loki")
        if selector is None:
            return
        start, end = investigation_window(self.context)
        arguments = {"logql": selector, "start": start, "end": end, "limit": self.max_logs}
        result = await self.call(
            "query_loki", arguments, self.tools._loki_adapter.query_range,
            selector, start, end, self.max_logs,
        )
        result["signal_role"] = "recovered_target"
        result["target_coverage"] = self._target_coverage.get(
            "loki", {"validated": False, "reason": "selector_not_validated"}
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
        if selector is None:
            selector = await self.selector("prometheus")
        else:
            coverage = _expression_target_coverage(selector, self.context)
            if coverage is None:
                return
            self._target_coverage["prometheus"] = coverage
        if selector is None:
            return
        excluded = set(excluded_names or ())
        excluded.update(_METRIC_REFERENCE.findall(primary_expression))
        catalog = await self.metric_catalog()
        start, end = investigation_window(self.context)
        selected_signatures: list[set[str]] = []
        attempts = 0
        candidates, audit = _related_metric_plan(catalog, self.context, excluded)
        self.calls.append(
            ToolResult(
                "select_prometheus_candidates",
                {"stage": "related_metric_recovery"},
                audit,
                0.0,
            )
        )
        for descriptor, signature in candidates:
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
        target_coverage = _expression_target_coverage(
            str(arguments.get("promql", "")), self.context
        )
        if target_coverage is None:
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
        result["target_coverage"] = target_coverage
