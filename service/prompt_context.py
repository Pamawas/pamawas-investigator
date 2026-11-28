"""Select and budget the context sent to an investigation model."""

import hashlib
import json
import re
from typing import Any

from prometheus_client import Counter, Histogram

from models import IncidentContext
from service.incident_prompt_context import prompt_data_bytes
from service.query_recovery import metric_semantic_terms

INCIDENT_BYTES = 8 * 1024
CATALOG_BYTES = 4 * 1024
CHECKPOINT_BYTES = 4 * 1024
POLICY_BYTES = 16 * 1024
HISTORY_BYTES = 32 * 1024
MODEL_INPUT_BYTES = 64 * 1024

_CATALOG_CANDIDATE_LIMIT = 12
_EXPRESSION_WORD = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")

prompt_size_bytes = Histogram(
    "pamawas_investigator_prompt_input_bytes",
    "Serialized investigation model context and tool definitions in bytes",
    buckets=(1024, 4096, 8192, 16384, 32768, 49152, 65536),
)
prompt_catalog_candidates = Histogram(
    "pamawas_investigator_prompt_catalog_candidates",
    "Provider catalog candidates selected for an investigation prompt",
    buckets=(0, 1, 2, 4, 8, 12, 16),
)
recovery_queries = Counter(
    "pamawas_investigator_query_recovery_total",
    "Telemetry queries issued by bounded recovery",
    ["provider"],
)
rejected_evidence = Counter(
    "pamawas_investigator_rejected_evidence_total",
    "Provider queries or results rejected by deterministic admission checks",
    ["provider", "reason"],
)


def _score_metrics(
    incident: dict[str, Any], metrics: dict[str, Any]
) -> list[tuple[int, str, str, dict]]:
    title = incident.get("incident", {}).get("title", "")
    domain = set(metric_semantic_terms(title))
    referenced = {
        name
        for event in incident.get("events", [])
        for name in _EXPRESSION_WORD.findall(event.get("expression", ""))
    }
    ranked = []
    for name, descriptor in metrics.items():
        if not isinstance(descriptor, dict) or len(name) > 160:
            continue
        exact = name in referenced
        text = f"{name} {descriptor.get('help') or ''}"
        score = 100 * exact + len(domain.intersection(metric_semantic_terms(text)))
        if score > 0:
            reason = "alert_expression" if exact else "incident_domain_overlap"
            ranked.append((score, name, reason, descriptor))
    return sorted(ranked, key=lambda item: (-item[0], item[1]))


def catalog_prompt(
    context: IncidentContext, rendered_incident: str
) -> tuple[str, dict[str, Any]]:
    """Render bounded, deterministic hints from the run's pinned catalog."""
    incident = json.loads(rendered_incident)
    catalog = context.telemetry_catalog or {}
    metrics = catalog.get("prometheus", {})
    ranked = _score_metrics(incident, metrics)
    coverage = context.loki_target_coverage or {}
    payload: dict[str, Any] = {"prometheus": []}
    selected: list[dict[str, Any]] = []
    if coverage.get("validated"):
        payload["loki"] = {key: coverage[key] for key in ("label", "value", "selector")}
        selected.append({
            "provider": "loki", "name": coverage["label"], "score": 1,
            "reason": "live_exact_target_match",
        })
    elif coverage:
        payload["loki"] = {"coverage_gap": coverage.get("reason")}

    for score, name, reason, descriptor in ranked[:_CATALOG_CANDIDATE_LIMIT]:
        candidate = {
            "name": name,
            "type": str(descriptor.get("type") or "")[:48],
            "unit": str(descriptor.get("unit") or "")[:48],
            "help": str(descriptor.get("help") or "")[:160],
        }
        payload["prometheus"].append(candidate)
        if len(prompt_data_bytes(payload)) > CATALOG_BYTES:
            payload["prometheus"].pop()
            continue
        selected.append({
            "provider": "prometheus", "name": name, "score": score, "reason": reason,
        })

    rendered = prompt_data_bytes(payload).decode() if selected or coverage else ""
    return rendered, {
        "catalog_revision_id": context.telemetry_catalog_revision_id,
        "catalog_content_hash": context.telemetry_catalog_content_hash,
        "selected_candidates": selected,
        "prometheus_entries": len(metrics),
        "prometheus_below_relevance_floor": len(metrics) - len(ranked),
        "prometheus_omitted_for_budget": len(ranked) - len(payload["prometheus"]),
        "loki_target_coverage": {
            key: coverage[key] for key in ("validated", "reason", "label", "start", "end", "probes")
            if key in coverage
        },
        "incident_omissions": incident.get("omissions", {}),
        "incident_and_catalog_hash": hashlib.sha256(
            (rendered_incident + "\n" + rendered).encode()
        ).hexdigest(),
    }


def checkpoint_prompt(tools: list[dict], findings: list[dict], budget: int) -> str:
    """Omit complete records until the resume document fits as valid JSON."""
    payload = {
        "prior_tool_executions": [],
        "prior_validated_findings": [],
        "tool_executions_omitted": len(tools),
        "findings_omitted": len(findings),
    }
    for entries, name, omitted in (
        (findings, "prior_validated_findings", "findings_omitted"),
        (tools, "prior_tool_executions", "tool_executions_omitted"),
    ):
        for entry in reversed(entries):
            payload[name].insert(0, entry)
            payload[omitted] -= 1
            if len(prompt_data_bytes(payload)) > budget:
                payload[name].pop(0)
                payload[omitted] += 1
    rendered = prompt_data_bytes(payload)
    return rendered.decode() if len(rendered) <= budget else "{}"


def _encode(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def budget_model_input(
    messages: list[dict], fixed_count: int, tools: list[dict], tool_choice: Any
) -> tuple[list[dict], dict[str, Any]]:
    """Keep a contiguous suffix of complete conversation turns under the request cap."""
    fixed, history = messages[:fixed_count], messages[fixed_count:]
    request = {"messages": fixed, "tools": tools, "tool_choice": tool_choice}
    base_bytes = len(_encode(request))
    if base_bytes > MODEL_INPUT_BYTES:
        raise RuntimeError("Fixed investigation context exceeds model input budget")
    remaining = min(HISTORY_BYTES, MODEL_INPUT_BYTES - base_bytes)
    boundaries = [0, *(i for i, msg in enumerate(history) if i and msg["role"] == "assistant")]
    start = len(history)
    for boundary in reversed(boundaries):
        entries = history[boundary:start]
        cost = sum(len(_encode(msg)) + 1 for msg in entries)
        if cost > remaining:
            break
        remaining -= cost
        start = boundary
    bounded = [*fixed, *history[start:]]
    encoded = _encode({**request, "messages": bounded})
    return bounded, {
        "input_bytes": len(encoded),
        "input_budget_bytes": MODEL_INPUT_BYTES,
        "input_hash": hashlib.sha256(encoded).hexdigest(),
        "history_budget_bytes": HISTORY_BYTES,
        "history_messages_omitted": start,
    }
