"""Main Investigator Service."""

import asyncio
import itertools
import json
import logging
import math
import os
import re
import time
import uuid
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta
from datetime import timezone as tz
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import openai
import psycopg2
from openai import OpenAI

from adapters import (
    ConditionEvaluation,
    ConditionEvaluatorRegistry,
    ConditionState,
    DeploymentAdapterError,
    GrafanaAlertingAdapter,
    GrafanaAlertingConfig,
    LokiError,
    PrometheusError,
    RelatedIncidentsError,
)
from config import Config
from metrics import (
    increment_db_errors,
    increment_deadline_exceeded,
    increment_findings_by_type,
    increment_forced_submission,
    increment_investigations,
    increment_max_tool_calls_reached,
)
from models import (
    ConditionReference,
    EvidenceSource,
    EvidenceType,
    Finding,
    IncidentContext,
    InvestigationState,
    LogEntry,
    Metric,
    ToolResult,
)
from service.daily_summary import write_day_summary as _write_day_summary
from service.evidence_policy import EVIDENCE_POLICY
from service.evidence_summary import (
    alert_observation,
    log_observations,
    metric_observations,
    numeric_points,
)
from service.persistence import InvestigationPersistence
from service.query_recovery import QueryRecovery, metric_semantic_terms
from tools import InvestigatorTools, ToolRegistry

logger = logging.getLogger(__name__)

PROMPT_VERSION = "pamawas-investigator-v23"

_FULL_INVESTIGATION_DATA_TOOLS = ("query_prometheus", "query_loki")
_VERIFICATION_DATA_TOOLS = ("query_prometheus",)
_DISCOVERY_TO_DATA_TOOL = {
    "get_prometheus_labels": "query_prometheus",
    "get_loki_labels": "query_loki",
}
_MAX_GROUNDED_FINDINGS_PER_SOURCE = 3
_SINGLE_QUOTED_LOGQL_LITERAL = re.compile(r"'((?:\\.|[^'\\])*)'")
_LOGQL_SELECTOR = re.compile(r"\{([^{}]*)\}")
_LOGQL_MATCHER = re.compile(
    r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(=~|!~|=|!=)\s*"((?:\\.|[^"\\])*)"'
)
_CAUSAL_FINDING_TYPES = (EvidenceType.LIKELY_CAUSE, EvidenceType.HYPOTHESIS)
_PROCESS_ONLY_UNKNOWN_PREFIXES = (
    "Investigation ended without a stated conclusion",
    "Investigation ended without a substantive conclusion",
)
_PRIVATE_RESULT_KEYS = frozenset({"_candidate_selection_audit"})
_RECOVERY_SIGNAL_ROLES = frozenset(
    {"recovered_primary", "recovered_target", "related", "related_candidate"}
)
_ACCEPTED_RECOVERY_SIGNAL_ROLES = frozenset(
    {"recovered_primary", "recovered_target", "related"}
)
_CANDIDATE_AUDIT_LIMIT = 12

_RELEVANCE_STOPWORDS = {
    "alert", "critical", "high", "incident", "production", "service", "warning",
}


def _relevance_terms(value: Any) -> set[str]:
    terms = set()
    for token in re.findall(r"[a-z0-9]+", str(value or "").lower()):
        if len(token) < 2 or token in _RELEVANCE_STOPWORDS:
            continue
        terms.add(token[:-1] if len(token) > 3 and token.endswith("s") else token)
    return terms


def _incident_relevance_terms(
    context: IncidentContext | None,
) -> tuple[set[str], set[str]]:
    if context is None:
        return set(), set()
    title_terms = _relevance_terms(context.title)
    target_terms = {
        token
        for target in context.affected_services or []
        for token in _relevance_terms(target)
    }
    return title_terms, target_terms


def _prometheus_metric_names(expression: str) -> list[str]:
    names = re.findall(
        r"(?<![a-zA-Z0-9_:])([a-zA-Z_:][a-zA-Z0-9_:]*)\s*(?=\{|\[)",
        expression,
    )
    if not names and re.fullmatch(r"[a-zA-Z_:][a-zA-Z0-9_:]*", expression.strip()):
        names = [expression.strip()]
    return list(dict.fromkeys(names))


def _prometheus_search_document(
    call: ToolResult, context: IncidentContext | None
) -> str:
    expression = str(call.arguments.get("promql") or "")
    parts = [expression]
    result_metadata = (
        call.result.get("metric_metadata", {})
        if isinstance(call.result, dict)
        else {}
    )
    if isinstance(result_metadata, dict):
        parts.append(json.dumps(result_metadata, sort_keys=True, default=str))
    catalog = (context.telemetry_catalog or {}).get("prometheus", {}) if context else {}
    for name in _prometheus_metric_names(expression):
        descriptor = catalog.get(name)
        if isinstance(descriptor, dict):
            parts.append(str(descriptor.get("_search_text") or ""))
            parts.append(json.dumps(descriptor, sort_keys=True, default=str))
    return " ".join(parts)


def _relevance_score(document: str, context: IncidentContext | None) -> int:
    title_terms, target_terms = _incident_relevance_terms(context)
    document_terms = _relevance_terms(document)
    # Title matches identify the incident domain/symptom. Target matches scope
    # the observation but cannot by themselves outrank a domain match.
    return (
        4 * len(title_terms & document_terms)
        + 6 * len(target_terms & document_terms)
    )


def _relative_relevance_floor(highest: int) -> int:
    if highest >= 8:
        return highest - 3
    if highest > 0:
        return highest
    # A zero maximum means none of the returned telemetry shares an incident
    # domain or target term. Treat that as no eligible evidence rather than
    # admitting every arbitrary call at the same zero score.
    return 1


def _prometheus_relevance_score(
    call: ToolResult, context: IncidentContext | None
) -> int:
    """Score PromQL against the incident and the persisted live metric catalog."""
    expression = str(call.arguments.get("promql") or "")
    if not expression:
        return -1
    if context is None:
        return 0
    return _relevance_score(_prometheus_search_document(call, context), context)


def _select_relevant_prometheus_calls(
    calls: list[ToolResult], context: IncidentContext | None, limit: int
) -> list[ToolResult]:
    scored = [
        (index, _prometheus_relevance_score(call, context), call)
        for index, call in enumerate(calls)
    ]
    scored = [item for item in scored if item[1] >= 0]
    if not scored:
        return []
    highest = max(item[1] for item in scored)
    floor = _relative_relevance_floor(highest)
    selected = [item for item in scored if item[1] >= floor]
    selected.sort(key=lambda item: (-item[1], item[0]))
    return [item[2] for item in selected[:limit]]


def _loki_relevance_score(call: ToolResult, context: IncidentContext | None) -> int:
    """Score returned live log content without deployment-specific vocabularies."""
    if context is None:
        return 0
    result = call.result if isinstance(call.result, dict) else {}
    streams = result.get("data", {}).get("result", [])
    payload_parts: list[str] = []
    for stream in streams if isinstance(streams, list) else []:
        if not isinstance(stream, dict):
            continue
        payload_parts.append(json.dumps(stream.get("stream", {}), sort_keys=True))
        payload_parts.extend(
            str(point[1])
            for point in stream.get("values", [])
            if isinstance(point, list) and len(point) >= 2
        )
    document = " ".join([str(call.arguments.get("logql") or ""), *payload_parts])
    return _relevance_score(document, context)


def _select_relevant_loki_calls(
    calls: list[ToolResult], context: IncidentContext | None, limit: int
) -> list[ToolResult]:
    scored = [
        (index, _loki_relevance_score(call, context), call)
        for index, call in enumerate(calls)
    ]
    scored = [item for item in scored if item[1] >= 0]
    if not scored:
        return []
    floor = _relative_relevance_floor(max(item[1] for item in scored))
    scored = [item for item in scored if item[1] >= floor]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [item[2] for item in scored[:limit]]


def _select_condition_reference(
    events: list[dict[str, Any]],
) -> tuple[ConditionReference | None, str | None]:
    """Select one immutable source condition from normalized alert events."""
    references: dict[tuple[Any, ...], ConditionReference] = {}
    for event in events:
        raw_reference = event.get("condition_ref")
        if raw_reference is None:
            continue
        try:
            reference = (
                raw_reference
                if isinstance(raw_reference, ConditionReference)
                else ConditionReference.from_dict(raw_reference)
            )
        except ValueError as exc:
            event_id = str(event.get("id") or "unknown event")
            return None, f"{event_id} carries an invalid condition reference ({exc})"
        # Labels are bounded source metadata, not condition identity. Legacy
        # backfills and new notifications may retain different label subsets
        # for the same source-owned instance; that must not create a false
        # conflict. The required locator fields remain exact.
        identity = (
            reference.version,
            reference.source,
            reference.evaluator,
            reference.resource_type,
            reference.resource_id,
            reference.instance_id,
            reference.tenant_id,
        )
        references.setdefault(identity, reference)
    if not references:
        return None, "no typed condition reference was attached to the incident"
    if len(references) > 1:
        return None, "the incident contains conflicting condition references"
    return next(iter(references.values())), None


def _condition_reference_prompt(context: IncidentContext | None) -> str:
    if context is None or context.condition_ref is None:
        return (
            "No authoritative condition reference is available. Telemetry may explain the "
            "incident, but it must not be used to declare recovery.\n"
        )
    reference = context.condition_ref
    state = context.condition_state or "not evaluated"
    return (
        f"Authoritative condition: {reference.source}/{reference.evaluator}, "
        f"{reference.resource_type}={reference.resource_id}, instance={reference.instance_id}. "
        f"Current evaluator state: {state}. Telemetry may explain this condition but cannot "
        "override its source state or independently establish recovery.\n"
    )


def _condition_evaluation_finding(
    reference: ConditionReference | None,
    evaluation: ConditionEvaluation,
) -> Finding:
    detail = (
        f"condition evaluator={evaluation.source}/{evaluation.evaluator}; "
        f"reason={evaluation.reason_code}"
    )
    if reference is not None:
        detail += (
            f"; resource={reference.resource_type}:{reference.resource_id}; "
            f"instance={reference.instance_id}"
        )
    if evaluation.state == ConditionState.NORMAL and evaluation.authoritative:
        return Finding(
            type=EvidenceType.FACT,
            content=(
                "The authoritative alert source reports the exact condition instance as "
                "normal, which conflicts with the incident remaining open."
            ),
            source=EvidenceSource.ALERT,
            source_detail=detail,
            confidence=1.0,
        )
    if evaluation.state in {ConditionState.FIRING, ConditionState.PENDING}:
        return Finding(
            type=EvidenceType.FACT,
            content=(
                "The authoritative alert source reports the exact condition instance as "
                f"{evaluation.state.value}; recovery is not established."
            ),
            source=EvidenceSource.ALERT,
            source_detail=detail,
            confidence=1.0,
        )
    return Finding(
        type=EvidenceType.UNKNOWN,
        content=(
            "The authoritative condition state could not establish recovery: "
            f"{evaluation.detail}"
        ),
        source=EvidenceSource.SYSTEM,
        source_detail=detail,
        confidence=0.0,
    )

# Use datetime.UTC from Python 3.11+, fallback for older versions
try:
    from datetime import UTC
except ImportError:
    UTC = tz.utc


# Openers that mean the model is narrating its process rather than stating a
# conclusion. A text-only reply that begins with one of these is working notes,
# not a finding: rendered under "🎯 Likely cause" on a critical alert card it
# read as a wall of chain-of-thought ("Let me analyze the data more carefully.
# I need to convert the Unix timestamps to understand the timeline...").
_REASONING_OPENERS = (
    "let me",
    "i need to",
    "i should",
    "i will",
    "first, i",
    "now let me",
    "looking at",
    "let's",
    "to understand",
    "to determine",
    "i'll",
    "we need to",
)


# Longest text that can still be a finding. A finding is a claim — a sentence or
# two — not an essay. This also bounds what a report has to render.
MAX_FINDING_CHARS = 1200

# Phrases that appear when the model narrates its process rather than stating a
# result: "Let me check...", "Wait —", "Actually, let me reconsider".
_PROCESS_MARKERS = (
    "let me",
    "let's ",
    "i need to",
    "i should ",
    "i'll ",
    "i will ",
    "wait —",
    "wait,",
    "hmm",
    "actually,",
    "but actually",
    "let me step back",
    "the key question is",
    "i already ",
    "let me think",
    "let me reconsider",
    "to summarize what i",
    "checking ",
)


def _is_reasoning_preamble(text: str) -> bool:
    """True when a reply is process narration rather than a stated conclusion.

    The first version of this only looked at the opening words and then gave up
    if the text mentioned "likely cause" anywhere. That was trivially defeated:
    a 6,292-character essay of working notes began with "Let me organize what I
    know so far:" and contained the phrase "what the most likely cause is", so
    the conclusion check passed and the whole essay was stored as a hypothesis
    and published as the likely cause on a critical card.

    Length is the reliable signal. A finding is one or two sentences; anything
    long enough to contain a reconsidered line of reasoning is reasoning, no
    matter which phrases it happens to include.
    """
    lowered = " ".join((text or "").lower().split())
    if not lowered:
        return False

    # An essay is not a finding. This alone catches the case above.
    if len(lowered) > MAX_FINDING_CHARS:
        return True

    opens_with_narration = any(lowered.startswith(opener) for opener in _REASONING_OPENERS)
    marker_hits = sum(1 for marker in _PROCESS_MARKERS if marker in lowered)

    if opens_with_narration:
        # A finding is the conclusion, never narration about the next query.
        # Short working notes were the production escape hatch: "Let me query
        # Loki..." contains only one marker and is under 400 characters.
        return True

    # Not opening with narration, but visibly reasoning throughout.
    return marker_hits >= 3


def _plain_typography(text: str) -> str:
    """Replace the punctuation the model reaches for with plain punctuation.

    Em and en dashes are the clearest writing tell there is: they show up in
    generated prose far more often than in anything a person types, and they
    render badly in a proportional-font chat card. The prompt asks for plain
    sentences, but a style instruction is a preference, not a constraint, so the
    typography is also normalized where every finding passes through.
    """
    if not text:
        return text
    cleaned = re.sub(r"\s*[\N{EM DASH}\N{EN DASH}\N{HORIZONTAL BAR}]\s*", ", ", text)
    cleaned = re.sub(r",\s*,", ",", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip()


def _normalize_finding(finding: Finding) -> Finding:
    """Apply the rules every finding must satisfy before it is stored.

    Typography: no em or en dashes, which are the clearest generated-prose tell
    and render as a stray line in a proportional font.

    Confidence on an UNKNOWN: zero. The model was submitting `unknown` findings
    at 0.9 confidence — "I am 90% confident that I do not know" — and the report
    printed `UNKNOWN (90%)`, which reads as a strong claim about a non-answer.
    Confidence belongs to a claim; an UNKNOWN is the absence of one.
    """
    content = _plain_typography(finding.content)
    confidence = finding.confidence
    if finding.type == EvidenceType.UNKNOWN:
        confidence = 0.0
    if content == finding.content and confidence == finding.confidence:
        return finding
    return replace(finding, content=content, confidence=confidence)


def _is_trigger_restatement(finding: Finding, incident_id: str) -> bool:
    """Return whether a finding merely restates the incident that triggered the run.

    The current incident is input to the investigation, not newly gathered
    evidence. Related-incident lookups remain useful because they add recurrence
    information that was not already guaranteed by receiving this incident.
    """
    if finding.type != EvidenceType.FACT or finding.source != EvidenceSource.INCIDENTS:
        return False

    detail = (finding.source_detail or "").lower()
    if any(
        marker in detail
        for marker in (
            "get_related_incidents",
            "related incident",
            "post-trigger",
            "lifecycle event",
        )
    ):
        return False

    referenced_ids = set(re.findall(r"inc_[a-zA-Z0-9-]+", finding.source_detail or ""))
    if any(referenced_id != incident_id for referenced_id in referenced_ids):
        return False
    return True


def _drop_trigger_restatements(
    findings: list[Finding], incident_id: str
) -> list[Finding]:
    """Remove tautological facts copied from the triggering incident payload."""
    kept: list[Finding] = []
    dropped = 0
    for finding in findings:
        if _is_trigger_restatement(finding, incident_id):
            dropped += 1
            logger.warning(
                "Discarding current-incident restatement from findings: %s",
                finding.content[:160],
            )
            continue
        kept.append(finding)
    if dropped and not kept:
        kept.append(
            Finding(
                type=EvidenceType.UNKNOWN,
                content=(
                    "The investigation produced no new evidence beyond the triggering "
                    "incident metadata."
                ),
                source=EvidenceSource.SYSTEM,
                source_detail="current-incident restatement filter",
                confidence=0.0,
            )
        )
    return kept


def _prometheus_has_numeric_samples(result: Any) -> bool:
    """Return whether a Prometheus response contains usable numeric evidence."""
    if not isinstance(result, dict) or result.get("status") != "success":
        return False
    if result.get("counter_only"):
        return False
    data = result.get("data")
    if not isinstance(data, dict):
        return False
    series = data.get("result")
    if not isinstance(series, list):
        return False

    return any(numeric_points(item) for item in series)


def _prometheus_metric_candidate_plan(
    promql: str,
    metric_catalog: Any,
    limit: int = 12,
    context: IncidentContext | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Admit only real metrics with positive non-generic query-domain overlap."""
    metric_names = _prometheus_metric_names(str(promql))
    # Selector labels and values describe scope, not the metric domain. Using
    # them for admission recreated the production bug where a shared cluster
    # label made unrelated Argo CD and Kubernetes metrics look relevant.
    query_terms = metric_semantic_terms(" ".join(metric_names))
    incident_terms = metric_semantic_terms(context.title) if context else []
    required_terms = (
        set(query_terms).intersection(incident_terms)
        if context is not None
        else set(query_terms)
    )
    query_counts = Counter(query_terms)
    query_bigrams = set(itertools.pairwise(query_terms))
    normalized_query = "_".join(query_terms)

    def normalize(item: Any) -> dict[str, str] | None:
        if isinstance(item, str) and item.strip():
            return {"name": item, "type": "unknown", "help": "", "unit": ""}
        if not isinstance(item, dict) or not str(item.get("name") or "").strip():
            return None
        return {
            "name": str(item["name"]),
            "type": str(item.get("type") or "unknown"),
            "help": str(item.get("help") or ""),
            "unit": str(item.get("unit") or ""),
        }

    normalized = [
        normalize(item) for item in metric_catalog
    ] if isinstance(metric_catalog, list) else []
    candidates: dict[str, dict[str, str]] = {}
    for item in normalized:
        if item is None:
            continue
        previous = candidates.get(item["name"])
        quality = (
            int(item["type"].lower() != "unknown"),
            int(bool(item["unit"])),
            int(bool(item["help"])),
            item["type"],
            item["unit"],
            item["help"],
        )
        previous_quality = (
            int(previous["type"].lower() != "unknown"),
            int(bool(previous["unit"])),
            int(bool(previous["help"])),
            previous["type"],
            previous["unit"],
            previous["help"],
        ) if previous is not None else None
        if previous_quality is None or quality > previous_quality:
            candidates[item["name"]] = item

    ranked: list[
        tuple[tuple[int, int, int, int], str, dict[str, str], list[str]]
    ] = []
    rejected: list[dict[str, str]] = []
    rejected_total = 0
    query_term_set = set(query_terms)
    for item in candidates.values():
        candidate_terms = metric_semantic_terms(
            f"{item['name']} {item['help']} {item['unit']}"
        )
        overlap = sorted(required_terms.intersection(candidate_terms))
        if not overlap:
            rejected_total += 1
            if len(rejected) < _CANDIDATE_AUDIT_LIMIT:
                rejected.append(
                    {
                        "name": item["name"],
                        "reason": "no_incident_domain_overlap",
                    }
                )
            continue
        score = (
            sum(query_counts[term] for term in set(candidate_terms)),
            len(overlap),
            len(query_bigrams.intersection(itertools.pairwise(candidate_terms))),
            int(
                SequenceMatcher(
                    None, normalized_query, "_".join(candidate_terms)
                ).ratio()
                * 1000
            ),
        )
        ranked.append((score, item["name"], item, overlap))

    ranked.sort(key=lambda item: (*(-part for part in item[0]), item[1]))
    selected = [item for _, _, item, _ in ranked[:limit]]
    audit = {
        "policy_version": 1,
        "stage": "empty_prometheus_query_recovery",
        "catalog_revision_id": (
            context.telemetry_catalog_revision_id if context else None
        ),
        "catalog_content_hash": (
            context.telemetry_catalog_content_hash if context else None
        ),
        "query_domain_terms": sorted(query_term_set),
        "incident_domain_terms": sorted(set(incident_terms)),
        "required_domain_terms": sorted(required_terms),
        "accepted": [
            {
                "name": name,
                "domain_overlap": overlap,
                "score": {
                    "weighted_domain_overlap": score[0],
                    "distinct_domain_overlap": score[1],
                    "ordered_domain_overlap": score[2],
                    "semantic_similarity_milli": score[3],
                },
            }
            for score, name, _, overlap in ranked[:_CANDIDATE_AUDIT_LIMIT]
        ],
        "accepted_total": len(ranked),
        "rejected": rejected,
        "rejected_total": rejected_total,
    }
    return selected, audit


def _prometheus_metric_candidates(
    promql: str,
    metric_catalog: Any,
    limit: int = 12,
    context: IncidentContext | None = None,
) -> list[dict[str, str]]:
    """Compatibility wrapper returning only admitted retry candidates."""
    return _prometheus_metric_candidate_plan(promql, metric_catalog, limit, context)[0]


def _public_tool_result(result: Any) -> Any:
    """Remove persisted harness audits before a result enters model context."""
    if not isinstance(result, dict):
        return result
    return {
        key: value
        for key, value in result.items()
        if key not in _PRIVATE_RESULT_KEYS and key != "_private"
    }


def _is_private_tool_call(call: ToolResult) -> bool:
    return isinstance(call.result, dict) and call.result.get("_private") is True


def _has_admissible_recovery_coverage(call: ToolResult) -> bool:
    """Fail closed for harness-recovered signals without exact target coverage."""
    if not isinstance(call.result, dict):
        return False
    role = call.result.get("signal_role")
    if role not in _RECOVERY_SIGNAL_ROLES:
        return True
    if role not in _ACCEPTED_RECOVERY_SIGNAL_ROLES:
        return False
    coverage = call.result.get("target_coverage")
    return isinstance(coverage, dict) and coverage.get("validated") is True


def _successful_tool_calls(
    tool_calls: list[ToolResult], tool_name: str
) -> list[ToolResult]:
    """Return successful executions of one data-query tool."""
    return [
        call
        for call in tool_calls
        if call.tool_name == tool_name
        and isinstance(call.result, dict)
        and call.result.get("status") == "success"
        and _has_admissible_recovery_coverage(call)
        and (
            tool_name != "query_prometheus"
            or _prometheus_has_numeric_samples(call.result)
        )
    ]


def _data_query_identity(call: ToolResult) -> tuple[str, str]:
    """Identify a telemetry signal independently of its requested time range."""
    argument = "promql" if call.tool_name == "query_prometheus" else "logql"
    expression = " ".join(str(call.arguments.get(argument, "")).split())
    return call.tool_name, expression


def _representative_distinct_query_calls(
    calls: list[ToolResult], argument: str
) -> list[ToolResult]:
    """Keep one useful timeline per expression, preferring the widest window."""
    order: list[str] = []
    selected: dict[str, tuple[float, ToolResult]] = {}
    for call in calls:
        expression = " ".join(str(call.arguments.get(argument, "")).split())
        try:
            start = datetime.fromisoformat(
                str(call.arguments.get("start", "")).replace("Z", "+00:00")
            )
            end = datetime.fromisoformat(
                str(call.arguments.get("end", "")).replace("Z", "+00:00")
            )
            span = max(0.0, (end - start).total_seconds())
        except (TypeError, ValueError):
            span = 0.0
        if expression not in selected:
            order.append(expression)
        previous = selected.get(expression)
        if previous is None or span >= previous[0]:
            selected[expression] = (span, call)
    return [selected[expression][1] for expression in order]


def _required_data_tools(verification: bool = False) -> tuple[str, ...]:
    """Return data-query tools required before findings may be accepted."""
    return _VERIFICATION_DATA_TOOLS if verification else _FULL_INVESTIGATION_DATA_TOOLS


def _missing_required_data_tools(
    tool_calls: list[ToolResult], verification: bool = False
) -> tuple[str, ...]:
    """Return data queries still needed before findings may be accepted.

    A full investigation needs usable evidence from at least one relevant
    telemetry provider, not successful coverage from every provider. It should
    gather two distinct data observations when possible so the alert signal is
    not mistaken for its own cause. Those observations may both come from the
    same provider. Verification remains Prometheus-specific.
    """
    successful: list[ToolResult] = []
    seen: set[tuple[str, str]] = set()
    for call in tool_calls:
        if not (
            call.tool_name in _required_data_tools(verification)
            and isinstance(call.result, dict)
            and call.result.get("status") == "success"
            and _has_admissible_recovery_coverage(call)
            and (
                call.tool_name != "query_prometheus"
                or _prometheus_has_numeric_samples(call.result)
            )
        ):
            continue
        signature = _data_query_identity(call)
        if signature in seen:
            continue
        seen.add(signature)
        successful.append(call)

    if verification:
        successful_names = {call.tool_name for call in successful}
        return tuple(
            tool_name
            for tool_name in _required_data_tools(True)
            if tool_name not in successful_names
        )
    if len(successful) >= 2:
        return ()
    if successful:
        # Ask the provider that already has target coverage for one distinct
        # causal query instead of forcing an unrelated provider to succeed.
        return (successful[-1].tool_name,)
    return _FULL_INVESTIGATION_DATA_TOOLS


def _pending_data_tools(state: InvestigationState, verification: bool) -> tuple[str, ...]:
    """Return providers still owed an attempt before the report can be written."""
    missing = _missing_required_data_tools(state.tool_calls, verification)
    if verification:
        return missing
    return tuple(name for name in missing if name not in state.exhausted_data_tools)


def _exhaustion_reason(
    exhausted_by: str,
    tool_calls: list[ToolResult],
    verification: bool,
) -> str:
    """Describe why an investigation ended without accepted findings."""
    if exhausted_by == "deadline":
        return "Investigation deadline exceeded"
    if exhausted_by == "required_data":
        missing = _missing_required_data_tools(tool_calls, verification)
        return (
            "Investigation reached its submission reserve without successful "
            f"required data queries: {', '.join(missing)}"
        )
    return "Investigation exhausted its turn budget"


def _metric_from_series(series: Any, query_name: str) -> Metric | None:
    """Convert the latest sample of one Prometheus result series."""
    if not isinstance(series, dict):
        return None
    raw_labels = series.get("metric")
    labels = dict(raw_labels) if isinstance(raw_labels, dict) else {}
    name = str(labels.pop("__name__", query_name))
    points = series.get("values")
    if not isinstance(points, list) or not points:
        point = series.get("value")
        points = [point] if isinstance(point, list) else []
    if not points:
        return None

    point = points[-1]
    if not isinstance(point, list) or len(point) < 2:
        return None
    try:
        at = datetime.fromtimestamp(float(point[0]), tz=UTC)
        value = float(point[1])
    except (OSError, OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None

    normalized_labels = {str(key): str(value) for key, value in labels.items()}
    return Metric(name=name, value=value, labels=normalized_labels, at=at)


def _metrics_from_tool_calls(
    tool_calls: list[ToolResult], max_items: int = 1000
) -> list[Metric]:
    """Extract bounded metric evidence from the queries the investigator executed.

    The query result already carries the data lake's real metric and label schema.
    Reusing it avoids a second enrichment query built from a deployment-specific
    selector. One latest sample per returned series is enough for report evidence;
    the complete response remains available in tool_executions.
    """
    samples: list[Metric] = []
    seen: set[tuple[Any, ...]] = set()

    for call in _successful_tool_calls(tool_calls, "query_prometheus"):
        data = call.result.get("data")
        if not isinstance(data, dict):
            continue
        result = data.get("result")
        if not isinstance(result, list):
            continue

        query_name = str(call.arguments.get("promql") or "prometheus_query")
        for series in result:
            sample = _metric_from_series(series, query_name)
            if sample is None:
                continue
            identity = (
                sample.name,
                tuple(sorted(sample.labels.items())),
                sample.at,
                sample.value,
            )
            if identity in seen:
                continue
            seen.add(identity)
            samples.append(sample)
            if len(samples) >= max_items:
                return samples

    return samples


def _log_entry_from_point(point: Any, stream: str) -> LogEntry | None:
    """Convert one Loki result point, accepting nanosecond or second timestamps."""
    if not isinstance(point, list) or len(point) < 2:
        return None
    try:
        raw_timestamp = float(point[0])
        seconds = (
            raw_timestamp / 1_000_000_000
            if abs(raw_timestamp) > 1_000_000_000_000
            else raw_timestamp
        )
        timestamp = datetime.fromtimestamp(seconds, tz=UTC)
    except (OSError, OverflowError, TypeError, ValueError):
        return None
    return LogEntry(timestamp=timestamp, stream=stream, line=str(point[1]))


def _log_entries_from_stream(stream_result: Any) -> list[LogEntry]:
    """Convert all valid points returned for one schema-independent Loki stream."""
    if not isinstance(stream_result, dict):
        return []
    raw_stream = stream_result.get("stream")
    stream = json.dumps(
        raw_stream if isinstance(raw_stream, dict) else {},
        sort_keys=True,
        separators=(",", ":"),
    )
    values = stream_result.get("values")
    if not isinstance(values, list):
        return []
    entries = [_log_entry_from_point(point, stream) for point in values]
    return [entry for entry in entries if entry is not None]


def _logs_from_tool_calls(
    tool_calls: list[ToolResult], max_items: int = 1000
) -> list[LogEntry]:
    """Extract bounded log evidence while preserving the returned stream schema."""
    entries: list[LogEntry] = []
    seen: set[tuple[datetime, str, str]] = set()

    for call in _successful_tool_calls(tool_calls, "query_loki"):
        data = call.result.get("data")
        if not isinstance(data, dict):
            continue
        result = data.get("result")
        if not isinstance(result, list):
            continue

        for stream_result in result:
            for entry in _log_entries_from_stream(stream_result):
                identity = (entry.timestamp, entry.stream, entry.line)
                if identity in seen:
                    continue
                seen.add(identity)
                entries.append(entry)
                if len(entries) >= max_items:
                    return entries

    return entries


def _pod_timezone():
    """Return the pod-configured timezone, with a neutral UTC fallback."""
    timezone_name = os.getenv("TZ", "UTC")
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        logger.warning("Invalid TZ %r; formatting investigation evidence in UTC", timezone_name)
        return UTC


def _format_query_timestamp(value: Any) -> str:
    """Render an RFC3339 query boundary in the pod timezone."""
    rendered = str(value or "")
    try:
        parsed = datetime.fromisoformat(rendered.replace("Z", "+00:00"))
    except ValueError:
        return rendered
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(_pod_timezone()).strftime("%Y-%m-%d %H:%M:%S %Z")


def _normalize_logql_literals(logql: str) -> str:
    """Convert unsupported single-quoted LogQL literals without changing selectors."""
    return _SINGLE_QUOTED_LOGQL_LITERAL.sub(
        lambda match: json.dumps(match.group(1), ensure_ascii=False),
        logql,
    )


def _logql_matchers(logql: str) -> list[tuple[str, str, str]]:
    """Extract bounded selector matchers from normalized LogQL."""
    selector = _LOGQL_SELECTOR.search(logql)
    if selector is None:
        return []
    matchers: list[tuple[str, str, str]] = []
    for match in _LOGQL_MATCHER.finditer(selector.group(1)):
        raw_value = match.group(3)
        try:
            value = json.loads(f'"{raw_value}"')
        except json.JSONDecodeError:
            value = raw_value
        matchers.append((match.group(1), match.group(2), str(value)))
    return matchers[:8]


def _loki_has_entries(result: Any) -> bool:
    """Return whether a Loki response contains at least one log entry."""
    if not isinstance(result, dict) or result.get("status") != "success":
        return False
    data = result.get("data")
    streams = data.get("result") if isinstance(data, dict) else None
    if not isinstance(streams, list):
        return False
    return any(
        isinstance(stream, dict)
        and isinstance(stream.get("values"), list)
        and bool(stream["values"])
        for stream in streams
    )


def _loki_value_matches(operator: str, expected: str, values: list[str]) -> bool:
    """Validate one positive LogQL matcher against discovered label values."""
    if operator == "=":
        return expected in values
    if operator == "=~":
        try:
            pattern = re.compile(expected)
        except re.error:
            return False
        return any(pattern.search(value) is not None for value in values)
    return True


def _query_window(call: ToolResult) -> str:
    """Render one tool call's time window using the pod timezone."""
    arguments = call.arguments if isinstance(call.arguments, dict) else {}
    start = _format_query_timestamp(arguments.get("start"))
    end = _format_query_timestamp(arguments.get("end"))
    return f"{start} to {end}"


def _query_source_detail(call: ToolResult, argument_name: str) -> str:
    """Identify one executed data query without assuming deployment labels."""
    arguments = call.arguments if isinstance(call.arguments, dict) else {}
    query = str(arguments.get(argument_name) or "")
    return f"{call.tool_name}: {query} [{_query_window(call)}]"




def _prometheus_query_finding(
    call: ToolResult, context: IncidentContext | None = None
) -> Finding:
    """Build reader-ready evidence from the executed metric samples."""
    detail = _query_source_detail(call, "promql")
    content = metric_observations(
        call.result, _format_query_timestamp, str(call.arguments.get("promql", "")), context
    )
    if not content:
        return Finding(
            type=EvidenceType.UNKNOWN,
            content="The metric query returned no usable observations for the incident window.",
            source=EvidenceSource.PROMETHEUS,
            source_detail=detail,
            confidence=0.0,
        )
    return Finding(
        type=EvidenceType.FACT,
        content=content,
        source=EvidenceSource.PROMETHEUS,
        source_detail=detail,
        confidence=1.0,
    )


def _loki_query_finding(
    call: ToolResult, context: IncidentContext | None = None
) -> Finding:
    """Describe actual log events without dumping stream labels and source paths."""
    detail = _query_source_detail(call, "logql")
    data = call.result.get("data") if isinstance(call.result, dict) else None
    streams = data.get("result") if isinstance(data, dict) else None
    content = log_observations(
        streams if isinstance(streams, list) else [], _format_query_timestamp,
        context, str(call.arguments.get("logql", "")),
    )
    if not content:
        return Finding(
            type=EvidenceType.UNKNOWN,
            content="No matching log events were retrieved for the incident window.",
            source=EvidenceSource.LOKI,
            source_detail=detail,
            confidence=0.0,
        )
    return Finding(
        type=EvidenceType.FACT,
        content=content,
        source=EvidenceSource.LOKI,
        source_detail=detail,
        confidence=1.0,
    )


def _causal_query_details(tool_calls: list[ToolResult]) -> str:
    """Identify the successful provider queries available to a causal synthesis."""
    details: list[str] = []
    seen: set[tuple[str, str]] = set()
    for tool_name, argument_name in (
        ("query_prometheus", "promql"),
        ("query_loki", "logql"),
    ):
        for call in _successful_tool_calls(tool_calls, tool_name):
            identity = _data_query_identity(call)
            if identity in seen:
                continue
            seen.add(identity)
            details.append(_query_source_detail(call, argument_name))
    return "; ".join(details)


def _normalized_causal_syntheses(
    findings: list[Finding], tool_calls: list[ToolResult]
) -> list[Finding]:
    """Rewrite every model causal candidate to grounded query provenance.

    Requiring the model to copy an exact PromQL/LogQL string into
    ``source_detail`` discarded otherwise useful hypotheses. The harness owns
    provenance, so it attaches the actual executed queries. The shared evidence
    policy chooses the one candidate whose mechanism has real support later.
    """
    candidates = [
        finding for finding in findings
        if finding.type in _CAUSAL_FINDING_TYPES
        and not _finding_claims_no_provider_data(finding)
    ]
    detail = _causal_query_details(tool_calls)
    if not candidates or not detail:
        return []
    return [
        replace(
            candidate,
            source=EvidenceSource.LLM,
            source_detail=f"synthesis from {detail}",
            confidence=min(
                float(candidate.confidence),
                (
                    EVIDENCE_POLICY.likely_cause_confidence_cap
                    if candidate.type is EvidenceType.LIKELY_CAUSE
                    else EVIDENCE_POLICY.hypothesis_confidence_cap
                ),
            ),
        )
        for candidate in candidates
    ]


def _has_causal_verdict(findings: list[Finding]) -> bool:
    """Return whether findings explicitly answer the root-cause question."""
    for finding in findings:
        if finding.type in _CAUSAL_FINDING_TYPES:
            return True
        if finding.type == EvidenceType.UNKNOWN:
            content = finding.content.lower()
            if any(marker in content for marker in ("root cause", "causal", "cause")):
                return True
    return False


def _is_root_cause_unknown(finding: Finding) -> bool:
    """Return whether an UNKNOWN is the run's causal verdict, not a source gap."""
    if finding.type != EvidenceType.UNKNOWN:
        return False
    content = finding.content.lower()
    return any(marker in content for marker in ("root cause", "causal", "cause is"))


def _finding_claims_no_provider_data(finding: Finding) -> bool:
    """Detect broad no-data claims that can contradict grounded provider facts."""
    content = " ".join((finding.content or "").lower().split())
    markers = (
        "no prometheus",
        "prometheus data queries did not return",
        "prometheus series and no loki",
        "no telemetry",
        "no successful data",
        "no usable series",
        "no time-series data",
        "no metric data",
        "both returned empty",
        "both returned failed",
        "no matching log events",
        "no usable data from query_loki",
        "no loki log entries",
    )
    return any(marker in content for marker in markers)


def _canonical_finding_text(text: str) -> str:
    """Normalize prose enough to collapse repeated observations."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _deduplicate_findings(findings: list[Finding]) -> list[Finding]:
    """Collapse exact and near-identical rows while preserving the best claim."""
    kept: list[Finding] = []
    for finding in findings:
        canonical = _canonical_finding_text(finding.content)
        duplicate_at = None
        for index, existing in enumerate(kept):
            if finding.type != existing.type or finding.source != existing.source:
                continue
            other = _canonical_finding_text(existing.content)
            if canonical == other or (
                canonical and other and SequenceMatcher(None, canonical, other).ratio() >= 0.96
            ):
                duplicate_at = index
                break
        if duplicate_at is None:
            kept.append(finding)
            continue
        existing = kept[duplicate_at]
        if finding.confidence > existing.confidence:
            kept[duplicate_at] = finding
        logger.warning("Discarding duplicate %s finding: %s", finding.type, finding.content[:160])
    return kept


def _drop_provider_contradictions(findings: list[Finding]) -> list[Finding]:
    """Remove no-data claims contradicted by usable facts in the same run."""
    has_prometheus_fact = any(
        finding.type == EvidenceType.FACT and finding.source == EvidenceSource.PROMETHEUS
        for finding in findings
    )
    has_loki_fact = any(
        finding.type == EvidenceType.FACT and finding.source == EvidenceSource.LOKI
        for finding in findings
    )
    kept: list[Finding] = []
    for finding in findings:
        if finding.type != EvidenceType.UNKNOWN or not _finding_claims_no_provider_data(finding):
            kept.append(finding)
            continue
        lowered = finding.content.lower()
        contradicts_prometheus = has_prometheus_fact and any(
            marker in lowered
            for marker in ("prometheus", "metric", "telemetry", "time-series", "both returned")
        )
        contradicts_loki = has_loki_fact and any(
            marker in lowered for marker in ("loki", "log events", "log entries", "both returned")
        )
        if contradicts_prometheus or contradicts_loki:
            logger.warning(
                "Discarding provider no-data claim contradicted by grounded facts: %s",
                finding.content[:200],
            )
            continue
        kept.append(finding)
    return kept


def _fact_is_relevant_to_incident(finding: Finding, context: IncidentContext | None) -> bool:
    """Direct current-incident telemetry can support a bounded hypothesis."""
    return finding.type == EvidenceType.FACT and finding.source in {
        EvidenceSource.ALERT,
        EvidenceSource.PROMETHEUS,
        EvidenceSource.LOKI,
    }


def _fallback_causal_finding(
    findings: list[Finding], tool_calls: list[ToolResult], context: IncidentContext | None
) -> Finding:
    """Return honest uncertainty when no submitted finding names a mechanism."""
    facts = [
        finding for finding in findings
        if _fact_is_relevant_to_incident(finding, context)
    ]
    if not facts:
        return Finding(
            type=EvidenceType.UNKNOWN,
            content=(
                "Root cause is unconfirmed. The retrieved observations do not identify "
                "a responsible component or support a plausible causal mechanism."
            ),
            source=EvidenceSource.SYSTEM,
            source_detail="causal-verdict enforcement",
            confidence=0.0,
        )

    observations = " ".join(
        finding.content.strip().rstrip(".") for finding in facts[:2]
    )
    return Finding(
        type=EvidenceType.UNKNOWN,
        content=(
            "Root cause was not established. The run confirmed these observations: "
            f"{observations}. None of the retrieved signals demonstrated which component "
            "initiated the condition."
        ),
        source=EvidenceSource.SYSTEM,
        source_detail="causal-verdict enforcement",
        confidence=0.0,
    )


def _ensure_causal_verdict(
    findings: list[Finding], tool_calls: list[ToolResult], context: IncidentContext | None = None
) -> list[Finding]:
    """Guarantee exactly one explicit, evidence-bounded causal verdict."""
    candidates = [finding for finding in findings if finding.type in _CAUSAL_FINDING_TYPES]
    causal_unknowns = [finding for finding in findings if _is_root_cause_unknown(finding)]
    facts = [finding for finding in findings if finding.type == EvidenceType.FACT]

    selected: Finding
    if candidates:
        validated = [
            result
            for candidate in candidates
            if (result := EVIDENCE_POLICY.validate(candidate, facts, context)) is not None
        ]
        selected = (
            max(
                validated,
                key=lambda finding: (
                    finding.type is EvidenceType.LIKELY_CAUSE,
                    finding.confidence,
                ),
            )
            if validated
            else _fallback_causal_finding(findings, tool_calls, context)
        )
    else:
        # Provider gaps do not decide the causal verdict. If no submitted
        # hypothesis names an evidence-backed mechanism, the harness records
        # one honest root-cause UNKNOWN from the complete fact set.
        selected = _fallback_causal_finding(findings, tool_calls, context)

    remaining = [
        finding for finding in findings
        if finding not in candidates and finding not in causal_unknowns
    ]
    return [*remaining, selected]


def _ensure_actions(
    findings: list[Finding], tool_calls: list[ToolResult],
    context: IncidentContext | None = None,
) -> list[Finding]:
    """Keep operator actions available even when the model or a provider fails."""
    if any(finding.type == EvidenceType.RECOMMENDATION for finding in findings):
        return findings
    facts = [finding for finding in findings if finding.type == EvidenceType.FACT]
    scope = "; ".join(
        dict.fromkeys(finding.source_detail for finding in facts if finding.source_detail)
    ) or "the incident target and historical window"
    if facts:
        content = (
            "Align the primary alert measurement with each collected related signal over the "
            "same window. Identify which signal consistently begins first or changes with the "
            "symptom, inspect the component that owns that signal, and verify the primary "
            "measurement after mitigation."
        )
    else:
        content = (
            "Collect one target-scoped primary measurement and at least one distinct resource, "
            "dependency, or event signal over the same window before changing the system."
        )
    action = Finding(
        type=EvidenceType.RECOMMENDATION,
        content=content,
        source=EvidenceSource.SYSTEM,
        source_detail=scope,
        confidence=1.0,
    )
    return [*findings, action]


def _ensure_causal_verdict_for_mode(
    findings: list[Finding], tool_calls: list[ToolResult], verification: bool,
    context: IncidentContext | None = None,
) -> list[Finding]:
    """Apply causal-verdict enforcement only to full investigations."""
    if verification:
        return findings
    return _ensure_actions(
        _ensure_causal_verdict(findings, tool_calls, context),
        tool_calls,
        context,
    )


def _drop_superseded_process_unknowns(findings: list[Finding]) -> list[Finding]:
    """Remove model-process gaps once finalization produced a causal verdict."""
    if not _has_causal_verdict(findings):
        return findings
    return [
        finding
        for finding in findings
        if not (
            finding.type == EvidenceType.UNKNOWN
            and finding.source == EvidenceSource.SYSTEM
            and finding.content.startswith(_PROCESS_ONLY_UNKNOWN_PREFIXES)
        )
    ]


def _order_and_link_findings(findings: list[Finding]) -> list[Finding]:
    """Put findings in reader order and create durable evidence relationships."""
    facts = [finding for finding in findings if finding.type == EvidenceType.FACT]
    causal = [
        finding for finding in findings
        if finding.type in _CAUSAL_FINDING_TYPES or _is_root_cause_unknown(finding)
    ][:1]
    unknowns = [
        finding for finding in findings
        if finding.type == EvidenceType.UNKNOWN and not _is_root_cause_unknown(finding)
    ]
    recommendations = [
        finding for finding in findings if finding.type == EvidenceType.RECOMMENDATION
    ]
    ordered = [*facts, *causal, *unknowns, *recommendations]
    fact_indexes = list(range(len(facts)))
    causal_index = len(facts) if causal else None
    linked: list[Finding] = []
    for index, finding in enumerate(ordered):
        supports = list(finding.supports_evidence)
        if index == causal_index and not supports:
            supports = fact_indexes
        elif finding.type == EvidenceType.RECOMMENDATION and not supports:
            supports = [causal_index] if causal_index is not None else fact_indexes
        linked.append(replace(finding, supports_evidence=supports))
    return linked


def _finalize_findings(
    findings: list[Finding],
    tool_calls: list[ToolResult],
    context: IncidentContext | None,
    incident_id: str,
    verification: bool,
) -> list[Finding]:
    """Apply all validation once before any terminal evidence is persisted."""
    finalized = [_sanitize_finding(finding) for finding in findings]
    finalized = _drop_trigger_restatements(finalized, incident_id)
    if not verification:
        finalized = _ground_findings_in_data_queries(finalized, tool_calls, context)
    finalized = _drop_provider_contradictions(_deduplicate_findings(finalized))
    finalized = _ensure_causal_verdict_for_mode(
        finalized, tool_calls, verification, context
    )
    finalized = _drop_superseded_process_unknowns(finalized)
    finalized = _deduplicate_findings(finalized)
    return _order_and_link_findings(finalized) if not verification else finalized


def _ground_findings_in_data_queries(
    findings: list[Finding], tool_calls: list[ToolResult], context: IncidentContext | None = None
) -> list[Finding]:
    """Replace model-authored telemetry claims with exact executed-query summaries.

    A weak model can call the required tools and still cite a different query or
    copy values from the triggering incident. Prometheus/Loki findings therefore
    come from parsed tool results, not model prose. Model synthesis and operator
    recommendations remain available under their non-provider sources.
    """
    kept = [
        finding
        for finding in findings
        if (finding.type == EvidenceType.RECOMMENDATION
            or finding.source not in (EvidenceSource.PROMETHEUS, EvidenceSource.LOKI))
        and finding.type not in _CAUSAL_FINDING_TYPES
    ]
    grounded: list[Finding] = []
    alert_content, alert_detail = alert_observation(context, _format_query_timestamp)
    if alert_content:
        grounded.append(
            Finding(
                type=EvidenceType.FACT,
                content=alert_content,
                source=EvidenceSource.ALERT,
                source_detail=alert_detail,
                confidence=1.0,
            )
        )
    prometheus_calls = _successful_tool_calls(tool_calls, "query_prometheus")
    if alert_content:
        prometheus_calls = [
            call for call in prometheus_calls
            if call.result.get("signal_role") != "recovered_primary"
        ]
    executed = {str(call.arguments.get("promql", "")) for call in prometheus_calls}
    prometheus_calls = [
        call for call in prometheus_calls
        if f"rate({call.arguments.get('promql', '')}[5m])" not in executed
    ]
    prometheus_calls = _representative_distinct_query_calls(
        prometheus_calls, "promql"
    )
    prometheus_calls = _select_relevant_prometheus_calls(
        prometheus_calls, context, _MAX_GROUNDED_FINDINGS_PER_SOURCE
    )
    loki_calls = _select_relevant_loki_calls(
        _successful_tool_calls(tool_calls, "query_loki"),
        context,
        _MAX_GROUNDED_FINDINGS_PER_SOURCE,
    )
    grounded.extend(
        _prometheus_query_finding(call, context)
        for call in prometheus_calls
    )
    grounded.extend(
        _loki_query_finding(call, context)
        for call in loki_calls
    )
    grounded.extend(
        _normalized_causal_syntheses(
            findings, [*prometheus_calls, *loki_calls]
        )
    )
    return grounded + kept


def _ground_partial_data_queries(
    tool_calls: list[ToolResult], verification: bool, context: IncidentContext | None = None
) -> list[Finding]:
    """Return deterministic full-investigation evidence gathered before termination."""
    if verification:
        return []
    return _ground_findings_in_data_queries([], tool_calls, context)


def _decode_cached_tool_result(cached: Any) -> dict[str, Any] | None:
    """Decode one cached tool result, ignoring malformed cache entries."""
    try:
        parsed = json.loads(cached) if isinstance(cached, str) else cached
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _cached_verification_results(
    seen_calls: dict[str, str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return usable Prometheus and Loki results from the repeated-call cache."""
    prometheus_result = None
    loki_result = None

    for signature, cached in seen_calls.items():
        parsed = _decode_cached_tool_result(cached)
        if parsed is None:
            continue

        if signature.startswith("query_prometheus:"):
            prometheus_result = parsed
        elif signature.startswith("query_loki:"):
            loki_result = parsed

    return prometheus_result, loki_result


def _cached_result_items(result: dict[str, Any] | None) -> list[Any]:
    """Read result items from either the adapter or direct API response shape."""
    if result is None or result.get("status") not in (None, "success"):
        return []
    data = result.get("data")
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    items = data.get("result")
    return items if isinstance(items, list) else []


def _prometheus_verification_finding(
    result: dict[str, Any] | None,
) -> Finding | None:
    """Record cached Prometheus evidence without assuming metric semantics."""
    series = _cached_result_items(result)
    if not series:
        return None
    return Finding(
        type=EvidenceType.FACT,
        content=(
            f"Prometheus response contained {len(series)} result series for the "
            "repeated live query."
        ),
        source=EvidenceSource.PROMETHEUS,
        source_detail="cached query_prometheus response after repeated call",
        confidence=1.0,
    )


def _loki_verification_finding(result: dict[str, Any] | None) -> Finding | None:
    """Record cached Loki evidence without interpreting log content as recovery."""
    streams = _cached_result_items(result)
    if not streams:
        return None
    line_count = sum(
        len(stream.get("values", []))
        for stream in streams
        if isinstance(stream, dict) and isinstance(stream.get("values"), list)
    )
    return Finding(
        type=EvidenceType.FACT,
        content=(
            f"Loki response contained {line_count} log lines across "
            f"{len(streams)} result streams."
        ),
        source=EvidenceSource.LOKI,
        source_detail="cached query_loki response after repeated call",
        confidence=0.8,
    )


def _is_placeholder_or_dummy(text: str) -> bool:
    """True when text is a placeholder, dummy value, or lacks substantive content."""
    cleaned = text.strip().lower()
    dummy_exact = {
        "placeholder",
        "todo",
        "tbd",
        "none",
        "null",
        "n/a",
        "na",
        "test",
        "asdf",
        "unknown",
        "...",
        "-",
    }
    if cleaned in dummy_exact:
        return True
    if (
        cleaned.startswith("placeholder")
        or cleaned.startswith("todo:")
        or cleaned.startswith("tbd:")
    ):
        return True
    return False


def _sanitize_finding(finding: Finding) -> Finding:
    """Keep working notes and placeholder text out of a finding, whichever path produced it.

    A model that narrates its reasoning does so through both exits of the loop:
    as a plain text reply, and as a `submit_findings` call whose content is the
    transcript rather than a conclusion. The text path rejects that; this covers
    the tool path, so an essay cannot reach the report by being submitted
    instead of typed.

    Additionally, lazy placeholder submissions like 'placeholder', 'TBD', or 'TODO'
    are rejected and replaced with honest terminal claims.
    """
    content = _plain_typography(finding.content)
    if _is_placeholder_or_dummy(content):
        logger.warning(
            "Discarding a %s finding with placeholder content: %s",
            finding.type,
            content[:120],
        )
        return Finding(
            type=EvidenceType.UNKNOWN,
            content=(
                "Investigation ended without a substantive conclusion; the model "
                "submitted placeholder text."
            ),
            source=EvidenceSource.SYSTEM,
            confidence=0.0,
        )

    if not _is_reasoning_preamble(content):
        if content != finding.content:
            return replace(finding, content=content)
        return finding
    logger.warning(
        "Discarding a %s finding that was working notes (%d chars): %s",
        finding.type,
        len(finding.content),
        " ".join(finding.content.split())[:120],
    )
    return Finding(
        type=EvidenceType.UNKNOWN,
        content=(
            "Investigation ended without a stated conclusion; the model submitted "
            "its working notes instead of a finding."
        ),
        source=EvidenceSource.SYSTEM,
        confidence=0.0,
    )


def unwrap_tool_call_envelope(name: str, args: dict) -> tuple[str, dict]:
    """Unwrap a tool call that arrived wrapped in a second function envelope.

    Some [OI]-compatible endpoints return the whole tool call as the arguments
    of a generic `functions` call:

        {"name": "functions", "arguments": {"name": "query_loki",
                                            "arguments": {"logql": "..."}}}

    Treated literally, every such call is an unknown tool and the model gets an
    error back for a call it made correctly — production runs burned ten calls
    this way, each one a round trip that produced nothing.
    """
    if name != "functions" or not isinstance(args, dict):
        return name, args

    inner_name = args.get("name")
    if not isinstance(inner_name, str) or not inner_name:
        return name, args

    inner_args = args.get("arguments")
    if inner_args is None:
        # A no-argument tool (get_loki_labels) arrives as {"name": "..."} with
        # no arguments key at all.
        logger.warning("Unwrapped nested tool call envelope: functions -> %s", inner_name)
        return inner_name, {}
    if isinstance(inner_args, str):
        try:
            inner_args = json.loads(inner_args)
        except json.JSONDecodeError as e:
            logger.warning("Malformed inner arguments in tool envelope: %s", e)
            return inner_name, {"_parse_error": str(e), "_raw": inner_args}
    if not isinstance(inner_args, dict):
        return name, args

    logger.warning("Unwrapped nested tool call envelope: functions -> %s", inner_name)
    return inner_name, inner_args


def _resolve_tool_call(tool_call) -> tuple[str, dict]:
    """Return the real tool name and arguments for a tool call.

    Some endpoints send ``arguments`` as the JSON string ``"null"`` or as a
    bare ``None`` for no-argument tool calls.  ``json.loads("null")`` returns
    ``None``, and passing that into ``unwrap_tool_call_envelope`` (which does
    ``args.get(...)``) raises ``'NoneType' object is not subscriptable``.
    Normalise to an empty dict so every downstream handler can safely call
    ``function_args.get("key")``.

    If the LLM output was truncated or contains malformed JSON, catch the
    JSONDecodeError and pass a structured error payload so the loop can feed
    the error back to the model instead of aborting the whole investigation.
    """
    name = tool_call.function.name
    raw = tool_call.function.arguments
    if raw is None:
        args = {}
    elif not isinstance(raw, str):
        args = raw if isinstance(raw, dict) else {}
    else:
        try:
            parsed = json.loads(raw)
            args = parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Malformed tool call arguments for %s: %s (raw=%r)", name, e, raw)
            args = {"_parse_error": str(e), "_raw": raw}
    return unwrap_tool_call_envelope(name, args)


def _resolved_tool_name(tool_call) -> str:
    """The tool name after unwrapping a nested envelope, without parsing args."""
    name = tool_call.function.name
    if name != "functions":
        return name
    try:
        args = json.loads(tool_call.function.arguments)
    except (json.JSONDecodeError, TypeError):
        return name
    inner = args.get("name") if isinstance(args, dict) else None
    return inner if isinstance(inner, str) and inner else name


# Floor for a single LLM call. Below this a call cannot return anything useful,
# so failing fast and letting the investigation submit what it has beats opening
# another connection.
_MIN_CALL_TIMEOUT_SECONDS = 5.0

# Ceiling for a single LLM call. It exists so a hung endpoint cannot consume the
# whole investigation on one attempt, not to say how long a healthy call takes.
MAX_CALL_TIMEOUT_SECONDS = 180.0


def _call_timeout(
    request_timeout: float, budget_seconds: float | None, attempts_left: int = 1
) -> float:
    """How long one LLM request may take, given what is left of the budget.

    This used to be `min(request_timeout, budget)`, which capped every call at
    30 seconds regardless of how much budget remained. That is a paid-API
    assumption: a free endpoint generating a tool-call response routinely needs
    60 to 120 seconds, so every call was cut off mid-generation, retried against
    the same slow endpoint, and eventually failed over to a second slow endpoint
    that timed out the same way. Production logs show exactly that — six
    consecutive `Request timed out` lines for one incident — and the run ended
    with no findings after burning its whole budget on timeouts.

    The budget, not a fixed number, is the authority. Each attempt gets an equal
    share of what is left, which lets a slow model finish while still leaving
    room for a retry. `request_timeout` is now a floor for the normal case, not a
    cap.

    Two bounds keep the share honest. It never exceeds what is left, so a call
    cannot promise time the investigation does not have; and it never exceeds
    `MAX_CALL_TIMEOUT_SECONDS`, so one hung endpoint cannot consume the whole
    run on a single attempt. The only exception to the first bound is the
    floor: with seconds left, a 5-second attempt is still worth making.
    """
    if budget_seconds is None:
        return request_timeout
    share = max(request_timeout, budget_seconds / max(1, attempts_left))
    return max(
        _MIN_CALL_TIMEOUT_SECONDS,
        min(share, MAX_CALL_TIMEOUT_SECONDS, budget_seconds),
    )


def _remaining_budget(started: float, budget_seconds: float | None) -> float | None:
    """Seconds left of the budget, measured from when the call began."""
    if budget_seconds is None:
        return None
    import time as _time

    return max(0.0, budget_seconds - (_time.monotonic() - started))


def _is_timeout(error: Exception) -> bool:
    """True when the endpoint accepted the request but did not answer in time."""
    return isinstance(error, (openai.APITimeoutError, TimeoutError))


def _out_of_time(remaining: float | None, attempt: int) -> bool:
    """True when a further retry cannot finish inside what is left.

    The first attempt always runs: with the budget gone it still gets the floor
    timeout, because returning an error without trying is strictly worse.
    """
    if remaining is None or attempt == 0:
        return False
    return remaining <= _MIN_CALL_TIMEOUT_SECONDS


class InvestigatorLLM:
    """OpenAI-compatible LLM client with endpoint redundancy for investigator."""

    # Bounded retry policy: transient network/5xx errors are retried with
    # exponential backoff; anything else fails fast.
    MAX_RETRIES = 3
    INITIAL_BACKOFF_SECONDS = 0.5
    MAX_BACKOFF_SECONDS = 4.0

    # How long to back off when the LLM endpoint rate-limits us (HTTP 429)
    # and no other endpoint is available to fail over to.
    #
    # Previously this was 30 s regardless of the Retry-After header value
    # the endpoint sends.  With two concurrent investigations + a daily-
    # summary call each sleeping 30-90 s, all three threads held their full
    # messages[] history in RAM simultaneously and pushed the pod past its
    # 1 Gi limit (observed: exit code 137, restart #39/#40, 2026-11-22).
    #
    # The new policy:
    #   1. Honour Retry-After when the response carries one, capped at this
    #      constant so a misconfigured endpoint cannot park a thread forever.
    #   2. Fail over to the next endpoint immediately on 429 when one exists
    #      instead of sleeping - the other endpoint is not rate-limited yet.
    #   3. Only sleep (up to this cap) when we are truly out of endpoints.
    RATE_LIMIT_BACKOFF_SECONDS = 30.0

    def __init__(self, config: Config):
        config.validate()
        # Floor for a single request, from LLM_REQUEST_TIMEOUT_SECONDS. It is a
        # floor rather than a cap: `_call_timeout` gives each attempt a share of
        # the remaining investigation budget, so a slow free endpoint is allowed
        # the time it actually needs. Tuning this below the model's real latency
        # is what made every call time out.
        self.request_timeout = float(getattr(config, "llm_request_timeout_seconds", 0) or 60.0)

        # Build one aligned ordered endpoint list. Plural configuration is
        # validated strictly; when it is absent, the legacy single-endpoint
        # settings remain the documented fallback.
        endpoints: list[tuple[str, str | None, str | None]] = []
        if config.llm_base_urls:
            endpoints = list(
                zip(
                    config.llm_base_urls,
                    config.llm_api_keys or [],
                    config.llm_models or [],
                    strict=True,
                )
            )
        else:
            endpoints = [(config.llm_base_url, config.llm_api_key, config.llm_model)]

        self.endpoints: list[tuple[str, str | None, str | None]] = []
        self.endpoint_labels: list[str] = []
        self.quarantined_endpoints: set[int] = set()
        # Each endpoint gets its own OpenAI client. Endpoints without a
        # configured API key are skipped with a warning so the service can
        # still start when LLM credentials are missing (investigations will
        # fail gracefully at call time rather than crashing on startup).
        self.clients: list[OpenAI] = []
        self.models: list[str] = []
        for url, key, model in endpoints:
            try:
                self.clients.append(
                    OpenAI(base_url=url, api_key=key or "", timeout=self.request_timeout)
                )
                self.models.append(model or config.llm_model)
                self.endpoints.append((url, key, model))
                self.endpoint_labels.append(self._endpoint_identity(url))
            except openai.OpenAIError as e:
                logger.warning(
                    "Skipping LLM endpoint %s: missing or invalid credentials (%s)",
                    self._endpoint_identity(url),
                    e,
                )

        if not self.clients:
            logger.warning(
                "No LLM endpoints with valid credentials configured; "
                "investigations will fail until an API key is provided"
            )

    @staticmethod
    def _endpoint_identity(url: str) -> str:
        """Return a log-safe endpoint identity with credentials/query removed."""
        try:
            parsed = urlsplit(url)
            host = parsed.hostname or "unknown-host"
            if parsed.port is not None:
                host = f"{host}:{parsed.port}"
            path = parsed.path.rstrip("/")
            return f"{host}{path}" if path else host
        except (TypeError, ValueError):
            return "invalid-endpoint"

    @staticmethod
    def _is_terminal_endpoint_error(error: Exception) -> bool:
        """Return whether retrying this endpoint cannot repair its configuration."""
        return isinstance(error, openai.APIStatusError) and error.status_code == 404

    def _is_retryable(self, error: Exception) -> bool:
        """Return True for transient errors worth retrying / failing over."""
        if isinstance(error, openai.APITimeoutError):
            return True
        if isinstance(error, openai.APIConnectionError):
            return True
        if isinstance(error, openai.APIStatusError):
            # 429 Too Many Requests: the LLM endpoint is rate-limiting us.
            # Back off (respecting Retry-After when advertised) instead of
            # failing fast so an investigation can complete once the quota
            # resets.
            if error.status_code == 429:
                return True
            # 5xx server errors are transient; retry them too.
            if error.status_code >= 500:
                return True
        return False

    def _should_fail_over(
        self,
        error: Exception,
        endpoint_index: int,
        attempt: int,
        call_timeout: float,
        endpoint_label: str,
        session: str,
    ) -> bool:
        """Decide whether to abandon this endpoint and try the next one.

        A timeout is a capacity signal, not a blip: the endpoint accepted the
        request and could not answer inside the time it was given. Retrying the
        same slow endpoint spends the budget waiting for the same result, so the
        next endpoint is tried instead. Retries stay for connection errors and
        5xx responses, which really are transient.

        A 429 from a free aggregator (freellmapi) means every model on that
        endpoint is exhausted right now.  Sleeping 30 s and retrying the same
        endpoint wastes the budget and keeps the messages[] list in RAM the
        whole time.  When another endpoint is available, fail over immediately;
        only sleep when we have no other choice.

        With only one endpoint configured there is nothing to fail over to, so
        the normal retry policy applies and the endpoint gets another attempt.
        """
        retryable = self._is_retryable(error)
        has_another_endpoint = endpoint_index < len(self.clients) - 1

        if _is_timeout(error) and has_another_endpoint:
            logger.warning(
                "LLM call timed out on %s after %.0fs; failing over instead of "
                "retrying the same endpoint (session=%s)",
                endpoint_label,
                call_timeout,
                session,
            )
            return True

        # 429 with another endpoint available: fail over immediately instead of
        # sleeping.  Sleeping 30 s per attempt while holding the full message
        # history caused the pod to OOM when several investigations ran at once.
        is_rate_limited = (
            isinstance(error, openai.APIStatusError) and error.status_code == 429
        )
        if is_rate_limited and has_another_endpoint:
            logger.warning(
                "LLM endpoint %s returned 429; failing over immediately (session=%s)",
                endpoint_label,
                session,
            )
            return True

        if retryable and attempt < self.MAX_RETRIES - 1:
            return False

        if retryable:
            logger.warning(
                "LLM retries exhausted on %s (%d); failing over",
                endpoint_label,
                self.MAX_RETRIES,
            )
        else:
            logger.error("LLM call failed (non-retryable, session=%s): %s", session, error)
        return True

    def _compute_backoff(
        self,
        error: Exception,
        attempt: int,
        started: float,
        budget_seconds: float | None,
    ) -> float:
        """Calculate backoff duration for a failed LLM call."""
        if isinstance(error, openai.APIStatusError) and error.status_code == 429:
            retry_after: float | None = None
            try:
                ra = error.response.headers.get("Retry-After")
                if ra is not None:
                    retry_after = float(ra)
            except (AttributeError, ValueError):
                pass
            fallback = self.RATE_LIMIT_BACKOFF_SECONDS
            backoff = min(
                retry_after if retry_after is not None else fallback,
                fallback,
            )
            # Never sleep past the remaining budget - there is no point waiting
            # 30 s for a rate limit to clear if the investigation deadline is in 10 s.
            remaining_now = _remaining_budget(started, budget_seconds)
            if remaining_now is not None:
                floor_reserve = remaining_now - _MIN_CALL_TIMEOUT_SECONDS
                backoff = min(backoff, max(0.0, floor_reserve))
            return backoff

        return min(
            self.INITIAL_BACKOFF_SECONDS * (2**attempt),
            self.MAX_BACKOFF_SECONDS,
        )

    def _call_endpoint(
        self,
        client: OpenAI,
        model: str,
        endpoint_index: int,
        endpoint_label: str,
        messages: list[dict],
        optional: dict,
        max_tokens: int,
        session: str,
        started: float,
        budget_seconds: float | None,
    ) -> tuple[Any | None, Exception | None]:
        """Try calling an endpoint up to MAX_RETRIES times."""
        import time as _time

        last_error: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            remaining = _remaining_budget(started, budget_seconds)
            if _out_of_time(remaining, attempt):
                break
            call_timeout = _call_timeout(
                self.request_timeout, remaining, self.MAX_RETRIES - attempt
            )
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    timeout=call_timeout,
                    **optional,
                )
                if endpoint_index > 0:
                    logger.warning(
                        "LLM failover succeeded on %s (session=%s)",
                        endpoint_label,
                        session,
                    )
                return response.choices[0].message, None
            except (
                openai.OpenAIError,
                RuntimeError,
                ValueError,
                KeyError,
                TypeError,
                TimeoutError,
            ) as e:
                last_error = e
                if self._is_terminal_endpoint_error(e):
                    self.quarantined_endpoints.add(endpoint_index)
                    logger.error(
                        "Quarantining LLM %s after terminal HTTP 404; later turns will "
                        "skip this endpoint (session=%s)",
                        endpoint_label,
                        session,
                    )
                    break
                if self._should_fail_over(
                    e, endpoint_index, attempt, call_timeout, endpoint_label, session
                ):
                    break
                backoff = self._compute_backoff(e, attempt, started, budget_seconds)
                if backoff <= 0:
                    break
                logger.warning(
                    "LLM call failed (attempt %d/%d) on %s, retrying in %.1fs (session=%s): %s",
                    attempt + 1,
                    self.MAX_RETRIES,
                    endpoint_label,
                    backoff,
                    session,
                    e,
                )
                _time.sleep(backoff)

        return None, last_error

    def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        max_tokens: int = 2000,
        session: str = "",
        budget_seconds: float | None = None,
    ):
        """Call the LLM with optional tool use; failing over across endpoints
        and raising the last error.

        `session` labels the call in logs (the incident id). The client is
        stateless — it holds no conversation — so the label exists to make a
        log line traceable to exactly one investigation when several run at
        once, not because any state is attached to it.
        """
        import time as _time

        # Tool kwargs are only sent when there are tools. Passing
        # tool_choice="none" alongside tools=None is rejected by some
        # OpenAI-compatible endpoints, and tool-free callers (the daily
        # summary) must not depend on that being tolerated.
        optional = {}
        if tools is not None:
            optional = {"tools": tools, "tool_choice": tool_choice}

        # Counted down across every attempt and every endpoint. A snapshot
        # would let 3 retries x 30s run 90 seconds past the point where the
        # investigation needed its reserve, which is exactly how a run with 13
        # successful tool calls ended with no conclusion.
        started = _time.monotonic()

        last_error: Exception | None = None
        attempted_endpoint = False
        for idx, (client, model) in enumerate(zip(self.clients, self.models)):
            if idx in self.quarantined_endpoints:
                continue
            attempted_endpoint = True
            identity = (
                self.endpoint_labels[idx]
                if idx < len(self.endpoint_labels)
                else f"endpoint-{idx + 1}"
            )
            endpoint_label = f"endpoint {idx + 1}/{len(self.clients)} ({identity})"
            message, err = self._call_endpoint(
                client=client,
                model=model,
                endpoint_index=idx,
                endpoint_label=endpoint_label,
                messages=messages,
                optional=optional,
                max_tokens=max_tokens,
                session=session,
                started=started,
                budget_seconds=budget_seconds,
            )
            if message is not None:
                return message
            if err is not None:
                last_error = err

        # All endpoints exhausted.
        if last_error is not None:
            raise last_error
        if not attempted_endpoint and self.quarantined_endpoints:
            raise RuntimeError("All configured LLM endpoints are quarantined")
        raise RuntimeError("LLM call failed after all endpoints exhausted")  # pragma: no cover


class PamawasInvestigator:
    """Main investigator orchestrator."""

    # How many turns in a row may produce prose without a tool call before the
    # loop stops asking the model to continue. A model that replies with text
    # three times running has stopped investigating; asking it a fourth time
    # only burns wall-clock budget that a paged engineer is waiting on.
    MAX_CONSECUTIVE_TEXT_TURNS = 3

    # How many turns in a row may produce only repeated tool calls (served from
    # cache) before the loop forces findings submission. A model that repeats
    # identical queries three times running is stuck in a loop; letting it spin
    # burns turn budget (up to 50/100 turns) and ends with no findings.
    MAX_CONSECUTIVE_REPEATED_TURNS = 3

    # Bound how many times the harness forces a missing data-query tool when
    # the endpoint ignores tool_choice or keeps producing unsuccessful calls.
    # Reaching this limit yields an honest system UNKNOWN, never model findings
    # unsupported by Prometheus/Loki data queries.
    MAX_REQUIRED_QUERY_FAILURES = 3

    # Wall-clock budget reserved at the end of an investigation for the model to
    # write its findings and for them to be persisted.
    #
    # Without a reserve the deadline was only noticed at the top of a turn, so a
    # run that spent its last minutes inside tool calls discovered it had no
    # time left only after the model had gathered evidence it never got to
    # report. Runs of 10m21s ended with "deadline exceeded without submitting
    # findings" despite every Prometheus and Loki query having succeeded.
    DEADLINE_RESERVE_FRACTION = 0.25
    DEADLINE_RESERVE_MIN_SECONDS = 90.0

    # How long before the reserve begins the model is told to wrap up. It gets a
    # chance to synthesise what it has rather than being cut off mid-exploration.
    DEADLINE_WARN_FRACTION = 0.5

    def __init__(self, config: Config):
        self.config = config
        self.llm = InvestigatorLLM(config)
        self.tools = InvestigatorTools(config)
        self.condition_evaluators = ConditionEvaluatorRegistry()
        if config.grafana_url.strip() and config.grafana_api_token.strip():
            self.condition_evaluators.register(
                "grafana",
                "managed_alerting",
                GrafanaAlertingAdapter(
                    GrafanaAlertingConfig(
                        base_url=config.grafana_url.strip(),
                        api_token=config.grafana_api_token.strip(),
                        org_id=config.grafana_org_id.strip(),
                        timeout_seconds=config.grafana_timeout_seconds,
                        max_response_bytes=config.grafana_max_response_bytes,
                        max_alerts=config.grafana_max_alerts,
                        max_rules=config.grafana_max_rules,
                        circuit_breaker_failure_threshold=(
                            config.circuit_breaker_failure_threshold
                        ),
                        circuit_breaker_timeout_seconds=(
                            config.circuit_breaker_timeout_seconds
                        ),
                        circuit_breaker_half_open_max_calls=(
                            config.circuit_breaker_half_open_max_calls
                        ),
                    )
                ),
            )
        elif config.grafana_url.strip():
            logger.warning(
                "Grafana URL is configured without an API token; authoritative "
                "Grafana condition evaluation is disabled and recovery will fail closed"
            )

        self.db_conn = None
        self.persistence = None
        self._connect_db()

    def _connect_db(self):
        """Establish database connection with timeout."""
        if not self.config.database_url:
            logger.warning("No DATABASE_URL provided, database features disabled")
            return

        try:
            self.db_conn = psycopg2.connect(
                self.config.database_url,
                connect_timeout=5,  # 5-second timeout to prevent startup block
            )
            # This connection is used only for read-only lookups (incident
            # context, health probes). autocommit keeps every statement out of
            # an explicit transaction so reads never leak open transactions.
            self.db_conn.autocommit = True
            self.persistence = InvestigationPersistence(self.config.database_url)
            logger.info("Connected to database")
        except (psycopg2.Error, RuntimeError) as e:
            logger.error("Failed to connect to database: %s", e)
            increment_db_errors()
            self.db_conn = None

    @staticmethod
    def _apply_catalog_revision(
        context: IncidentContext, snapshot: dict[str, Any]
    ) -> None:
        catalog = snapshot.get("catalog")
        if not isinstance(catalog, dict):
            raise RuntimeError("Telemetry catalog revision has invalid content")
        context.telemetry_catalog = catalog
        context.telemetry_catalog_revision_id = str(snapshot["revision_id"])
        context.telemetry_catalog_content_hash = str(snapshot["content_hash"])

    async def _refresh_telemetry_catalog(
        self, context: IncidentContext, run_id: str | None = None
    ) -> None:
        """Refresh live provider schemas and pin one immutable run revision.

        The database claim prevents concurrent investigator workers from
        stampeding Prometheus or Loki. Failed refreshes never erase the last
        successful catalog. A retry loads its already-pinned revision before
        considering refresh state, so provider changes cannot alter candidate
        selection partway through one logical run.
        """
        if self.persistence is None:
            return
        if not run_id:
            raise RuntimeError(
                "A persisted investigation must have a run ID before catalog pinning"
            )

        pinned = await asyncio.to_thread(
            self.persistence.load_pinned_telemetry_catalog, run_id
        )
        if pinned is not None:
            self._apply_catalog_revision(context, pinned)
            logger.info(
                "Loaded pinned telemetry catalog revision %s for run %s",
                context.telemetry_catalog_revision_id,
                run_id,
            )
            return

        async def refresh_provider(provider: str) -> None:
            claimed = await asyncio.to_thread(
                self.persistence.claim_telemetry_catalog_refresh,
                provider,
                self.config.telemetry_catalog_refresh_seconds,
            )
            if not claimed:
                return
            try:
                if provider == "prometheus":
                    response = await self.tools._prometheus_adapter.query_metric_catalog()
                    data = response.get("data") if isinstance(response, dict) else None
                    entries = data if isinstance(data, list) else []
                    resource_kind = "metric"
                else:
                    response = await self.tools._loki_adapter.query_labels()
                    data = response.get("data") if isinstance(response, dict) else None
                    entries = [
                        {"name": str(label)} for label in data or []
                        if str(label).strip()
                    ] if isinstance(data, list) else []
                    resource_kind = "label"
                if not entries:
                    raise ValueError(f"{provider} discovery returned no catalog entries")
                await asyncio.to_thread(
                    self.persistence.store_telemetry_catalog,
                    provider,
                    resource_kind,
                    entries,
                )
                logger.info(
                    "Refreshed %s telemetry catalog with %d entries",
                    provider,
                    len(entries),
                )
            except (
                PrometheusError,
                LokiError,
                RuntimeError,
                ValueError,
                TypeError,
                KeyError,
                psycopg2.Error,
            ) as e:
                logger.warning(
                    "%s telemetry catalog refresh failed with %s; keeping last snapshot",
                    provider,
                    type(e).__name__,
                )
                await asyncio.to_thread(
                    self.persistence.mark_telemetry_catalog_refresh_failed,
                    provider,
                    type(e).__name__,
                )

        await asyncio.gather(
            refresh_provider("prometheus"),
            refresh_provider("loki"),
        )
        pinned = await asyncio.to_thread(
            self.persistence.pin_telemetry_catalog_revision, run_id
        )
        self._apply_catalog_revision(context, pinned)
        logger.info(
            "Pinned telemetry catalog revision %s (%s entries) to run %s",
            context.telemetry_catalog_revision_id,
            pinned.get("entry_count", 0),
            run_id,
        )

    async def _evaluate_condition(
        self, context: IncidentContext
    ) -> ConditionEvaluation:
        """Evaluate current source state before any telemetry investigation."""
        if context.condition_ref is None and context.condition_ref_error:
            evaluation = ConditionEvaluation.unknown(
                None,
                "invalid_condition_reference",
                context.condition_ref_error,
            )
        else:
            evaluation = await self.condition_evaluators.evaluate(context.condition_ref)
        context.condition_state = evaluation.state.value
        context.condition_state_reason = evaluation.reason_code
        return evaluation

    async def close(self):
        """Close investigator tools and database connections."""
        await self.condition_evaluators.close()
        await self.tools.close()
        if self.db_conn and not self.db_conn.closed:
            self.db_conn.close()
        if self.persistence:
            self.persistence.close()

    def _get_incident_context(self, incident_id: str) -> IncidentContext:
        """Retrieve incident context from the database."""
        if not self.db_conn:
            logger.warning("No database connection, returning mock context")
            return IncidentContext(
                incident_id=incident_id,
                title=f"Incident {incident_id}",
                events=[],
                started_at="2026-08-13T02:00:00Z",
            )

        try:
            with self.db_conn.cursor() as cursor:
                # Get incident info
                cursor.execute(
                    """
                    SELECT id, title, status, started_at, resolved_at, severity, affected_services
                    FROM incidents WHERE id = %s
                """,
                    (incident_id,),
                )
                incident_row = cursor.fetchone()

                if not incident_row:
                    return IncidentContext(
                        incident_id=incident_id,
                        title=f"Incident {incident_id} not found",
                        events=[],
                        started_at="",
                        error=f"Incident {incident_id} not found",
                    )

                # Get events for this incident
                # NOTE: the column is `occurred_at`, not `timestamp`
                cursor.execute(
                    """
                    SELECT e.id, e.source, e.type, e.occurred_at, e.service, e.environment,
                           e.severity, e.title, e.status, e.labels, e.raw_payload,
                           e.condition_ref
                    FROM events e
                    JOIN incident_events ie ON e.id = ie.event_id
                    WHERE ie.incident_id = %s
                    ORDER BY e.occurred_at, e.id
                """,
                    (incident_id,),
                )
                events = cursor.fetchall()
                event_context = [
                    {
                        "id": e[0],
                        "source": e[1],
                        "type": e[2],
                        "timestamp": str(e[3]),
                        "service": e[4],
                        "environment": e[5],
                        "severity": e[6],
                        "title": e[7],
                        "status": e[8],
                        "labels": e[9],
                        "raw_payload": e[10],
                        "condition_ref": e[11],
                    }
                    for e in events
                ]
                condition_ref, condition_ref_error = _select_condition_reference(event_context)

                return IncidentContext(
                    incident_id=incident_row[0],
                    title=incident_row[1],
                    status=incident_row[2],
                    started_at=str(incident_row[3]),
                    resolved_at=str(incident_row[4]) if incident_row[4] else None,
                    severity=incident_row[5],
                    affected_services=incident_row[6],
                    events=event_context,
                    condition_ref=condition_ref,
                    condition_ref_error=condition_ref_error,
                )
            # db_conn runs in autocommit mode, so each statement completes
            # immediately and no transaction is left open.
        except (psycopg2.Error, RuntimeError) as e:
            logger.error("Failed to get incident context: %s", e)
            # Roll back the aborted transaction so the connection remains usable
            try:
                self.db_conn.rollback()
            except psycopg2.Error:
                pass
            return IncidentContext(
                incident_id=incident_id,
                title="Error loading incident",
                events=[],
                started_at="",
                error=str(e),
            )

    def _truncate_context(self, text: str, limit: int) -> str:
        """Truncate context to prevent excessive growth."""
        if len(text) <= limit:
            return text
        # Keep beginning and end, accounting for truncation marker
        marker = "\n... [TRUNCATED] ...\n"
        marker_len = len(marker)
        half = (limit - marker_len) // 2
        return text[:half] + marker + text[-half:]

    def _build_system_prompt(self, context: IncidentContext | None = None) -> str:
        """Build the bounded, evidence-grounded investigator system prompt."""
        scope = ""
        if context is not None:
            affected = ", ".join(context.affected_services or []) or "unspecified"
            scope = (
                "\n=== SESSION SCOPE ===\n"
                f"This session investigates exactly one incident: {context.incident_id}.\n"
                f"Title: {context.title}\n"
                f"Affected service(s) for this incident: {affected}\n"
                "Every finding you submit must describe THIS incident. Evidence "
                "about a different host, service or alert is not this incident's "
                "cause, even when it looks similar: say so and leave it out. Use "
                "get_related_incidents only to tell the operator whether this "
                "condition keeps recurring — never as the cause of this "
                "occurrence.\n"
                f"{_condition_reference_prompt(context)}"
                "=== END SESSION SCOPE ===\n\n"
            )
        return scope + (
            "You are Pamawas Investigator. Determine what happened, its scope, the best "
            "supported explanation, and the next operator action. Be accurate before being "
            "complete. An honest UNKNOWN is better than an unsupported conclusion.\n\n"
            "OPERATING CONTRACT:\n"
            "- The current incident is the investigation premise, not a finding. Its ID, "
            "title, severity, target, start time, status, affected services, and original "
            "alert firing are already known. Use them to scope queries; do not submit them "
            "as evidence.\n"
            "- New evidence comes from returned tool results, additional post-trigger events, "
            "or related history that answers an unresolved question.\n"
            "- Tool definitions supplied to this request are authoritative. Never claim a "
            "tool or data source was used, available, or unavailable unless its definition "
            "or result is present.\n"
            "- Prometheus and Loki are the primary evidence sources in this tool set. Consider "
            "both for a full investigation, but call each only to answer a distinct question, "
            "not merely to complete a checklist.\n"
            "- Keep this incident separate from neighbouring incidents. Related incidents "
            "may establish recurrence, but cannot establish this occurrence's cause.\n\n"
            "PROMETHEUS AND LOKI STEERING:\n"
            "- Use query_prometheus to establish whether the reported condition exists in "
            "telemetry, its magnitude, when it began, how it changed, and whether it recovered.\n"
            "Prefer the alert's original expression and identifiers when available.\n"
            "- Use query_loki over the same target and time window to look for discrete errors, "
            "restarts, state changes, saturation messages, dependency failures, or other events "
            "that can explain or contradict the metric timeline.\n"
            "- If the Prometheus selector schema is unknown, call get_prometheus_labels once. "
            "If the Loki selector schema is unknown, call get_loki_labels once. Then request "
            "values for one promising returned label and immediately run the data query. Never "
            "repeat catalogs and never treat catalog output as evidence.\n"
            "- A full investigation MUST obtain usable evidence from at least one relevant "
            "provider. Attempt the other provider only when it can answer a distinct causal "
            "question and its discovered schema covers the incident target. Some targets have "
            "metrics without logs or logs without metrics; absence from one provider is a source "
            "gap, not a reason to discard evidence from the other. Catalog and label discovery "
            "never supply incident facts.\n"
            "- The alert's primary metric or log confirms the symptom, not its cause. After that "
            "first observation, use the same provider or another relevant provider for at least "
            "one materially distinct causal query when available. Changing only the time range "
            "of the same expression refines its timeline but does not create a second causal "
            "signal.\n"
            "- Prometheus HTTP success with zero numeric series is not evidence and does not "
            "satisfy the gate. When the harness returns metric_candidates, retry with an exact "
            "returned metric name and an evidence-appropriate PromQL transformation.\n"
            "- Preserve measurement semantics and units for every alert type. Report a "
            "percentage only when the alert/provider supplies one or the executed query has "
            "an explicit evidence-backed denominator. Transform counters with rate/increase "
            "and label the derived unit honestly (for example bytes per second, requests per "
            "second, or operations per second); never relabel an arbitrary counter rate "
            "as percent.\n"
            "- A zero-entry Loki result is accepted only after the harness validates every "
            "selector label and positive value against Loki's discovered schema. If validation "
            "fails, retry using the labels and values returned by the harness.\n"
            "- If a data query fails, correct it using only discovered schema and make a bounded, "
            "materially different attempt. If a source remains unavailable, do not fabricate "
            "evidence; the harness will preserve successful observations and request a "
            "partial write-up with the source gap reported as UNKNOWN.\n"
            "- Do not stop after validating the alert metric. Metrics describe what changed; "
            "logs may explain why. Conversely, a log line without matching telemetry does not "
            "prove it caused the incident.\n\n"
            "DECISION PROCEDURE:\n"
            "1. Establish the reported symptom, target, start time, current status, and "
            "reported blast radius from the incident events.\n"
            "2. Identify the smallest set of unresolved questions needed to explain the "
            "incident. Consider the primary signal, correlated symptoms, dependencies, "
            "recent changes when a change source exists, and recurrence when history matters.\n"
            "3. Query the provider that carries the primary signal and its timeline. Use the "
            "other provider for explanatory or contradictory evidence only when it has relevant "
            "target coverage. Query related incidents when recurrence changes the operator "
            "action.\n"
            "4. Use the alert's original identifiers, expression, and time window when available. "
            "When a backend schema is unknown, perform one bounded discovery pass, validate "
            "a candidate key and value, then run the data query. Never invent schema.\n"
            "5. Compare the result with the incident timeline and competing explanations. "
            "Correlation alone is not a cause; a cause needs temporal alignment and a "
            "plausible mechanism supported by evidence.\n"
            "6. Stop when the primary question has a supported answer, or when the remaining "
            "uncertainty cannot be reduced with the available tools. Then submit findings.\n\n"
            "TOOL DISCIPLINE:\n"
            "- Every call must answer one distinct unresolved question.\n"
            "- Discovery output describes schema; it is not incident evidence.\n"
            "- Never repeat an identical or equivalent call. After an empty result, make at "
            "most one materially different attempt using validated schema.\n"
            "- Treat unavailable, error, and empty as different outcomes. An empty result is "
            "meaningful only after validating the selector and time window.\n"
            "- Preserve the returned measurement semantics and units. Do not reinterpret an "
            "availability signal as utilization, a cumulative counter as a point value, or a "
            "raw number as a percentage without an evidence-backed transformation.\n"
            "- Each turn must add evidence, test a different hypothesis, or submit.\n\n"
            "EVIDENCE CONTRACT:\n"
            "- Evidence content must be ready for a human incident report before submission. "
            "State what changed, readable values with known units, and local observation times. "
            "Describe important log events in plain sentences. Keep exact queries and "
            "identifiers in source_detail; do not put query syntax, sample counts, complete "
            "label maps, source-code paths, or scientific notation in content.\n"
            "- FACT: a new observation from a successful data query, a post-trigger lifecycle "
            "event, or related history. Restating the triggering alert is not a finding.\n"
            f"{EVIDENCE_POLICY.prompt_contract()}"
            "- RECOMMENDATION: a concrete next action tied to a finding or unknown.\n"
            "Use source=prometheus only for a successful Prometheus data query, source=loki "
            "only for returned log data, source=incidents only for new related-history or "
            "post-trigger lifecycle evidence, "
            "source=llm for synthesis or a tool without a dedicated source enum, and "
            "source=system for harness limits. Set source_detail to the exact executed query, "
            "tool call, or exact incident/event identifier. A catalog call cannot support a "
            "FACT. Never claim a source was not checked after its result is already present.\n\n"
            "SUBMISSION:\n"
            "- Use submit_findings; prose alone does not complete the investigation.\n"
            "- Submit after at least one relevant provider has returned usable data. If another "
            "provider was attempted but lacks target coverage or exhausts bounded attempts, record "
            "that source gap and continue. Do not submit after catalog discovery alone.\n"
            "- Every full investigation MUST contain exactly one causal verdict: LIKELY_CAUSE "
            "when the mechanism is strongly supported, HYPOTHESIS when it is plausible but "
            "unconfirmed, or UNKNOWN explicitly stating that the root cause was not established. "
            f"The {EVIDENCE_POLICY.version} validator may downgrade or replace a model verdict "
            "that does not meet the policy printed above. "
            "A recommendation without a causal verdict is incomplete. Use source=llm for a "
            "causal synthesis and cite the exact executed query in source_detail. Explain "
            "the observed metric change or specific log event that supports the mechanism. "
            "A log count, warning level, or temporal overlap alone is insufficient.\n"
            "- Report only substantive findings: the key observed facts, that one causal "
            "verdict, important unknowns, and at least one specific recommendation.\n"
            "- Never submit a FACT whose only content is that this incident exists, fired, "
            "has a severity, targets a service, or started at the supplied time.\n"
            "- Keep each finding to one or two plain sentences. State the conclusion and its "
            "evidence, not your investigation process. Do not pad the report.\n"
            "- Confidence is confidence in the stated claim. UNKNOWN findings use 0.0.\n"
        )

    def _build_verification_prompt(self, context: IncidentContext | None = None) -> str:
        """Build the system prompt for a stale-incident verification.

        A verification answers one question — "is this condition actually still
        down?" — that decides whether a silent incident is closed or kept open.
        It is deliberately narrower than a full investigation: the model is told
        to determine the current state of the affected target and stop, not to
        chase a root cause.
        """
        now_dt = datetime.now(tz.utc)
        now_iso = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        fifteen_min_ago = (now_dt - timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ")

        affected = "unspecified"
        if context is not None:
            affected = ", ".join(context.affected_services or []) or "unspecified"
        incident = context.incident_id if context else "unknown"
        title = context.title if context else "unknown"

        return (
            "\n=== SESSION SCOPE (VERIFICATION) ===\n"
            f"This session verifies one incident: {incident}.\n"
            f"Title: {title}\n"
            f"Affected service(s): {affected}\n"
            f"CURRENT LIVE TIME (NOW UTC): {now_iso}\n\n"
            f"{_condition_reference_prompt(context)}\n"
            "Answer ONE question: is this condition actually still down right now?\n"
            "- Check ONLY the current/live state right now.\n"
            f"- Query Prometheus for the live window (e.g. start='{fifteen_min_ago}', "
            f"end='{now_iso}').\n"
            "- Call get_prometheus_labels / get_loki_labels first if the label schema is "
            "unknown.\n"
            "- Label discovery is not evidence. A successful query_prometheus call is "
            "required before submit_findings is accepted.\n"
            "- DO NOT query historical time ranges from hours ago.\n"
            "- DO NOT investigate past incident history or root causes.\n"
            "- Submit a FACT: 'The target has recovered' if current metrics are healthy, "
            "or 'The target is still down' if currently failing.\n"
            "- An UNKNOWN must not claim recovery.\n"
            "=== END SESSION SCOPE ===\n"
        )

    def _build_initial_message(
        self, context: IncidentContext, verification: bool = False
    ) -> dict:
        """Build the initial user message with incident context."""
        now_dt = datetime.now(tz.utc)
        now_iso = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        fifteen_min_ago = (now_dt - timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ")

        if verification:
            return {
                "role": "user",
                "content": (
                    f"Verify whether this condition has recovered as of right now.\n\n"
                    f"Current evaluation time: {now_iso}\n"
                    f"Incident ID: {context.incident_id}\n"
                    f"Title: {context.title}\n"
                    f"Status: {context.status}\n"
                    f"Originally started at: {context.started_at}\n"
                    f"Severity: {context.severity}\n"
                    f"Affected services: {', '.join(context.affected_services or [])}\n\n"
                    f"Original triggering events:\n"
                    f"{json.dumps(context.events, indent=2)[: self.config.truncation_limit]}\n\n"
                    f"Query Prometheus metrics for the live window (e.g. {fifteen_min_ago} "
                    f"to {now_iso}) to determine if the target is healthy right now. "
                    "Submit your findings immediately."
                ),
            }

        return {
            "role": "user",
            "content": (
                f"Investigate this incident:\n\n"
                f"Current time: {now_iso}\n"
                f"Incident ID: {context.incident_id}\n"
                f"Title: {context.title}\n"
                f"Status: {context.status}\n"
                f"Started at: {context.started_at}\n"
                f"Resolved at: {context.resolved_at or 'not resolved'}\n"
                f"Severity: {context.severity}\n"
                f"Affected services: {', '.join(context.affected_services or [])}\n\n"
                f"Events in this incident:\n"
                f"{json.dumps(context.events, indent=2)[: self.config.truncation_limit]}\n\n"
                "Begin your investigation by understanding what happened. Use your "
                "tools to gather evidence and form hypotheses."
            ),
        }

    @staticmethod
    def _checkpoint_object(value: Any, fallback: Any) -> Any:
        """Normalize JSONB values returned by psycopg2 or lightweight fakes."""
        if value is None:
            return fallback
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return fallback
        return value

    def _hydrate_state_from_checkpoint(
        self,
        state: InvestigationState,
        expected_run_kind: str,
    ) -> str | None:
        """Reconstruct bounded state for a retry of the same persisted run."""
        if not self.persistence or not state.request_key_hash:
            return None
        checkpoint = self.persistence.load_run_checkpoint(
            state.incident_id, state.request_key_hash
        )
        if not isinstance(checkpoint, dict):
            return None

        run_kind = str(checkpoint.get("run_kind") or "investigation")
        if run_kind != expected_run_kind:
            raise RuntimeError(
                f"Run kind mismatch for {state.incident_id}: expected "
                f"{expected_run_kind}, found {run_kind}"
            )
        checkpoint_run_id = str(checkpoint.get("run_id") or "")
        if state.run_id and checkpoint_run_id and state.run_id != checkpoint_run_id:
            raise RuntimeError(
                f"Run ID mismatch for {state.incident_id}: expected {state.run_id}, "
                f"found {checkpoint_run_id}"
            )
        state.run_id = checkpoint_run_id or state.run_id
        state.run_kind = run_kind
        checkpoint_revision_id = checkpoint.get("catalog_revision_id")
        state.catalog_revision_id = (
            str(checkpoint_revision_id) if checkpoint_revision_id else None
        )

        restored_tools: list[ToolResult] = []
        for row in checkpoint.get("tool_executions") or []:
            if not isinstance(row, (list, tuple)) or len(row) < 5:
                continue
            _, tool_name, raw_arguments, raw_result, duration_ms = row[:5]
            arguments = self._checkpoint_object(raw_arguments, {})
            result = self._checkpoint_object(raw_result, {})
            restored_tools.append(
                ToolResult(
                    tool_name=str(tool_name),
                    arguments=arguments if isinstance(arguments, dict) else {},
                    result=result if isinstance(result, dict) else {"summary": result},
                    duration_ms=float(duration_ms or 0),
                )
            )

        restored_findings = self._restore_checkpoint_findings(
            checkpoint.get("evidence") or [], state.run_id
        )

        state.tool_calls = restored_tools
        state.findings = restored_findings
        public_tools = [call for call in restored_tools if not _is_private_tool_call(call)]
        state.tool_call_count = len(public_tools)
        # LLM turns are not persisted because raw conversation transcripts are
        # deliberately excluded. Count each restored tool execution as one
        # consumed turn: conservative, bounded, and never resets a retry to a
        # fresh budget after substantial work was already completed.
        state.turn_count = min(len(public_tools), state.max_tool_calls)
        for call in public_tools:
            signature = self._call_signature(call.tool_name, call.arguments)
            state.seen_calls[signature] = self._truncate_result(call.result)
            state.seen_call_counts.setdefault(signature, 0)

        state.resumed_from_checkpoint = bool(restored_tools or restored_findings)
        status = str(checkpoint.get("status") or "")
        state.completed = status == "completed"
        if state.resumed_from_checkpoint:
            logger.info(
                "Reconstructed run %s with %d tool execution(s) and %d finding(s)",
                state.run_id,
                len(restored_tools),
                len(restored_findings),
            )
        return status or None

    @staticmethod
    def _restore_checkpoint_findings(
        raw_rows: list[Any], run_id: str | None
    ) -> list[Finding]:
        """Restore finding relationships from durable evidence IDs to indexes."""
        rows = [
            row
            for row in raw_rows
            if isinstance(row, (list, tuple)) and len(row) >= 9
        ]
        rows.sort(key=lambda row: int(row[8]))
        indexes = {str(row[0]): index for index, row in enumerate(rows)}
        restored: list[Finding] = []
        for row in rows:
            try:
                restored.append(
                    Finding(
                        type=EvidenceType(str(row[1])),
                        content=str(row[2]),
                        source=EvidenceSource(str(row[3])),
                        source_detail=str(row[4] or ""),
                        confidence=float(row[5]),
                        supports_evidence=[
                            indexes[str(evidence_id)]
                            for evidence_id in (row[6] or [])
                            if str(evidence_id) in indexes
                        ],
                        contradicts_evidence=[
                            indexes[str(evidence_id)]
                            for evidence_id in (row[7] or [])
                            if str(evidence_id) in indexes
                        ],
                    )
                )
            except (TypeError, ValueError):
                logger.warning("Skipping invalid checkpoint evidence for run %s", run_id)
        return restored

    def _build_resume_message(self, state: InvestigationState) -> dict:
        """Build the compact conversation checkpoint supplied on a resumed turn."""
        tools = [
            {
                "sequence": index,
                "tool": call.tool_name,
                "arguments_redacted": call.arguments,
                "result_summary": _public_tool_result(call.result),
            }
            for index, call in enumerate(state.tool_calls)
            if not _is_private_tool_call(call)
        ]
        findings = [finding.to_dict() for finding in state.findings]
        checkpoint = self._truncate_context(
            json.dumps(
                {"prior_tool_executions": tools, "prior_validated_findings": findings},
                default=str,
                sort_keys=True,
            ),
            self.config.truncation_limit,
        )
        return {
            "role": "user",
            "content": (
                "Resume the SAME persisted investigation attempt. Raw prompts and raw "
                "provider responses are intentionally not retained. Use this redacted, "
                "bounded checkpoint as prior conversation state; do not repeat identical "
                "queries and do not assume details absent from the summaries.\n\n"
                f"{checkpoint}"
            ),
        }

    def _resume_messages(self, state: InvestigationState) -> list[dict]:
        """Return the optional reconstructed context without branching in the loop."""
        return [self._build_resume_message(state)] if state.resumed_from_checkpoint else []

    def write_day_summary(self, facts: dict) -> str:
        """Write the daily report's prose summary from the given facts.

        LLM usage stays in this service; the reporter only calls the endpoint.
        Returns "" on any failure so the digest still delivers.
        """
        return _write_day_summary(self.llm, facts)

    def _handle_error_context(self, context: IncidentContext) -> list[Finding]:
        """Handle error context and return error finding."""
        logger.error(f"Failed to get incident context: {context.error}")
        return [
            Finding(
                type=EvidenceType.UNKNOWN,
                content=f"Failed to load incident context: {context.error}",
                source=EvidenceSource.SYSTEM,
                confidence=0.0,
            )
        ]

    @staticmethod
    def _check_verification_time_range(end: Any) -> str | None:
        """Check whether a verification query end timestamp is too old."""
        try:
            end_clean = str(end).strip()
            end_dt = datetime.fromisoformat(end_clean.replace("Z", "+00:00"))
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=tz.utc)
            now_dt = datetime.now(tz.utc)
            age_seconds = (now_dt - end_dt).total_seconds()
            if age_seconds > 1800:
                now_str = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                age_min = int(age_seconds / 60)
                return (
                    f"Verification checks current live health, but your query end timestamp "
                    f"({end}) is {age_min} minutes old. "
                    f"Please query the live window ending at current time: {now_str}."
                )
        except (ValueError, TypeError, AttributeError):
            pass
        return None

    async def _invalid_loki_selector_values(
        self, matchers: list[tuple[str, str, str]]
    ) -> dict[str, dict[str, Any]] | None:
        """Return invalid positive selector values, or None for a legacy adapter."""
        invalid_values: dict[str, dict[str, Any]] = {}
        for label, operator, expected in matchers:
            if operator not in ("=", "=~"):
                continue
            try:
                response = await self.tools._loki_adapter.query_label_values(label)
            except AttributeError:
                logger.warning(
                    "Loki adapter has no label-value method; skipping validation"
                )
                return None
            except LokiError as exc:
                invalid_values[label] = {"error": str(exc)}
                continue
            raw_values = response.get("data", []) if isinstance(response, dict) else []
            values = [str(value) for value in raw_values]
            if not _loki_value_matches(operator, expected, values):
                invalid_values[label] = {
                    "operator": operator,
                    "requested": expected,
                    "available_values": values[:20],
                }
        return invalid_values

    async def _validate_empty_loki_result(
        self, logql: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Treat empty Loki data as evidence only after validating its selector."""
        matchers = _logql_matchers(logql)
        if not matchers:
            return {
                "status": "empty",
                "error": (
                    "Loki returned no entries and the selector could not be parsed for "
                    "schema validation. Retry with a selector built from discovered labels."
                ),
                "data": result.get("data", {}),
            }

        try:
            catalog = await self.tools._loki_adapter.query_labels()
        except AttributeError:
            logger.warning("Loki adapter has no label-catalog method; skipping validation")
            return result
        except LokiError as exc:
            return {
                "status": "empty",
                "error": f"Loki selector validation failed: {exc}",
                "data": result.get("data", {}),
            }

        available_labels = catalog.get("data", []) if isinstance(catalog, dict) else []
        available_set = {str(label) for label in available_labels}
        missing_labels = sorted({label for label, _, _ in matchers} - available_set)
        if missing_labels:
            return {
                "status": "empty",
                "error": (
                    "Loki returned no entries because selector labels were not present "
                    f"in the discovered schema: {', '.join(missing_labels)}."
                ),
                "data": result.get("data", {}),
                "available_labels": sorted(available_set),
            }

        invalid_values = await self._invalid_loki_selector_values(matchers)
        if invalid_values is None:
            return result
        if invalid_values:
            return {
                "status": "empty",
                "error": (
                    "Loki returned no entries and one or more selector values were not "
                    "validated. Retry using returned label values."
                ),
                "data": result.get("data", {}),
                "invalid_selector_values": invalid_values,
            }
        return result

    async def _execute_tool(  # noqa: C901
        self,
        function_name: str,
        function_args: dict,
        verification: bool = False,
        context: IncidentContext | None = None,
    ):
        """Execute a tool call and return the result.

        Runs in the asyncio event loop, so it awaits the async adapter
        methods directly (not the sync `self.tools.*` wrappers, which call
        ``loop.run_until_complete`` and would fail inside a running loop).
        Loki/Prometheus parse errors are caught here and returned as
        structured error dicts (status="error") so a single bad query
        does not terminate the whole investigation run.
        """
        try:
            if not isinstance(function_args, dict) or "_parse_error" in function_args:
                err_msg = (
                    function_args.get("_parse_error", "tool arguments must be an object")
                    if isinstance(function_args, dict)
                    else "tool arguments must be an object"
                )
                return (
                    {
                        "status": "error",
                        "error": f"{function_name}: invalid arguments JSON ({err_msg})",
                    },
                    None,
                    False,
                )

            if function_name == "query_prometheus":
                # Guard against a missing start/end/promql argument. A tool call
                # from the LLM may omit required keys; a direct [] lookup would
                # raise KeyError and abort the whole investigation with a useless
                # "Investigation error: 'start'" finding.
                promql = function_args.get("promql")
                start = function_args.get("start")
                end = function_args.get("end")
                if promql is None or start is None or end is None:
                    return (
                        {
                            "status": "error",
                            "error": "query_prometheus requires promql, start, end",
                        },
                        None,
                        False,
                    )
                if verification:
                    stale_error = self._check_verification_time_range(end)
                    if stale_error is not None:
                        return (
                            {"status": "error", "error": stale_error},
                            None,
                            False,
                        )
                result = await self.tools._prometheus_adapter.query_range(promql, start, end)
                if (
                    isinstance(result, dict)
                    and result.get("status") == "success"
                    and not _prometheus_has_numeric_samples(result)
                ):
                    if context is not None and context.telemetry_catalog_revision_id:
                        pinned = (context.telemetry_catalog or {}).get("prometheus", {})
                        metric_catalog: Any = (
                            list(pinned.values()) if isinstance(pinned, dict) else []
                        )
                    else:
                        metric_catalog = []
                        try:
                            catalog = (
                                await self.tools._prometheus_adapter.query_metric_catalog()
                            )
                            metric_catalog = catalog.get("data", [])
                        except (AttributeError, PrometheusError) as exc:
                            logger.warning(
                                "Prometheus metric-name discovery failed after empty "
                                "query: %s",
                                exc,
                            )
                    candidates, candidate_audit = _prometheus_metric_candidate_plan(
                        promql, metric_catalog, context=context
                    )
                    logger.warning(
                        "Prometheus query returned no numeric series; retry candidates: %s",
                        candidates,
                    )
                    return (
                        {
                            "status": "empty",
                            "error": (
                                "Prometheus returned no usable numeric series. This "
                                "does not satisfy the evidence gate. Retry with a "
                                "materially different expression using an exact metric "
                                "name from metric_candidates. Preserve only selectors "
                                "validated by the incident or discovered schema; use "
                                "rate/increase for counter metrics when appropriate."
                            ),
                            "data": result.get("data", {}),
                            "metric_candidates": candidates,
                            "_candidate_selection_audit": candidate_audit,
                        },
                        None,
                        False,
                    )
                return result, None, False

            elif function_name == "get_prometheus_labels":
                label = function_args.get("label") or ""
                if label:
                    result = await self.tools._prometheus_adapter.query_label_values(label)
                else:
                    result = await self.tools._prometheus_adapter.query_labels()
                return result, None, False

            elif function_name == "get_loki_labels":
                label = function_args.get("label") or ""
                if label:
                    result = await self.tools._loki_adapter.query_label_values(label)
                else:
                    result = await self.tools._loki_adapter.query_labels()
                return result, None, False

            elif function_name == "query_loki":
                logql = function_args.get("logql")
                start = function_args.get("start")
                end = function_args.get("end")
                if logql is None or start is None or end is None:
                    return (
                        {
                            "status": "error",
                            "error": "query_loki requires logql, start, end",
                        },
                        None,
                        False,
                    )
                if _LOGQL_SELECTOR.search(str(logql)) is None:
                    return (
                        {
                            "status": "error",
                            "error": (
                                "query_loki requires a LogQL stream selector such as "
                                "{job=\"...\"}; do not send PromQL or a metric expression"
                            ),
                        },
                        None,
                        False,
                    )
                normalized_logql = _normalize_logql_literals(str(logql))
                if normalized_logql != logql:
                    logger.warning(
                        "Normalized single-quoted LogQL literals before execution"
                    )
                    logql = normalized_logql
                    function_args["logql"] = normalized_logql
                raw_limit = function_args.get("limit", 100)
                try:
                    limit = int(raw_limit)
                except (TypeError, ValueError):
                    limit = 100
                bounded_limit = min(max(limit, 1), self.config.loki_max_log_lines)
                if bounded_limit != raw_limit:
                    logger.warning(
                        "Bounded Loki query limit from %r to %d",
                        raw_limit,
                        bounded_limit,
                    )
                function_args["limit"] = bounded_limit
                if verification:
                    stale_error = self._check_verification_time_range(end)
                    if stale_error is not None:
                        return (
                            {"status": "error", "error": stale_error},
                            None,
                            False,
                        )
                result = await self.tools._loki_adapter.query_range(
                    logql,
                    start,
                    end,
                    bounded_limit,
                )
                if (
                    isinstance(result, dict)
                    and result.get("status") == "success"
                    and not _loki_has_entries(result)
                ):
                    result = await self._validate_empty_loki_result(logql, result)
                return result, None, False

            elif function_name == "get_related_incidents":
                if verification:
                    return (
                        {
                            "status": "error",
                            "error": (
                                "get_related_incidents is disabled during verification. "
                                "Query Prometheus for current live metrics."
                            ),
                        },
                        None,
                        False,
                    )
                service = function_args.get("service")
                keywords = function_args.get("symptom_keywords") or []
                if service is None:
                    return (
                        {
                            "status": "error",
                            "error": "get_related_incidents requires service",
                        },
                        None,
                        False,
                    )
                result = await self.tools._related_incidents_adapter.find_related(service, keywords)
                return result, None, False

            elif function_name == "submit_findings":
                raw = function_args.get("findings")
                if not isinstance(raw, list) or not raw:
                    return (
                        {
                            "status": "error",
                            "error": "submit_findings requires a non-empty findings list",
                        },
                        None,
                        False,
                    )
                findings = []
                for f in raw:
                    if not isinstance(f, dict):
                        return (
                            {
                                "status": "error",
                                "error": "submit_findings: each finding must be an object",
                            },
                            None,
                            False,
                        )
                    ftype = f.get("type")
                    content = f.get("content")
                    source = f.get("source")
                    confidence = f.get("confidence")
                    if not ftype or content is None or not source or confidence is None:
                        return (
                            {
                                "status": "error",
                                "error": "submit_findings: finding missing required fields",
                            },
                            None,
                            False,
                        )
                    # A finding's source decides how the report labels it and
                    # whether the claim is checkable, so an unrecognized value
                    # is a tool-call error the model must correct rather than
                    # something to coerce into a default.
                    try:
                        evidence_source = EvidenceSource(source)
                    except ValueError:
                        return (
                            {
                                "status": "error",
                                "error": (
                                    "submit_findings: source must be one of "
                                    "prometheus, loki, incidents, llm, system"
                                ),
                            },
                            None,
                            False,
                        )
                    findings.append(
                        Finding(
                            type=EvidenceType(ftype),
                            content=content,
                            source=evidence_source,
                            confidence=confidence,
                            source_detail=f.get("source_detail") or "",
                        )
                    )
                result = self.tools.submit_findings(findings)
                return result, findings, True  # result, findings, is_final
        except (
            LokiError,
            PrometheusError,
            DeploymentAdapterError,
            RelatedIncidentsError,
        ) as e:
            # A query/tool failure is a structured error, not a run-ending
            # crash. Return it as an error dict so the loop can continue.
            logger.warning("Tool %s returned an error: %s", function_name, e)
            return {"status": "error", "error": str(e)}, None, False
        except Exception as e:
            # Unexpected crash — propagate so the investigation-loop handler
            # (not this function) decides how to persist it.
            logger.error("Tool %s crashed unexpectedly: %s", function_name, e)
            raise
        else:
            return {"error": f"Unknown tool: {function_name}"}, None, False

    def _truncate_result(self, result: any) -> str:
        """Truncate a model-visible result while retaining private audits in state."""
        result_str = json.dumps(_public_tool_result(result))
        if len(result_str) > self.config.truncation_limit:
            result_str = self._truncate_context(result_str, self.config.truncation_limit)
        return result_str

    @staticmethod
    def _call_signature(function_name: str, function_args: dict) -> str:
        """A stable identity for one tool call.

        Arguments are sorted and serialized so that two spellings of the same
        request — same keys in a different order — compare equal. A model
        re-asking a question rarely re-types it identically.
        """
        try:
            rendered = json.dumps(function_args or {}, sort_keys=True, default=str)
        except (TypeError, ValueError):
            rendered = str(function_args)
        return f"{function_name}:{rendered}"

    def _serve_repeated_call(
        self,
        tool_call,
        function_name: str,
        function_args: dict,
        state: InvestigationState,
        messages: list[dict],
    ) -> bool:
        """Answer a repeated tool call from the first result.

        Returns True when the call was served from the cache, in which case the
        caller must not execute it. The API requires a message for every tool
        call id, so the model still gets a reply — the same result it already
        has, plus a note saying so. `tool_call_count` is deliberately not
        incremented: the point is that a repeat costs no budget.

        `submit_findings` is never cached. It is the terminal action, and its
        arguments carry the findings themselves.
        """
        if function_name == "submit_findings":
            return False

        signature = self._call_signature(function_name, function_args)
        previous = state.seen_calls.get(signature)
        if previous is None:
            return False

        repeat_count = state.seen_call_counts.get(signature, 0) + 1
        state.seen_call_counts[signature] = repeat_count

        logger.warning(
            "Repeated tool call served from cache (repeat #%d): %s (args=%s)",
            repeat_count,
            function_name,
            function_args,
        )
        if repeat_count >= 2:
            note = (
                f"You have already made this exact call {repeat_count + 1} times. "
                "Its result is unchanged. STOP REPEATING THIS QUERY. "
                "Either query a completely different metric/log stream or "
                "submit your findings now with submit_findings.\n\n" + previous
            )
        else:
            note = (
                "You already made this exact call earlier in this "
                "investigation. Its result is unchanged and repeated below. "
                "Do not call it again. Either query something different or "
                "submit your findings with submit_findings.\n\n" + previous
            )

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": note,
            }
        )
        return True

    def _handle_tool_result(
        self,
        tool_call,
        function_name: str,
        function_args: dict,
        result: any,
        tool_start: float,
        state: InvestigationState,
        messages: list[dict],
    ):
        """Process tool result and update state."""
        result_str = self._truncate_result(result)

        # Add tool result to conversation
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result_str})

        state.tool_calls.append(
            ToolResult(
                tool_name=function_name,
                # The resolved arguments, not the envelope: a nested call would
                # otherwise be recorded as {"name": ..., "arguments": {...}}.
                arguments=function_args,
                result=result,
                duration_ms=(time.time() - tool_start) * 1000,
            )
        )

        state.tool_call_count += 1

    def _check_max_tools_reached(
        self, state: InvestigationState, must_submit: bool, messages: list[dict]
    ):
        """Check if max tool calls reached and prompt for submission if needed."""
        if state.tool_call_count >= state.max_tool_calls and not must_submit:
            logger.info("Reached max tool calls, forcing findings submission")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have reached the maximum number of tool calls. "
                        "Please submit your findings using the submit_findings tool."
                    ),
                }
            )

    def _handle_text_response(
        self, response, state: InvestigationState, must_submit: bool, messages: list[dict]
    ):
        """Handle text response (no tool calls)."""
        logger.info(f"LLM response (no tool calls): {response.content[:100]}...")
        state.consecutive_text_turns += 1

        if must_submit:
            # Either the turn budget or the patience budget is gone, and the
            # turn was a forced submission — the API was told it may only call
            # submit_findings. A text reply therefore means the endpoint ignored
            # tool_choice. Take what the model said as a hypothesis rather than
            # discarding it: a partial answer beats the null "deadline exceeded"
            # result this path used to produce after burning the full wall-clock
            # budget.
            logger.warning(
                "Text response without tool use (%d consecutive) - recording it as a hypothesis",
                state.consecutive_text_turns,
            )
            content = (response.content or "").strip()
            if not content:
                # Nothing was written, so there is no hypothesis to record.
                # Inventing one ("Investigation completed via text response")
                # put a sentence on the report that the model never said.
                state.findings.append(
                    Finding(
                        type=EvidenceType.UNKNOWN,
                        content=(
                            "Investigation ended without a stated conclusion; "
                            "the model returned no text and no findings."
                        ),
                        source=EvidenceSource.SYSTEM,
                        confidence=0.0,
                    )
                )
                state.completed = True
                return
            if _is_reasoning_preamble(content):
                # The model narrated its process instead of stating a
                # conclusion ("Let me analyze the data more carefully. I need to
                # convert the Unix timestamps..."). Storing that as a finding
                # put a wall of chain-of-thought under "🎯 Likely cause" on a
                # critical card. There is no conclusion here to record, but the
                # run still needs an honest terminal row so the report says the
                # investigation ended without one rather than showing nothing.
                logger.warning("Discarding reasoning preamble as a finding: %s", content[:120])
                state.findings.append(
                    Finding(
                        type=EvidenceType.UNKNOWN,
                        content=(
                            "Investigation ended without a stated conclusion; the "
                            "model only described its process."
                        ),
                        source=EvidenceSource.SYSTEM,
                        confidence=0.0,
                    )
                )
                state.completed = True
                return
            state.findings.append(
                Finding(
                    type=EvidenceType.HYPOTHESIS,
                    content=_plain_typography(content),
                    source=EvidenceSource.LLM,
                    confidence=0.3,
                )
            )
            state.completed = True
        else:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Continue your investigation. Use your tools to gather more "
                        "evidence, or submit your findings when ready."
                    ),
                }
            )

    async def _handle_submit_findings(
        self,
        findings: list[Finding],
        incident_id: str,
        request_key_hash: str,
        state: InvestigationState,
        verification: bool = False,
    ):
        """Handle submit_findings tool call."""
        logger.info("Investigation completed via submit_findings")
        findings = [*findings, *state.findings]
        findings = _finalize_findings(
            findings, state.tool_calls, state.context, incident_id, verification
        )
        state.findings = findings
        state.completed = True

        if self.persistence:
            await self._persist_investigation(
                incident_id=incident_id,
                request_key_hash=request_key_hash,
                findings=findings,
                tool_calls=state.tool_calls,
                model_provider=self.config.llm_base_url,
                model_name=self.config.llm_model,
                prompt_version=PROMPT_VERSION,
                tool_contract=1,
                max_tool_calls=state.max_tool_calls,
                service=state.service,
                run_kind=state.run_kind,
            )

    async def _close_verification_on_repeat_budget(
        self,
        incident_id: str,
        request_key_hash: str,
        state: InvestigationState,
        function_name: str,
    ) -> None:
        """After 2 repeated tool calls in verification, synthesize findings from cached results.

        The model had two chances to query live data; repeating further adds no new
        information. Preserve the cached Prometheus and Loki evidence without inferring
        recovery from any provider-specific metric name, label, or value threshold.
        """
        logger.warning(
            "Verification for incident %s exceeded repeat budget on %s; "
            "synthesizing findings from cached tool results",
            incident_id,
            function_name,
        )

        prometheus_result, loki_result = _cached_verification_results(state.seen_calls)
        cached_findings = (
            _prometheus_verification_finding(prometheus_result),
            _loki_verification_finding(loki_result),
        )
        state.findings.extend(finding for finding in cached_findings if finding is not None)

        # Fallback if we still have no findings
        if not any(
            not finding.source_detail.startswith("condition evaluator=")
            for finding in state.findings
        ):
            state.findings.append(
                Finding(
                    type=EvidenceType.UNKNOWN,
                    content="Unable to determine target status from cached verification data",
                    source=EvidenceSource.SYSTEM,
                    confidence=0.0,
                )
            )

        # Mark as completed so the loop terminates with a verdict.
        state.completed = True
        state.findings = _finalize_findings(
            state.findings, state.tool_calls, state.context, incident_id, True
        )
        await self._persist_investigation(
            incident_id=incident_id,
            request_key_hash=request_key_hash,
            findings=state.findings,
            tool_calls=state.tool_calls,
            model_provider=self.config.llm_base_url,
            model_name=self.config.llm_model,
            prompt_version=PROMPT_VERSION,
            tool_contract=1,
            max_tool_calls=state.max_tool_calls,
            service=state.service,
            run_kind=state.run_kind,
        )

    async def investigate_async(
        self,
        incident_id: str,
        request_key_hash: str = "",
        run_id: str | None = None,
    ) -> list[Finding]:
        """Run the bounded investigation loop; every terminal path persists
        the run and its evidence.

        Concurrency is bounded outside this method, by the worker pool that
        claims runs from the database. The deadline below starts here, so time
        spent queued is not charged against the investigation's own budget.
        """
        logger.info(f"Starting investigation for incident {incident_id}")
        increment_investigations()

        # Overall investigation deadline (wall-clock). Once exceeded, we stop
        # calling tools and submit whatever partial findings we have.
        deadline = time.monotonic() + self.config.investigation_timeout_seconds

        # Get incident context
        context = self._get_incident_context(incident_id)
        if hasattr(context, "error") and context.error:
            return await self._handle_error_context_path(
                incident_id=incident_id,
                request_key_hash=request_key_hash,
                context=context,
            )
        # Initialize investigation state
        state = InvestigationState(
            incident_id=incident_id,
            findings=[],
            tool_calls=[],
            max_tool_calls=self.config.max_tool_calls,
            request_key_hash=request_key_hash,
            run_id=run_id,
            run_kind="investigation",
            service=context.affected_services[0] if context.affected_services else None,
        )
        checkpoint_status = self._hydrate_state_from_checkpoint(state, "investigation")
        if checkpoint_status == "completed":
            return state.findings
        if self.persistence is not None and not state.run_id:
            state.run_id, _ = self.persistence.create_investigation_run(
                incident_id=incident_id,
                request_key_hash=request_key_hash,
                model_provider=self.config.llm_base_url,
                model_name=self.config.llm_model,
                prompt_version=PROMPT_VERSION,
                tool_contract=1,
                max_tool_calls=state.max_tool_calls,
                run_kind=state.run_kind,
            )

        evaluation = await self._evaluate_condition(context)
        state.findings = [
            finding
            for finding in state.findings
            if not finding.source_detail.startswith("condition evaluator=")
        ]
        state.findings.append(
            _condition_evaluation_finding(context.condition_ref, evaluation)
        )
        await self._refresh_telemetry_catalog(context, state.run_id)
        state.catalog_revision_id = context.telemetry_catalog_revision_id

        return await self._run_investigation_loop(
            incident_id=incident_id,
            request_key_hash=request_key_hash,
            state=state,
            context=context,
            deadline=deadline,
        )

    async def verify_incident(
        self,
        incident_id: str,
        request_key_hash: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Return a source-authoritative verdict for one stale incident.

        This audits a stale-open incident: the Scheduler discovered an incident
        silent past INCIDENT_STALE_AFTER, enqueued it in stale_verifications,
        and the async stale-verification worker executes this bounded check. A
        normal state from the condition's own evaluator is the only path to a
        recovered verdict. Metrics, logs, titles, and model prose cannot close it.
        """
        request_key_hash = request_key_hash or (
            f"verification:{incident_id}:{uuid.uuid4().hex}"
        )
        context = self._get_incident_context(incident_id)
        if hasattr(context, "error") and context.error:
            findings = self._handle_error_context(context)
            await self._persist_investigation(
                incident_id=incident_id,
                request_key_hash=request_key_hash,
                findings=findings,
                tool_calls=[],
                model_provider=self.config.llm_base_url,
                model_name=self.config.llm_model,
                prompt_version=PROMPT_VERSION,
                tool_contract=1,
                max_tool_calls=self.config.verify_max_tool_calls,
                service=(context.affected_services or [None])[0],
                run_kind="verification",
            )
            return {
                "verdict": "inconclusive",
                "reason": context.error,
                "findings": [f.to_dict() for f in findings],
            }

        evaluation = await self._evaluate_condition(context)
        finding = _condition_evaluation_finding(context.condition_ref, evaluation)
        findings = [finding]
        if evaluation.authoritative and evaluation.state == ConditionState.NORMAL:
            verdict = "recovered"
        elif evaluation.authoritative and evaluation.state in {
            ConditionState.FIRING,
            ConditionState.PENDING,
        }:
            verdict = "still_down"
        else:
            verdict = "inconclusive"
        reason = finding.content

        await self._persist_investigation(
            incident_id=incident_id,
            request_key_hash=request_key_hash,
            findings=findings,
            tool_calls=[],
            model_provider=self.config.llm_base_url,
            model_name=self.config.llm_model,
            prompt_version=PROMPT_VERSION,
            tool_contract=1,
            max_tool_calls=self.config.verify_max_tool_calls,
            service=(context.affected_services or [None])[0],
            run_kind="verification",
        )

        return {
            "verdict": verdict,
            "reason": reason,
            "findings": [f.to_dict() for f in findings],
            "condition_state": evaluation.state.value,
            "authoritative": evaluation.authoritative,
        }

    async def _handle_error_context_path(
        self,
        incident_id: str,
        request_key_hash: str,
        context: IncidentContext,
    ) -> list[Finding]:
        """Persist an error-context finding and return the findings."""
        findings = _finalize_findings(
            self._handle_error_context(context), [], context, incident_id, False
        )
        await self._persist_investigation(
            incident_id=incident_id,
            request_key_hash=request_key_hash,
            findings=findings,
            tool_calls=[],
            model_provider=self.config.llm_base_url,
            model_name=self.config.llm_model,
            prompt_version=PROMPT_VERSION,
            tool_contract=1,
            max_tool_calls=self.config.max_tool_calls,
            service=context.affected_services[0] if context.affected_services else None,
        )
        return findings

    def _submission_pressure(
        self,
        incident_id: str,
        remaining: float,
        warned: bool,
        forced_already: bool,
        state: InvestigationState,
        messages: list[dict],
        verification: bool = False,
    ) -> tuple[bool, bool]:
        """Decide whether this turn must be a forced submission.

        Two things can force it: the wall-clock deadline entering its reserve,
        and the model spending its no-tool patience on prose instead of a
        submission. Keeping both here means the loop stays one branch.
        """
        force_submit, warned = self._deadline_action(
            incident_id=incident_id,
            remaining=remaining,
            warned=warned,
            forced_already=forced_already,
            messages=messages,
        )
        if force_submit or forced_already:
            return force_submit, warned
        if self._escalate_text_patience(
            incident_id, state, messages, verification=verification
        ):
            return True, warned
        return (
            self._escalate_repeat_patience(
                incident_id, state, messages, verification=verification
            ),
            warned,
        )

    def _enforce_missing_data_queries(
        self,
        incident_id: str,
        state: InvestigationState,
        messages: list[dict],
        reason: str,
        verification: bool = False,
    ) -> bool:
        """Activate the hard data-query gate when required sources are missing."""
        missing = _pending_data_tools(state, verification)
        if not missing:
            return False

        state.required_query_enforcement = True
        logger.warning(
            "Incident %s cannot submit findings yet; enforcing required data queries: %s",
            incident_id,
            ", ".join(missing),
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{reason} Discovery calls do not count as incident evidence. "
                    f"Before submission, execute successful data queries with: "
                    f"{', '.join(missing)}. The harness will reject findings until "
                    "those queries succeed. Use only schema learned from tool results; "
                    "do not invent label names or values."
                ),
            }
        )
        return True

    def _escalate_repeat_patience(
        self,
        incident_id: str,
        state: InvestigationState,
        messages: list[dict],
        verification: bool = False,
    ) -> bool:
        """Force a structured submission once the model repeatedly queries cached data.

        A model that issues identical tool calls across consecutive turns has
        stopped discovering new information and entered an exploratory loop.
        Forcing submit_findings prevents spinning through the entire turn budget
        and ensures the evidence already collected is reported.

        Returns True when this turn must be a forced submission.
        """
        if state.consecutive_repeated_turns < self.MAX_CONSECUTIVE_REPEATED_TURNS:
            return False
        if self._enforce_missing_data_queries(
            incident_id=incident_id,
            state=state,
            messages=messages,
            reason="You are repeating discovery instead of querying data.",
            verification=verification,
        ):
            state.consecutive_repeated_turns = 0
            return False
        logger.warning(
            "Incident %s issued %d consecutive turns of repeated tool calls; "
            "forcing findings submission",
            incident_id,
            state.consecutive_repeated_turns,
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "You are repeatedly issuing the same queries without gathering new data. "
                    "Stop querying and call submit_findings now to record your conclusions "
                    "based on the evidence you already gathered."
                ),
            }
        )
        increment_forced_submission()
        return True

    def _escalate_text_patience(
        self,
        incident_id: str,
        state: InvestigationState,
        messages: list[dict],
        verification: bool = False,
    ) -> bool:
        """Force a structured submission once the model stops calling tools.

        A model that has written several turns without calling a tool and
        without submitting is not going to submit on its own. Giving up here is
        what produced cards that said "UNKNOWN (0%): the model only described
        its process" on a critical incident even though the model had just seen
        real evidence: its prose was discarded while a structured answer was
        still one forced call away.

        Returns True when this turn must be a forced submission.
        """
        if state.consecutive_text_turns < self.MAX_CONSECUTIVE_TEXT_TURNS:
            return False
        if self._enforce_missing_data_queries(
            incident_id=incident_id,
            state=state,
            messages=messages,
            reason="You stopped before gathering the required data.",
            verification=verification,
        ):
            state.consecutive_text_turns = 0
            return False
        logger.warning(
            "Incident %s replied %d times without a tool call; forcing findings "
            "submission instead of recording a null result",
            incident_id,
            state.consecutive_text_turns,
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Stop writing prose. Call submit_findings now and record what you "
                    "have established from the evidence you already gathered, "
                    "including anything you are still unsure about."
                ),
            }
        )
        increment_forced_submission()
        return True

    def _deadline_action(
        self,
        incident_id: str,
        remaining: float,
        warned: bool,
        forced_already: bool,
        messages: list[dict],
    ) -> tuple[bool, bool]:
        """Decide what the remaining wall-clock budget means for this turn.

        Returns (force_submit, warned). The budget is spent in three phases:
        exploring, then a nudge to wrap up, then a hard requirement to submit.
        The reserve exists so evidence already gathered reaches the report
        instead of being thrown away as "deadline exceeded".
        """
        reserve = max(
            self.DEADLINE_RESERVE_MIN_SECONDS,
            self.config.investigation_timeout_seconds * self.DEADLINE_RESERVE_FRACTION,
        )
        if remaining <= reserve:
            if not forced_already:
                logger.warning(
                    "Incident %s is into its deadline reserve (%.0fs left); "
                    "forcing findings submission",
                    incident_id,
                    remaining,
                )
                increment_forced_submission()
            return True, warned

        if not warned and remaining <= reserve * (1 + self.DEADLINE_WARN_FRACTION):
            # A softer warning first: the model can still choose to submit on
            # its own terms, which reads better than a forced summary.
            logger.info(
                "Incident %s has %.0fs left; asking the model to wrap up",
                incident_id,
                remaining,
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"You have about {int(remaining)} seconds left. Stop "
                        "investigating and submit your findings now with "
                        "submit_findings, reporting what you have established "
                        "so far."
                    ),
                }
            )
            return False, True

        return False, warned

    async def _run_investigation_loop(
        self,
        incident_id: str,
        request_key_hash: str,
        state: InvestigationState,
        context: IncidentContext,
        deadline: float,
        verification: bool = False,
    ) -> list[Finding]:
        """Run the bounded investigation tool-calling loop and persist final findings."""
        state.context = context
        # Build system prompt and initial message
        system_prompt = (
            self._build_verification_prompt(context)
            if verification
            else self._build_system_prompt(context)
        )
        initial_message = self._build_initial_message(
            context, verification=verification
        )

        messages = [{"role": "system", "content": system_prompt}, initial_message]
        messages.extend(self._resume_messages(state))

        # Investigation loop. Bounded on TURNS (LLM calls), not tool calls: a
        # turn may issue zero tool calls (a prose reply), and if only tool calls
        # advanced the counter such a turn would loop forever until the
        # wall-clock deadline and end with a synthesized UNKNOWN.
        # The reserve exists so the model has time to write up what it found.
        # Exploring must stop when the reserve opens, which is earlier than the
        # hard deadline; tool execution enforces this boundary.
        reserve = max(
            self.DEADLINE_RESERVE_MIN_SECONDS,
            self.config.investigation_timeout_seconds * self.DEADLINE_RESERVE_FRACTION,
        )
        explore_deadline = deadline - reserve

        exhausted_by = ""
        warned_about_deadline = False
        forced_submission_started = False
        while state.turn_count < state.max_tool_calls:
            # Enforce overall deadline: if no time remains, stop and submit.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                exhausted_by = "deadline"
                logger.warning(
                    "Investigation deadline exceeded for incident %s, submitting partial findings",
                    incident_id,
                )
                break

            force_submit, warned_about_deadline = self._submission_pressure(
                incident_id=incident_id,
                remaining=remaining,
                warned=warned_about_deadline,
                forced_already=forced_submission_started,
                state=state,
                messages=messages,
                verification=verification,
            )
            missing_required = _pending_data_tools(state, verification)
            if force_submit and missing_required and time.monotonic() >= explore_deadline:
                exhausted_by = "required_data"
                logger.warning(
                    "Incident %s reached its submission reserve without successful "
                    "required data queries: %s",
                    incident_id,
                    ", ".join(missing_required),
                )
                break
            if force_submit:
                forced_submission_started = True

            state.turn_count += 1
            logger.info(
                "Investigation turn %d/%d (tool calls so far: %d)",
                state.turn_count,
                state.max_tool_calls,
                state.tool_call_count,
            )

            try:
                result = await self._handle_investigation_turn(
                    incident_id=incident_id,
                    request_key_hash=request_key_hash,
                    state=state,
                    messages=messages,
                    force_submit=force_submit,
                    deadline=deadline,
                    explore_deadline=explore_deadline,
                    verification=verification,
                )
                if result is not None:
                    return result
            except (
                openai.OpenAIError,
                RuntimeError,
                ValueError,
                KeyError,
                TypeError,
                TimeoutError,
            ) as e:
                logger.error("Error in investigation loop: %s", e)
                state.findings.append(
                    Finding(
                        type=EvidenceType.UNKNOWN,
                        content=(
                            "Investigation ended because an internal processing error occurred; "
                            "no raw exception details were included in report evidence."
                        ),
                        source=EvidenceSource.SYSTEM,
                        source_detail=f"internal error class: {type(e).__name__}",
                        confidence=0.0,
                    )
                )
                state.findings = _finalize_findings(
                    state.findings,
                    state.tool_calls,
                    state.context,
                    incident_id,
                    verification,
                )
                await self._persist_investigation(
                    incident_id=incident_id,
                    request_key_hash=request_key_hash,
                    findings=state.findings,
                    tool_calls=state.tool_calls,
                    model_provider=self.config.llm_base_url,
                    model_name=self.config.llm_model,
                    prompt_version=PROMPT_VERSION,
                    tool_contract=1,
                    max_tool_calls=state.max_tool_calls,
                    service=state.service,
                    run_kind=state.run_kind,
                )
                return state.findings

        # If we exit the loop without submitting findings (deadline or turn
        # budget), synthesize a submission from whatever partial findings and
        # tool results we have.
        if not any(
            not finding.source_detail.startswith("condition evaluator=")
            for finding in state.findings
        ):
            reason = _exhaustion_reason(
                exhausted_by, state.tool_calls, verification
            )
            state.findings.extend(
                _ground_partial_data_queries(state.tool_calls, verification, state.context)
            )
            if state.tool_calls:
                successful_data_queries = len(_required_data_tools(verification)) - len(
                    _missing_required_data_tools(state.tool_calls, verification)
                )
                content = (
                    f"{reason} without submitting findings. "
                    f"{successful_data_queries} required data queries succeeded; "
                    "discovery calls were not treated as evidence."
                )
            else:
                content = f"{reason} without calling any tools"
            state.findings.append(
                Finding(
                    type=EvidenceType.UNKNOWN,
                    content=content,
                    source=EvidenceSource.SYSTEM,
                    confidence=0.0,
                )
            )

        state.findings = _finalize_findings(
            state.findings,
            state.tool_calls,
            state.context,
            incident_id,
            verification,
        )
        increment_findings_by_type(state.findings)
        if exhausted_by in ("deadline", "required_data"):
            increment_deadline_exceeded()
        else:
            increment_max_tool_calls_reached()
        await self._persist_investigation(
            incident_id=incident_id,
            request_key_hash=request_key_hash,
            findings=state.findings,
            tool_calls=state.tool_calls,
            model_provider=self.config.llm_base_url,
            model_name=self.config.llm_model,
            prompt_version=PROMPT_VERSION,
            tool_contract=1,
            max_tool_calls=state.max_tool_calls,
            service=state.service,
            run_kind=state.run_kind,
        )

        return state.findings

    @staticmethod
    def _tool_definition(tool_name: str, verification: bool) -> dict | None:
        """Return one tool definition by name."""
        return next(
            (
                tool
                for tool in ToolRegistry.get_tools(verification=verification)
                if tool.get("function", {}).get("name") == tool_name
            ),
            None,
        )

    def _reject_submission_without_required_data(
        self,
        tool_call,
        state: InvestigationState,
        messages: list[dict],
        verification: bool,
    ) -> bool:
        """Reject submit_findings until required data-query tools have succeeded."""
        missing = _pending_data_tools(state, verification)
        if not missing:
            return False

        state.required_query_enforcement = True
        error = (
            "submit_findings rejected: discovery and incident metadata are not "
            "evidence. Execute successful data queries with "
            f"{', '.join(missing)} before submitting findings."
        )
        logger.warning(error)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({"status": "error", "error": error}),
            }
        )
        return True

    def _record_required_query_failure(
        self,
        state: InvestigationState,
        messages: list[dict],
        verification: bool,
    ) -> bool:
        """Bound forced-query failures; return True when the run must terminate."""
        missing = _pending_data_tools(state, verification)
        if not missing:
            state.required_query_enforcement = False
            state.required_query_failures = 0
            return False

        state.required_query_enforcement = True
        state.required_query_failures += 1
        if state.required_query_failures < self.MAX_REQUIRED_QUERY_FAILURES:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The required data query did not succeed. Retry with a valid, "
                        "materially different query using only schema returned by the "
                        f"tools. Still missing: {', '.join(missing)}."
                    ),
                }
            )
            return False

        if not verification and all(
            _successful_tool_calls(state.tool_calls, name) for name in missing
        ):
            # The provider already returned usable primary evidence; only the
            # additional, materially distinct causal query failed. Waive the
            # second-observation preference without falsely claiming that the
            # provider itself was unavailable.
            state.exhausted_data_tools.update(missing)
            state.required_query_enforcement = False
            messages.append({"role": "user", "content": (
                "Additional distinct query attempts are exhausted. Submit a low-confidence "
                "cause assessment and a concrete test based on the successful observation. "
                "Do not claim the successful provider returned no data."
            )})
            return False

        if not verification and any(
            _successful_tool_calls(state.tool_calls, name)
            for name in _FULL_INVESTIGATION_DATA_TOOLS
        ):
            state.exhausted_data_tools.update(missing)
            state.required_query_enforcement = False
            state.findings.append(
                Finding(
                    type=EvidenceType.UNKNOWN,
                    content=f"No usable data from {', '.join(missing)} after bounded attempts.",
                    source=EvidenceSource.SYSTEM,
                    source_detail="required data-query enforcement",
                    confidence=0.0,
                )
            )
            messages.append({"role": "user", "content": (
                "Provider retries are exhausted. Submit a cause assessment and suggested "
                "actions from the successful query evidence now. Include the source gap "
                "as UNKNOWN. One provider can support a cause; two are not mandatory."
            )})
            return False

        # Keep trustworthy evidence from every provider query that did succeed.
        # A broken source must not hide deterministic evidence already returned
        # by another source. Causal synthesis remains explicitly labeled and
        # confidence-capped rather than being promoted to provider fact.
        state.findings = _ground_partial_data_queries(
            state.tool_calls, verification, state.context
        )
        state.findings.append(
            Finding(
                type=EvidenceType.UNKNOWN,
                content=(
                    "Investigation stopped because required data queries did not "
                    f"succeed after enforced attempts: {', '.join(missing)}."
                ),
                source=EvidenceSource.SYSTEM,
                source_detail="required data-query enforcement",
                confidence=0.0,
            )
        )
        state.findings = _finalize_findings(
            state.findings,
            state.tool_calls,
            state.context,
            state.incident_id,
            verification,
        )
        state.completed = True
        return True

    def _record_recovery_call(
        self, call: ToolResult, state: InvestigationState
    ) -> None:
        """Persist one harness call without exposing private audits to the model cache."""
        state.tool_calls.append(call)
        if _is_private_tool_call(call):
            logger.info("Persisting private harness audit %s", call.tool_name)
            return
        state.tool_call_count += 1
        state.seen_calls[self._call_signature(call.tool_name, call.arguments)] = (
            self._truncate_result(call.result)
        )
        logger.info("Harness recovery tool %s with args %s", call.tool_name, call.arguments)

    async def _recover_data_query(
        self, name: str, arguments: dict, result: dict,
        state: InvestigationState, messages: list[dict],
        explore_deadline: float | None, verification: bool,
    ) -> None:
        """Run one bounded recovery pass per provider before asking the model to retry."""
        if (
            verification or name not in _FULL_INVESTIGATION_DATA_TOOLS
            or name in state.recovered_data_tools or state.context is None
            or not state.context.started_at
        ):
            return
        if name == "query_loki" and result.get("status") == "success":
            return
        remaining = 60.0 if explore_deadline is None else explore_deadline - time.monotonic()
        if remaining <= 0:
            return
        state.recovered_data_tools.add(name)
        recovery = QueryRecovery(self.tools, state.context, self.config.loki_max_log_lines)
        try:
            async with asyncio.timeout(min(60.0, remaining)):
                if name == "query_loki":
                    await recovery.recover_loki()
                elif result.get("status") == "success":
                    await recovery.counter_rate(arguments)
                    await recovery.recover_related_prometheus(
                        primary_expression=str(arguments.get("promql") or "")
                    )
                else:
                    candidates = result.get("metric_candidates", [])
                    await recovery.recover_prometheus(candidates)
        except (AttributeError, TimeoutError, ValueError) as exc:
            logger.warning("Bounded %s query recovery ended: %s", name, exc)
        if recovery.counter_metric:
            result["counter_only"] = True
            state.seen_calls[self._call_signature(name, arguments)] = self._truncate_result(result)
        for call in recovery.calls:
            self._record_recovery_call(call, state)
        if recovery.calls:
            summaries = _ground_findings_in_data_queries([], recovery.calls, state.context)
            messages.append({"role": "user", "content": (
                "The harness executed schema-discovered recovery queries. Use these "
                "observations and exact queries for the cause assessment and suggested actions:\n"
                + json.dumps({
                    "observations": [finding.to_dict() for finding in summaries],
                    "target_coverage": recovery.target_coverage,
                })
            )})

    async def _handle_investigation_turn(  # noqa: C901
        self,
        incident_id: str,
        request_key_hash: str,
        state: InvestigationState,
        messages: list[dict],
        force_submit: bool = False,
        deadline: float | None = None,
        explore_deadline: float | None = None,
        verification: bool = False,
    ) -> list[Finding] | None:
        """Run a single investigation turn (LLM call + tool handling);
        returns findings if complete."""
        is_final_turn = state.turn_count >= state.max_tool_calls
        missing_required = _pending_data_tools(state, verification)
        force_submit = force_submit or bool(state.exhausted_data_tools)
        can_start_data_query = (
            explore_deadline is None or time.monotonic() < explore_deadline
        )
        forced_required_query = (
            missing_required[0]
            if state.required_query_enforcement
            and missing_required
            and can_start_data_query
            else None
        )
        # A forced submission is a hard requirement, not a hint: the model is
        # told to call submit_findings and the API refuses anything else, so the
        # turn cannot be spent on another query with no time left to report it.
        must_submit = (is_final_turn or force_submit) and forced_required_query is None
        if forced_required_query is not None:
            required_tool = self._tool_definition(forced_required_query, verification)
            tools = (
                [required_tool]
                if required_tool is not None
                else ToolRegistry.get_tools(verification=verification)
            )
            tool_choice = {
                "type": "function",
                "function": {"name": forced_required_query},
            }
        elif force_submit:
            submit_tool = [
                t
                for t in ToolRegistry.get_tools(verification=verification)
                if t.get("function", {}).get("name") == "submit_findings"
            ]
            tools = (
                submit_tool
                if submit_tool
                else ToolRegistry.get_tools(verification=verification)
            )
            tool_choice = {"type": "function", "function": {"name": "submit_findings"}}
        elif is_final_turn:
            tools = ToolRegistry.get_tools(verification=verification)
            tool_choice = {"type": "function", "function": {"name": "submit_findings"}}
        else:
            tools = ToolRegistry.get_tools(verification=verification)
            tool_choice = "auto"

        # Call the LLM off the asyncio event loop: chat_completion is a
        # synchronous SDK call (retries with backoff) that would block health
        # probes and kill the background investigation task.
        #
        # An exploring turn may spend what is left before the reserve opens, not
        # what is left before the hard deadline. Using the deadline here let one
        # slow explore call eat the reserve that exists to write the findings,
        # and the run then ended with nothing to report. The forced-submission
        # turn is the write-up itself, so it may use the whole remainder.
        budget_deadline = deadline if must_submit else (explore_deadline or deadline)
        budget = None if budget_deadline is None else max(0.0, budget_deadline - time.monotonic())
        response = await asyncio.to_thread(
            self.llm.chat_completion,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            session=incident_id,
            budget_seconds=budget,
        )

        # Add the assistant's response to the conversation
        messages.append({"role": "assistant", "content": response.content or ""})

        # Handle tool calls
        tool_calls = getattr(response, "tool_calls", None)

        # A forced turn that comes back with anything other than submit_findings
        # means the endpoint ignored tool_choice. Running that query would spend
        # the reserve that exists to write the findings, and the run would end
        # with nothing to show — the exact failure the reserve was added to
        # prevent. Treat the reply as the final answer instead.
        if (
            force_submit
            and forced_required_query is None
            and tool_calls
            and not any(_resolved_tool_name(tc) == "submit_findings" for tc in tool_calls)
        ):
            logger.warning(
                "Forced submission turn returned tool calls instead of findings; "
                "completing with the model's text"
            )
            tool_calls = None

        if tool_calls:
            # Real progress: the model is still investigating, so the
            # no-tool patience budget starts over.
            state.consecutive_text_turns = 0
            has_fresh_tool_call = False
            forced_required_query_succeeded = False
            for tool_call in tool_calls:
                function_name, function_args = _resolve_tool_call(tool_call)

                if (
                    forced_required_query is not None
                    and function_name != forced_required_query
                ):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(
                                {
                                    "status": "error",
                                    "error": (
                                        f"Only {forced_required_query} is allowed until "
                                        "the required data-query gate is satisfied."
                                    ),
                                }
                            ),
                        }
                    )
                    continue

                if function_name == "submit_findings" and (
                    self._reject_submission_without_required_data(
                        tool_call=tool_call,
                        state=state,
                        messages=messages,
                        verification=verification,
                    )
                ):
                    continue

                # The reserve is a wall, not a checkpoint. A single turn may
                # carry several tool calls and each can take up to the tool
                # timeout, so the reserve can be consumed entirely inside one
                # turn: a real run made 13 successful tool calls, blew through
                # its 90-second reserve, hit the hard deadline at the top of the
                # next iteration and submitted nothing. No new tool call starts
                # once the reserve opens — submit_findings always runs, because
                # that is the write-up the reserve exists to protect.
                explore_until = explore_deadline if explore_deadline is not None else deadline
                if (
                    explore_until is not None
                    and function_name != "submit_findings"
                    and time.monotonic() >= explore_until
                ):
                    logger.warning(
                        "Investigation is into its reserve; not starting further tool calls"
                    )
                    return None

                logger.info(f"Calling tool: {function_name} with args: {function_args}")

                # Answer a repeated question from the first answer instead of
                # paying for it again. A real run asked for the same Loki label
                # list eleven times, received the same fifteen names each time,
                # and reached the tool limit having learned nothing after the
                # second call.
                repeated = self._serve_repeated_call(
                    tool_call, function_name, function_args, state, messages
                )
                if repeated:
                    required_data_tool = _DISCOVERY_TO_DATA_TOOL.get(function_name)
                    if required_data_tool in _pending_data_tools(state, verification):
                        state.required_query_enforcement = True
                    # Verification is a single "is the target still down?"
                    # question. A repeated query returns the same data, adds
                    # nothing new, and only wastes budget before the verdict.
                    # Close after 2 repeats; non-verification investigations
                    # keep their own escalation path (MAX_CONSECUTIVE_REPEATED_TURNS).
                    if verification and state.seen_call_counts.get(
                        self._call_signature(function_name, function_args), 0
                    ) >= 2:
                        logger.warning(
                            "Verification for incident %s exceeded repeat budget "
                            "on %s; closing with the evidence gathered so far",
                            incident_id,
                            function_name,
                        )
                        await self._close_verification_on_repeat_budget(
                            incident_id=incident_id,
                            request_key_hash=request_key_hash,
                            state=state,
                            function_name=function_name,
                        )
                        return state.findings
                    continue

                has_fresh_tool_call = True
                tool_start = time.time()
                result, findings, is_final = await self._execute_tool(
                    function_name,
                    function_args,
                    verification=verification,
                    context=state.context,
                )

                if is_final:
                    await self._handle_submit_findings(
                        findings,
                        incident_id,
                        request_key_hash,
                        state,
                        verification=verification,
                    )
                    increment_findings_by_type(state.findings)
                    return state.findings

                state.seen_calls[self._call_signature(function_name, function_args)] = (
                    self._truncate_result(result)
                )

                self._handle_tool_result(
                    tool_call, function_name, function_args, result, tool_start, state, messages
                )
                await self._recover_data_query(
                    function_name, function_args, result, state, messages,
                    explore_deadline, verification,
                )
                if (
                    function_name == forced_required_query
                    and function_name not in _pending_data_tools(state, verification)
                ):
                    forced_required_query_succeeded = True

                # Incremental persistence after each non-final tool call
                # so that a crash or deadline still leaves a terminal run.
                await self._persist_incremental(
                    incident_id=incident_id,
                    request_key_hash=request_key_hash,
                    findings=state.findings,
                    tool_calls=state.tool_calls,
                    model_provider=self.config.llm_base_url,
                    model_name=self.config.llm_model,
                    prompt_version=PROMPT_VERSION,
                    tool_contract=1,
                    max_tool_calls=state.max_tool_calls,
                    run_kind=state.run_kind,
                )

                self._check_max_tools_reached(state, must_submit, messages)

            if forced_required_query is not None:
                if forced_required_query_succeeded:
                    state.required_query_failures = 0
                    still_missing = _pending_data_tools(state, verification)
                    state.required_query_enforcement = bool(still_missing)
                    if still_missing:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "The previous required data query succeeded. "
                                    f"Now execute: {', '.join(still_missing)}."
                                ),
                            }
                        )
                elif self._record_required_query_failure(
                    state=state,
                    messages=messages,
                    verification=verification,
                ):
                    increment_findings_by_type(state.findings)
                    await self._persist_investigation(
                        incident_id=incident_id,
                        request_key_hash=request_key_hash,
                        findings=state.findings,
                        tool_calls=state.tool_calls,
                        model_provider=self.config.llm_base_url,
                        model_name=self.config.llm_model,
                        prompt_version=PROMPT_VERSION,
                        tool_contract=1,
                        max_tool_calls=state.max_tool_calls,
                        service=state.service,
                        run_kind=state.run_kind,
                    )
                    return state.findings

            if has_fresh_tool_call:
                state.consecutive_repeated_turns = 0
            else:
                state.consecutive_repeated_turns += 1
            return None

        state.consecutive_repeated_turns = 0
        if forced_required_query is not None or (must_submit and missing_required):
            state.consecutive_text_turns += 1
            if self._record_required_query_failure(
                state=state,
                messages=messages,
                verification=verification,
            ):
                increment_findings_by_type(state.findings)
                await self._persist_investigation(
                    incident_id=incident_id,
                    request_key_hash=request_key_hash,
                    findings=state.findings,
                    tool_calls=state.tool_calls,
                    model_provider=self.config.llm_base_url,
                    model_name=self.config.llm_model,
                    prompt_version=PROMPT_VERSION,
                    tool_contract=1,
                    max_tool_calls=state.max_tool_calls,
                    service=state.service,
                    run_kind=state.run_kind,
                )
                return state.findings
            return None
        self._handle_text_response(response, state, must_submit, messages)
        if state.completed:
            state.findings = _finalize_findings(
                state.findings,
                state.tool_calls,
                state.context,
                incident_id,
                verification,
            )
            increment_findings_by_type(state.findings)
            await self._persist_investigation(
                incident_id=incident_id,
                request_key_hash=request_key_hash,
                findings=state.findings,
                tool_calls=state.tool_calls,
                model_provider=self.config.llm_base_url,
                model_name=self.config.llm_model,
                prompt_version=PROMPT_VERSION,
                tool_contract=1,
                max_tool_calls=state.max_tool_calls,
                service=state.service,
                run_kind=state.run_kind,
            )
            return state.findings
        return None

    async def _persist_investigation(
        self,
        incident_id: str,
        request_key_hash: str,
        findings: list[Finding],
        tool_calls: list[ToolResult],
        model_provider: str,
        model_name: str,
        prompt_version: str,
        tool_contract: int,
        max_tool_calls: int,
        service: str | None = None,
        run_kind: str = "investigation",
    ):
        """Persist investigation run, tool executions, and evidence in a
        transaction (includes enrichment)."""
        if not self.persistence:
            logger.warning("No persistence layer available, skipping investigation persistence")
            return

        try:
            # Create or get investigation run
            run_id, _ = self.persistence.create_investigation_run(
                incident_id=incident_id,
                request_key_hash=request_key_hash,
                model_provider=model_provider,
                model_name=model_name,
                prompt_version=prompt_version,
                tool_contract=tool_contract,
                max_tool_calls=max_tool_calls,
                run_kind=run_kind,
            )

            # Update status to running
            self.persistence.update_run_status(run_id, "running")

            # Insert tool executions
            for i, tool_call in enumerate(tool_calls):
                self.persistence.insert_tool_execution(
                    run_id=run_id,
                    sequence_no=i,
                    tool_name=tool_call.tool_name,
                    arguments=tool_call.arguments,
                    result=tool_call.result,
                    status="completed",
                    duration_ms=int(tool_call.duration_ms),
                )

            # Persist evidence from the exact successful data-lake queries already
            # executed. This is schema-neutral: no second request guesses a label
            # name or maps affected_services[0] onto a deployment-specific key.
            enriched_metrics = _metrics_from_tool_calls(
                tool_calls, max_items=self.config.prometheus_max_series
            )
            enriched_logs = _logs_from_tool_calls(
                tool_calls, max_items=self.config.loki_max_log_lines
            )
            logger.info(
                "Extracted %d metrics and %d logs from executed tool results",
                len(enriched_metrics),
                len(enriched_logs),
            )

            # Every finding reaches the report and the API through this one
            # call, so normalize it here rather than trusting the prompt.
            findings = [_normalize_finding(f) for f in findings]

            # Insert evidence with enrichment
            self.persistence.insert_evidence(
                incident_id=incident_id,
                run_id=run_id,
                findings=findings,
                metrics=enriched_metrics,
                logs=enriched_logs,
                replace_existing=True,
            )

            # Mark run as completed
            self.persistence.update_run_status(
                run_id=run_id,
                status="completed",
                completed_at=datetime.now(UTC),
            )

            logger.info(
                f"Persisted investigation run {run_id} with {len(findings)} findings, "
                f"{len(enriched_metrics)} metrics, "
                f"{len(enriched_logs)} logs"
            )

        except (psycopg2.Error, RuntimeError) as e:
            logger.error("Failed to persist investigation: %s", e)
            # Try to mark run as failed (terminal state; 'failed_terminal' is
            # the only terminal failure value the DB check constraint permits)
            if "run_id" in locals():
                try:
                    self.persistence.update_run_status(run_id, "failed_terminal")
                except (psycopg2.Error, RuntimeError):
                    pass

    async def _persist_incremental(
        self,
        incident_id: str,
        request_key_hash: str,
        findings: list[Finding],
        tool_calls: list[ToolResult],
        model_provider: str,
        model_name: str,
        prompt_version: str,
        tool_contract: int,
        max_tool_calls: int,
        run_kind: str = "investigation",
    ):
        """Persist investigation incrementally after each tool call; upserts
        run and appends tool executions/evidence."""
        if not self.persistence:
            logger.warning("No persistence layer available, skipping incremental persistence")
            return

        try:
            run_id, _ = self.persistence.create_investigation_run(
                incident_id=incident_id,
                request_key_hash=request_key_hash,
                model_provider=model_provider,
                model_name=model_name,
                prompt_version=prompt_version,
                tool_contract=tool_contract,
                max_tool_calls=max_tool_calls,
                run_kind=run_kind,
            )

            # Mark the run as running. Not only when it was just created: the
            # HTTP handler creates the run as `queued` before the work starts,
            # so by the time the first tool result is persisted `created_new`
            # is always False and the run stayed `queued` for its entire life.
            # Anything watching for a terminal status — the scheduler's
            # high-severity follow-up, an operator checking whether a run is
            # alive — saw a run that never started.
            self.persistence.update_run_status(run_id, "running")

            # Insert any new tool executions not yet persisted. We track by
            # sequence_no; insert all tool calls each time is idempotent if the
            # table deduplicates on (run_id, sequence_no), otherwise we only
            # insert beyond the last persisted sequence.
            last_seq = self._last_persisted_sequence(run_id)
            for i, tool_call in enumerate(tool_calls):
                if i <= last_seq:
                    continue
                self.persistence.insert_tool_execution(
                    run_id=run_id,
                    sequence_no=i,
                    tool_name=tool_call.tool_name,
                    arguments=tool_call.arguments,
                    result=tool_call.result,
                    status="completed",
                    duration_ms=int(tool_call.duration_ms),
                )

            # Insert evidence for new findings (append-only)
            self._insert_incremental_evidence(incident_id, run_id, findings)

        except (psycopg2.Error, RuntimeError) as e:
            logger.error("Failed to persist incrementally: %s", e)

    def _last_persisted_sequence(self, run_id: str) -> int:
        """Return the highest persisted tool-execution sequence number for a run."""
        conn = None
        try:
            conn = self.persistence._get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT COALESCE(MAX(sequence_no), -1) FROM tool_executions WHERE run_id = %s",
                    (run_id,),
                )
                row = cursor.fetchone()
            conn.commit()
            return row[0] if row else -1
        except (psycopg2.Error, RuntimeError):
            if conn is not None:
                try:
                    conn.rollback()
                except (psycopg2.Error, AttributeError):
                    pass
            return -1
        finally:
            if conn is not None:
                self.persistence._put_connection(conn)

    def _insert_incremental_evidence(
        self, incident_id: str, run_id: str, findings: list[Finding]
    ) -> None:
        """Insert incremental evidence for new findings (append-only)."""
        if not findings:
            return
        self.persistence.insert_evidence(
            incident_id=incident_id,
            run_id=run_id,
            findings=findings,
            metrics=[],
            logs=[],
        )

    def investigate(self, incident_id: str, request_key_hash: str = "") -> list[Finding]:
        """Synchronous wrapper for investigate_async (backwards compatibility)."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If we're already in an async context, we can't use run_until_complete
            # This shouldn't happen in tests, but just in case
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, self.investigate_async(incident_id, request_key_hash)
                )
                return future.result()
        else:
            return asyncio.run(self.investigate_async(incident_id, request_key_hash))
