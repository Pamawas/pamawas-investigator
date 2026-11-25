"""Typed, bounded incident data supplied to the model as untrusted JSON."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from typing import Any

from models import IncidentContext
from service.evidence_summary import annotation_measurement

CONTEXT_SCHEMA = "pamawas-incident-context-v1"
MAX_EVENTS = 16
MAX_LABELS_PER_EVENT = 16
MAX_CONDITION_LABELS = 16
MAX_AFFECTED_SERVICES = 16
MAX_ANNOTATIONS_PER_EVENT = 8
MAX_ANNOTATIONS_TOTAL = 16
MAX_ANNOTATION_TEXT = 256
MAX_TEXT = 256
MAX_TITLE = 512
MAX_EXPRESSION = 2048

_LABEL_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:/-]{0,127}$")
_EXPRESSION_KEYS = ("expression", "expr", "promql", "logql", "query")
_ANNOTATION_CONTAINER_KEYS = frozenset(
    {"annotations", "commonAnnotations", "common_annotations"}
)
_TOP_LEVEL_FREE_TEXT_KEYS = ("message",)


def _bounded_text(value: Any, limit: int = MAX_TEXT) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    rendered = " ".join(str(value).split())
    if not rendered:
        return None
    return rendered[:limit]


def _label_value(value: Any) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if not isinstance(value, (str, int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return _bounded_text(value, MAX_TEXT)


def _bounded_labels(
    raw: Any,
    targets: set[str],
    preferred_keys: set[str],
    limit: int,
) -> tuple[dict[str, str], int]:
    if not isinstance(raw, dict):
        return {}, 0
    candidates: list[tuple[tuple[int, int, str], str, str]] = []
    rejected = 0
    for raw_key, raw_value in raw.items():
        key = str(raw_key).strip()
        value = _label_value(raw_value)
        if not _LABEL_KEY.fullmatch(key) or value is None:
            rejected += 1
            continue
        priority = (
            0 if value in targets else 1,
            0 if key in preferred_keys else 1,
            key,
        )
        candidates.append((priority, key, value))
    candidates.sort(key=lambda item: item[0])
    selected = {key: value for _, key, value in candidates[:limit]}
    return selected, rejected + max(0, len(candidates) - len(selected))


def _expression(raw_payload: Any) -> str | None:
    if not isinstance(raw_payload, dict):
        return None
    containers = [raw_payload]
    alerts = raw_payload.get("alerts")
    if isinstance(alerts, list):
        containers.extend(alert for alert in alerts if isinstance(alert, dict))
    for container in containers:
        for key in _EXPRESSION_KEYS:
            expression = _bounded_text(container.get(key), MAX_EXPRESSION)
            if expression is not None:
                return expression
    return None


@dataclass(frozen=True)
class PromptMeasurement:
    value: float
    unit: str

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"value": self.value}
        if self.unit:
            result["unit"] = self.unit
        return result


@dataclass(frozen=True)
class PromptAnnotation:
    path: str
    text: str
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "path": self.path,
            "text": self.text,
            "trust": "untrusted_external_text",
        }
        if self.truncated:
            result["truncated"] = True
        return result


def _annotation_text(value: Any) -> tuple[str | None, bool]:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None, False
    rendered = " ".join(str(value).split())
    if not rendered:
        return None, False
    return rendered[:MAX_ANNOTATION_TEXT], len(rendered) > MAX_ANNOTATION_TEXT


def _annotation_entries(
    raw_payload: Any,
) -> tuple[tuple[PromptAnnotation, ...], int, int]:
    if not isinstance(raw_payload, dict):
        return (), 0, 0
    entries: list[PromptAnnotation] = []
    omitted = 0

    for key in sorted(_ANNOTATION_CONTAINER_KEYS):
        selected, rejected, _shortened = _annotation_map_entries(
            raw_payload.get(key), key
        )
        entries.extend(selected)
        omitted += rejected
    for key in _TOP_LEVEL_FREE_TEXT_KEYS:
        text, shortened = _annotation_text(raw_payload.get(key))
        if text is not None:
            entries.append(PromptAnnotation(f"payload.{key}", text, shortened))
    nested, rejected, _shortened = _nested_alert_annotations(
        raw_payload.get("alerts")
    )
    entries.extend(nested)
    omitted += rejected

    entries.sort(key=lambda item: (item.path, item.text))
    selected = tuple(entries[:MAX_ANNOTATIONS_PER_EVENT])
    return (
        selected,
        omitted + max(0, len(entries) - len(selected)),
        sum(annotation.truncated for annotation in selected),
    )


def _annotation_map_entries(
    value: Any, prefix: str
) -> tuple[list[PromptAnnotation], int, int]:
    if not isinstance(value, dict):
        return [], 0, 0
    entries: list[PromptAnnotation] = []
    omitted = 0
    truncated = 0
    for raw_key in sorted(value, key=str):
        key = _bounded_text(raw_key, 128)
        text, shortened = _annotation_text(value.get(raw_key))
        if key is None or text is None:
            omitted += 1
            continue
        entries.append(PromptAnnotation(f"{prefix}.{key}", text, shortened))
        truncated += int(shortened)
    return entries, omitted, truncated


def _nested_alert_annotations(
    alerts: Any,
) -> tuple[list[PromptAnnotation], int, int]:
    if not isinstance(alerts, list):
        return [], 0, 0
    entries: list[PromptAnnotation] = []
    omitted = 0
    truncated = 0
    for index, alert in enumerate(alerts):
        if not isinstance(alert, dict):
            omitted += 1
            continue
        for key in sorted(_ANNOTATION_CONTAINER_KEYS):
            selected, rejected, shortened = _annotation_map_entries(
                alert.get(key), f"alerts[{index}].{key}"
            )
            entries.extend(selected)
            omitted += rejected
            truncated += shortened
    return entries, omitted, truncated


@dataclass(frozen=True)
class PromptEvent:
    event_id: str
    source: str | None
    event_type: str | None
    occurred_at: str | None
    service: str | None
    environment: str | None
    severity: str | None
    title: str | None
    status: str | None
    labels: dict[str, str]
    measurement: PromptMeasurement | None
    expression: str | None
    annotations: tuple[PromptAnnotation, ...]

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"id": self.event_id}
        for key, value in (
            ("source", self.source),
            ("type", self.event_type),
            ("occurred_at", self.occurred_at),
            ("service", self.service),
            ("environment", self.environment),
            ("severity", self.severity),
            ("title", self.title),
            ("status", self.status),
            ("expression", self.expression),
        ):
            if value is not None:
                result[key] = value
        if self.labels:
            result["labels"] = dict(self.labels)
        if self.measurement is not None:
            result["measurement"] = self.measurement.to_dict()
        if self.annotations:
            result["untrusted_annotations"] = [
                annotation.to_dict() for annotation in self.annotations
            ]
        return result


@dataclass(frozen=True)
class PromptIncidentContext:
    incident: dict[str, Any]
    events: tuple[PromptEvent, ...]
    condition: dict[str, Any] | None
    events_omitted: int
    labels_omitted: int
    affected_services_omitted: int
    annotations_omitted: int
    annotation_text_truncated: int

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": CONTEXT_SCHEMA,
            "incident": dict(self.incident),
            "events": [event.to_dict() for event in self.events],
            "omissions": {
                "events_omitted": self.events_omitted,
                "labels_omitted": self.labels_omitted,
                "affected_services_omitted": self.affected_services_omitted,
                "annotations_omitted": self.annotations_omitted,
                "annotation_text_truncated": self.annotation_text_truncated,
                "raw_payload_excluded": True,
                "free_form_annotations_typed": True,
            },
        }
        if self.condition is not None:
            result["authoritative_condition"] = dict(self.condition)
        return result


def _selected_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    ordered = sorted(
        (event for event in events if isinstance(event, dict)),
        key=lambda event: (
            str(event.get("timestamp") or ""),
            str(event.get("id") or ""),
        ),
    )
    if len(ordered) <= MAX_EVENTS:
        return ordered, 0
    head = MAX_EVENTS // 2
    tail = MAX_EVENTS - head
    return [*ordered[:head], *ordered[-tail:]], len(ordered) - MAX_EVENTS


def _bound_event_annotations(
    events: list[PromptEvent],
) -> tuple[list[PromptEvent], int]:
    """Select annotations round-robin so one noisy event cannot consume the budget."""
    selected: list[list[PromptAnnotation]] = [[] for _ in events]
    kept = 0
    for annotation_index in range(MAX_ANNOTATIONS_PER_EVENT):
        for event_index, event in enumerate(events):
            if annotation_index >= len(event.annotations):
                continue
            if kept >= MAX_ANNOTATIONS_TOTAL:
                omitted = sum(len(item.annotations) for item in events) - kept
                bounded = [
                    replace(event, annotations=tuple(selected[index]))
                    for index, event in enumerate(events)
                ]
                return bounded, omitted
            selected[event_index].append(event.annotations[annotation_index])
            kept += 1
    return events, 0


def build_prompt_context(context: IncidentContext) -> PromptIncidentContext:
    targets = {
        target
        for value in context.affected_services or []
        if (target := _bounded_text(value, MAX_TEXT)) is not None
    }
    preferred_keys = (
        set(context.condition_ref.labels) if context.condition_ref is not None else set()
    )
    selected_events, events_omitted = _selected_events(context.events or [])
    title = _bounded_text(context.title, MAX_TITLE)

    affected = sorted(targets)[:MAX_AFFECTED_SERVICES]
    affected_services_omitted = max(0, len(targets) - len(affected))
    labels_omitted = 0
    annotations_omitted = 0
    annotation_text_truncated = 0
    incident: dict[str, Any] = {
        "id": _bounded_text(context.incident_id, 128) or "unknown",
        "status": _bounded_text(context.status, 64) or "unknown",
        "started_at": _bounded_text(context.started_at, 64) or "unknown",
        "resolved_at": _bounded_text(context.resolved_at, 64),
        "severity": _bounded_text(context.severity, 64) or "unknown",
        "affected_services": affected,
    }
    if title is not None:
        incident["title"] = title

    prompt_events: list[PromptEvent] = []
    for event in selected_events:
        labels, omitted = _bounded_labels(
            event.get("labels"), targets, preferred_keys, MAX_LABELS_PER_EVENT
        )
        labels_omitted += omitted
        raw_payload = event.get("raw_payload")
        measured = annotation_measurement(raw_payload)
        annotations, omitted_annotations, _shortened_annotations = _annotation_entries(
            raw_payload
        )
        annotations_omitted += omitted_annotations
        measurement = (
            PromptMeasurement(measured[0], _bounded_text(measured[1], 64) or "")
            if measured is not None
            else None
        )
        prompt_events.append(
            PromptEvent(
                event_id=_bounded_text(event.get("id"), 128) or "unknown",
                source=_bounded_text(event.get("source"), 64),
                event_type=_bounded_text(event.get("type"), 64),
                occurred_at=_bounded_text(event.get("timestamp"), 64),
                service=_bounded_text(event.get("service"), MAX_TEXT),
                environment=_bounded_text(event.get("environment"), 64),
                severity=_bounded_text(event.get("severity"), 64),
                title=_bounded_text(event.get("title"), MAX_TITLE),
                status=_bounded_text(event.get("status"), 64),
                labels=labels,
                measurement=measurement,
                expression=_expression(raw_payload),
                annotations=annotations,
            )
        )

    prompt_events, globally_omitted_annotations = _bound_event_annotations(
        prompt_events
    )
    annotations_omitted += globally_omitted_annotations
    annotation_text_truncated = sum(
        annotation.truncated
        for event in prompt_events
        for annotation in event.annotations
    )

    condition = None
    if context.condition_ref is not None:
        condition_labels, omitted = _bounded_labels(
            context.condition_ref.labels,
            targets,
            preferred_keys,
            MAX_CONDITION_LABELS,
        )
        labels_omitted += omitted
        condition = {
            "source": _bounded_text(context.condition_ref.source, 128),
            "evaluator": _bounded_text(context.condition_ref.evaluator, 128),
            "resource_type": _bounded_text(context.condition_ref.resource_type, 128),
            "resource_id": _bounded_text(context.condition_ref.resource_id, MAX_TEXT),
            "instance_id": _bounded_text(context.condition_ref.instance_id, MAX_TEXT),
            "tenant_id": _bounded_text(context.condition_ref.tenant_id, 128),
            "state": _bounded_text(context.condition_state, 64) or "not evaluated",
            "state_reason": _bounded_text(context.condition_state_reason, 128),
        }
        condition = {key: value for key, value in condition.items() if value is not None}
        if condition_labels:
            condition["labels"] = condition_labels

    return PromptIncidentContext(
        incident=incident,
        events=tuple(prompt_events),
        condition=condition,
        events_omitted=events_omitted,
        labels_omitted=labels_omitted,
        affected_services_omitted=affected_services_omitted,
        annotations_omitted=annotations_omitted,
        annotation_text_truncated=annotation_text_truncated,
    )


def _json_bytes(value: dict[str, Any]) -> bytes:
    rendered = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    # Prevent data values from terminating the surrounding XML-like prompt
    # delimiter. These remain ordinary JSON unicode escapes after decoding.
    rendered = (
        rendered.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return rendered.encode("utf-8")


def _render_within_budget(payload: dict[str, Any], budget: int) -> str | None:
    rendered = _json_bytes(payload)
    return rendered.decode("utf-8") if len(rendered) <= budget else None


def _drop_event_collection(
    payload: dict[str, Any], field: str, omission_field: str, budget: int
) -> str | None:
    for event in reversed(payload["events"]):
        removed = event.pop(field, None)
        if removed:
            payload["omissions"][omission_field] += len(removed)
        if (rendered := _render_within_budget(payload, budget)) is not None:
            return rendered
    return None


def _drop_event_fields(
    payload: dict[str, Any], fields: tuple[str, ...], budget: int
) -> str | None:
    for field in fields:
        for event in reversed(payload["events"]):
            event.pop(field, None)
            if (rendered := _render_within_budget(payload, budget)) is not None:
                return rendered
    return None


def _drop_middle_events(payload: dict[str, Any], budget: int) -> str | None:
    events = payload["events"]
    while len(events) > 2:
        events.pop(-2)
        payload["omissions"]["events_omitted"] += 1
        if (rendered := _render_within_budget(payload, budget)) is not None:
            return rendered
    return None


def _drop_affected_services(payload: dict[str, Any], budget: int) -> str | None:
    affected = payload["incident"].get("affected_services", [])
    while affected:
        affected.pop()
        payload["omissions"]["affected_services_omitted"] += 1
        if (rendered := _render_within_budget(payload, budget)) is not None:
            return rendered
    return None


def _drop_remaining_events(payload: dict[str, Any], budget: int) -> str | None:
    events = payload["events"]
    while events:
        events.pop()
        payload["omissions"]["events_omitted"] += 1
        if (rendered := _render_within_budget(payload, budget)) is not None:
            return rendered
    return None


def _drop_event_labels(payload: dict[str, Any], budget: int) -> str | None:
    return _drop_event_collection(payload, "labels", "labels_omitted", budget)


def _drop_event_annotations(payload: dict[str, Any], budget: int) -> str | None:
    return _drop_event_collection(
        payload, "untrusted_annotations", "annotations_omitted", budget
    )


def _drop_optional_event_fields(
    payload: dict[str, Any], budget: int
) -> str | None:
    return _drop_event_fields(
        payload, ("expression", "title", "service", "environment"), budget
    )


def _drop_condition_labels(payload: dict[str, Any], budget: int) -> str | None:
    condition = payload.get("authoritative_condition")
    if not isinstance(condition, dict):
        return None
    labels = condition.pop("labels", None)
    if labels:
        payload["omissions"]["labels_omitted"] += len(labels)
    return _render_within_budget(payload, budget)


def _reduce_prompt_payload(payload: dict[str, Any], budget: int) -> str | None:
    reducers = (
        _drop_event_labels,
        _drop_optional_event_fields,
        _drop_middle_events,
        _drop_condition_labels,
        _drop_affected_services,
        _drop_event_annotations,
        _drop_remaining_events,
    )
    for reducer in reducers:
        if (rendered := reducer(payload, budget)) is not None:
            return rendered
    return None


def render_prompt_context(context: IncidentContext, max_bytes: int) -> str:
    """Render valid JSON within a deterministic total byte budget."""
    budget = max(2, int(max_bytes))
    payload = build_prompt_context(context).to_dict()
    if (rendered := _render_within_budget(payload, budget)) is not None:
        return rendered

    payload["omissions"]["budget_reduced"] = True
    if (rendered := _reduce_prompt_payload(payload, budget)) is not None:
        return rendered

    minimal = {
        "schema": CONTEXT_SCHEMA,
        "incident": {"id": _bounded_text(context.incident_id, 64) or "unknown"},
        "omissions": {"budget_reduced": True},
    }
    if (rendered := _render_within_budget(minimal, budget)) is not None:
        return rendered
    return "{}"
