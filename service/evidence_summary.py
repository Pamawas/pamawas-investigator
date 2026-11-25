"""Readable observations built from telemetry, before evidence is persisted."""

import json
import math
import re
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

from models import IncidentContext

_TIMESTAMP_PREFIX = re.compile(r"^\s*\d{4}-\d{2}-\d{2}T\S+\s+")
_LEVEL_PREFIX = re.compile(r"^(?:WARN|ERRO|ERROR|INFO|DEBUG|TRACE|FATAL)\s+", re.IGNORECASE)
_CODE_PREFIX = re.compile(r"^(?:\[[^\]]+\]\s*)?[\w./-]+\.[a-zA-Z]+:\d+:?\s*")
_METRIC_NAME = re.compile(r"([a-zA-Z_:][a-zA-Z0-9_:]*)\s*\{")
_PROMQL_MATCHER = re.compile(
    r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(=|=~)\s*"((?:\\.|[^"\\])*)"'
)
_LEVELS = {"warn", "warning", "error", "erro", "info", "debug", "trace", "unknown"}
_ANNOTATION_MEASUREMENT = re.compile(
    r"^\s*(?P<number>[-+]?\d[\d,]*(?:\.\d+)?)"
    r"\s*(?P<unit>[%A-Za-zµ°][%A-Za-z0-9µ°/._ -]*)?\s*$"
)
_MEASUREMENT_KEYS = ("value", "current_value", "metric_value", "current")


def numeric_points(series: Any) -> list[tuple[float, float]]:
    if not isinstance(series, dict):
        return []
    points = series.get("values")
    if not isinstance(points, list):
        points = [series.get("value")]
    parsed: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, list) or len(point) < 2:
            continue
        try:
            at, value = float(point[0]), float(point[1])
        except (ValueError, TypeError):
            continue
        if math.isfinite(at) and math.isfinite(value):
            parsed.append((at, value))
    return sorted(parsed)


def incident_bounds(context: IncidentContext | None) -> tuple[float, float] | None:
    if context is None or not context.started_at:
        return None
    try:
        start = datetime.fromisoformat(context.started_at.replace("Z", "+00:00"))
        end = (
            datetime.fromisoformat(context.resolved_at.replace("Z", "+00:00"))
            if context.resolved_at else datetime.now(UTC)
        )
    except ValueError:
        return None
    return (
        (start - timedelta(minutes=5)).timestamp(),
        (end + timedelta(minutes=5)).timestamp(),
    )


def short_time(timestamp: float, format_timestamp) -> str:
    rendered = format_timestamp(datetime.fromtimestamp(timestamp, UTC).isoformat())
    return rendered.split(" ", 1)[-1] if " " in rendered else rendered


def human_number(value: float, unit: str = "") -> str:
    if unit == "bytes" or unit.startswith("bytes "):
        suffix = unit.removeprefix("bytes")
        for label in ("B", "KiB", "MiB", "GiB", "TiB"):
            if abs(value) < 1024 or label == "TiB":
                return f"{value:,.2f} {label}{suffix}"
            value /= 1024
    precision = 2
    if 0 < abs(value) < 0.01:
        precision = min(12, max(6, math.ceil(-math.log10(abs(value))) + 2))
    number = f"{value:,.{precision}f}".rstrip("0").rstrip(".")
    return number + ("%" if unit == "%" else f" {unit}" if unit else "")


def _payload_annotation_maps(raw_payload: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_payload, dict):
        return []
    annotation_maps: list[dict[str, Any]] = []
    for key in ("annotations", "commonAnnotations", "common_annotations"):
        value = raw_payload.get(key)
        if isinstance(value, dict):
            annotation_maps.append(value)
    alerts = raw_payload.get("alerts")
    if isinstance(alerts, list):
        for alert in alerts:
            if not isinstance(alert, dict):
                continue
            annotations = alert.get("annotations")
            if isinstance(annotations, dict):
                annotation_maps.append(annotations)
    return annotation_maps


def _parse_annotation_measurement(value: Any) -> tuple[float, str] | None:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    match = _ANNOTATION_MEASUREMENT.fullmatch(str(value))
    if match is None:
        return None
    try:
        number = float(match.group("number").replace(",", ""))
    except ValueError:
        return None
    unit = (match.group("unit") or "").strip()
    if unit.lower() in {"percent", "percentage", "pct"}:
        unit = "%"
    return number, unit


def _measurement_from_annotations(raw_payload: Any) -> tuple[float, str] | None:
    """Extract one explicit measurement without interpreting alert-specific fields."""
    candidates: list[tuple[int, str, str, tuple[float, str]]] = []
    priority = {key: index for index, key in enumerate(_MEASUREMENT_KEYS)}
    for annotations in _payload_annotation_maps(raw_payload):
        canonical_map = json.dumps(annotations, sort_keys=True, default=str)
        for key in sorted(annotations):
            measurement = _parse_annotation_measurement(annotations.get(key))
            if measurement is not None:
                candidates.append((
                    priority.get(key, len(priority)),
                    key,
                    canonical_map,
                    measurement,
                ))
    return min(candidates)[-1] if candidates else None


def alert_observation(
    context: IncidentContext | None, format_timestamp
) -> tuple[str, str]:
    """Summarize explicit firing/resolution measurements from alert payloads."""
    if context is None:
        return "", ""
    observations = []
    events = sorted(
        context.events or [],
        key=lambda event: (
            str(event.get("timestamp") or ""),
            str(event.get("id") or ""),
            json.dumps(event.get("raw_payload") or {}, sort_keys=True, default=str),
        ),
    )
    for event in events:
        measurement = _measurement_from_annotations(event.get("raw_payload"))
        if measurement is None:
            continue
        value, unit = measurement
        timestamp = format_timestamp(event.get("timestamp"))
        short_timestamp = timestamp.split(" ", 1)[-1] if " " in timestamp else timestamp
        observations.append(
            {
                "event_id": str(event.get("id") or "unknown"),
                "status": str(event.get("status") or "observed").lower(),
                "timestamp": short_timestamp,
                "measurement": human_number(value, unit),
            }
        )
    if not observations:
        return "", ""

    first = observations[0]
    last = observations[-1]
    title = context.title or "The alert measurement"
    if len(observations) == 1:
        content = (
            f"Alert payload reported {title} at {first['measurement']} when "
            f"{first['status']} at {first['timestamp']}."
        )
    else:
        content = (
            f"Alert payload reported {title} at {first['measurement']} when "
            f"{first['status']} at {first['timestamp']}; it reported "
            f"{last['measurement']} when {last['status']} at {last['timestamp']}."
        )
    detail = "alert events: " + ", ".join(
        dict.fromkeys(item["event_id"] for item in observations)
    )
    return content, detail


def metric_caption(expression: str, metadata: dict) -> str:
    """Name the metric from provider metadata or the executed expression.

    Incident titles are symptoms, not metric metadata. Reusing a title after a
    generic word overlap made unrelated values look like the alert's primary
    measurement (for example, rendering Ceph used bytes as latency).
    """
    name = metadata.get("name")
    if not name:
        names = _METRIC_NAME.findall(expression)
        name = (
            names[0] if names
            else expression if re.fullmatch(r"[\w:]+", expression) else "Metric"
        )
    return str(name).replace("_", " ").removesuffix(" total").capitalize()


def metric_unit(expression: str, metadata: dict) -> str:
    unit = str(metadata.get("unit") or "").strip()
    if unit.lower() in ("percent", "percentage"):
        return "%"
    if metadata.get("type") == "counter" and re.search(r"\b(?:i?rate)\s*\(", expression):
        return f"{unit or 'units'} per second"
    return unit


def _selector_scope_score(labels: dict[str, str], expression: str) -> int:
    score = 0
    for key, operator, raw_value in _PROMQL_MATCHER.findall(expression or ""):
        if operator != "=":
            continue
        try:
            value = str(json.loads(f'"{raw_value}"')).lower()
        except (json.JSONDecodeError, TypeError):
            value = raw_value.lower()
        if labels.get(key.lower()) == value:
            score += 6
    return score


def _context_scope_score(
    labels: dict[str, str], context: IncidentContext | None
) -> int:
    if context is None:
        return 0
    values = list(labels.values())
    score = 0

    for target in context.affected_services or []:
        normalized = str(target).strip().lower()
        if normalized and any(
            normalized == value or normalized in value or value in normalized
            for value in values
        ):
            score += 5

    # Only use title tokens that contain an explicit identifier. Descriptive
    # alert words must not scope a returned series.
    identities = {
        token.lower()
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_.-]*\d+[a-zA-Z0-9_.-]*", context.title or "")
    }
    if any(identity in value for identity in identities for value in values):
        score += 4
    return score


def _series_scope_score(
    item: dict, expression: str, context: IncidentContext | None
) -> int:
    """Score whether a returned series belongs to the incident target.

    Prometheus can legitimately return several devices or daemons for a broad
    query. Choosing the two most volatile series then promoted neighbouring
    components into another target's report. Explicit query selectors and stable
    target identifiers therefore outrank volatility; when no series can be scoped, the
    caller conservatively falls back to the provider result rather than hiding
    all telemetry.
    """
    raw_labels = item.get("metric", {}) if isinstance(item, dict) else {}
    labels = {str(key).lower(): str(value).lower() for key, value in raw_labels.items()}
    return _selector_scope_score(labels, expression) + _context_scope_score(
        labels, context
    )


def metric_observations(
    result: dict, format_timestamp, expression: str = "",
    context: IncidentContext | None = None,
) -> str:
    metadata = result.get("metric_metadata", {})
    series = result.get("data", {}).get("result", [])
    bounds = incident_bounds(context)
    candidates = []
    for item in series:
        points = numeric_points(item)
        if bounds:
            points = [(at, value) for at, value in points if bounds[0] <= at <= bounds[1]]
        if points:
            candidates.append((item, points))
    if not candidates:
        return ""
    all_candidates = candidates
    scoped = [
        pair
        for pair in candidates
        if _series_scope_score(pair[0], expression, context) > 0
    ]
    if scoped:
        candidates = scoped
    candidates.sort(key=lambda pair: (
        -(max(value for _, value in pair[1]) - min(value for _, value in pair[1])),
        json.dumps(pair[0].get("metric", {}), sort_keys=True, default=str),
    ))
    label_values: dict[str, set[str]] = {}
    # Use the original result set to retain the label that distinguished a
    # selected target from neighbouring series. After scoping to one OSD, for
    # example, ceph_daemon is still important reader context.
    for item, _ in all_candidates:
        for key, value in item.get("metric", {}).items():
            label_values.setdefault(str(key), set()).add(str(value))
    varying = [key for key, values in sorted(label_values.items()) if len(values) > 1]
    caption = metric_caption(expression, metadata)
    unit = metric_unit(expression, metadata)
    observations = []
    for item, points in candidates[:2]:
        labels = item.get("metric", {})
        dimension = ", ".join(f"{key}={labels[key]}" for key in varying[:2] if key in labels)
        name = f"{caption} ({dimension})" if dimension else caption
        peak_at, peak = max(points, key=lambda point: point[1])
        latest_at, latest = points[-1]
        observations.append(
            f"{name} peaked at {human_number(peak, unit)} "
            f"at {short_time(peak_at, format_timestamp)}; "
            f"the last queried value was {human_number(latest, unit)} "
            f"at {short_time(latest_at, format_timestamp)}."
        )
    return " ".join(observations)


def log_identity(stream: dict, excluded: set[str]) -> str:
    """Select a useful returned identifier without assuming a label key."""
    values = Counter(str(value) for value in stream.values())
    candidates = [
        value for value in values
        if value not in excluded and value.lower() not in _LEVELS
        and re.search(r"[a-zA-Z]", value)
    ]
    if not candidates:
        return "Logs"
    return max(
        candidates,
        key=lambda value: (values[value], "-" in value, len(value), value),
    )


def readable_log_line(line: str) -> str:
    try:
        structured = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        structured = None
    if isinstance(structured, dict):
        # Locate prose by its value, without assuming a message/body field name.
        prose = [value for value in structured.values()
                 if isinstance(value, str) and len(value.split()) >= 3]
        if prose:
            line = max(prose, key=lambda value: len(value.split()))
        else:
            return ""
    message = _TIMESTAMP_PREFIX.sub("", line)
    message = _LEVEL_PREFIX.sub("", message)
    message = _CODE_PREFIX.sub("", message)
    message = re.sub(
        r"rpc error:\s*code\s*=\s*(\w+)\s+desc\s*=\s*", r"\1: ", message
    )
    message = " ".join(message.split())
    if len(message) > 180:
        message = message[:120].rsplit(" ", 1)[0] + " … " + message[-50:]
    return message[:1].upper() + message[1:]


def log_observations(
    streams: list, format_timestamp, context: IncidentContext | None = None,
    expression: str = "",
) -> str:
    bounds = incident_bounds(context)
    excluded = set(context.affected_services or []) if context else set()
    excluded.update(re.findall(r'"([^"\\]+)"', expression))
    events = []
    for stream in streams:
        if not isinstance(stream, dict):
            continue
        for point in stream.get("values", []):
            if not isinstance(point, list) or len(point) < 2:
                continue
            try:
                seconds = float(point[0])
                if abs(seconds) > 1e12:
                    seconds /= 1e9
                short_time(seconds, format_timestamp)
            except (OverflowError, OSError, TypeError, ValueError):
                continue
            if bounds and not bounds[0] <= seconds <= bounds[1]:
                continue
            message = readable_log_line(str(point[1]))
            if not message:
                continue
            severity = sum(word in message.lower() for word in ("error", "failed", "disconnect"))
            identity = log_identity(stream.get("stream", {}), excluded)
            events.append((severity, seconds, identity, message))
    selected = sorted(events, reverse=True)[:2]
    selected.sort(key=lambda event: event[1])
    return " ".join(
        f"{identity} at {short_time(seconds, format_timestamp)}: {message.rstrip('.')}."
        for _, seconds, identity, message in selected
    )


def select_metric_queries(calls: list, context: IncidentContext | None, limit: int) -> list:
    """Keep one primary signal plus bounded, explicitly planned related signals."""
    if not calls:
        return []
    if context is None:
        return calls[:limit]
    words = {word for word in re.findall(r"[a-z0-9]+", context.title.lower()) if len(word) >= 3}
    primary = [
        call for call in calls
        if call.result.get("signal_role") != "related"
        if words.intersection(re.findall(
            r"[a-z0-9]+", str(call.arguments.get("promql", "")).lower()
        ))
    ]
    if not primary:
        primary = [call for call in calls if call.result.get("signal_role") != "related"][:1]
    related = [call for call in calls if call.result.get("signal_role") == "related"]
    selected = [*primary[:1], *related]
    return selected[:limit]
