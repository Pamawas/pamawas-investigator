"""Readable observations built from telemetry, before evidence is persisted."""

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
_LEVELS = {"warn", "warning", "error", "erro", "info", "debug", "trace", "unknown"}


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


def metric_caption(expression: str, metadata: dict, title: str = "") -> str:
    if title:
        return title
    name = metadata.get("name")
    if not name:
        names = _METRIC_NAME.findall(expression)
        name = (
            names[0] if names
            else expression if re.fullmatch(r"[\w:]+", expression) else "Metric"
        )
    return str(name).replace("_", " ").removesuffix(" total").capitalize()


def metric_unit(expression: str, metadata: dict) -> str:
    unit = metadata.get("unit") or ""
    if unit in ("percent", "percentage"):
        return "%"
    if not unit:
        names = _METRIC_NAME.findall(expression)
        tokens = "_".join(names or [str(metadata.get("name", ""))]).split("_")
        unit = next((candidate for candidate in ("bytes", "seconds") if candidate in tokens), "")
    if re.match(r"^\s*100\s*-", expression) and re.search(r"\*\s*100\b", expression):
        return "%"
    if metadata.get("type") == "counter" and re.search(r"\b(?:i?rate)\s*\(", expression):
        return f"{unit or 'units'} per second"
    return unit


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
    candidates.sort(
        key=lambda pair: max(value for _, value in pair[1]) - min(value for _, value in pair[1]),
        reverse=True,
    )
    label_values: dict[str, set[str]] = {}
    for item, _ in candidates:
        for key, value in item.get("metric", {}).items():
            label_values.setdefault(str(key), set()).add(str(value))
    varying = [key for key, values in sorted(label_values.items()) if len(values) > 1]
    title = context.title if context else ""
    title_words = set(re.findall(r"[a-z0-9]+", title.lower()))
    query_words = set(re.findall(r"[a-z0-9]+", expression.lower()))
    caption = metric_caption(expression, metadata, title if title_words & query_words else "")
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
    return max(candidates, key=lambda value: (values[value], "-" in value, len(value)))


def readable_log_line(line: str) -> str:
    message = _TIMESTAMP_PREFIX.sub("", line)
    message = _LEVEL_PREFIX.sub("", message)
    message = _CODE_PREFIX.sub("", message)
    message = " ".join(message.split())
    if len(message) > 180:
        message = message[:120].rsplit(" ", 1)[0] + " … " + message[-50:]
    return message


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
            severity = sum(word in message.lower() for word in ("error", "failed", "disconnect"))
            identity = log_identity(stream.get("stream", {}), excluded)
            events.append((severity, seconds, identity, message))
    selected = sorted(events, reverse=True)[:2]
    selected.sort(key=lambda event: event[1])
    return " ".join(
        f"{identity} at {short_time(seconds, format_timestamp)}: {message}"
        for _, seconds, identity, message in selected
    )


def select_metric_queries(calls: list, context: IncidentContext | None, limit: int) -> list:
    """Keep the primary signal instead of whichever queries happened to run last."""
    if not calls:
        return []
    if context is None:
        return calls[:limit]
    words = {word for word in re.findall(r"[a-z0-9]+", context.title.lower()) if len(word) >= 3}
    relevant = [
        call for call in calls
        if words.intersection(re.findall(
            r"[a-z0-9]+", str(call.arguments.get("promql", "")).lower()
        ))
    ]
    return (relevant or calls[:1])[:limit]
