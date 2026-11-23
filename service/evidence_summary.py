"""Small, meaningful summaries of the actual telemetry values and events."""

import json
import math
from datetime import UTC, datetime
from typing import Any


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


def metric_observations(result: dict, format_timestamp) -> str:
    """Describe counter rates by series instead of pooling lifetime counters."""
    metadata = result.get("metric_metadata", {})
    if not metadata:
        return ""
    series = result.get("data", {}).get("result", [])
    candidates = [(item, numeric_points(item)) for item in series]
    candidates = [(item, points) for item, points in candidates if points]
    candidates.sort(
        key=lambda pair: max(value for _, value in pair[1]) - min(value for _, value in pair[1]),
        reverse=True,
    )
    unit = metadata.get("unit") or "metric units"
    if not metadata.get("unit") and "seconds" in metadata.get("name", "").split("_"):
        unit = "seconds"
    if metadata.get("type") == "counter":
        unit = f"{unit}/s"
    observations = []
    for item, points in candidates[:2]:
        labels = json.dumps(item.get("metric", {}), sort_keys=True, separators=(",", ":"))
        peak_at, peak = max(points, key=lambda point: point[1])
        latest_at, latest = points[-1]
        def at(value):
            return format_timestamp(datetime.fromtimestamp(value, UTC).isoformat())
        observations.append(
            f"Series {labels} peaked at {peak:.4g} {unit} at {at(peak_at)} "
            f"and ended at {latest:.4g} {unit} at {at(latest_at)}."
        )
    return " ".join(observations)


def log_observations(streams: list, format_timestamp) -> str:
    """Include concrete returned events, giving errors priority over routine lines."""
    events = []
    for stream in streams:
        if not isinstance(stream, dict):
            continue
        labels = json.dumps(stream.get("stream", {}), sort_keys=True, separators=(",", ":"))
        for point in stream.get("values", []):
            if not isinstance(point, list) or len(point) < 2:
                continue
            line = str(point[1])
            severity = sum(word in line.lower() for word in ("error", "failed", "disconnect"))
            events.append((severity, str(point[0]), labels, line))
    events.sort(reverse=True)
    observations = []
    for _, timestamp, labels, line in events[:2]:
        try:
            seconds = float(timestamp)
            if abs(seconds) > 1e12:
                seconds /= 1e9
            at = format_timestamp(datetime.fromtimestamp(seconds, UTC).isoformat())
        except (OverflowError, OSError, ValueError):
            at = timestamp
        observations.append(f"At {at}, stream {labels}: {line[:220]}")
    return " ".join(observations)
