"""Data models for the Pamawas Investigator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

# Python 3.9 compatibility for datetime.UTC (added in 3.11)
try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc


class EvidenceType(StrEnum):
    """Evidence classification types.

    FACT/LIKELY_CAUSE/HYPOTHESIS/UNKNOWN describe what the investigation found.
    RECOMMENDATION is a different kind of output: the concrete action an
    operator should take, derived from the finding. It is kept separate so the
    report can present actions as actions rather than mixing them into the
    evidence list, where they read as more claims about the incident.
    """

    FACT = "fact"
    LIKELY_CAUSE = "likely_cause"
    HYPOTHESIS = "hypothesis"
    UNKNOWN = "unknown"
    RECOMMENDATION = "recommendation"


class EvidenceSource(StrEnum):
    """Which system a finding's evidence came from.

    This is a closed set so the report can say, per finding, whether it rests on
    metrics, logs, history, or the model's own reasoning — and so the sources
    can be counted. It used to be free text, which produced dozens of one-off
    strings ('Prometheus up metric', 'Loki logs query', ...) that answered
    nothing and could not be aggregated.
    """

    PROMETHEUS = "prometheus"
    LOKI = "loki"
    INCIDENTS = "incidents"
    # The model's own synthesis across the above (correlations, inferences).
    LLM = "llm"
    # The investigator harness itself (timeout, internal error).
    SYSTEM = "system"


@dataclass
class Finding:
    """Represents a piece of evidence or finding from the investigation."""

    type: EvidenceType
    content: str
    source: EvidenceSource
    confidence: float  # 0.0 to 1.0
    # Exact query, stream selector, or incident identifier the finding rests on.
    # The source alone says which system answered; this says what was asked.
    source_detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "content": self.content,
            "source": self.source.value,
            "source_detail": self.source_detail,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Finding:
        """Create Finding from dictionary."""
        return cls(
            type=EvidenceType(data["type"]),
            content=data["content"],
            source=EvidenceSource(data["source"]),
            confidence=data["confidence"],
            source_detail=data.get("source_detail", ""),
        )


@dataclass
class IncidentContext:
    """Incident context loaded from database."""

    incident_id: str
    title: str
    status: str = ""
    started_at: str = ""
    resolved_at: str | None = None
    severity: str = ""
    affected_services: list[str] | None = None
    events: list[dict[str, Any]] | None = None
    error: str | None = None

    def __post_init__(self):
        if self.affected_services is None:
            self.affected_services = []
        if self.events is None:
            self.events = []


@dataclass
class ToolResult:
    """Result of a tool call."""

    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    duration_ms: float
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(tz=UTC)


@dataclass
class InvestigationState:
    """State of an ongoing investigation."""

    incident_id: str
    findings: list[Finding]
    tool_calls: list[ToolResult]
    tool_call_count: int = 0
    # LLM turns consumed. Bounded by max_tool_calls: a turn is one LLM call,
    # which may issue zero or many tool calls. The loop is bounded on turns so
    # a model that replies with prose (no tool calls) cannot spin forever —
    # previously only tool calls advanced the counter, so text-only turns
    # looped until the wall-clock deadline and the run ended with a
    # synthesized UNKNOWN.
    turn_count: int = 0
    max_tool_calls: int = 100
    # Consecutive turns that produced no tool call. A model that stops calling
    # tools is done investigating but forgets to submit; without this the loop
    # burns its whole turn budget (up to 100 LLM calls) and then the wall-clock
    # deadline, ending with a null result a human could have had in one turn.
    consecutive_text_turns: int = 0
    # Consecutive turns where all tool calls were repeats served from cache.
    # A model that repeats the same query over and over is stuck in a loop.
    consecutive_repeated_turns: int = 0
    # Set when the harness must force real data queries instead of allowing
    # more discovery, prose, or submission attempts. Discovery calls never
    # satisfy this gate.
    required_query_enforcement: bool = False
    # Consecutive forced-query turns that failed to produce a successful data
    # query. This bounds endpoints that ignore tool_choice or keep emitting
    # invalid calls.
    required_query_failures: int = 0
    # Provider attempts are bounded per run; an exhausted provider must not
    # prevent a write-up of evidence returned by the other provider.
    exhausted_data_tools: set[str] = field(default_factory=set)
    recovered_data_tools: set[str] = field(default_factory=set)
    context: IncidentContext | None = None
    completed: bool = False
    # Results already produced in this run, keyed by (tool, arguments). A model
    # that asks the same question again is answered from here instead of paying
    # for it twice.
    #
    # Production runs showed why this matters: one investigation issued eleven
    # identical `get_loki_labels {}` calls, got the same fifteen label names
    # every time, and hit the tool limit having learned nothing after the second
    # call. Every run that exhausted its budget had this shape. The prompt asks
    # the model not to repeat itself, but a prompt is a preference and this is
    # a guarantee.
    seen_calls: dict[str, str] = field(default_factory=dict)
    seen_call_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    request_key_hash: str | None = None
    # Legacy affected-target hint retained for request compatibility. It is not
    # mapped onto any Prometheus or Loki label without schema validation.
    service: str | None = None


@dataclass
class Metric:
    """Represents a single Prometheus metric sample."""

    name: str
    value: float
    labels: dict[str, str]
    at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "labels": self.labels,
            "at": self.at.isoformat() if isinstance(self.at, datetime) else self.at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Metric:
        """Create Metric from dictionary."""
        at = data.get("at")
        if isinstance(at, str):
            try:
                at = datetime.fromisoformat(at.replace("Z", "+00:00"))
            except ValueError:
                at = datetime.now(tz=UTC)
        elif at is None:
            at = datetime.now(tz=UTC)
        return cls(
            name=data["name"],
            value=data["value"],
            labels=data.get("labels", {}),
            at=at,
        )


@dataclass
class LogEntry:
    """Represents a single Loki log line."""

    timestamp: datetime
    stream: str
    line: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        ts = self.timestamp
        return {
            "timestamp": ts.isoformat() if isinstance(ts, datetime) else ts,
            "stream": self.stream,
            "line": self.line,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogEntry:
        """Create LogEntry from dictionary."""
        ts = data.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                ts = datetime.now(tz=UTC)
        elif ts is None:
            ts = datetime.now(tz=UTC)
        return cls(
            timestamp=ts,
            stream=data.get("stream", ""),
            line=data.get("line", ""),
        )


@dataclass
class Evidence:
    """Evidence created during investigation."""

    id: str
    incident_id: str
    run_id: str
    type: str
    content: str
    source: str
    confidence: float
    supports_evidence: list[str] = None
    contradicts_evidence: list[str] = None
    ordinal: int = 0
    created_at: datetime = None
    version: int = 1
    updated_at: datetime = None
    metrics: list[Metric] = None
    logs: list[LogEntry] = None

    def __post_init__(self):
        if self.supports_evidence is None:
            self.supports_evidence = []
        if self.contradicts_evidence is None:
            self.contradicts_evidence = []
        if self.metrics is None:
            self.metrics = []
        if self.logs is None:
            self.logs = []
        if self.created_at is None:
            self.created_at = datetime.now(tz=UTC)
        if self.updated_at is None:
            self.updated_at = datetime.now(tz=UTC)
