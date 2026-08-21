"""Data models for the Pamawas Investigator."""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class EvidenceType(StrEnum):
    """Evidence classification types."""
    FACT = "fact"
    LIKELY_CAUSE = "likely_cause"
    HYPOTHESIS = "hypothesis"
    UNKNOWN = "unknown"


@dataclass
class Finding:
    """Represents a piece of evidence or finding from the investigation."""
    type: EvidenceType
    content: str
    source: str
    confidence: float  # 0.0 to 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Finding:
        """Create Finding from dictionary."""
        return cls(
            type=EvidenceType(data["type"]),
            content=data["content"],
            source=data["source"],
            confidence=data["confidence"],
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
    max_tool_calls: int = 6
    completed: bool = False
    error: str | None = None
    request_key_hash: str | None = None
