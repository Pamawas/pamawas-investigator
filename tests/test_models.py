from datetime import UTC, datetime

import pytest

from models import EvidenceType, Finding, IncidentContext, InvestigationState, ToolResult


def test_finding_round_trip():
    finding = Finding(EvidenceType.LIKELY_CAUSE, "deployment", "events", 0.8)
    assert Finding.from_dict(finding.to_dict()) == finding


def test_finding_rejects_unknown_evidence_type():
    with pytest.raises(ValueError):
        Finding.from_dict({"type": "invalid", "content": "x", "source": "y", "confidence": 0})


def test_incident_context_lists_are_not_shared():
    first = IncidentContext("one", "First")
    second = IncidentContext("two", "Second")
    first.events.append({"id": 1})
    first.affected_services.append("api")
    assert second.events == []
    assert second.affected_services == []


def test_tool_result_sets_timezone_aware_timestamp():
    result = ToolResult("prometheus", {}, {}, 1.5)
    assert isinstance(result.timestamp, datetime)
    assert result.timestamp.tzinfo is UTC


def test_investigation_state_defaults():
    state = InvestigationState("inc", [], [])
    assert state.tool_call_count == 0
    assert state.max_tool_calls == 6
    assert state.completed is False
    assert state.error is None
