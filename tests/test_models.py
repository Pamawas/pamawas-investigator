from datetime import UTC, datetime

import pytest

from models import (
    ConditionReference,
    EvidenceSource,
    EvidenceType,
    Finding,
    IncidentContext,
    InvestigationState,
    ToolResult,
)


def test_finding_round_trip():
    finding = Finding(EvidenceType.LIKELY_CAUSE, "deployment", EvidenceSource.INCIDENTS, 0.8)
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


def test_condition_reference_round_trip_is_provider_neutral():
    raw = {
        "version": 1,
        "source": "grafana",
        "evaluator": "managed_alerting",
        "resource_type": "alert_rule",
        "resource_id": "rule-uid",
        "instance_id": "instance-fingerprint",
        "tenant_id": "1",
        "labels": {"service": "api"},
    }

    reference = ConditionReference.from_dict(raw)

    assert reference.to_dict() == raw


@pytest.mark.parametrize(
    "raw",
    [
        {
            "version": 1,
            "source": "grafana",
            "evaluator": "managed_alerting",
            "resource_type": "alert_rule",
            "resource_id": "rule-uid",
            "instance_id": "instance-fingerprint",
            "labels": {},
            "threshold": 1,
        },
        {
            "version": 1,
            "source": "grafana",
            "evaluator": "managed_alerting",
            "resource_type": "alert_rule",
            "resource_id": "rule-uid",
            "instance_id": "",
            "labels": {},
        },
    ],
)
def test_condition_reference_rejects_unknown_fields_and_missing_identity(raw):
    with pytest.raises(ValueError):
        ConditionReference.from_dict(raw)


def test_tool_result_sets_timezone_aware_timestamp():
    result = ToolResult("prometheus", {}, {}, 1.5)
    assert isinstance(result.timestamp, datetime)
    assert result.timestamp.tzinfo is UTC


def test_investigation_state_defaults():
    state = InvestigationState("inc", [], [])
    assert state.tool_call_count == 0
    assert state.max_tool_calls == 100
    assert state.completed is False
    assert state.error is None


def test_recommendation_is_a_distinct_evidence_type():
    """Recommendations answer "what do I do", not "what happened".

    They carry their own type so the report can render them as actions instead
    of mixing them into the evidence list as more claims about the incident.
    """
    rec = Finding(EvidenceType.RECOMMENDATION, "Roll back the deploy", EvidenceSource.LLM, 0.8)
    assert rec.to_dict()["type"] == "recommendation"
    assert Finding.from_dict(rec.to_dict()) == rec


def test_finding_records_which_system_it_came_from():
    """A reader must be able to tell a metric from a log from an inference."""
    finding = Finding(
        EvidenceType.FACT,
        "up went to 0",
        EvidenceSource.PROMETHEUS,
        1.0,
        source_detail='up{instance="koba-labs"}',
    )
    assert finding.to_dict()["source"] == "prometheus"
    assert finding.to_dict()["source_detail"] == 'up{instance="koba-labs"}'


def test_finding_rejects_a_source_outside_the_closed_set():
    """Free-text sources produced dozens of unaggregatable one-off strings."""
    with pytest.raises(ValueError):
        Finding.from_dict(
            {"type": "fact", "content": "x", "source": "Prometheus up metric", "confidence": 0}
        )
