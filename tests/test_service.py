import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from models import EvidenceType, IncidentContext
from service.investigator import InvestigatorLLM, PamawasInvestigator


def tool_call(name, arguments, call_id="call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def response(content="", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def test_llm_initialization_and_completion(config):
    message = response("answer")
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [SimpleNamespace(message=message)]
    with patch("service.investigator.OpenAI", return_value=client) as openai:
        llm = InvestigatorLLM(config)
        result = llm.chat_completion([{"role": "user", "content": "hi"}], tools=[])
    assert result is message
    openai.assert_called_once_with(base_url=config.llm_base_url, api_key=config.llm_api_key)
    client.chat.completions.create.assert_called_once_with(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        tool_choice="auto",
        temperature=0.1,
        max_tokens=2000,
    )


def test_connect_db_success_and_failure(config):
    config.database_url = "postgres://db"
    connection = MagicMock()
    with (
        patch("service.investigator.OpenAI"),
        patch("service.investigator.psycopg2.connect", return_value=connection),
    ):
        assert PamawasInvestigator(config).db_conn is connection
    with (
        patch("service.investigator.OpenAI"),
        patch("service.investigator.psycopg2.connect", side_effect=RuntimeError("offline")),
        patch("service.investigator.increment_db_errors") as increment,
    ):
        assert PamawasInvestigator(config).db_conn is None
    increment.assert_called_once_with()


def test_get_incident_context_without_database(investigator):
    context = investigator._get_incident_context("inc-1")
    assert context.incident_id == "inc-1"
    assert context.events == []


def test_get_incident_context_loads_incident_and_events(investigator, cursor):
    investigator.db_conn = MagicMock()
    investigator.db_conn.cursor.return_value = cursor
    cursor.fetchone.return_value = ("inc-1", "Outage", "open", "2026-01-01", None, "high", ["api"])
    cursor.fetchall.return_value = [
        (
            "evt-1",
            "alerts",
            "alarm",
            "2026-01-01",
            "api",
            "prod",
            "high",
            "Down",
            "open",
            {},
            {"x": 1},
        )
    ]

    context = investigator._get_incident_context("inc-1")

    assert context.title == "Outage"
    assert context.affected_services == ["api"]
    assert context.events[0]["raw_payload"] == {"x": 1}
    assert cursor.execute.call_count == 2


def test_get_incident_context_reports_missing_incident(investigator, cursor):
    investigator.db_conn = MagicMock()
    investigator.db_conn.cursor.return_value = cursor
    cursor.fetchone.return_value = None
    context = investigator._get_incident_context("missing")
    assert context.error == "Incident missing not found"


def test_truncate_context_preserves_ends_and_limit(investigator):
    text = "a" * 100 + "z" * 100
    result = investigator._truncate_context(text, 80)
    assert len(result) <= 80
    assert result.startswith("a") and result.endswith("z")
    assert "TRUNCATED" in result


def test_investigate_returns_context_error(investigator):
    investigator._get_incident_context = MagicMock(
        return_value=IncidentContext("x", "x", error="db failed")
    )
    findings = investigator.investigate("x")
    assert findings[0].type is EvidenceType.UNKNOWN
    assert "db failed" in findings[0].content


def test_investigate_executes_tool_then_submits_findings(investigator):
    investigator._get_incident_context = MagicMock(return_value=IncidentContext("inc", "Outage"))
    # Mock the async adapters directly - use AsyncMock for async methods
    from unittest.mock import AsyncMock
    investigator.tools._prometheus_adapter.query_range = AsyncMock(
        return_value={"status": "success", "data": [1]}
    )
    investigator.tools.submit_findings = MagicMock(return_value={"status": "success"})
    investigator.llm.chat_completion = MagicMock(
        side_effect=[
            response(
                tool_calls=[
                    tool_call("query_prometheus", {"promql": "up", "start": "s", "end": "e"})
                ]
            ),
            response(
                tool_calls=[
                    tool_call(
                        "submit_findings",
                        {
                            "findings": [
                                {
                                    "type": "fact",
                                    "content": "down",
                                    "source": "prometheus",
                                    "confidence": 0.9,
                                }
                            ]
                        },
                    )
                ]
            ),
        ]
    )

    findings = investigator.investigate("inc")

    assert findings[0].type is EvidenceType.FACT
    investigator.tools._prometheus_adapter.query_range.assert_called_once_with("up", "s", "e")
    assert investigator.llm.chat_completion.call_count == 2


def test_investigate_final_text_response_becomes_hypothesis(investigator):
    investigator.config.max_tool_calls = 1
    investigator._get_incident_context = MagicMock(return_value=IncidentContext("inc", "Outage"))
    investigator.llm.chat_completion = MagicMock(
        return_value=response("Possible resource exhaustion")
    )
    findings = investigator.investigate("inc")
    assert findings == [pytest.approx(findings[0])]
    assert findings[0].type is EvidenceType.HYPOTHESIS
    assert findings[0].confidence == 0.3


def test_investigate_returns_unknown_when_llm_raises(investigator):
    investigator._get_incident_context = MagicMock(return_value=IncidentContext("inc", "Outage"))
    investigator.llm.chat_completion = MagicMock(side_effect=RuntimeError("LLM unavailable"))
    finding = investigator.investigate("inc")[0]
    assert finding.type is EvidenceType.UNKNOWN
    assert "LLM unavailable" in finding.content


def test_investigate_handles_large_tool_result_without_invalid_json(investigator):
    investigator.config.max_tool_calls = 1
    investigator.config.truncation_limit = 80
    investigator._get_incident_context = MagicMock(return_value=IncidentContext("inc", "Outage"))
    investigator.tools.query_prometheus = MagicMock(return_value={"data": "x" * 500})
    investigator.llm.chat_completion = MagicMock(
        return_value=response(
            tool_calls=[tool_call("query_prometheus", {"promql": "up", "start": "s", "end": "e"})]
        )
    )
    finding = investigator.investigate("inc")[0]
    assert finding.type is EvidenceType.UNKNOWN
    assert "maximum tool calls" in finding.content
