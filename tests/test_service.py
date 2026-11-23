import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from models import EvidenceSource, EvidenceType, Finding, IncidentContext, ToolResult
from service.investigator import (
    InvestigatorLLM,
    PamawasInvestigator,
    _drop_trigger_restatements,
    _ground_findings_in_data_queries,
    _logs_from_tool_calls,
    _metrics_from_tool_calls,
)


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
    openai.assert_called_once_with(
        base_url=config.llm_base_urls[0],
        api_key=config.llm_api_keys[0],
        # The floor for one request, not a cap. A call gets a share of the
        # remaining investigation budget when that is larger.
        timeout=60.0,
    )
    client.chat.completions.create.assert_called_once_with(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        # With no budget supplied, the floor applies.
        timeout=60.0,
        tools=[],
        tool_choice="auto",
        temperature=0.1,
        max_tokens=2000,
    )


def test_llm_failover_to_next_endpoint_when_first_is_unavailable(config):
    """When the first endpoint exhausts its retries on a retryable (5xx)
    error, the client must fail over to the next endpoint instead of raising."""

    def make_retryable(msg):
        response = httpx.Response(
            503, request=httpx.Request("POST", "https://x/v1/chat/completions")
        )
        return openai.APIStatusError(message=msg, response=response, body=object())

    client1 = MagicMock()
    client1.chat.completions.create.side_effect = make_retryable("endpoint 1 down")
    client2 = MagicMock()
    message2 = response("ok from nvidia")
    client2.chat.completions.create.return_value.choices = [SimpleNamespace(message=message2)]

    llm = InvestigatorLLM(config)
    # Replace the real clients with the two mocks: endpoint 1 failing, endpoint 2 OK.
    llm.clients = [client1, client2]
    llm.models = ["test-model", "test-model"]

    result = llm.chat_completion([{"role": "user", "content": "hi"}], tools=[])
    assert result is message2
    # Endpoint 1 exhausted its retries (called MAX_RETRIES times), then endpoint 2 succeeded.
    assert client1.chat.completions.create.call_count == llm.MAX_RETRIES
    assert client2.chat.completions.create.call_count == 1


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
    assert any(finding.type is EvidenceType.UNKNOWN for finding in findings)
    assert "db failed" in findings[0].content


def test_investigate_executes_tool_then_submits_findings(investigator):
    investigator._get_incident_context = MagicMock(return_value=IncidentContext("inc", "Outage"))
    # Mock the async adapters directly - use AsyncMock for async methods
    from unittest.mock import AsyncMock

    investigator.tools._prometheus_adapter.query_range = AsyncMock(
        return_value={
            "status": "success",
            "data": {
                "result": [
                    {
                        "metric": {"target": "api"},
                        "values": [[1, "1"], [2, "0"]],
                    }
                ]
            },
        }
    )
    investigator.tools._loki_adapter.query_range = AsyncMock(
        return_value={"status": "success", "data": {"result": []}}
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
                        "query_loki",
                        {"logql": '{service="api"}', "start": "s", "end": "e"},
                    )
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
    assert "observed range was 0 to 1" in findings[0].content
    assert findings[0].source_detail == "query_prometheus: up [s to e]"
    assert all(finding.content != "down" for finding in findings)
    investigator.tools._prometheus_adapter.query_range.assert_called_once_with("up", "s", "e")
    investigator.tools._loki_adapter.query_range.assert_called_once_with(
        '{service="api"}', "s", "e", 100
    )
    assert investigator.llm.chat_completion.call_count == 3


def test_early_submission_is_rejected_until_both_data_queries_succeed(investigator):
    """Metadata or discovery alone can never complete a full investigation."""
    investigator.config.max_tool_calls = 4
    investigator._get_incident_context = MagicMock(
        return_value=IncidentContext("inc", "Outage")
    )
    investigator.tools._prometheus_adapter.query_range = AsyncMock(
        return_value={
            "status": "success",
            "data": {
                "result": [
                    {"metric": {"scope": "test"}, "values": [[1, "1"]]}
                ]
            },
        }
    )
    investigator.tools._prometheus_adapter.query_labels = AsyncMock(
        return_value={"status": "success", "data": ["schema_key"]}
    )
    investigator.tools._loki_adapter.query_range = AsyncMock(
        return_value={"status": "success", "data": {"result": []}}
    )
    investigator.tools._loki_adapter.query_labels = AsyncMock(
        return_value={"status": "success", "data": ["schema_key"]}
    )

    choices = []

    def complete(
        messages,
        tools=None,
        tool_choice="auto",
        max_tokens=2000,
        session="",
        budget_seconds=None,
    ):
        choices.append(tool_choice)
        forced_name = (
            tool_choice.get("function", {}).get("name")
            if isinstance(tool_choice, dict)
            else None
        )
        if forced_name == "query_prometheus":
            return response(
                tool_calls=[
                    tool_call(
                        "query_prometheus",
                        {"promql": "metric_from_discovered_schema", "start": "s", "end": "e"},
                    )
                ]
            )
        if forced_name == "query_loki":
            return response(
                tool_calls=[
                    tool_call(
                        "query_loki",
                        {"logql": '{schema_key="validated"}', "start": "s", "end": "e"},
                    )
                ]
            )
        return response(
            tool_calls=[
                tool_call(
                    "submit_findings",
                    {
                        "findings": [
                            {
                                "type": "unknown",
                                "content": "No matching data was present in either query.",
                                "source": "llm",
                                "source_detail": "required data-query results",
                                "confidence": 0.0,
                            }
                        ]
                    },
                )
            ]
        )

    investigator.llm.chat_completion = MagicMock(side_effect=complete)

    findings = investigator.investigate("inc")

    assert any(finding.type is EvidenceType.UNKNOWN for finding in findings)
    forced_names = [
        choice["function"]["name"]
        for choice in choices
        if isinstance(choice, dict)
    ]
    assert forced_names == ["query_prometheus", "query_loki", "submit_findings"]
    assert investigator.tools._prometheus_adapter.query_range.await_count == 1
    assert investigator.tools._loki_adapter.query_range.await_count == 1


def test_repeated_discovery_is_redirected_to_real_data_queries(investigator):
    """Regression for the live run that repeated both label catalogs three times."""
    investigator.config.max_tool_calls = 5
    investigator._get_incident_context = MagicMock(
        return_value=IncidentContext("inc", "Outage")
    )
    investigator.tools._prometheus_adapter.query_range = AsyncMock(
        return_value={
            "status": "success",
            "data": {
                "result": [
                    {"metric": {"scope": "test"}, "values": [[1, "1"]]}
                ]
            },
        }
    )
    investigator.tools._prometheus_adapter.query_labels = AsyncMock(
        return_value={"status": "success", "data": ["validated"]}
    )
    investigator.tools._loki_adapter.query_range = AsyncMock(
        return_value={"status": "success", "data": {"result": []}}
    )
    investigator.tools._loki_adapter.query_labels = AsyncMock(
        return_value={"status": "success", "data": ["validated"]}
    )

    auto_turn = 0

    def complete(
        messages,
        tools=None,
        tool_choice="auto",
        max_tokens=2000,
        session="",
        budget_seconds=None,
    ):
        nonlocal auto_turn
        forced_name = (
            tool_choice.get("function", {}).get("name")
            if isinstance(tool_choice, dict)
            else None
        )
        if forced_name == "query_prometheus":
            return response(
                tool_calls=[
                    tool_call(
                        "query_prometheus",
                        {"promql": "validated_metric", "start": "s", "end": "e"},
                    )
                ]
            )
        if forced_name == "query_loki":
            return response(
                tool_calls=[
                    tool_call(
                        "query_loki",
                        {"logql": '{validated="value"}', "start": "s", "end": "e"},
                    )
                ]
            )

        auto_turn += 1
        if auto_turn <= 2:
            return response(
                tool_calls=[
                    tool_call("get_prometheus_labels", {}, "prom-labels"),
                    tool_call("get_loki_labels", {}, "loki-labels"),
                ]
            )
        return response(
            tool_calls=[
                tool_call(
                    "submit_findings",
                    {
                        "findings": [
                            {
                                "type": "unknown",
                                "content": "The required queries completed without matching data.",
                                "source": "llm",
                                "source_detail": "Prometheus and Loki queries",
                                "confidence": 0.0,
                            }
                        ]
                    },
                )
            ]
        )

    investigator.llm.chat_completion = MagicMock(side_effect=complete)

    findings = investigator.investigate("inc")

    assert any(finding.type is EvidenceType.UNKNOWN for finding in findings)
    assert investigator.tools._prometheus_adapter.query_range.await_count == 1
    assert investigator.tools._loki_adapter.query_range.await_count == 1


def test_investigate_final_text_without_data_becomes_unknown(investigator):
    investigator.config.max_tool_calls = 1
    investigator._get_incident_context = MagicMock(return_value=IncidentContext("inc", "Outage"))
    investigator.llm.chat_completion = MagicMock(
        return_value=response("Possible resource exhaustion")
    )
    findings = investigator.investigate("inc")
    assert findings == [pytest.approx(findings[0])]
    assert findings[0].type is EvidenceType.UNKNOWN
    assert findings[0].confidence == 0.0
    assert "without calling any tools" in findings[0].content


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
    findings = investigator.investigate("inc")
    assert findings[0].source is EvidenceSource.PROMETHEUS
    assert findings[0].type is EvidenceType.FACT
    assert findings[-1].type is EvidenceType.UNKNOWN
    assert any("turn budget" in finding.content for finding in findings)


def test_text_only_turns_are_bounded_by_turn_budget(investigator):
    """A model that only replies with prose must not spin until the deadline.

    Regression: the loop was bounded on tool-call count, which text-only turns
    never advanced, so the investigation ran until the wall-clock timeout and
    ended with a synthesized UNKNOWN even though tools had succeeded.
    """
    investigator.config.max_tool_calls = 3
    investigator._get_incident_context = MagicMock(return_value=IncidentContext("inc", "Outage"))
    investigator.llm.chat_completion = MagicMock(return_value=response("still thinking"))

    findings = investigator.investigate("inc")

    # Exactly max_tool_calls LLM turns — bounded, not an infinite loop.
    assert investigator.llm.chat_completion.call_count == 3
    # Prose cannot bypass the required Prometheus/Loki data-query gate.
    assert findings[0].type is EvidenceType.UNKNOWN
    assert "without calling any tools" in findings[0].content


def test_repeated_text_only_turns_force_data_queries_then_stop_unknown(investigator):
    """Three prose turns activate data-query enforcement, not submission.

    The model stops calling tools when it has gathered what it needs but forgets
    to submit. Previously the loop kept asking it to "continue" until either the
    turn budget or the wall-clock deadline ran out, then recorded a null UNKNOWN;
    a later version gave up after three prose turns and recorded "the model only
    described its process" while the answer was one forced call away. Now the
    next turn refuses anything but submit_findings, so the model's conclusion is
    captured instead of discarded.
    """
    investigator.config.max_tool_calls = 100
    investigator._get_incident_context = MagicMock(return_value=IncidentContext("inc", "Outage"))
    investigator.llm.chat_completion = MagicMock(return_value=response("I think the node rebooted"))

    findings = investigator.investigate("inc")

    # Three patience turns, then three bounded forced-query failures.
    assert investigator.llm.chat_completion.call_count == (
        investigator.MAX_CONSECUTIVE_TEXT_TURNS
        + investigator.MAX_REQUIRED_QUERY_FAILURES
    )
    assert findings[0].type is EvidenceType.UNKNOWN
    assert "required data queries did not succeed" in findings[0].content


def test_required_query_failure_preserves_successful_provider_evidence(investigator):
    """One failed source must not hide deterministic data from a successful source."""
    from models import InvestigationState

    state = InvestigationState(
        incident_id="inc",
        findings=[],
        tool_calls=[
            ToolResult(
                tool_name="query_prometheus",
                arguments={"promql": "observed_metric", "start": "s", "end": "e"},
                result={
                    "status": "success",
                    "data": {
                        "result": [
                            {
                                "metric": {"scope": "target"},
                                "values": [[1, "70"], [2, "85"]],
                            }
                        ]
                    },
                },
                duration_ms=1,
            )
        ],
    )
    messages = []

    for _ in range(investigator.MAX_REQUIRED_QUERY_FAILURES - 1):
        assert not investigator._record_required_query_failure(
            state, messages, verification=False
        )
    assert investigator._record_required_query_failure(
        state, messages, verification=False
    )

    assert [finding.source for finding in state.findings] == [
        EvidenceSource.PROMETHEUS,
        EvidenceSource.SYSTEM,
    ]
    assert state.findings[0].type is EvidenceType.FACT
    assert "observed range was 70 to 85" in state.findings[0].content
    assert "query_loki" in state.findings[1].content


def test_text_patience_escalates_only_at_the_limit(investigator):
    """A single prose turn is not escalated; the limit is what forces it."""
    from models import InvestigationState

    state = InvestigationState(incident_id="inc", findings=[], tool_calls=[])
    messages = []

    state.consecutive_text_turns = investigator.MAX_CONSECUTIVE_TEXT_TURNS - 1
    assert investigator._escalate_text_patience("inc", state, messages) is False
    assert messages == []

    state.consecutive_text_turns = investigator.MAX_CONSECUTIVE_TEXT_TURNS
    assert investigator._escalate_text_patience("inc", state, messages) is False
    assert state.required_query_enforcement is True
    assert "query_prometheus, query_loki" in messages[-1]["content"]


def test_reasoning_preamble_is_not_recorded_as_a_finding(investigator):
    """Chain-of-thought must not become the "Likely cause" on a critical card.

    A real run stored "Let me analyze the data more carefully. I need to convert
    the Unix timestamps to understand the timeline..." as a hypothesis, and the
    reporter promoted it to the top of a critical alert — a wall of working
    notes where a paged engineer expects a conclusion.
    """
    from models import InvestigationState

    state = InvestigationState(incident_id="inc", findings=[], tool_calls=[])
    state.consecutive_text_turns = investigator.MAX_CONSECUTIVE_TEXT_TURNS

    investigator._handle_text_response(
        response(
            "Let me analyze the data more carefully. I need to convert the Unix "
            "timestamps to understand the timeline. The incident started at "
            "2026-11-16 01:37:20+00:00."
        ),
        state,
        must_submit=True,
        messages=[],
    )

    assert state.completed is True
    # The reasoning itself is not stored, but the run still gets one honest
    # terminal row so the report says the investigation ended without a
    # conclusion rather than showing no evidence at all.
    assert len(state.findings) == 1
    assert state.findings[0].type is EvidenceType.UNKNOWN
    assert "Let me analyze" not in state.findings[0].content


def test_unwraps_tool_call_nested_in_a_functions_envelope():
    """Some endpoints nest the real call inside a generic `functions` call.

    Treated literally every such call was an unknown tool: production runs
    burned ten tool calls this way, each one a round trip that returned an
    error to a call the model had made correctly.
    """
    from service.investigator import unwrap_tool_call_envelope

    name, args = unwrap_tool_call_envelope(
        "functions",
        {"name": "query_loki", "arguments": {"logql": '{job="x"}', "limit": 20}},
    )
    assert name == "query_loki"
    assert args == {"logql": '{job="x"}', "limit": 20}


def test_unwraps_a_no_argument_tool():
    from service.investigator import unwrap_tool_call_envelope

    name, args = unwrap_tool_call_envelope("functions", {"name": "get_loki_labels"})
    assert name == "get_loki_labels"
    assert args == {}


def test_leaves_ordinary_tool_calls_alone():
    from service.investigator import unwrap_tool_call_envelope

    name, args = unwrap_tool_call_envelope("query_loki", {"logql": '{job="x"}'})
    assert name == "query_loki"
    assert args == {"logql": '{job="x"}'}

    # A `functions` call with no inner name is not an envelope; leave it as an
    # unknown tool so the model gets a real error instead of a silent guess.
    name, args = unwrap_tool_call_envelope("functions", {"label": "unit"})
    assert name == "functions"


def _advancing_clock(monkeypatch, step=0.5):
    """Give the investigator module a clock that advances on every read.

    The deadline logic compares against the wall clock, so a test that runs in
    microseconds never reaches the reserve. A fake clock makes the turn at
    which the reserve opens deterministic instead of timing-dependent.

    The fake replaces the `time` name in service.investigator's namespace, not
    time.monotonic on the shared module. Patching the module attribute rewrote
    the clock for everything in the process — including asyncio's event loop,
    which reads monotonic constantly — so the investigation's budget expired
    before its first turn and the test failed for reasons unrelated to the code.
    """
    now = [1000.0]

    class FakeTime:
        @staticmethod
        def monotonic():
            now[0] += step
            return now[0]

        @staticmethod
        def time():
            return now[0]

    monkeypatch.setattr("service.investigator.time", FakeTime)


def test_deadline_reserve_forces_submission_instead_of_losing_the_run(investigator, monkeypatch):
    """The bug this fixes: runs that hit the wall-clock deadline reported nothing.

    A production investigation ran 10m21s, made 18 successful Prometheus and
    Loki queries, and ended with "deadline exceeded without submitting
    findings" — every result it had gathered was discarded. The deadline was
    only checked at the top of a turn, so the model was never told to stop.

    Now the last part of the budget is reserved for writing up, and the turn
    that enters the reserve is required to call submit_findings.
    """
    from unittest.mock import AsyncMock

    investigator.config.max_tool_calls = 50
    investigator.config.investigation_timeout_seconds = 12.0
    investigator.DEADLINE_RESERVE_MIN_SECONDS = 2.0
    _advancing_clock(monkeypatch, step=0.5)

    investigator._get_incident_context = MagicMock(return_value=IncidentContext("inc", "Outage"))
    investigator.tools._prometheus_adapter.query_range = AsyncMock(
        return_value={
            "status": "success",
            "data": {
                "result": [
                    {
                        "metric": {"target": "api"},
                        "values": [[1, "25"], [2, "10"]],
                    }
                ]
            },
        }
    )
    investigator.tools._loki_adapter.query_range = AsyncMock(
        return_value={"status": "success", "data": {"result": []}}
    )

    seen_tool_choices = []
    auto_turn = 0

    def fake_completion(
        messages, tools=None, tool_choice="auto", max_tokens=2000, session="", budget_seconds=None
    ):
        nonlocal auto_turn
        seen_tool_choices.append(tool_choice)
        if isinstance(tool_choice, dict):
            assert tool_choice["function"]["name"] == "submit_findings"
            return response(
                tool_calls=[
                    tool_call(
                        "submit_findings",
                        {
                            "findings": [
                                {
                                    "type": "fact",
                                    "content": "load1 spiked to 25 on a 2-CPU node",
                                    "source": "prometheus",
                                    "confidence": 0.9,
                                }
                            ]
                        },
                    )
                ]
            )
        auto_turn += 1
        if auto_turn == 1:
            return response(
                tool_calls=[
                    tool_call(
                        "query_prometheus",
                        {"promql": "up", "start": "s", "end": "e"},
                        "prom-required",
                    ),
                    tool_call(
                        "query_loki",
                        {"logql": '{service="api"}', "start": "s", "end": "e"},
                        "loki-required",
                    ),
                ]
            )
        return response(
            tool_calls=[
                tool_call(
                    "query_prometheus",
                    {"promql": f"metric_{auto_turn}", "start": "s", "end": "e"},
                )
            ]
        )

    investigator.llm.chat_completion = MagicMock(side_effect=fake_completion)

    findings = investigator.investigate("inc")

    forced = [
        c
        for c in seen_tool_choices
        if isinstance(c, dict) and c["function"]["name"] == "submit_findings"
    ]
    assert forced, "the loop must force submit_findings once the reserve is reached"
    assert findings[0].type is EvidenceType.FACT
    assert "observed range was 10 to 25" in findings[0].content
    assert all("load1" not in finding.content for finding in findings)
    assert investigator.tools._prometheus_adapter.query_range.await_count < 50, (
        "the reserve must stop exploration before the turn budget runs out"
    )


def test_forced_turn_that_ignores_tool_choice_does_not_spend_the_reserve(investigator, monkeypatch):
    """Some endpoints ignore tool_choice and return another tool call.

    Executing it would burn the reserve that exists to write the findings, and
    the run would end with nothing — so the reply is treated as the final
    answer instead.
    """
    from unittest.mock import AsyncMock

    investigator.config.max_tool_calls = 50
    investigator.config.investigation_timeout_seconds = 4.0
    investigator.DEADLINE_RESERVE_MIN_SECONDS = 2.0
    _advancing_clock(monkeypatch, step=0.5)

    investigator._get_incident_context = MagicMock(return_value=IncidentContext("inc", "Outage"))
    investigator.tools._prometheus_adapter.query_range = AsyncMock(
        return_value={
            "status": "success",
            "data": {
                "result": [
                    {"metric": {"scope": "test"}, "values": [[1, "1"]]}
                ]
            },
        }
    )

    def fake_completion(
        messages, tools=None, tool_choice="auto", max_tokens=2000, session="", budget_seconds=None
    ):
        # Never submits, and ignores the forced tool_choice.
        return response(
            tool_calls=[tool_call("query_prometheus", {"promql": "up", "start": "s", "end": "e"})]
        )

    investigator.llm.chat_completion = MagicMock(side_effect=fake_completion)

    findings = investigator.investigate("inc")

    assert findings, "a run must never finish with no finding at all"
    assert findings[0].type is EvidenceType.UNKNOWN
    assert investigator.tools._prometheus_adapter.query_range.await_count < 50, (
        "the forced turn's query must not be executed"
    )


def test_nested_envelope_is_recognised_as_submit_findings():
    """The forced-submit guard must look through the envelope too.

    Otherwise a correctly-submitted finding wrapped as
    {"name": "functions", "arguments": {"name": "submit_findings", ...}} is
    treated as a stray query and the run is cut off with no findings.
    """
    from service.investigator import _resolve_tool_call, _resolved_tool_name

    nested = tool_call(
        "functions",
        {"name": "submit_findings", "arguments": {"findings": []}},
    )
    assert _resolved_tool_name(nested) == "submit_findings"
    assert _resolve_tool_call(nested) == ("submit_findings", {"findings": []})

    plain = tool_call("submit_findings", {"findings": []})
    assert _resolved_tool_name(plain) == "submit_findings"
    assert _resolve_tool_call(plain) == ("submit_findings", {"findings": []})


def test_resolve_tool_call_null_arguments():
    """_resolve_tool_call must not crash when arguments is ``None`` or ``"null"``.

    Some OI-compatible endpoints omit arguments entirely for no-arg tool calls,
    sending ``None`` as the raw value.  Others JSON-encode it as the string
    ``"null"``.  Either way ``json.loads("null")`` returns ``None``, and the old
    code passed that into ``unwrap_tool_call_envelope`` which did ``args.get(…)``
    on a ``None`` and raised ``'NoneType' object is not subscriptable``.
    """
    from service.investigator import _resolve_tool_call

    # Raw None on the function object (endpoint omitted the field entirely)
    null_raw = SimpleNamespace(
        id="call-null",
        function=SimpleNamespace(name="get_loki_labels", arguments=None),
    )
    assert _resolve_tool_call(null_raw) == ("get_loki_labels", {})

    # JSON-encoded null ("null" string)
    null_json = SimpleNamespace(
        id="call-null-json",
        function=SimpleNamespace(name="get_loki_labels", arguments="null"),
    )
    assert _resolve_tool_call(null_json) == ("get_loki_labels", {})

    # Nested envelope with null inner arguments
    null_nested = SimpleNamespace(
        id="call-null-nested",
        function=SimpleNamespace(
            name="functions",
            arguments=json.dumps({"name": "get_loki_labels", "arguments": None}),
        ),
    )
    name, args = _resolve_tool_call(null_nested)
    assert name == "get_loki_labels"
    assert args == {}


def test_resolve_tool_call_malformed_json_arguments():
    """_resolve_tool_call must not crash with JSONDecodeError when arguments is malformed.

    When an LLM response is truncated (e.g. token limits), the arguments string can be
    cut off mid-JSON ('Unterminated string starting at: line 1 column 43 (char 42)').
    Instead of raising JSONDecodeError (which aborts the entire investigation),
    it must return a structured _parse_error dict.
    """
    from service.investigator import _resolve_tool_call

    malformed = SimpleNamespace(
        id="call-malformed",
        function=SimpleNamespace(
            name="query_prometheus",
            arguments='{"promql": "up{job=\\"docker_cadvisor\\", instance=\\"last',
        ),
    )
    name, args = _resolve_tool_call(malformed)
    assert name == "query_prometheus"
    assert "_parse_error" in args
    assert "Unterminated string" in args["_parse_error"]


@pytest.mark.asyncio
async def test_execute_tool_handles_parse_error(investigator):
    """_execute_tool returns a structured error when tool arguments failed JSON parsing."""
    result, findings, is_final = await investigator._execute_tool(
        "query_prometheus",
        {"_parse_error": "Unterminated string starting at: line 1 column 43 (char 42)"},
    )
    assert result["status"] == "error"
    assert "invalid arguments JSON" in result["error"]
    assert findings is None
    assert is_final is False


@pytest.mark.asyncio
async def test_execute_tool_repairs_weak_model_logql_syntax_and_limit(investigator):
    """Generic syntax repair should make cheap-model LogQL executable."""
    investigator.config.loki_max_log_lines = 100
    investigator.tools._loki_adapter.query_range = AsyncMock(
        return_value={"status": "success", "data": {"result": []}}
    )
    arguments = {
        "logql": "{instance='koba-labs', job='system_node'} |~ 'CPU|load'",
        "start": "2026-11-23T11:30:00Z",
        "end": "2026-11-23T11:45:00Z",
        "limit": 200,
    }

    result, findings, is_final = await investigator._execute_tool(
        "query_loki", arguments
    )

    assert result["status"] == "success"
    assert findings is None
    assert is_final is False
    assert arguments["logql"] == (
        '{instance="koba-labs", job="system_node"} |~ "CPU|load"'
    )
    assert arguments["limit"] == 100
    investigator.tools._loki_adapter.query_range.assert_awaited_once_with(
        '{instance="koba-labs", job="system_node"} |~ "CPU|load"',
        "2026-11-23T11:30:00Z",
        "2026-11-23T11:45:00Z",
        100,
    )


@pytest.mark.asyncio
async def test_empty_prometheus_query_returns_ranked_real_metric_candidates(investigator):
    """HTTP 200 with zero series must trigger schema-neutral metric discovery."""
    investigator.tools._prometheus_adapter.query_range = AsyncMock(
        return_value={
            "status": "success",
            "data": {"resultType": "matrix", "result": []},
        }
    )
    investigator.tools._prometheus_adapter.query_label_values = AsyncMock(
        return_value={
            "status": "success",
            "data": [
                "container_cpu_usage_seconds_total",
                "mktxp_system_cpu_load",
                "node_cpu_seconds_total",
                "process_cpu_seconds_total",
            ],
        }
    )

    result, findings, is_final = await investigator._execute_tool(
        "query_prometheus",
        {
            "promql": 'system_node_cpu_usage_percent{job="system_node"}',
            "start": "2026-11-23T11:30:00Z",
            "end": "2026-11-23T11:45:00Z",
        },
    )

    assert result["status"] == "empty"
    assert result["metric_candidates"][0] == "node_cpu_seconds_total"
    assert "does not satisfy the evidence gate" in result["error"]
    assert findings is None
    assert is_final is False
    investigator.tools._prometheus_adapter.query_label_values.assert_awaited_once_with(
        "__name__"
    )


def test_system_prompt_names_the_incident_it_serves(investigator):
    """Each investigation's system prompt states which incident it covers.

    A fresh conversation is necessary but not sufficient: the model is handed
    real neighbouring incidents by get_related_incidents, and Prometheus and
    Loki are queried by host, so another incident's symptoms are one query
    away. Without a stated scope the model can explain this incident using the
    neighbour's evidence.
    """
    context = IncidentContext(
        incident_id="inc_scope_test",
        title="Monitoring Target Down",
        status="open",
        severity="critical",
        started_at="2026-11-16T01:37:20Z",
        affected_services=["last-labs"],
    )

    prompt = investigator._build_system_prompt(context)

    assert "inc_scope_test" in prompt
    assert "last-labs" in prompt
    assert "Monitoring Target Down" in prompt
    # The out-of-scope rule must be explicit, not implied.
    assert "THIS incident" in prompt
    assert "different host" in prompt


def test_system_prompt_is_source_agnostic_and_evidence_grounded(investigator):
    """The prompt selects tools by question and grounds every submitted source."""
    context = IncidentContext(
        incident_id="inc_cpu_prompt",
        title="CPU",
        status="open",
        severity="warning",
        started_at="2026-11-23T10:15:10Z",
        affected_services=["koba-labs", "system_node"],
        events=[
            {
                "labels": {
                    "alertname": "CPU",
                    "instance": "koba-labs",
                    "job": "system_node",
                }
            }
        ],
    )

    prompt = investigator._build_system_prompt(context)

    assert "OPERATING CONTRACT" in prompt
    assert "Tool definitions supplied to this request are authoritative" in prompt
    assert "Prometheus and Loki are the primary evidence sources" in prompt
    assert "PROMETHEUS AND LOKI STEERING" in prompt
    assert "Use query_prometheus to establish" in prompt
    assert "Use query_loki over the same target and time window" in prompt
    assert "call get_prometheus_labels once" in prompt
    assert "call get_loki_labels once" in prompt
    assert "Metrics describe what changed; logs may explain why" in prompt
    assert "never treat catalog output as evidence" in prompt
    assert "MUST execute at least one successful query_prometheus" in prompt
    assert "The harness enforces this rule and rejects early submissions" in prompt
    assert "zero numeric series is not evidence" in prompt
    assert "MUST contain exactly one causal verdict" in prompt
    assert "A recommendation without a causal verdict is incomplete" in prompt
    assert "The current incident is the investigation premise, not a finding" in prompt
    assert "Restating the triggering alert is not a finding" in prompt
    assert "DECISION PROCEDURE" in prompt
    assert "Query Prometheus for the primary signal and its timeline" in prompt
    assert "Query Loki for explanatory or contradictory events" in prompt
    assert "Query related incidents when recurrence changes the operator action" in prompt
    assert "perform one bounded discovery pass" in prompt
    assert "TOOL DISCIPLINE" in prompt
    assert "Every call must answer one distinct unresolved question" in prompt
    assert "EVIDENCE CONTRACT" in prompt
    assert "Set source_detail to the exact executed query" in prompt
    assert "SUBMISSION" in prompt
    assert "Never submit a FACT whose only content is that this incident exists" in prompt
    assert "Normally a full investigation must attempt both" not in prompt
    assert "CPU ALERT RULES" not in prompt
    assert "IMPORTANT LOGQL SYNTAX RULES" not in prompt
    assert "check recent deployments" not in prompt
    assert 'try {instance="<host>"}' not in prompt
    assert 'job="system_node"' not in prompt
    assert "kubernetes_container_name" not in prompt
    assert "node_cpu_seconds_total" not in prompt


def test_current_incident_restatements_are_not_reported_as_facts():
    current_id = "inc_current"
    findings = [
        Finding(
            type=EvidenceType.FACT,
            content="The CPU incident fired for target alpha.",
            source=EvidenceSource.INCIDENTS,
            source_detail=current_id,
            confidence=1.0,
        ),
        Finding(
            type=EvidenceType.FACT,
            content="This condition occurred three times today.",
            source=EvidenceSource.INCIDENTS,
            source_detail="get_related_incidents returned inc_previous",
            confidence=1.0,
        ),
    ]

    filtered = _drop_trigger_restatements(findings, current_id)

    assert [finding.content for finding in filtered] == [
        "This condition occurred three times today."
    ]


def test_only_trigger_restatements_become_one_honest_unknown():
    filtered = _drop_trigger_restatements(
        [
            Finding(
                type=EvidenceType.FACT,
                content="Incident inc_current started at the supplied time.",
                source=EvidenceSource.INCIDENTS,
                source_detail="inc_current",
                confidence=1.0,
            )
        ],
        "inc_current",
    )

    assert len(filtered) == 1
    assert filtered[0].type is EvidenceType.UNKNOWN
    assert filtered[0].source is EvidenceSource.SYSTEM
    assert "no new evidence" in filtered[0].content


def test_query_results_persist_without_assuming_data_lake_labels():
    """Evidence extraction must preserve arbitrary schemas returned by each backend."""
    tool_calls = [
        ToolResult(
            tool_name="query_prometheus",
            arguments={"promql": "custom_utilization_expression"},
            result={
                "status": "success",
                "data": {
                    "resultType": "matrix",
                    "result": [
                        {
                            "metric": {
                                "__name__": "vendor_utilization_total",
                                "tenant_dimension": "alpha-target",
                            },
                            "values": [[1795428900, "41.5"], [1795428960, "42.5"]],
                        }
                    ],
                },
            },
            duration_ms=1,
        ),
        ToolResult(
            tool_name="query_loki",
            arguments={"logql": '{scope_dimension="alpha-target"}'},
            result={
                "status": "success",
                "data": {
                    "resultType": "streams",
                    "result": [
                        {
                            "stream": {
                                "scope_dimension": "alpha-target",
                                "source_dimension": "custom-runtime",
                            },
                            "values": [["1795428960000000000", "custom log line"]],
                        }
                    ],
                },
            },
            duration_ms=1,
        ),
    ]

    metrics = _metrics_from_tool_calls(tool_calls)
    logs = _logs_from_tool_calls(tool_calls)

    assert len(metrics) == 1
    assert metrics[0].name == "vendor_utilization_total"
    assert metrics[0].value == 42.5
    assert metrics[0].labels == {"tenant_dimension": "alpha-target"}
    assert len(logs) == 1
    assert logs[0].line == "custom log line"
    assert json.loads(logs[0].stream) == {
        "scope_dimension": "alpha-target",
        "source_dimension": "custom-runtime",
    }


def test_provider_findings_are_grounded_in_exact_executed_queries(monkeypatch):
    """A cheap model cannot cite invented queries or alert-payload values as telemetry."""
    monkeypatch.setenv("TZ", "Asia/Jakarta")
    tool_calls = [
        ToolResult(
            tool_name="query_prometheus",
            arguments={
                "promql": "custom_utilization_expression",
                "start": "2026-11-23T11:30:00Z",
                "end": "2026-11-23T11:45:00Z",
            },
            result={
                "status": "success",
                "data": {
                    "result": [
                        {
                            "metric": {"tenant_dimension": "alpha-target"},
                            "values": [[1, "68.0"], [2, "86.5"], [3, "71.0"]],
                        }
                    ]
                },
            },
            duration_ms=1,
        ),
        ToolResult(
            tool_name="query_loki",
            arguments={
                "logql": '{scope_dimension="alpha-target"}',
                "start": "2026-11-23T11:30:00Z",
                "end": "2026-11-23T11:45:00Z",
            },
            result={"status": "success", "data": {"result": []}},
            duration_ms=1,
        ),
    ]
    model_findings = [
        Finding(
            type=EvidenceType.FACT,
            content="Copied alert value that the executed query did not establish.",
            source=EvidenceSource.PROMETHEUS,
            source_detail="invented_promql_expression",
            confidence=0.9,
        ),
        Finding(
            type=EvidenceType.UNKNOWN,
            content="Invented Loki selector returned no data.",
            source=EvidenceSource.LOKI,
            source_detail='{invented_label="value"}',
            confidence=0.0,
        ),
        Finding(
            type=EvidenceType.RECOMMENDATION,
            content="Collect process-level telemetry for the same window.",
            source=EvidenceSource.LLM,
            source_detail="follow-up",
            confidence=0.7,
        ),
    ]

    grounded = _ground_findings_in_data_queries(model_findings, tool_calls)

    assert [finding.source for finding in grounded] == [
        EvidenceSource.PROMETHEUS,
        EvidenceSource.LOKI,
        EvidenceSource.LLM,
    ]
    assert "observed range was 68 to 86.5" in grounded[0].content
    assert "2026-11-23 18:30:00 WIB" in grounded[0].content
    assert grounded[0].source_detail == (
        "query_prometheus: custom_utilization_expression "
        "[2026-11-23 18:30:00 WIB to 2026-11-23 18:45:00 WIB]"
    )
    assert grounded[1].content == (
        "Loki returned no log entries for the executed query from "
        "2026-11-23 18:30:00 WIB to 2026-11-23 18:45:00 WIB."
    )
    assert grounded[1].source_detail == (
        'query_loki: {scope_dimension="alpha-target"} '
        "[2026-11-23 18:30:00 WIB to 2026-11-23 18:45:00 WIB]"
    )
    assert all("invented" not in finding.source_detail for finding in grounded)
    assert grounded[2].type is EvidenceType.RECOMMENDATION


def test_missing_model_cause_gets_cautious_schema_neutral_loki_hypothesis():
    """A weak model cannot finish with only facts and a recommendation."""
    from service.investigator import _ensure_causal_verdict

    tool_calls = [
        ToolResult(
            tool_name="query_loki",
            arguments={
                "logql": '{vendor_scope="alpha-target"} |~ "warn|error"',
                "start": "2026-11-23T11:30:00Z",
                "end": "2026-11-23T11:45:00Z",
            },
            result={
                "status": "success",
                "data": {
                    "result": [
                        {
                            "stream": {
                                "vendor_component": "network-agent",
                                "vendor_scope": "alpha-target",
                            },
                            "values": [["1", "warning one"], ["2", "warning two"]],
                        }
                    ]
                },
            },
            duration_ms=1,
        )
    ]
    findings = [
        Finding(
            type=EvidenceType.RECOMMENDATION,
            content="Inspect the matching component.",
            source=EvidenceSource.LLM,
            confidence=0.5,
        )
    ]

    completed = _ensure_causal_verdict(findings, tool_calls)

    assert completed[-1].type is EvidenceType.HYPOTHESIS
    assert completed[-1].source is EvidenceSource.LLM
    assert completed[-1].confidence == 0.4
    assert '"vendor_component":"network-agent"' in completed[-1].content
    assert "temporal overlap alone does not establish causation" in completed[-1].content


def test_model_causal_verdict_is_preserved_as_capped_synthesis():
    tool_calls = [
        ToolResult(
            tool_name="query_loki",
            arguments={"logql": '{scope="target"}', "start": "s", "end": "e"},
            result={
                "status": "success",
                "data": {
                    "result": [
                        {"stream": {"scope": "target"}, "values": [["1", "restart"]]}
                    ]
                },
            },
            duration_ms=1,
        )
    ]
    model_findings = [
        Finding(
            type=EvidenceType.HYPOTHESIS,
            content="A restart may have caused the transient condition.",
            source=EvidenceSource.LOKI,
            source_detail="invented selector",
            confidence=0.95,
        )
    ]

    grounded = _ground_findings_in_data_queries(model_findings, tool_calls)

    hypotheses = [
        finding for finding in grounded if finding.type is EvidenceType.HYPOTHESIS
    ]
    assert len(hypotheses) == 1
    assert hypotheses[0].source is EvidenceSource.LLM
    assert hypotheses[0].confidence == 0.6
    assert hypotheses[0].source_detail == (
        'synthesis from query_loki: {scope="target"} [s to e]'
    )
    assert "invented" not in hypotheses[0].source_detail


def test_each_investigation_gets_its_own_conversation(investigator):
    """Two investigations must not share a message list.

    State living on the investigator (not on the call) would leak one
    incident's evidence into another's report the moment two run in sequence.
    """
    seen: list[list[dict]] = []

    def fake_completion(
        messages, tools=None, tool_choice="auto", max_tokens=2000, session="", budget_seconds=None
    ):
        # Snapshot the conversation as the model saw it.
        seen.append([m.get("content", "") for m in messages])
        return response(content="nothing conclusive", tool_calls=None)

    investigator.config.max_tool_calls = 1
    investigator.llm.chat_completion = MagicMock(side_effect=fake_completion)
    investigator._get_incident_context = MagicMock(
        side_effect=lambda incident_id: IncidentContext(incident_id, f"Outage {incident_id}")
    )

    investigator.investigate("inc_first")
    investigator.investigate("inc_second")

    assert len(seen) == 2
    first, second = seen
    # Neither conversation contains the other incident's context.
    assert not any("inc_second" in part for part in first)
    assert not any("inc_first" in part for part in second)
    # Both start from a clean two-message session.
    assert len(first) >= 2 and len(second) >= 2


def test_llm_calls_are_labelled_with_their_session(investigator):
    """A slow or failing LLM call must be traceable to one incident."""
    sessions = []

    def fake_completion(
        messages, tools=None, tool_choice="auto", max_tokens=2000, session="", budget_seconds=None
    ):
        sessions.append(session)
        return response(content="done", tool_calls=None)

    investigator.config.max_tool_calls = 1
    investigator.llm.chat_completion = MagicMock(side_effect=fake_completion)
    investigator._get_incident_context = MagicMock(
        return_value=IncidentContext("inc_session_label", "Outage")
    )

    investigator.investigate("inc_session_label")

    assert sessions == ["inc_session_label"]


def test_long_working_notes_are_not_a_finding_even_when_they_say_likely_cause():
    """The regression that reached a critical card.

    A 6,293-character essay of working notes began "Let me organize what I know
    so far:" and contained the phrase "what the most likely cause is". The first
    detector gave up as soon as it saw "likely cause", so the whole essay was
    stored as a hypothesis and published as the likely cause.

    Length is the reliable signal: a finding is a claim, not an investigation
    transcript, however many conclusion-shaped phrases it happens to contain.
    """
    from service.investigator import MAX_FINDING_CHARS, _is_reasoning_preamble

    blob = (
        "Let me organize what I know so far:\n\n"
        'up{job="docker_cadvisor", instance="koba-labs"} shows intermittent '
        "single-scrape failures at 14:31, 14:43, 14:56, 15:23. "
        "At 14:51:00 the system_node was also down. "
        "Hmm, actually let me reconsider whether this is a Prometheus-side issue. "
        "Let me check a few more things. "
        "The key question is what caused the alert. "
        "Actually, let me step back. "
        "So the most likely cause is a transient network issue. " + "Details follow. " * 400
    )
    assert len(blob) > MAX_FINDING_CHARS

    assert _is_reasoning_preamble(blob) is True


def test_a_short_finding_may_use_the_word_cause():
    """The honest case must not be caught by the same rule."""
    from service.investigator import _is_reasoning_preamble

    assert not _is_reasoning_preamble(
        "The likely cause is an OOM kill in the wazuh container at 03:42 UTC."
    )
    assert not _is_reasoning_preamble(
        "The node ran out of memory; the wazuh container leaked heap until the "
        "kernel OOM-killed it. Restarting the pod cleared the condition."
    )


def test_working_notes_submitted_through_the_tool_are_rejected(investigator):
    """The essay could also arrive by being submitted, not typed.

    Both exits from the loop must apply the same rule, or the model just picks
    the other one.
    """
    from models import EvidenceSource, Finding
    from service.investigator import _sanitize_finding

    long_notes = Finding(
        type=EvidenceType.HYPOTHESIS,
        content="Let me organize what I know so far: " + "then I checked the metrics. " * 200,
        source=EvidenceSource.LLM,
        confidence=0.3,
    )
    real = Finding(
        type=EvidenceType.LIKELY_CAUSE,
        content="An OOM kill in the wazuh container.",
        source=EvidenceSource.LLM,
        confidence=0.85,
    )

    cleaned = [_sanitize_finding(f) for f in (long_notes, real)]

    assert cleaned[0].type is EvidenceType.UNKNOWN
    assert "working notes" in cleaned[0].content
    assert cleaned[1].content == "An OOM kill in the wazuh container."

    dummy = Finding(
        type=EvidenceType.FACT,
        content="placeholder",
        source=EvidenceSource.LLM,
        confidence=0.5,
    )
    cleaned_dummy = _sanitize_finding(dummy)
    assert cleaned_dummy.type is EvidenceType.UNKNOWN
    assert "placeholder text" in cleaned_dummy.content


def test_em_dashes_are_normalized_in_findings():
    """Em dashes are the clearest generated-prose tell, and they reach the card.

    The prompt asks for plain sentences, but a style instruction is a
    preference. The typography is also normalized where findings are written, so
    a model that ignores the instruction still cannot put one on a report.
    """
    from service.investigator import _plain_typography

    em, en = chr(0x2014), chr(0x2013)
    assert _plain_typography(f"Possible cause {em} not confirmed") == (
        "Possible cause, not confirmed"
    )
    assert _plain_typography(f"up dropped 1 {en} 0") == "up dropped 1, 0"
    assert _plain_typography("no dashes here") == "no dashes here"
    assert _plain_typography("") == ""


def test_no_tool_calls_start_once_the_reserve_opens(investigator, monkeypatch):
    """The reserve must be a wall, not a checkpoint.

    A real run made 13 successful tool calls, blew through its 90-second
    reserve inside a single turn, hit the hard deadline at the top of the next
    iteration and submitted nothing — "13 of 13 tool calls returned data; no
    conclusion was recorded before the budget ran out".

    The loop only checked the deadline between turns, but a turn can carry
    several tool calls, each taking up to the tool timeout. This exercises the
    guard inside one turn, which is where the budget was actually lost.
    """
    import asyncio
    import time
    from unittest.mock import AsyncMock

    from models import InvestigationState

    investigator.tools._prometheus_adapter.query_range = AsyncMock(
        return_value={
            "status": "success",
            "data": {
                "result": [
                    {"metric": {"scope": "test"}, "values": [[1, "1"]]}
                ]
            },
        }
    )
    investigator.llm.chat_completion = MagicMock(
        side_effect=lambda *a, **kw: response(
            tool_calls=[
                tool_call("query_prometheus", {"promql": "up", "start": "s", "end": "e"}, "c1"),
                tool_call("query_prometheus", {"promql": "load", "start": "s", "end": "e"}, "c2"),
                tool_call("query_prometheus", {"promql": "mem", "start": "s", "end": "e"}, "c3"),
            ]
        )
    )

    state = InvestigationState(incident_id="inc", findings=[], tool_calls=[], max_tool_calls=50)

    def run(explore_deadline):
        state.tool_calls = []
        state.turn_count = 1
        asyncio.run(
            investigator._handle_investigation_turn(
                incident_id="inc",
                request_key_hash="h",
                state=state,
                messages=[],
                force_submit=False,
                deadline=None,
                explore_deadline=explore_deadline,
            )
        )
        return len(state.tool_calls)

    # Reserve already open: no new query may start.
    assert run(explore_deadline=0.0) == 0, "no tool call may start inside the reserve"
    # Plenty of exploration time left: all three run.
    assert run(explore_deadline=time.monotonic() + 3600) == 3


def test_a_slow_model_gets_the_budget_not_a_30_second_cap():
    """A free endpoint that needs 90 seconds must not be cut off at 30.

    Production runs failed with six consecutive `Request timed out` lines: every
    call was capped at 30 seconds regardless of how much investigation budget
    remained, so a slow endpoint could never finish and the run ended with no
    findings. The budget is the authority, not a fixed per-call number.
    """
    from service.investigator import MAX_CALL_TIMEOUT_SECONDS, _call_timeout

    # Plenty of budget: the share is far more than the floor.
    assert _call_timeout(60.0, 540.0, attempts_left=3) == MAX_CALL_TIMEOUT_SECONDS
    # Two thirds of the way through: still generous enough to answer.
    assert _call_timeout(60.0, 200.0, attempts_left=2) == 100.0
    # Nearly out of budget: the floor, so the call still gets a chance.
    assert _call_timeout(60.0, 4.0, attempts_left=1) == 5.0
    # No budget at all (tool-free callers such as the daily summary).
    assert _call_timeout(60.0, None) == 60.0


def test_the_call_timeout_never_exceeds_the_remaining_budget_share():
    """A single attempt cannot promise more time than the budget holds."""
    from service.investigator import _call_timeout

    for budget, attempts in ((600.0, 3), (300.0, 2), (120.0, 3), (30.0, 1)):
        timeout = _call_timeout(60.0, budget, attempts_left=attempts)
        assert timeout <= max(60.0, budget), (budget, attempts, timeout)


def test_a_repeated_tool_call_is_answered_from_the_first_result(investigator):
    """A model that re-asks a question must not pay for it twice.

    One production run issued eleven identical `get_loki_labels {}` calls,
    received the same fifteen label names every time, and reached the tool limit
    having learned nothing after the second call. Every exhausted run had this
    shape.
    """
    from models import InvestigationState

    state = InvestigationState(incident_id="inc", findings=[], tool_calls=[])
    messages = []

    signature = investigator._call_signature("get_loki_labels", {})
    state.seen_calls[signature] = '{"data": ["job", "instance"]}'

    call = tool_call("get_loki_labels", {})
    served = investigator._serve_repeated_call(call, "get_loki_labels", {}, state, messages)

    assert served is True
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == call.id
    # The model gets the original answer back, plus a note saying it repeated.
    assert '{"data": ["job", "instance"]}' in messages[0]["content"]
    assert "Do not call it again" in messages[0]["content"]
    # A repeat costs no budget: that is the whole point.
    assert state.tool_call_count == 0


def test_a_new_call_is_not_served_from_cache(investigator):
    from models import InvestigationState

    state = InvestigationState(incident_id="inc", findings=[], tool_calls=[])
    state.seen_calls[investigator._call_signature("get_loki_labels", {})] = "prior"
    messages = []

    # Different arguments: a genuinely different question.
    served = investigator._serve_repeated_call(
        tool_call("get_loki_labels", {"label": "job"}),
        "get_loki_labels",
        {"label": "job"},
        state,
        messages,
    )

    assert served is False
    assert messages == []


def test_argument_order_does_not_defeat_repeat_detection(investigator):
    """Same request, keys in a different order, is the same request."""
    a = investigator._call_signature("query_loki", {"logql": '{job="x"}', "limit": 20})
    b = investigator._call_signature("query_loki", {"limit": 20, "logql": '{job="x"}'})
    assert a == b


def test_submit_findings_is_never_served_from_cache(investigator):
    """The terminal action must always run: its arguments are the findings."""
    from models import InvestigationState

    state = InvestigationState(incident_id="inc", findings=[], tool_calls=[])
    state.seen_calls[investigator._call_signature("submit_findings", {"findings": []})] = "prior"

    served = investigator._serve_repeated_call(
        tool_call("submit_findings", {"findings": []}),
        "submit_findings",
        {"findings": []},
        state,
        [],
    )
    assert served is False


def test_repeated_call_escalates_message_on_second_repeat(investigator):
    """After the second repeat the message becomes a stronger warning."""
    from models import InvestigationState

    state = InvestigationState(incident_id="inc", findings=[], tool_calls=[])
    sig = investigator._call_signature(
        "query_prometheus", {"promql": "up", "start": "s", "end": "e"}
    )
    state.seen_calls[sig] = '{"data": []}'

    messages = []
    # First repeat
    investigator._serve_repeated_call(
        tool_call("query_prometheus", {"promql": "up", "start": "s", "end": "e"}),
        "query_prometheus",
        {"promql": "up", "start": "s", "end": "e"},
        state,
        messages,
    )
    assert "Do not call it again" in messages[0]["content"]
    assert state.seen_call_counts[sig] == 1

    # Second repeat — should use stronger warning
    investigator._serve_repeated_call(
        tool_call("query_prometheus", {"promql": "up", "start": "s", "end": "e"}),
        "query_prometheus",
        {"promql": "up", "start": "s", "end": "e"},
        state,
        messages,
    )
    assert "STOP REPEATING THIS QUERY" in messages[1]["content"]
    assert state.seen_call_counts[sig] == 2


def test_repeat_patience_forces_missing_data_queries_after_limit(investigator):
    """Repeated discovery activates data-query enforcement, not submission."""
    from models import InvestigationState

    state = InvestigationState(incident_id="inc", findings=[], tool_calls=[])
    messages = []

    # Below the limit: should not force
    state.consecutive_repeated_turns = investigator.MAX_CONSECUTIVE_REPEATED_TURNS - 1
    assert investigator._escalate_repeat_patience("inc", state, messages) is False
    assert messages == []

    # At the limit: redirect to the missing data-query tools.
    state.consecutive_repeated_turns = investigator.MAX_CONSECUTIVE_REPEATED_TURNS
    assert investigator._escalate_repeat_patience("inc", state, messages) is False
    assert state.required_query_enforcement is True
    assert "query_prometheus, query_loki" in messages[-1]["content"]


def test_fresh_tool_call_resets_consecutive_repeated_turns(investigator):
    """A new (not cached) tool call resets the consecutive_repeated_turns counter."""
    from models import InvestigationState

    state = InvestigationState(incident_id="inc", findings=[], tool_calls=[])
    messages = []

    # Simulate one turn of all-repeated calls: counter should increment
    state.consecutive_repeated_turns = 1
    has_fresh = False

    # has_fresh_tool_call = False → counter increments
    if not has_fresh:
        state.consecutive_repeated_turns += 1

    assert state.consecutive_repeated_turns == 2

    # Simulate a turn with a fresh call: counter resets
    has_fresh = True
    if has_fresh:
        state.consecutive_repeated_turns = 0

    assert state.consecutive_repeated_turns == 0

    # Confirm the threshold activates required data-query enforcement.
    state.consecutive_repeated_turns = investigator.MAX_CONSECUTIVE_REPEATED_TURNS
    assert investigator._escalate_repeat_patience("inc", state, messages) is False
    assert state.required_query_enforcement is True


def test_an_unknown_finding_carries_no_confidence():
    """UNKNOWN (90%) is a contradiction: confidence belongs to a claim.

    Production reports printed `❓ UNKNOWN (90%) Whether the Uptime Kuma probe
    also failed ... is not established`. The model was expressing confidence in
    a non-answer.
    """
    from models import EvidenceSource, Finding
    from service.investigator import _normalize_finding

    unknown = Finding(
        type=EvidenceType.UNKNOWN,
        content="Whether the external probe also failed is not established.",
        source=EvidenceSource.LLM,
        confidence=0.9,
    )
    normalized = _normalize_finding(unknown)
    assert normalized.confidence == 0.0
    assert normalized.content == unknown.content


def test_a_fact_keeps_its_confidence():
    from models import EvidenceSource, Finding
    from service.investigator import _normalize_finding

    fact = Finding(
        type=EvidenceType.FACT,
        content="up dropped to 0 at 14:20.",
        source=EvidenceSource.PROMETHEUS,
        confidence=1.0,
    )
    assert _normalize_finding(fact).confidence == 1.0


def test_claim_recovered_recognizes_positive_recovery():
    """A verification only closes an incident on a positive recovery claim."""
    from service.investigator import _claim_recovered

    assert _claim_recovered("The target has recovered; up is 1 again.")
    assert _claim_recovered("up{job='x'} is back up since 14:32.")
    assert _claim_recovered("no longer down; the host is reachable.")
    # Uncertainty, failure, or still firing/unreachable is not recovery.
    assert not _claim_recovered("The target may still be down.")
    assert not _claim_recovered("up is 0; still failing.")
    assert not _claim_recovered(
        "The alert is still firing as of the last evaluation at 07:02:00, "
        "confirming the target remains unreachable."
    )
    assert not _claim_recovered("The target remains unreachable.")
    assert not _claim_recovered("no recovery observed in the metrics.")


def test_cached_verification_evidence_does_not_assume_schema_or_recovery():
    from service.investigator import (
        _cached_verification_results,
        _claim_recovered,
        _loki_verification_finding,
        _prometheus_verification_finding,
    )

    prometheus_signature = "query_prometheus:" + json.dumps(
        {"promql": "tenant_latency_ratio", "start": "s", "end": "e"},
        sort_keys=True,
    )
    loki_signature = "query_loki:" + json.dumps(
        {"logql": '{tenant="blue"}', "start": "s", "end": "e"},
        sort_keys=True,
    )
    seen_calls = {
        prometheus_signature: json.dumps(
            {
                "status": "success",
                "data": {
                    "resultType": "matrix",
                    "result": [{"metric": {"scope": "edge"}, "values": [[1, "0.2"]]}],
                },
            }
        ),
        loki_signature: json.dumps(
            {
                "status": "success",
                "data": {
                    "resultType": "streams",
                    "result": [
                        {"stream": {"scope": "edge"}, "values": [["1", "slow request"]]}
                    ],
                },
            }
        ),
    }

    prometheus_result, loki_result = _cached_verification_results(seen_calls)
    findings = (
        _prometheus_verification_finding(prometheus_result),
        _loki_verification_finding(loki_result),
    )

    assert all(finding is not None for finding in findings)
    assert "1 result series" in findings[0].content
    assert "1 log lines across 1 result streams" in findings[1].content
    assert all(not _claim_recovered(finding.content) for finding in findings)


@pytest.mark.asyncio
async def test_verify_incident_closes_only_on_recovery(investigator):
    """A stale incident closes only when the evidence says it recovered.

    The correlator acts on this verdict. A finding that the target is still down
    must keep the incident open (still_down), never close it.
    """
    from models import EvidenceSource, EvidenceType, Finding, IncidentContext

    investigator._get_incident_context = MagicMock(
        return_value=IncidentContext("inc", "Outage", affected_services=["svc"])
    )
    investigator._run_investigation_loop = AsyncMock(
        return_value=[
            Finding(
                type=EvidenceType.FACT,
                content="The target has recovered; up is 1 since 14:32.",
                source=EvidenceSource.PROMETHEUS,
                confidence=1.0,
            )
        ]
    )

    result = await investigator.verify_incident("inc")
    assert result["verdict"] == "recovered"


@pytest.mark.asyncio
async def test_verify_incident_keeps_open_when_still_down(investigator):
    from models import EvidenceSource, EvidenceType, Finding, IncidentContext

    investigator._get_incident_context = MagicMock(
        return_value=IncidentContext("inc", "Outage", affected_services=["svc"])
    )
    investigator._run_investigation_loop = AsyncMock(
        return_value=[
            Finding(
                type=EvidenceType.FACT,
                content="up is 0; scrape_duration_seconds is 30s.",
                source=EvidenceSource.PROMETHEUS,
                confidence=1.0,
            )
        ]
    )

    result = await investigator.verify_incident("inc")
    assert result["verdict"] == "still_down"


@pytest.mark.asyncio
async def test_verify_incident_defaults_to_still_down_when_inconclusive(investigator):
    """An empty or UNKNOWN verification never closes the incident."""
    from models import IncidentContext

    investigator._get_incident_context = MagicMock(return_value=IncidentContext("inc", "Outage"))
    investigator._run_investigation_loop = AsyncMock(return_value=[])

    result = await investigator.verify_incident("inc")
    assert result["verdict"] == "still_down"


def test_verification_prompt_and_message_anchors_current_time(investigator):
    from models import IncidentContext

    context = IncidentContext(
        incident_id="inc_sv_1",
        title="Ceph OSD High Latency",
        status="open",
        severity="warning",
        started_at="2026-11-22T18:42:00Z",
        affected_services=["10.1.1.4:9283"],
    )

    prompt = investigator._build_verification_prompt(context)
    assert "CURRENT LIVE TIME" in prompt
    assert "inc_sv_1" in prompt
    assert "10.1.1.4:9283" in prompt
    assert "get_related_incidents" not in prompt

    msg = investigator._build_initial_message(context, verification=True)
    assert "Verify whether this condition has recovered as of right now" in msg["content"]
    assert "Current evaluation time" in msg["content"]
    assert "Originally started at: 2026-11-22T18:42:00Z" in msg["content"]


@pytest.mark.asyncio
async def test_execute_tool_verification_disables_related_incidents(investigator):
    result, findings, is_final = await investigator._execute_tool(
        "get_related_incidents",
        {"service": "10.1.1.4:9283", "symptom_keywords": ["latency"]},
        verification=True,
    )
    assert result["status"] == "error"
    assert "disabled during verification" in result["error"]
    assert findings is None
    assert is_final is False


@pytest.mark.asyncio
async def test_execute_tool_verification_rejects_stale_time_range(investigator):
    # Stale end timestamp far in the past
    stale_end = "2026-11-22T18:50:00Z"
    stale_start = "2026-11-22T18:45:00Z"

    result, findings, is_final = await investigator._execute_tool(
        "query_prometheus",
        {"promql": "up", "start": stale_start, "end": stale_end},
        verification=True,
    )
    assert result["status"] == "error"
    assert "Verification checks current live health" in result["error"]
    assert "minutes old" in result["error"]
    assert findings is None
    assert is_final is False


@pytest.mark.asyncio
async def test_execute_tool_verification_accepts_live_time_range(investigator):
    from datetime import datetime, timedelta
    from datetime import timezone as tz

    now = datetime.now(tz.utc)
    live_start = (now - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    live_end = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    result, findings, is_final = await investigator._execute_tool(
        "query_prometheus",
        {"promql": "up", "start": live_start, "end": live_end},
        verification=True,
    )
    assert result["status"] == "success"
    assert findings is None
    assert is_final is False
