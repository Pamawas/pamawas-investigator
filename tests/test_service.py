import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai
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
    assert "turn budget" in finding.content


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
    # The final turn's prose is captured as a hypothesis rather than discarded.
    assert findings[0].type is EvidenceType.HYPOTHESIS
    assert findings[0].content == "still thinking"


def test_repeated_text_only_turns_force_a_submission(investigator):
    """Three prose-only turns escalate to a forced submit, not a null result.

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

    # Three patience turns, then one forced turn.
    assert investigator.llm.chat_completion.call_count == (
        investigator.MAX_CONSECUTIVE_TEXT_TURNS + 1
    )
    assert findings[0].type is EvidenceType.HYPOTHESIS
    assert findings[0].content == "I think the node rebooted"


def test_text_patience_escalates_only_at_the_limit(investigator):
    """A single prose turn is not escalated; the limit is what forces it."""
    from models import InvestigationState

    state = InvestigationState(incident_id="inc", findings=[], tool_calls=[])
    messages = []

    state.consecutive_text_turns = investigator.MAX_CONSECUTIVE_TEXT_TURNS - 1
    assert investigator._escalate_text_patience("inc", state, messages) is False
    assert messages == []

    state.consecutive_text_turns = investigator.MAX_CONSECUTIVE_TEXT_TURNS
    assert investigator._escalate_text_patience("inc", state, messages) is True
    assert "submit_findings" in messages[-1]["content"]


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
    investigator.config.investigation_timeout_seconds = 4.0
    investigator.DEADLINE_RESERVE_MIN_SECONDS = 2.0
    _advancing_clock(monkeypatch, step=0.5)

    investigator._get_incident_context = MagicMock(return_value=IncidentContext("inc", "Outage"))
    investigator.tools._prometheus_adapter.query_range = AsyncMock(
        return_value={"status": "success", "data": [1]}
    )

    seen_tool_choices = []

    def fake_completion(
        messages, tools=None, tool_choice="auto", max_tokens=2000, session="", budget_seconds=None
    ):
        seen_tool_choices.append(tool_choice)
        if isinstance(tool_choice, dict):
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
        return response(
            tool_calls=[tool_call("query_prometheus", {"promql": "up", "start": "s", "end": "e"})]
        )

    investigator.llm.chat_completion = MagicMock(side_effect=fake_completion)

    findings = investigator.investigate("inc")

    forced = [c for c in seen_tool_choices if isinstance(c, dict)]
    assert forced, "the loop must force submit_findings once the reserve is reached"
    assert findings[0].type is EvidenceType.FACT
    assert "load1" in findings[0].content
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
        return_value={"status": "success", "data": [1]}
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
        return_value={"status": "success", "data": [1]}
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
