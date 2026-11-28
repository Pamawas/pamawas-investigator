from unittest.mock import MagicMock

import pytest

from models import IncidentContext, InvestigationState
from service.prompt_context import CATALOG_BYTES, MODEL_INPUT_BYTES, budget_model_input


def test_catalog_selection_is_order_independent_and_bounded(investigator):
    investigator.config.truncation_limit = 8192
    metrics = {
        f"queue_depth_{index:02d}": {"help": "Queue depth " + "<abc>🎵" * 200}
        for index in range(30)
    }
    metrics["unrelated_metric"] = {"help": "Other activity"}
    context = IncidentContext(
        "inc", "Queue saturation", telemetry_catalog={"prometheus": metrics},
        telemetry_catalog_revision_id="tcr_pinned",
    )
    state = InvestigationState(incident_id="inc", findings=[], tool_calls=[])
    messages = investigator._initial_prompt_messages(state, context, False)
    audit = state.prompt_context_audit
    context.telemetry_catalog["prometheus"] = dict(reversed(list(metrics.items())))
    investigator._initial_prompt_messages(state, context, False)

    assert state.prompt_context_audit == audit
    assert audit["catalog_revision_id"] == "tcr_pinned"
    assert audit["sections"]["catalog"]["bytes"] <= CATALOG_BYTES
    assert audit["prometheus_below_relevance_floor"] == 1
    assert audit["prometheus_omitted_for_budget"] > 0
    assert audit["selected_candidates"][0] == {
        "name": "queue_depth_00", "provider": "prometheus", "score": 1,
        "reason": "incident_domain_overlap",
    }
    assert "<abc>" not in messages[2]["content"]


def test_prompt_budget_keeps_only_complete_recent_tool_turns():
    fixed = [{"role": "system", "content": "guardrails"}, {"role": "user", "content": "incident"}]
    history = [
        {"role": "assistant", "content": f"turn {turn}"}
        for turn in range(5)
    ]
    messages = [*fixed]
    for turn in history:
        messages.extend([turn, {"role": "tool", "content": "é" * 7000}])

    bounded, audit = budget_model_input(messages, len(fixed), [], "auto")

    assert bounded == [*fixed, *messages[-4:]]
    assert audit["history_messages_omitted"] == 6
    assert audit["input_bytes"] <= MODEL_INPUT_BYTES
    assert len(audit["input_hash"]) == 64


@pytest.mark.asyncio
async def test_prompt_audit_is_saved_before_sending_the_first_turn(investigator):
    investigator.persistence = MagicMock()
    state = InvestigationState(incident_id="inc", findings=[], tool_calls=[], run_id="irun_1")
    context = IncidentContext("inc", "Queue saturation", telemetry_catalog_revision_id="tcr_pinned")
    messages = investigator._initial_prompt_messages(state, context, False)
    state.turn_count = 1
    plan = investigator._build_turn_plan(state, False, None, False)
    invocations = MagicMock()
    invocations.attach_mock(investigator.persistence.save_prompt_context_audit, "audit")
    investigator.llm.chat_completion = invocations.model

    await investigator._run_turn_model("inc", messages, state, plan, None, None)

    assert [call[0] for call in invocations.mock_calls] == ["audit", "model"]
    run_id, audit = invocations.audit.call_args.args
    assert run_id == "irun_1"
    assert audit["turn"] == 1
    assert audit["catalog_revision_id"] == "tcr_pinned"
    assert 0 < audit["input_bytes"] <= MODEL_INPUT_BYTES
    assert "messages" not in audit
