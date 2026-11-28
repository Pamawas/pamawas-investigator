from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from models import EvidenceType, IncidentContext, InvestigationPolicy, ToolResult
from service.investigator import _finalize_findings


def _fixture():
    policy = InvestigationPolicy(
        "queue", 2, 10, {"owner": "ops"}, "prometheus", "gauge", "items", 1,
        "Inspect the queue consumers.", "operations/queues",
    )
    context = IncidentContext(
        "inc", "Queue saturation", affected_services=["queue-a"],
        events=[{"labels": {"owner": "ops"}}],
        telemetry_catalog={"prometheus": {"queue_depth": {"type": "gauge", "unit": "items"}}},
        investigation_policies=(replace(policy, version=1), policy),
    )
    result = {
        "status": "success", "data": {"result": [{
            "metric": {"stream_target": "queue-a"}, "values": [[1, "1"], [2, "4"]],
        }]},
    }
    call = ToolResult(
        "query_prometheus", {"promql": 'queue_depth{stream_target="queue-a"}'}, result, 5,
    )
    return policy, context, call


@pytest.mark.asyncio
async def test_reviewed_policy_selected_from_grounded_fact_and_saved_with_run(investigator):
    _, context, call = _fixture()
    findings = _finalize_findings([], [call], context, "inc", False)
    action = next(finding for finding in findings if finding.type == EvidenceType.RECOMMENDATION)
    assert action.recommendation_policy == ("queue", 2)
    assert action.content == "Inspect the queue consumers.\nRunbook operations/queues"
    assert action.supports_evidence

    investigator.persistence = MagicMock()
    investigator.persistence.create_investigation_run.return_value = ("irun_1", False)
    await investigator._persist_investigation(
        "inc", "request", findings, [call], "provider", "model", "v25", 1, 3,
    )
    completed = investigator.persistence.update_run_status.call_args.kwargs
    assert completed["status"] == "completed"
    assert completed["recommendation_policy"] == ("queue", 2)


@pytest.mark.parametrize("missing", ["scope", "labels", "semantics", "independence"])
def test_policy_uses_safe_default_when_required_evidence_is_missing(missing):
    policy, context, call = _fixture()
    if missing == "scope":
        call.result["data"]["result"][0]["metric"]["stream_target"] = "another-target"
    elif missing == "labels":
        context.events = [{"labels": {"owner": "another-team"}, "title": "owner=ops"}]
    elif missing == "semantics":
        policy = replace(policy, metric_unit="seconds")
    else:
        policy = replace(policy, minimum_observations=2)
    context.investigation_policies = (policy,)
    repeated = replace(call, arguments={**call.arguments, "start": "other-window"})

    findings = _finalize_findings([], [call, repeated], context, "inc", False)

    action = next(finding for finding in findings if finding.type == EvidenceType.RECOMMENDATION)
    assert action.recommendation_policy is None
    assert action.source_detail == "default investigation action"
