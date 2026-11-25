from __future__ import annotations

import httpx
import pytest

from adapters import (
    ConditionEvaluatorRegistry,
    ConditionState,
    GrafanaAlertingAdapter,
    GrafanaAlertingConfig,
)
from models import ConditionReference


def condition_ref(instance_id: str = "fp-primary") -> ConditionReference:
    return ConditionReference(
        version=1,
        source="grafana",
        evaluator="managed_alerting",
        resource_type="alert_rule",
        resource_id="rule-uid",
        instance_id=instance_id,
        tenant_id="1",
        labels={"service": "api"},
    )


def rule_payload(state: str, *, include_rule: bool = True):
    rules = []
    if include_rule:
        rules.append({"uid": "rule-uid", "state": state, "health": "ok"})
    return {"status": "success", "data": {"groups": [{"rules": rules}]}}


def active_alert(fingerprint: str, *, rule_uid: str = "rule-uid"):
    return {
        "fingerprint": fingerprint,
        "labels": {"__alert_rule_uid__": rule_uid, "service": "api"},
        "status": {"state": "active"},
    }


def adapter_for(rules, alerts, *, rules_status: int = 200, alerts_status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer token"
        if request.url.path.endswith("/api/v1/rules"):
            return httpx.Response(rules_status, json=rules, request=request)
        if request.url.path.endswith("/api/v2/alerts"):
            return httpx.Response(alerts_status, json=alerts, request=request)
        return httpx.Response(404, request=request)

    return GrafanaAlertingAdapter(
        GrafanaAlertingConfig(base_url="http://grafana", api_token="token"),
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_exact_active_instance_is_firing():
    adapter = adapter_for(rule_payload("firing"), [active_alert("fp-primary")])

    evaluation = await adapter.evaluate(condition_ref())

    assert evaluation.state is ConditionState.FIRING
    assert evaluation.authoritative is True
    await adapter.close()


@pytest.mark.asyncio
async def test_explicit_normal_rule_and_absent_instance_is_recovered():
    adapter = adapter_for(rule_payload("normal"), [])

    evaluation = await adapter.evaluate(condition_ref())

    assert evaluation.state is ConditionState.NORMAL
    assert evaluation.reason_code == "instance_normal"
    await adapter.close()


@pytest.mark.asyncio
async def test_pending_and_no_data_never_become_normal():
    pending = adapter_for(rule_payload("pending"), [])
    no_data = adapter_for(rule_payload("nodata"), [])

    pending_result = await pending.evaluate(condition_ref())
    no_data_result = await no_data.evaluate(condition_ref())

    assert pending_result.state is ConditionState.PENDING
    assert no_data_result.state is ConditionState.NO_DATA
    await pending.close()
    await no_data.close()


@pytest.mark.asyncio
async def test_normal_state_requires_explicitly_healthy_rule_evaluation():
    adapter = adapter_for(
        {
            "status": "success",
            "data": {
                "groups": [
                    {
                        "rules": [
                            {
                                "uid": "rule-uid",
                                "state": "normal",
                                "health": "unknown",
                            }
                        ]
                    }
                ]
            },
        },
        [],
    )

    evaluation = await adapter.evaluate(condition_ref())

    assert evaluation.state is ConditionState.UNKNOWN
    assert evaluation.authoritative is False
    await adapter.close()


@pytest.mark.asyncio
async def test_api_failure_and_missing_rule_are_unknown():
    failed = adapter_for({}, [], rules_status=503)
    missing = adapter_for(rule_payload("normal", include_rule=False), [])

    failed_result = await failed.evaluate(condition_ref())
    missing_result = await missing.evaluate(condition_ref())

    assert failed_result.state is ConditionState.UNKNOWN
    assert failed_result.authoritative is False
    assert missing_result.state is ConditionState.UNKNOWN
    assert missing_result.reason_code == "rule_not_found"
    await failed.close()
    await missing.close()


@pytest.mark.asyncio
async def test_multiple_instances_are_matched_by_exact_fingerprint():
    alerts = [active_alert("fp-other"), active_alert("fp-primary")]
    firing = adapter_for(rule_payload("firing"), alerts)
    recovered = adapter_for(rule_payload("firing"), [active_alert("fp-other")])

    firing_result = await firing.evaluate(condition_ref())
    recovered_result = await recovered.evaluate(condition_ref())

    assert firing_result.state is ConditionState.FIRING
    assert recovered_result.state is ConditionState.NORMAL
    assert recovered_result.reason_code == "instance_resolved_other_instances_firing"
    await firing.close()
    await recovered.close()


@pytest.mark.asyncio
async def test_duplicate_instance_identity_is_ambiguous():
    adapter = adapter_for(
        rule_payload("firing"),
        [active_alert("fp-primary"), active_alert("fp-primary")],
    )

    evaluation = await adapter.evaluate(condition_ref())

    assert evaluation.state is ConditionState.UNKNOWN
    assert evaluation.reason_code == "ambiguous_alert_instance"
    await adapter.close()


@pytest.mark.asyncio
async def test_malformed_or_out_of_scope_alert_set_is_unknown():
    missing_fingerprint = {
        "labels": {"__alert_rule_uid__": "rule-uid", "service": "api"},
    }
    adapter = adapter_for(
        rule_payload("firing"),
        [active_alert("fp-other"), missing_fingerprint],
    )

    evaluation = await adapter.evaluate(condition_ref())

    assert evaluation.state is ConditionState.UNKNOWN
    assert evaluation.reason_code == "invalid_alert_instance"
    await adapter.close()


@pytest.mark.asyncio
async def test_malformed_rule_response_is_unknown():
    adapter = adapter_for(
        {"status": "success", "data": {"groups": [{"rules": ["not-an-object"]}]}},
        [],
    )

    evaluation = await adapter.evaluate(condition_ref())

    assert evaluation.state is ConditionState.UNKNOWN
    assert evaluation.reason_code == "grafana_api_unavailable"
    await adapter.close()


@pytest.mark.asyncio
async def test_registry_returns_unknown_for_unsupported_source():
    registry = ConditionEvaluatorRegistry()
    reference = ConditionReference(
        version=1,
        source="other",
        evaluator="native",
        resource_type="condition",
        resource_id="condition-1",
        instance_id="instance-1",
        labels={},
    )

    evaluation = await registry.evaluate(reference)

    assert evaluation.state is ConditionState.UNKNOWN
    assert evaluation.reason_code == "unsupported_condition_evaluator"
