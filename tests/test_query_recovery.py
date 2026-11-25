from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from models import EvidenceSource, EvidenceType, Finding, IncidentContext
from service.investigator import _ensure_causal_verdict_for_mode, _ground_findings_in_data_queries
from service.query_recovery import QueryRecovery


def context():
    return IncidentContext(
        "inc_recovery", "CPU", started_at="2026-11-23T11:35:10Z",
        resolved_at="2026-11-23T11:40:10Z", affected_services=["alpha-target", "metrics-job"],
        events=[{"labels": {"metric_scope": "alpha-target", "metric_job": "metrics-job"}}],
    )


@pytest.mark.asyncio
async def test_loki_recovery_uses_its_own_keys_and_values():
    adapter = SimpleNamespace(
        query_labels=AsyncMock(return_value={"status": "success", "data": ["log_scope"]}),
        query_label_values=AsyncMock(return_value={"status": "success", "data": ["alpha-target"]}),
        query_range=AsyncMock(return_value={
            "status": "success", "data": {"result": [{
                "stream": {"log_scope": "alpha-target", "component": "worker"},
                "values": [["1795433800000000000", "worker restarted after timeout"]],
            }]},
        }),
    )
    recovery = QueryRecovery(SimpleNamespace(_loki_adapter=adapter), context(), 100)

    await recovery.recover_loki()

    adapter.query_range.assert_awaited_once_with(
        '{log_scope="alpha-target"}', "2026-11-23T11:30:10+00:00",
        "2026-11-23T11:45:10+00:00", 100,
    )
    grounded = _ground_findings_in_data_queries([], recovery.calls, recovery.context)
    assert "Worker restarted after timeout" in grounded[0].content
    assert "metric_job" not in grounded[0].source_detail


@pytest.mark.asyncio
async def test_loki_recovery_stops_after_incident_identity_labels_do_not_match():
    """Missing Loki target coverage does not trigger unrelated label enumeration."""
    async def label_values(label):
        values = {
            "instance": ["armbian-2", "koba-labs"],
            "job": ["system_journal", "kube_logs"],
        }
        return {"status": "success", "data": values[label]}

    adapter = SimpleNamespace(
        query_labels=AsyncMock(return_value={
            "status": "success",
            "data": ["boot_id", "cluster", "filename", "instance", "job"],
        }),
        query_label_values=AsyncMock(side_effect=label_values),
        query_range=AsyncMock(),
    )
    incident = IncidentContext(
        "inc_ceph",
        "Ceph OSD High Latency",
        started_at="2026-11-24T06:51:00Z",
        resolved_at="2026-11-24T06:52:00Z",
        affected_services=["10.1.1.4:9283", "osd.1"],
        events=[{"labels": {
            "instance": "10.1.1.4:9283",
            "job": "proxmox_ceph",
            "ceph_daemon": "osd.1",
        }}],
    )
    recovery = QueryRecovery(SimpleNamespace(_loki_adapter=adapter), incident, 100)

    await recovery.recover_loki()

    assert [call.args[0] for call in adapter.query_label_values.await_args_list] == [
        "instance",
        "job",
    ]
    adapter.query_range.assert_not_awaited()


@pytest.mark.asyncio
async def test_selector_discovers_arbitrary_provider_label_name():
    adapter = SimpleNamespace(
        query_labels=AsyncMock(return_value={
            "status": "success", "data": ["opaque_dimension"],
        }),
        query_label_values=AsyncMock(return_value={
            "status": "success", "data": ["alpha-target"],
        }),
        query_range=AsyncMock(return_value={"status": "success", "data": {"result": []}}),
    )
    incident = IncidentContext(
        "inc_arbitrary",
        "Arbitrary Alert",
        started_at="2026-11-24T06:51:00Z",
        resolved_at="2026-11-24T06:52:00Z",
        affected_services=["alpha-target"],
        events=[{"labels": {"different_source_key": "alpha-target"}}],
    )
    recovery = QueryRecovery(SimpleNamespace(_loki_adapter=adapter), incident, 100)

    await recovery.recover_loki()

    adapter.query_label_values.assert_awaited_once_with("opaque_dimension")
    adapter.query_range.assert_awaited_once_with(
        '{opaque_dimension="alpha-target"}',
        "2026-11-24T06:46:00+00:00",
        "2026-11-24T06:57:00+00:00",
        100,
    )


@pytest.mark.asyncio
async def test_counter_recovery_preserves_rate_query_and_meaningful_observations(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Jakarta")
    adapter = SimpleNamespace(
        query_metric_catalog=AsyncMock(return_value={"status": "success", "data": [{
            "name": "vendor_activity_seconds_total", "type": "counter", "unit": "seconds",
        }]}),
        query_range=AsyncMock(return_value={"status": "success", "data": {"result": [{
            "metric": {"metric_scope": "alpha-target", "phase": "busy"},
            "values": [[1795433700, "0.85"], [1795434000, "0.6"]],
        }]}}),
    )
    recovery = QueryRecovery(SimpleNamespace(_prometheus_adapter=adapter), context(), 100)
    original = 'vendor_activity_seconds_total{metric_scope="alpha-target"}'

    await recovery.counter_rate({"promql": original})

    executed = recovery.calls[-1]
    assert executed.arguments["promql"] == f"rate({original}[5m])"
    grounded = _ground_findings_in_data_queries([], recovery.calls, recovery.context)
    assert "0.85 seconds per second" in grounded[0].content
    assert "18:35:00 WIB" in grounded[0].content
    assert "{" not in grounded[0].content


@pytest.mark.asyncio
async def test_recovery_queries_primary_and_two_metadata_ranked_related_signals(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Jakarta")

    async def query_range(expression, _start, _end):
        values_by_expression = {
            'vendor_compute_utilization{metric_scope="alpha-target"}': (
                [1795433700, "85.6"], [1795434000, "63.51"]
            ),
            'vendor_memory_available_bytes{metric_scope="alpha-target"}': (
                [1795433700, "2147483648"], [1795434000, "3221225472"]
            ),
            'rate(vendor_storage_read_bytes_total{metric_scope="alpha-target"}[5m])': (
                [1795433700, "1048576"], [1795434000, "524288"]
            ),
        }
        return {
            "status": "success",
            "data": {"result": [{
                "metric": {"metric_scope": "alpha-target"},
                "values": list(values_by_expression[expression]),
            }]},
        }

    adapter = SimpleNamespace(
        query_labels=AsyncMock(return_value={
            "status": "success", "data": ["metric_scope"],
        }),
        query_label_values=AsyncMock(return_value={
            "status": "success", "data": ["alpha-target"],
        }),
        query_metric_catalog=AsyncMock(return_value={
            "status": "success",
            "data": [
                {
                    "name": "vendor_compute_utilization",
                    "type": "gauge",
                    "help": "Compute utilization",
                    "unit": "percent",
                },
                {
                    "name": "vendor_memory_available_bytes",
                    "type": "gauge",
                    "help": "Available memory",
                    "unit": "bytes",
                },
                {
                    "name": "vendor_storage_read_bytes_total",
                    "type": "counter",
                    "help": "Bytes read from storage",
                    "unit": "bytes",
                },
            ],
        }),
        query_range=AsyncMock(side_effect=query_range),
    )
    recovery = QueryRecovery(SimpleNamespace(_prometheus_adapter=adapter), context(), 100)

    await recovery.recover_prometheus([
        {
            "name": "vendor_compute_utilization",
            "type": "gauge",
            "help": "Compute utilization",
            "unit": "percent",
        },
    ])

    assert [call.args[0] for call in adapter.query_range.await_args_list] == [
        'vendor_compute_utilization{metric_scope="alpha-target"}',
        'vendor_memory_available_bytes{metric_scope="alpha-target"}',
        'rate(vendor_storage_read_bytes_total{metric_scope="alpha-target"}[5m])',
    ]
    grounded = _ground_findings_in_data_queries([], recovery.calls, context())
    contents = " ".join(finding.content for finding in grounded)
    assert "85.6%" in contents
    assert "3.00 GiB" in contents
    assert "1.00 MiB per second" in contents


def test_partial_evidence_always_has_actions_and_a_causal_verdict():
    findings = [Finding(
        EvidenceType.UNKNOWN, "Loki unavailable after attempts", EvidenceSource.SYSTEM, 0.0,
    )]
    completed = _ensure_causal_verdict_for_mode(findings, [], False)
    assert any(finding.type == EvidenceType.RECOMMENDATION for finding in completed)
    assert any(finding.source_detail == "causal-verdict enforcement" for finding in completed)


def test_weak_model_run_recovers_queries_and_keeps_actions(investigator):
    from tests.test_service import response, tool_call

    investigator._get_incident_context = MagicMock(return_value=context())
    investigator.tools._prometheus_adapter.query_metric_catalog = AsyncMock(
        return_value={"status": "success", "data": [{
            "name": "vendor_activity_seconds_total", "type": "counter", "unit": "seconds",
        }]}
    )
    investigator.tools._prometheus_adapter.query_range = AsyncMock(side_effect=[
        {"status": "success", "data": {"result": [{
            "metric": {"metric_scope": "alpha-target"},
            "values": [[1795433700, "40000"], [1795434000, "40500"]],
        }]}},
        {"status": "success", "data": {"result": [{
            "metric": {"metric_scope": "alpha-target"},
            "values": [[1795433700, "0.85"], [1795434000, "0.6"]],
        }]}},
    ])
    investigator.tools._loki_adapter.query_labels = AsyncMock(
        return_value={"status": "success", "data": ["log_scope"]}
    )
    investigator.tools._loki_adapter.query_label_values = AsyncMock(
        return_value={"status": "success", "data": ["alpha-target"]}
    )
    investigator.tools._loki_adapter.query_range = AsyncMock(side_effect=[
        {"status": "error", "error": "invalid selector"},
        {"status": "success", "data": {"result": [{
            "stream": {"log_scope": "alpha-target"},
            "values": [["1795433800000000000", "worker restarted after timeout"]],
        }]}},
    ])
    investigator.llm.chat_completion = MagicMock(side_effect=[
        response(tool_calls=[tool_call("query_prometheus", {
            "promql": 'vendor_activity_seconds_total{metric_scope="alpha-target"}',
            "start": "s", "end": "e",
        })]),
        response(tool_calls=[tool_call("query_loki", {
            "logql": '{metric_job="metrics-job"}', "start": "s", "end": "e",
        })]),
        response(tool_calls=[tool_call("submit_findings", {"findings": [
            {"type": "hypothesis", "source": "loki", "confidence": 0.6,
             "content": "The workload restart may have contributed to the transient activity.",
             "source_detail": '{log_scope="alpha-target"}'},
            {"type": "recommendation", "source": "prometheus", "confidence": 0.8,
             "content": "Inspect the restarted worker and its resource consumption.",
             "source_detail": '{log_scope="alpha-target"}'},
        ]})]),
    ])

    findings = investigator.investigate("inc_recovery")

    assert any(finding.type == EvidenceType.HYPOTHESIS for finding in findings)
    assert any(finding.type == EvidenceType.RECOMMENDATION for finding in findings)
    assert any("0.85 seconds per second" in finding.content for finding in findings)
    assert any("Worker restarted after timeout" in finding.content for finding in findings)
    assert not any("40500" in finding.content for finding in findings)


@pytest.mark.asyncio
async def test_recovery_does_not_query_without_a_validated_target():
    adapter = SimpleNamespace(
        query_labels=AsyncMock(return_value={"status": "success", "data": ["log_scope"]}),
        query_label_values=AsyncMock(
            return_value={"status": "success", "data": ["another-target"]}
        ),
        query_range=AsyncMock(),
    )
    recovery = QueryRecovery(SimpleNamespace(_loki_adapter=adapter), context(), 100)
    await recovery.recover_loki()
    adapter.query_range.assert_not_awaited()


@pytest.mark.asyncio
async def test_condition_reference_target_label_is_preferred_over_broad_event_scope():
    async def label_values(label):
        values = {
            "cluster": ["production"],
            "opaque_target": ["alpha-target"],
        }
        return {"status": "success", "data": values[label]}

    adapter = SimpleNamespace(
        query_labels=AsyncMock(return_value={
            "status": "success", "data": ["cluster", "opaque_target"],
        }),
        query_label_values=AsyncMock(side_effect=label_values),
        query_range=AsyncMock(return_value={"status": "success", "data": {"result": []}}),
    )
    incident = IncidentContext(
        "inc_condition_ref",
        "Arbitrary latency condition",
        started_at="2026-11-24T06:51:00Z",
        resolved_at="2026-11-24T06:52:00Z",
        affected_services=["alpha-target"],
        events=[{
            "labels": {"cluster": "production"},
            "condition_ref": {"labels": {"opaque_target": "alpha-target"}},
        }],
    )
    recovery = QueryRecovery(SimpleNamespace(_loki_adapter=adapter), incident, 100)

    await recovery.recover_loki()

    assert [call.args[0] for call in adapter.query_label_values.await_args_list] == [
        "opaque_target"
    ]
    adapter.query_range.assert_awaited_once_with(
        '{opaque_target="alpha-target"}',
        "2026-11-24T06:46:00+00:00",
        "2026-11-24T06:57:00+00:00",
        100,
    )


@pytest.mark.asyncio
async def test_broad_cluster_match_cannot_authorize_recovery_query():
    adapter = SimpleNamespace(
        query_labels=AsyncMock(return_value={"status": "success", "data": ["cluster"]}),
        query_label_values=AsyncMock(return_value={
            "status": "success", "data": ["production"],
        }),
        query_range=AsyncMock(),
    )
    incident = IncidentContext(
        "inc_broad_scope",
        "Gateway latency",
        started_at="2026-11-24T06:51:00Z",
        resolved_at="2026-11-24T06:52:00Z",
        affected_services=["checkout-api"],
        events=[{"labels": {"cluster": "production"}}],
    )
    recovery = QueryRecovery(SimpleNamespace(_loki_adapter=adapter), incident, 100)

    await recovery.recover_loki()

    adapter.query_label_values.assert_awaited_once_with("cluster")
    adapter.query_range.assert_not_awaited()
    assert recovery.calls[0].result["target_coverage"] == {
        "validated": False,
        "reason": "no_exact_target_label_match",
    }


@pytest.mark.asyncio
async def test_caller_supplied_broad_selector_cannot_bypass_target_validation():
    adapter = SimpleNamespace(
        query_metric_catalog=AsyncMock(),
        query_range=AsyncMock(),
    )
    incident = IncidentContext(
        "inc_supplied_scope",
        "Gateway latency",
        started_at="2026-11-24T06:51:00Z",
        resolved_at="2026-11-24T06:52:00Z",
        affected_services=["checkout-api"],
    )
    recovery = QueryRecovery(SimpleNamespace(_prometheus_adapter=adapter), incident, 100)

    await recovery.recover_related_prometheus('{cluster="production"}')

    adapter.query_metric_catalog.assert_not_awaited()
    adapter.query_range.assert_not_awaited()


@pytest.mark.asyncio
async def test_related_recovery_rejects_unrelated_production_shaped_metrics():
    async def query_range(expression, _start, _end):
        assert expression == 'traefik_connections_open{opaque_scope="edge-api"}'
        return {
            "status": "success",
            "data": {"result": [{
                "metric": {"opaque_scope": "edge-api"},
                "values": [[1795502880, "12"], [1795503420, "4"]],
            }]},
        }

    adapter = SimpleNamespace(
        query_labels=AsyncMock(return_value={
            "status": "success", "data": ["opaque_scope"],
        }),
        query_label_values=AsyncMock(return_value={
            "status": "success", "data": ["edge-api"],
        }),
        query_metric_catalog=AsyncMock(return_value={
            "status": "success",
            "data": [
                {
                    "name": "argocd_cluster_cache_age_seconds",
                    "type": "gauge",
                    "help": "Age of the application cache",
                    "unit": "seconds",
                },
                {
                    "name": "kube_pod_service_account",
                    "type": "gauge",
                    "help": "Pod service account metadata",
                    "unit": "",
                },
                {
                    "name": "traefik_connections_open",
                    "type": "gauge",
                    "help": "Open proxy connections",
                    "unit": "connections",
                },
            ],
        }),
        query_range=AsyncMock(side_effect=query_range),
    )
    incident = IncidentContext(
        "inc_traefik",
        "Traefik Service High Latency",
        started_at="2026-11-24T06:51:00Z",
        resolved_at="2026-11-24T06:52:00Z",
        affected_services=["edge-api"],
        events=[{"condition_ref": {"labels": {"opaque_scope": "edge-api"}}}],
    )
    recovery = QueryRecovery(SimpleNamespace(_prometheus_adapter=adapter), incident, 100)

    await recovery.recover_related_prometheus()

    adapter.query_range.assert_awaited_once()
    query_call = next(call for call in recovery.calls if call.tool_name == "query_prometheus")
    assert query_call.result["target_coverage"]["validated"] is True
    assert query_call.result["signal_role"] == "related"
    audit = next(
        call for call in recovery.calls if call.tool_name == "select_prometheus_candidates"
    )
    rejected = {item["name"] for item in audit.result["rejected"]}
    assert rejected == {
        "argocd_cluster_cache_age_seconds",
        "kube_pod_service_account",
    }
    assert audit.result["_private"] is True
