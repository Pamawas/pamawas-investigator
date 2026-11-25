from unittest.mock import AsyncMock, MagicMock

import pytest

from models import (
    EvidenceSource,
    EvidenceType,
    Finding,
    IncidentContext,
    InvestigationState,
    ToolResult,
)
from service.evidence_summary import human_number, metric_unit
from service.investigator import _finalize_findings, _ground_findings_in_data_queries


def incident():
    return IncidentContext(
        "inc_reader", "CPU", started_at="2026-11-23T11:35:10Z",
        resolved_at="2026-11-23T11:40:10Z", affected_services=["alpha-host"],
    )


def query_calls():
    primary = ToolResult(
        "query_prometheus",
        {"promql": '100 - (rate(vendor_cpu_seconds_total{tenant_unit="alpha-host"}[5m]) * 100)',
         "start": "2026-11-23T11:00:00Z", "end": "2026-11-23T12:30:00Z"},
        {"status": "success", "metric_metadata": {"unit": "percent"},
         "data": {"result": [{
            "metric": {"tenant_unit": "alpha-host"},
            "values": [[1795432800, "99.9"], [1795433700, "86.5"], [1795434000, "71"]],
        }]}}, 1,
    )
    unrelated = [ToolResult(
        "query_prometheus", {"promql": f"unrelated_counter_{index}"},
        {"status": "success", "data": {"result": [{"value": [1795434000, "170599000"]}]}}, 1,
    ) for index in range(20)]
    logs = ToolResult(
        "query_loki", {"logql": '{vendor_scope="alpha-host"}'},
        {"status": "success", "data": {"result": [{
            "stream": {"vendor_scope": "alpha-host", "vendor_app": "mesh-agent",
                       "vendor_source": "mesh-agent", "verbosity": "warn"},
            "values": [["1795433805296000000",
                        "2026-11-23T11:36:45.296Z WARN shared/signal/client/grpc.go:287: "
                        "Disconnected from Signal service: EOF"]],
        }]}}, 1,
    )
    return [primary, *unrelated, logs]


def test_report_evidence_preserves_primary_signal_and_reads_as_observations(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Jakarta")
    findings = _ground_findings_in_data_queries([], query_calls(), incident())
    assert len(findings) == 2
    assert findings[0].content == (
        "Vendor cpu seconds peaked at 86.5% at 18:35:00 WIB; "
        "the last queried value was 71% at 18:40:00 WIB."
    )
    assert findings[1].content == (
        "mesh-agent at 18:36:45 WIB: Disconnected from Signal service: EOF."
    )
    content = " ".join(finding.content for finding in findings)
    for audit_text in ("{", "query_prometheus", "numeric samples", ".go:", "99.9", "170599000"):
        assert audit_text not in content
    assert "vendor_cpu_seconds_total" in findings[0].source_detail
    assert 'vendor_scope="alpha-host"' in findings[1].source_detail


def test_same_metric_windows_render_one_wide_timeline(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Jakarta")
    expression = (
        'ceph_osd_apply_latency_ms{ceph_daemon="osd.1", '
        'instance="10.1.1.4:9283", job="proxmox_ceph"}'
    )
    wide = ToolResult(
        "query_prometheus",
        {
            "promql": expression,
            "start": "2026-11-24T06:45:00Z",
            "end": "2026-11-24T07:00:00Z",
        },
        {"status": "success", "data": {"result": [{
            "metric": {"ceph_daemon": "osd.1", "instance": "10.1.1.4:9283"},
            "values": [[1795502880, "271"], [1795503420, "68"]],
        }]}},
        1,
    )
    incident_window = ToolResult(
        "query_prometheus",
        {
            "promql": expression,
            "start": "2026-11-24T06:50:00Z",
            "end": "2026-11-24T06:53:00Z",
        },
        {"status": "success", "data": {"result": [{
            "metric": {"ceph_daemon": "osd.1", "instance": "10.1.1.4:9283"},
            "values": [[1795503000, "215"], [1795503180, "84"]],
        }]}},
        1,
    )
    context = IncidentContext(
        "inc_ceph",
        "Ceph OSD High Latency",
        started_at="2026-11-24T06:51:00Z",
        resolved_at="2026-11-24T06:52:00Z",
        affected_services=["10.1.1.4:9283", "proxmox_ceph"],
    )

    findings = _ground_findings_in_data_queries(
        [], [wide, incident_window], context
    )

    assert len(findings) == 1
    assert "peaked at 271" in findings[0].content
    assert "last queried value was 68" in findings[0].content
    assert "13:45:00 WIB" in findings[0].source_detail
    assert "14:00:00 WIB" in findings[0].source_detail


@pytest.mark.asyncio
async def test_reader_ready_evidence_is_passed_to_persistence(investigator, monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Jakarta")
    investigator.persistence = MagicMock()
    investigator._persist_investigation = AsyncMock()
    state = InvestigationState("inc_reader", [], query_calls(), context=incident())
    model_findings = [Finding(
        EvidenceType.FACT, "Copied alert value 80.57%", EvidenceSource.PROMETHEUS, 1,
        source_detail="invented expression",
    )]
    await investigator._handle_submit_findings(model_findings, "inc_reader", "key", state)
    persisted = investigator._persist_investigation.await_args.kwargs["findings"]
    assert persisted[0].content == (
        "Vendor cpu seconds peaked at 86.5% at 18:35:00 WIB; "
        "the last queried value was 71% at 18:40:00 WIB."
    )
    assert not any("80.57" in finding.content for finding in persisted)
    assert any(finding.type == EvidenceType.RECOMMENDATION for finding in persisted)


def test_numbers_are_readable_without_changing_units_or_tiny_values():
    assert human_number(170599000) == "170,599,000"
    assert human_number(1048576, "bytes") == "1.00 MiB"
    assert human_number(1048576, "bytes per second") == "1.00 MiB per second"
    assert human_number(0.00000001) == "0.00000001"


def test_counter_rate_preserves_provider_declared_unit(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Jakarta")
    call = ToolResult(
        "query_prometheus",
        {
            "promql": (
                'rate(container_cpu_usage_seconds_total{instance="alpha-host"}[5m])'
            ),
            "start": "2026-11-23T11:30:00Z",
            "end": "2026-11-23T11:45:00Z",
        },
        {
            "status": "success",
            "metric_metadata": {
                "name": "container_cpu_usage_seconds_total",
                "type": "counter",
                "unit": "seconds",
            },
            "data": {"result": [{
                "metric": {"instance": "alpha-host"},
                "values": [[1795433700, "0.53"], [1795434000, "0.46"]],
            }]},
        },
        1,
    )

    findings = _ground_findings_in_data_queries([], [call], incident())

    assert findings[0].content == (
        "Container cpu usage seconds peaked at 0.53 seconds per second at 18:35:00 WIB; "
        "the last queried value was 0.46 seconds per second at 18:40:00 WIB."
    )


def test_counter_rate_does_not_infer_unit_from_metric_name(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Jakarta")
    call = ToolResult(
        "query_prometheus",
        {
            "promql": 'rate(vendor_payload_bytes_total{scope="alpha-host"}[5m])',
            "start": "2026-11-23T11:30:00Z",
            "end": "2026-11-23T11:45:00Z",
        },
        {
            "status": "success",
            "metric_metadata": {
                "name": "vendor_payload_bytes_total",
                "type": "counter",
                "unit": "",
            },
            "data": {"result": [{
                "metric": {"scope": "alpha-host"},
                "values": [[1795433700, "2.5"], [1795434000, "1.5"]],
            }]},
        },
        1,
    )

    findings = _ground_findings_in_data_queries([], [call], incident())

    assert "2.5 units per second" in findings[0].content
    assert "bytes per second" not in findings[0].content


def test_composite_same_metric_ratio_scaled_by_100_renders_percent(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Jakarta")
    expression = (
        'sum by (service) (rate(traefik_service_requests_total{service="grafana",'
        'code=~"5.."}[5m])) / sum by (service) '
        '(rate(traefik_service_requests_total{service="grafana"}[5m])) * 100'
    )
    context = IncidentContext(
        "inc_traefik_error",
        "Traefik Service High Error Rate",
        started_at="2026-11-24T15:28:40Z",
        resolved_at="2026-11-24T15:34:40Z",
        affected_services=["grafana"],
    )
    call = ToolResult(
        "query_prometheus",
        {
            "promql": expression,
            "start": "2026-11-24T15:23:40Z",
            "end": "2026-11-24T15:39:40Z",
        },
        {
            "status": "success",
            "metric_metadata": {
                "name": "traefik_service_requests_total",
                "type": "counter",
                "unit": "",
            },
            "data": {"result": [{
                "metric": {"service": "grafana"},
                "values": [[1795533940, "80"], [1795534780, "8.09"]],
            }]},
        },
        1,
    )

    findings = _ground_findings_in_data_queries([], [call], context)

    assert len(findings) == 1
    assert "peaked at 80%" in findings[0].content
    assert "last queried value was 8.09%" in findings[0].content


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ('metric_total{service="api"} * 100', ""),
        (
            'rate(metric_total{service="api"}[5m]) '
            '/ rate(metric_total{service="api"}[5m])',
            "",
        ),
        (
            'metric_total{path="/download*100"} / 10',
            "",
        ),
        (
            'sum(rate(bytes_total{service="api"}[5m])) '
            '/ sum(rate(seconds_total{service="api"}[5m])) * 100',
            "",
        ),
    ],
)
def test_metric_unit_does_not_guess_ambiguous_percentages(expression, expected):
    assert metric_unit(expression, {"type": "counter", "unit": ""}) == expected


def test_alert_payload_preserves_explicit_measurement_for_arbitrary_alert(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Jakarta")
    context = IncidentContext(
        "inc_thermal",
        "Thermal Saturation",
        started_at="2026-11-23T11:35:10Z",
        resolved_at="2026-11-23T11:40:10Z",
        affected_services=["alpha-host"],
        events=[
            {
                "id": "evt_fire",
                "timestamp": "2026-11-23T11:35:10Z",
                "status": "firing",
                "raw_payload": {"commonAnnotations": {"value": "91.2°C"}},
            },
            {
                "id": "evt_resolve",
                "timestamp": "2026-11-23T11:40:10Z",
                "status": "resolved",
                "raw_payload": {"commonAnnotations": {"value": "70.1°C"}},
            },
        ],
    )

    findings = _ground_findings_in_data_queries([], [], context)

    assert findings[0].source == EvidenceSource.ALERT
    assert findings[0].content == (
        "Alert payload reported Thermal Saturation at 91.2 °C when firing at "
        "18:35:10 WIB; it reported 70.1 °C when resolved at 18:40:10 WIB."
    )
    assert findings[0].source_detail == "alert events: evt_fire, evt_resolve"


def test_causal_synthesis_removes_superseded_process_only_gap(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Jakarta")
    process_gap = Finding(
        EvidenceType.UNKNOWN,
        "Investigation ended without a stated conclusion; the model only described its process.",
        EvidenceSource.SYSTEM,
        0.0,
    )

    findings = _finalize_findings(
        [process_gap], query_calls()[:1], incident(), "inc_reader", False
    )

    assert any(
        finding.type == EvidenceType.UNKNOWN and "Root cause was not established" in finding.content
        for finding in findings
    )
    assert not any(
        finding.type == EvidenceType.UNKNOWN
        and "without a stated conclusion" in finding.content
        for finding in findings
    )


def test_evidence_rules_apply_to_another_signal_and_arbitrary_schemas(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Jakarta")
    context = IncidentContext(
        "inc_latency", "Latency", started_at="2026-11-23T11:35:10Z",
        resolved_at="2026-11-23T11:40:10Z", affected_services=["beta-app"],
    )
    calls = [ToolResult(
        "query_prometheus", {"promql": 'vendor_latency{customer_key="beta-app"}'},
        {"status": "success", "metric_metadata": {"unit": "ms"},
         "data": {"result": [{"metric": {"customer_key": "beta-app"},
                              "values": [[1795433700, "1234.56"], [1795434000, "200"]]}]}}, 1,
    ), ToolResult(
        "query_loki", {"logql": '{application_key="beta-app"}'},
        {"status": "success", "data": {"result": [{
            "stream": {"vendor_component": "gateway-worker"},
            "values": [["1795433805296000000",
                        '{"unusual_body_key":"Upstream request failed after timeout",'
                        '"verbosity":"error"}']],
        }]}}, 1,
    )]
    first = _ground_findings_in_data_queries([], calls, context)
    second = _ground_findings_in_data_queries([], calls, context)
    assert first == second
    assert first[0].content == (
        "Vendor latency peaked at 1,234.56 ms at 18:35:00 WIB; "
        "the last queried value was 200 ms at 18:40:00 WIB."
    )
    assert first[1].content == (
        "gateway-worker at 18:36:45 WIB: Upstream request failed after timeout."
    )
    assert not any("CPU" in finding.content or "{" in finding.content for finding in first)


def test_metric_observation_excludes_neighbouring_osds(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Jakarta")
    context = IncidentContext(
        "inc_osd_4",
        "Ceph OSD osd.4 latency",
        started_at="2026-11-23T11:35:10Z",
        resolved_at="2026-11-23T11:40:10Z",
        affected_services=["ceph-osd.4"],
    )
    calls = [
        ToolResult(
            "query_prometheus",
            {
                "promql": "ceph_osd_apply_latency_ms",
                "start": "2026-11-23T11:30:00Z",
                "end": "2026-11-23T11:45:00Z",
            },
            {
                "status": "success",
                "metric_metadata": {"unit": "ms"},
                "data": {
                    "result": [
                        {
                            "metric": {"ceph_daemon": "osd.2", "osd": "2"},
                            "values": [[1795433700, "9000"], [1795434000, "8000"]],
                        },
                        {
                            "metric": {"ceph_daemon": "osd.4", "osd": "4"},
                            "values": [[1795433700, "120"], [1795434000, "30"]],
                        },
                    ]
                },
            },
            1,
        )
    ]

    findings = _ground_findings_in_data_queries([], calls, context)

    assert len(findings) == 1
    assert "osd.4" in findings[0].content
    assert "9,000" not in findings[0].content
