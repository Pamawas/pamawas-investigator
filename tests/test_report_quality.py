import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from adapters.circuit_breaker import CircuitBreaker, CircuitState
from models import EvidenceSource, EvidenceType, Finding, IncidentContext, ToolResult
from service.investigator import (
    _ensure_causal_verdict,
    _ground_findings_in_data_queries,
    _is_reasoning_preamble,
)


def metric_call(expression: str, value: str = "1", metadata: dict | None = None) -> ToolResult:
    return ToolResult(
        "query_prometheus",
        {"promql": expression, "start": "s", "end": "e"},
        {
            "status": "success",
            "metric_metadata": metadata or {},
            "data": {"result": [{"metric": {}, "values": [[1, value]]}]},
        },
        1,
    )


def log_call(selector: str, line: str, labels: dict | None = None) -> ToolResult:
    return ToolResult(
        "query_loki",
        {"logql": selector, "start": "s", "end": "e"},
        {
            "status": "success",
            "data": {"result": [{"stream": labels or {}, "values": [["1", line]]}]},
        },
        1,
    )


def test_postgres_rejects_mysql_metrics_and_netbird_logs():
    context = IncidentContext(
        "inc_pg",
        "PostgreSQL Slow Query",
        affected_services=["postgres"],
        telemetry_catalog={
            "prometheus": {
                "pg_stat_statements_mean_exec_time": {
                    "name": "pg_stat_statements_mean_exec_time",
                    "help": "PostgreSQL query duration",
                    "_search_text": "PostgreSQL query duration",
                },
                "mysql_global_status_last_query_cost": {
                    "name": "mysql_global_status_last_query_cost",
                    "help": "MySQL query cost",
                    "_search_text": "MySQL query cost",
                },
            },
            "loki": {},
        },
    )
    calls = [
        metric_call("pg_stat_statements_mean_exec_time"),
        metric_call("mysql_global_status_last_query_cost"),
        log_call('{app="postgres"}', "postgres slow statement duration 3s"),
        log_call('{app="netbird"}', "netbird peer connection error"),
    ]

    findings = _ground_findings_in_data_queries([], calls, context)
    details = " ".join(finding.source_detail for finding in findings)
    contents = " ".join(finding.content for finding in findings).lower()

    assert "pg_stat_statements" in details
    assert "mysql_global_status" not in details
    assert "postgres slow statement" in contents
    assert "netbird" not in contents


def test_traefik_rejects_webhook_metrics_and_unrelated_queue_logs():
    context = IncidentContext(
        "inc_traefik", "Traefik High Latency", affected_services=["monitoring-prometheus"]
    )
    calls = [
        metric_call("traefik_service_request_duration_seconds"),
        metric_call("apiserver_admission_webhook_admission_duration_seconds"),
        log_call(
            '{app="monitoring-prometheus"}',
            "traefik upstream timeout caused high request latency",
        ),
        log_call('{app="monitoring-prometheus"}', "nuq.queue_scrape completed normally"),
    ]

    findings = _ground_findings_in_data_queries([], calls, context)
    rendered = " ".join(
        [
            *(finding.source_detail for finding in findings),
            *(finding.content for finding in findings),
        ]
    ).lower()

    assert "traefik_service_request_duration" in rendered
    assert "admission_webhook" not in rendered
    assert "nuq.queue_scrape" not in rendered


def test_ceph_capacity_is_excluded_from_latency_evidence():
    context = IncidentContext("inc_ceph", "Ceph OSD High Latency", affected_services=["osd.1"])
    findings = _ground_findings_in_data_queries(
        [],
        [
            metric_call("ceph_cluster_total_used_bytes", "1099511627776", {"unit": "bytes"}),
            metric_call("ceph_osd_apply_latency_ms", "87", {"unit": "ms"}),
        ],
        context,
    )

    rendered = " ".join(finding.source_detail for finding in findings)
    assert "ceph_osd_apply_latency_ms" in rendered
    assert "ceph_cluster_total_used_bytes" not in rendered


def test_short_reasoning_note_is_never_a_finding():
    assert _is_reasoning_preamble("Let me query Loki for matching backend logs.")


def test_open_authentik_incident_surfaces_current_worker_conflict():
    context = IncidentContext(
        "inc_authentik",
        "Authentik Worker Absent",
        status="open",
        affected_services=["authentik-worker"],
    )
    findings = _ground_findings_in_data_queries(
        [], [metric_call('authentik_tasks_workers{deployment="authentik-worker"}', "1")], context
    )

    conflict = [
        finding for finding in findings
        if finding.source_detail == "incident-state reconciliation"
    ]
    assert len(conflict) == 1
    assert "conflicts with the still-open" in conflict[0].content


def test_likely_cause_requires_prometheus_and_loki_facts():
    findings = [
        Finding(EvidenceType.FACT, "Metric one.", EvidenceSource.PROMETHEUS, 1.0),
        Finding(EvidenceType.FACT, "Metric two.", EvidenceSource.PROMETHEUS, 1.0),
        Finding(EvidenceType.LIKELY_CAUSE, "A component saturated.", EvidenceSource.LLM, 0.9),
    ]

    completed = _ensure_causal_verdict(findings, [])
    causal = [finding for finding in completed if finding.type == EvidenceType.HYPOTHESIS]
    assert len(causal) == 1
    assert causal[0].confidence == 0.55


def test_metric_caption_does_not_borrow_incident_title():
    context = IncidentContext("inc_queue", "Database Queue Latency", affected_services=["db"])
    findings = _ground_findings_in_data_queries(
        [], [metric_call('nuq_queue_depth{service="db"}', "12")], context
    )

    assert findings[0].content.startswith("Nuq queue depth peaked")
    assert not findings[0].content.startswith("Database Queue Latency")


def test_circuit_breaker_stdlib_logging_does_not_raise(caplog):
    caplog.set_level(logging.INFO)
    breaker = CircuitBreaker(
        "prometheus", failure_threshold=1, timeout_seconds=0, half_open_max_calls=1
    )

    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    assert breaker.allow_call()
    assert breaker.state == CircuitState.HALF_OPEN
    breaker.record_success()

    assert breaker.state == CircuitState.CLOSED
    assert "breaker=prometheus" in caplog.text


@pytest.mark.asyncio
async def test_investigator_refreshes_and_freezes_live_catalog(investigator):
    persistence = MagicMock()
    persistence.claim_telemetry_catalog_refresh.return_value = True
    persistence.load_telemetry_catalog.return_value = {
        "prometheus": {
            "pg_stat_statements_mean_exec_time": {
                "name": "pg_stat_statements_mean_exec_time",
                "help": "PostgreSQL query duration",
                "_search_text": "PostgreSQL query duration",
            }
        },
        "loki": {"app": {"name": "app", "_search_text": "app"}},
    }
    investigator.persistence = persistence
    investigator.tools._prometheus_adapter.query_metric_catalog = AsyncMock(return_value={
        "status": "success",
        "data": [{
            "name": "pg_stat_statements_mean_exec_time",
            "help": "PostgreSQL query duration",
            "type": "gauge",
            "unit": "seconds",
        }],
    })
    investigator.tools._loki_adapter.query_labels = AsyncMock(return_value={
        "status": "success", "data": ["app", "namespace"],
    })
    context = IncidentContext("inc", "PostgreSQL Slow Query")

    await investigator._refresh_telemetry_catalog(context)

    assert persistence.store_telemetry_catalog.call_count == 2
    assert "pg_stat_statements_mean_exec_time" in context.telemetry_catalog["prometheus"]
    persistence.load_telemetry_catalog.assert_called_once_with()


@pytest.mark.asyncio
async def test_failed_catalog_refresh_keeps_last_db_snapshot(investigator):
    persistence = MagicMock()
    persistence.claim_telemetry_catalog_refresh.return_value = True
    persistence.load_telemetry_catalog.return_value = {
        "prometheus": {"up": {"name": "up", "_search_text": "up"}},
        "loki": {},
    }
    investigator.persistence = persistence
    investigator.tools._prometheus_adapter.query_metric_catalog = AsyncMock(
        side_effect=RuntimeError("provider unavailable")
    )
    investigator.tools._loki_adapter.query_labels = AsyncMock(
        side_effect=RuntimeError("provider unavailable")
    )
    context = IncidentContext("inc", "Target Down")

    await investigator._refresh_telemetry_catalog(context)

    assert context.telemetry_catalog["prometheus"]["up"]["name"] == "up"
    assert persistence.mark_telemetry_catalog_refresh_failed.call_count == 2
