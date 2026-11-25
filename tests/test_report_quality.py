import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from adapters.circuit_breaker import CircuitBreaker, CircuitState
from models import (
    EvidenceSource,
    EvidenceType,
    Finding,
    IncidentContext,
    ToolResult,
)
from service.investigator import (
    _ensure_causal_verdict,
    _finalize_findings,
    _ground_findings_in_data_queries,
    _is_reasoning_preamble,
    _logs_from_tool_calls,
    _metrics_from_tool_calls,
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


def test_zero_score_telemetry_is_not_admitted_as_incident_evidence():
    context = IncidentContext(
        "inc_pg",
        "PostgreSQL Slow Query",
        affected_services=["postgres"],
    )
    calls = [
        metric_call("ambient_temperature_celsius"),
        log_call('{job="backup-agent"}', "filesystem snapshot completed normally"),
    ]

    findings = _ground_findings_in_data_queries([], calls, context)

    assert findings == []


def test_contextless_telemetry_is_not_admitted_as_incident_evidence():
    calls = [
        metric_call('up{target="api"}'),
        log_call('{service="api"}', "api restarted after timeout"),
    ]

    assert _ground_findings_in_data_queries([], calls) == []


def test_recovered_telemetry_without_validated_target_coverage_is_not_evidence():
    context = IncidentContext(
        "inc_coverage",
        "Vendor latency",
        affected_services=["alpha-target"],
    )
    metric = metric_call('vendor_latency{scope="alpha-target"}', "82")
    metric.result["signal_role"] = "recovered_primary"
    metric.result["target_coverage"] = {
        "validated": False,
        "reason": "no_exact_target_label_match",
    }
    log = log_call(
        '{scope="alpha-target"}',
        "vendor request exceeded latency threshold",
    )
    log.result["signal_role"] = "recovered_target"
    log.result["target_coverage"] = {
        "validated": False,
        "reason": "no_exact_target_label_match",
    }

    assert _ground_findings_in_data_queries([], [metric, log], context) == []
    assert _metrics_from_tool_calls([metric]) == []
    assert _logs_from_tool_calls([log]) == []


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


def test_positive_telemetry_never_independently_declares_recovery():
    context = IncidentContext(
        "inc_capacity",
        "Capacity condition",
        status="open",
        affected_services=["worker-service"],
    )
    findings = _ground_findings_in_data_queries(
        [], [metric_call('active_capacity{service="worker-service"}', "1")], context
    )

    assert findings
    assert all(finding.source is not EvidenceSource.ALERT for finding in findings)
    assert all("recovery" not in finding.content.lower() for finding in findings)


def test_two_distinct_prometheus_facts_can_support_likely_cause():
    findings = [
        Finding(
            EvidenceType.FACT,
            "Vendor queue depth rose before request latency.",
            EvidenceSource.PROMETHEUS,
            1.0,
            "query_prometheus: vendor_queue_depth",
        ),
        Finding(
            EvidenceType.FACT,
            "Vendor worker saturation increased during the same interval.",
            EvidenceSource.PROMETHEUS,
            1.0,
            "query_prometheus: vendor_worker_saturation",
        ),
        Finding(
            EvidenceType.LIKELY_CAUSE,
            "Vendor worker saturation delayed queue processing and caused request latency.",
            EvidenceSource.LLM,
            0.9,
        ),
    ]

    completed = _ensure_causal_verdict(findings, [])
    causal = [finding for finding in completed if finding.type == EvidenceType.LIKELY_CAUSE]
    assert len(causal) == 1
    assert causal[0].confidence == 0.75
    assert causal[0].supports_evidence == [0, 1]


def test_provider_pair_does_not_make_unrelated_fact_causal_support():
    findings = [
        Finding(
            EvidenceType.FACT,
            "Vendor request latency increased.",
            EvidenceSource.PROMETHEUS,
            1.0,
            "query_prometheus: vendor_request_latency",
        ),
        Finding(
            EvidenceType.FACT,
            "The nightly backup completed successfully.",
            EvidenceSource.LOKI,
            1.0,
            "query_loki: backup",
        ),
        Finding(
            EvidenceType.LIKELY_CAUSE,
            "Worker saturation caused vendor request latency.",
            EvidenceSource.LLM,
            0.9,
        ),
    ]

    completed = _ensure_causal_verdict(findings, [])
    causal = [finding for finding in completed if finding.type == EvidenceType.HYPOTHESIS]
    assert len(causal) == 1
    assert causal[0].confidence == 0.55
    assert causal[0].supports_evidence == [0]


def test_full_finalizer_accepts_same_provider_independent_mechanism_support():
    context = IncidentContext(
        "inc_generic",
        "Vendor request degradation",
        affected_services=["alpha-target"],
    )
    calls = [
        metric_call(
            'vendor_request_latency_seconds{tenant="alpha-target"}',
            "4.2",
        ),
        metric_call(
            'vendor_request_worker_saturation_ratio{tenant="alpha-target"}',
            "0.94",
        ),
    ]
    submitted = [
        Finding(
            EvidenceType.LIKELY_CAUSE,
            "Vendor request worker saturation increased request latency.",
            EvidenceSource.LLM,
            0.9,
        )
    ]

    finalized = _finalize_findings(
        submitted,
        calls,
        context,
        context.incident_id,
        verification=False,
    )

    facts = [finding for finding in finalized if finding.type is EvidenceType.FACT]
    causes = [
        finding for finding in finalized if finding.type is EvidenceType.LIKELY_CAUSE
    ]
    assert len(facts) == 2
    assert len(causes) == 1
    assert causes[0].supports_evidence == [0, 1]


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
    persistence.load_pinned_telemetry_catalog.return_value = None
    persistence.pin_telemetry_catalog_revision.return_value = {
        "revision_id": "tcr_abc",
        "content_hash": "abc",
        "entry_count": 2,
        "catalog": {
            "prometheus": {
                "pg_stat_statements_mean_exec_time": {
                    "name": "pg_stat_statements_mean_exec_time",
                    "help": "PostgreSQL query duration",
                    "_search_text": "PostgreSQL query duration",
                }
            },
            "loki": {"app": {"name": "app", "_search_text": "app"}},
        },
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

    await investigator._refresh_telemetry_catalog(context, "irun_1")

    assert persistence.store_telemetry_catalog.call_count == 2
    assert "pg_stat_statements_mean_exec_time" in context.telemetry_catalog["prometheus"]
    assert context.telemetry_catalog_revision_id == "tcr_abc"
    persistence.pin_telemetry_catalog_revision.assert_called_once_with("irun_1")


@pytest.mark.asyncio
async def test_failed_catalog_refresh_keeps_last_db_snapshot(investigator):
    persistence = MagicMock()
    persistence.claim_telemetry_catalog_refresh.return_value = True
    persistence.load_pinned_telemetry_catalog.return_value = None
    persistence.pin_telemetry_catalog_revision.return_value = {
        "revision_id": "tcr_last_good",
        "content_hash": "last-good",
        "entry_count": 1,
        "catalog": {
            "prometheus": {"up": {"name": "up", "_search_text": "up"}},
            "loki": {},
        },
    }
    investigator.persistence = persistence
    investigator.tools._prometheus_adapter.query_metric_catalog = AsyncMock(
        side_effect=RuntimeError("provider unavailable")
    )
    investigator.tools._loki_adapter.query_labels = AsyncMock(
        side_effect=RuntimeError("provider unavailable")
    )
    context = IncidentContext("inc", "Target Down")

    await investigator._refresh_telemetry_catalog(context, "irun_2")

    assert context.telemetry_catalog["prometheus"]["up"]["name"] == "up"
    assert persistence.mark_telemetry_catalog_refresh_failed.call_count == 2


@pytest.mark.asyncio
async def test_retry_reuses_pinned_catalog_without_live_refresh(investigator):
    persistence = MagicMock()
    persistence.load_pinned_telemetry_catalog.return_value = {
        "revision_id": "tcr_original",
        "content_hash": "original-hash",
        "entry_count": 1,
        "catalog": {
            "prometheus": {
                "vendor_queue_depth": {
                    "name": "vendor_queue_depth",
                    "_search_text": "vendor queue depth",
                }
            },
            "loki": {},
        },
    }
    investigator.persistence = persistence
    context = IncidentContext("inc", "Vendor queue latency")

    await investigator._refresh_telemetry_catalog(context, "irun_retry")

    assert context.telemetry_catalog_revision_id == "tcr_original"
    assert "vendor_queue_depth" in context.telemetry_catalog["prometheus"]
    persistence.claim_telemetry_catalog_refresh.assert_not_called()
    persistence.pin_telemetry_catalog_revision.assert_not_called()
