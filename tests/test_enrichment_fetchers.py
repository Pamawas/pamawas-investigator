"""Regression tests for the evidence-enrichment fetchers.

Both fetchers must return the typed models (Metric / LogEntry), not bare
dicts. The persistence layer serializes enrichment with `to_dict()`, and a
dict carrying a raw `datetime` raised
"Object of type datetime is not JSON serializable" — which aborted
persistence and left the investigation run retry-looping with no evidence.
"""

import asyncio

from models import LogEntry, Metric
from service.loki_fetcher import LokiFetcher
from service.prometheus_fetcher import PrometheusFetcher


def test_prometheus_fetch_returns_metric_models():
    """Instant-query samples come back as Metric instances with a datetime."""
    fetcher = PrometheusFetcher(base_url="http://prometheus:9090")

    async def fake_query_instant(query, eval_time):
        return {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {"__name__": "up", "job": "docker_cadvisor"},
                        "value": [1794742200, "1"],
                    }
                ],
            },
        }

    fetcher._query_instant = fake_query_instant

    samples = asyncio.run(
        fetcher.fetch_async(
            incident_id="inc_1",
            start="2026-11-15T12:00:00Z",
            end="2026-11-15T12:05:00Z",
            service="grafana",
        )
    )

    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, Metric)
    assert sample.name == "up"
    assert sample.value == 1.0
    # to_dict is what persistence encodes; it must be JSON-safe.
    assert isinstance(sample.to_dict()["at"], str)


def test_loki_fetch_returns_log_entry_models():
    """Query-range lines come back as LogEntry instances with a datetime."""
    fetcher = LokiFetcher(base_url="http://loki:3100")

    async def fake_query_range(query, start, end, limit):
        return {
            "status": "success",
            "data": {
                "resultType": "streams",
                "result": [
                    {
                        "stream": {"kubernetes_container_name": "grafana"},
                        "values": [["1794742200000000000", "level=error boom"]],
                    }
                ],
            },
        }

    fetcher._query_range = fake_query_range

    entries = asyncio.run(
        fetcher.fetch_async(
            incident_id="inc_1",
            start="2026-11-15T12:00:00Z",
            end="2026-11-15T12:05:00Z",
            limit=10,
            service="grafana",
        )
    )

    assert len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, LogEntry)
    assert entry.line == "level=error boom"
    assert isinstance(entry.to_dict()["timestamp"], str)


def test_fetchers_skip_without_a_service():
    """No service means no truthful selector, so enrichment yields nothing."""
    assert (
        asyncio.run(
            PrometheusFetcher(base_url="http://prometheus:9090").fetch_async(
                incident_id="inc_1", start="", end="", service=None
            )
        )
        == []
    )
    assert (
        asyncio.run(
            LokiFetcher(base_url="http://loki:3100").fetch_async(
                incident_id="inc_1", start="", end="", service=None
            )
        )
        == []
    )
