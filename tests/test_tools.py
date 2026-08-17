import pytest

from adapters.fake import (
    FakeDeploymentTransport,
    FakeLokiTransport,
    FakePrometheusTransport,
    FakeRelatedIncidentsTransport,
)
from models import EvidenceType, Finding
from tools import InvestigatorTools, ToolRegistry


@pytest.fixture
def fake_prometheus():
    return FakePrometheusTransport()


@pytest.fixture
def fake_loki():
    return FakeLokiTransport()


@pytest.fixture
def fake_deployments():
    return FakeDeploymentTransport()


@pytest.fixture
def fake_related_incidents():
    return FakeRelatedIncidentsTransport()


@pytest.fixture
def tools(config, fake_prometheus, fake_loki, fake_deployments, fake_related_incidents):
    """Create InvestigatorTools with fake transports."""
    tools = InvestigatorTools(config)
    tools._prometheus_adapter = fake_prometheus
    tools._loki_adapter = fake_loki
    tools._deployment_adapter = fake_deployments
    tools._related_incidents_adapter = fake_related_incidents
    return tools


def test_tools_return_success_and_record_metrics(config, tools):
    """Test that tools return success and record metrics via fake transports."""
    # Test Prometheus
    result = tools.query_prometheus("up", "2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z")
    assert result["status"] == "success"
    assert "data" in result

    # Test Loki
    result = tools.query_loki("{job='api'}", "2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z")
    assert result["status"] == "success"
    assert "data" in result

    # Test deployments
    result = tools.get_recent_deployments("api", "2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z")
    assert result["status"] == "success"
    assert "data" in result
    assert len(result["data"]) > 0

    # Test related incidents
    result = tools.get_related_incidents("api", ["timeout"])
    assert result["status"] == "success"
    assert "data" in result
    assert len(result["data"]) > 0


def test_submit_findings_serializes_findings_and_records_metrics(config, tools):
    finding = Finding(EvidenceType.FACT, "latency rose", "prometheus", 1.0)
    result = tools.submit_findings([finding])
    assert result["findings"] == [finding.to_dict()]


def test_tool_registry_names_match_definitions():
    definitions = ToolRegistry.get_tools()
    names = ToolRegistry.get_tool_names()
    assert names == [item["function"]["name"] for item in definitions]
    assert names == [
        "query_prometheus",
        "query_loki",
        "get_recent_deployments",
        "get_related_incidents",
        "submit_findings",
    ]
    submit_schema = definitions[-1]["function"]["parameters"]
    assert submit_schema["required"] == ["findings"]


def test_fake_transports_track_calls(config, tools):
    """Test that fake transports track calls for verification."""
    # Prometheus
    tools.query_prometheus("up", "s", "e")
    assert tools._prometheus_adapter.call_count == 1
    assert tools._prometheus_adapter.last_call["promql"] == "up"

    # Loki
    tools.query_loki("{job='api'}", "s", "e")
    assert tools._loki_adapter.call_count == 1
    assert tools._loki_adapter.last_call["logql"] == "{job='api'}"

    # Deployments
    tools.get_recent_deployments("api", "s", "e")
    assert tools._deployment_adapter.call_count == 1
    assert tools._deployment_adapter.last_call["service"] == "api"

    # Related incidents
    tools.get_related_incidents("api", ["timeout"])
    assert tools._related_incidents_adapter.call_count == 1
    assert tools._related_incidents_adapter.last_call["service"] == "api"


def test_fake_transports_can_simulate_errors(config, tools):
    """Test that fake transports can simulate errors."""
    # Set error on prometheus
    tools._prometheus_adapter.set_error(Exception("connection refused"))
    result = tools.query_prometheus("up", "s", "e")
    assert result["status"] == "error"
    assert "connection refused" in result["error"]

    # Set timeout on loki
    tools._loki_adapter.set_timeout()
    result = tools.query_loki("{job='api'}", "s", "e")
    assert result["status"] == "error"
    assert "timeout" in result["error"].lower()
