from unittest.mock import patch

import pytest

from models import EvidenceType, Finding
from tools import InvestigatorTools, ToolRegistry


@pytest.mark.parametrize(
    ("method", "arguments", "metric"),
    [
        ("query_prometheus", ("up", "start", "end"), "prometheus"),
        ("query_loki", ("{job='api'}", "start", "end"), "loki"),
        ("get_recent_deployments", ("api", "start", "end"), "deployments"),
        ("get_related_incidents", ("api", ["timeout"]), "related_incidents"),
    ],
)
def test_tools_return_success_and_record_metrics(config, method, arguments, metric):
    tools = InvestigatorTools(config)
    with (
        patch("tools.tools.observe_tool_call_duration") as observe,
        patch("tools.tools.increment_tool_calls") as increment,
        patch("tools.tools.time.time", side_effect=[10.0, 10.25]),
    ):
        result = getattr(tools, method)(*arguments)
    assert result["status"] == "success"
    observe.assert_called_once_with(metric, 0.25)
    increment.assert_called_once_with(metric, "success")


def test_submit_findings_serializes_findings_and_records_metrics(config):
    finding = Finding(EvidenceType.FACT, "latency rose", "prometheus", 1.0)
    with patch("tools.tools.increment_tool_calls") as increment:
        result = InvestigatorTools(config).submit_findings([finding])
    assert result["findings"] == [finding.to_dict()]
    increment.assert_called_once_with("submit_findings", "success")


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
