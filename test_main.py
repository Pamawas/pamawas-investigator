import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import the investigator modules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    EvidenceType,
    Finding,
    InvestigationConfig,
    InvestigatorLLM,
    InvestigatorTools,
    PamawasInvestigator,
)


class TestEvidenceType:
    """Test evidence type enum"""

    def test_evidence_types(self):
        assert EvidenceType.FACT.value == "fact"
        assert EvidenceType.LIKELY_CAUSE.value == "likely_cause"
        assert EvidenceType.HYPOTHESIS.value == "hypothesis"
        assert EvidenceType.UNKNOWN.value == "unknown"


class TestFinding:
    """Test Finding dataclass"""

    def test_finding_creation(self):
        finding = Finding(
            type=EvidenceType.FACT,
            content="Test fact",
            source="prometheus",
            confidence=0.95
        )
        assert finding.type == EvidenceType.FACT
        assert finding.content == "Test fact"
        assert finding.source == "prometheus"
        assert finding.confidence == 0.95

    def test_finding_confidence_bounds(self):
        # Test valid bounds
        Finding(type=EvidenceType.FACT, content="Test", source="test", confidence=0.0)
        Finding(type=EvidenceType.FACT, content="Test", source="test", confidence=1.0)


class TestInvestigationConfig:
    """Test configuration loading"""

    def test_config_from_env(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgres://test:test@localhost/test")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("PROMETHEUS_URL", "http://prometheus:9090")
        monkeypatch.setenv("LOKI_URL", "http://loki:3100")
        monkeypatch.setenv("MAX_TOOL_CALLS", "8")

        config = InvestigationConfig.from_env()

        assert config.database_url == "postgres://test:test@localhost/test"
        assert config.llm_base_url == "https://api.openai.com/v1"
        assert config.llm_api_key == "test-key"
        assert config.llm_model == "gpt-4o-mini"
        assert config.prometheus_url == "http://prometheus:9090"
        assert config.loki_url == "http://loki:3100"
        assert config.max_tool_calls == 8

    def test_config_defaults(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgres://test:test@localhost/test")
        monkeypatch.setenv("LLM_API_KEY", "test-key")

        config = InvestigationConfig.from_env()

        assert config.max_tool_calls == 6
        assert config.context_truncation_limit == 8192


class TestInvestigatorLLM:
    """Test LLM client"""

    @patch('main.OpenAI')
    def test_llm_initialization(self, mock_openai):
        config = InvestigationConfig(
            database_url="postgres://test:test@localhost/test",
            llm_base_url="https://api.openai.com/v1",
            llm_api_key="test-key",
            llm_model="gpt-4o-mini",
            prometheus_url="http://prometheus:9090",
            loki_url="http://loki:3100",
            deployments_url="http://deployments:8080",
            related_incidents_url="http://related:8080",
        )

        llm = InvestigatorLLM(config)

        assert llm.model == "gpt-4o-mini"
        mock_openai.assert_called_once_with(
            base_url="https://api.openai.com/v1",
            api_key="test-key"
        )


class TestInvestigatorTools:
    """Test investigator tools"""

    def setup_method(self):
        self.config = InvestigationConfig(
            database_url="postgres://test:test@localhost/test",
            llm_base_url="https://api.openai.com/v1",
            llm_api_key="test-key",
            llm_model="gpt-4o-mini",
            prometheus_url="http://prometheus:9090",
            loki_url="http://loki:3100",
            deployments_url="http://deployments:8080",
            related_incidents_url="http://related:8080",
        )
        self.tools = InvestigatorTools(self.config)

    def test_query_prometheus(self):
        result = self.tools.query_prometheus(
            promql='rate(http_requests_total[5m])',
            start='2026-01-01T00:00:00Z',
            end='2026-01-01T01:00:00Z'
        )

        assert result["status"] == "success"
        assert "data" in result
        assert result["data"]["resultType"] == "matrix"

    def test_query_loki(self):
        result = self.tools.query_loki(
            logql='{job="api-server"} |= "error"',
            start='2026-01-01T00:00:00Z',
            end='2026-01-01T01:00:00Z',
            limit=50
        )

        assert result["status"] == "success"
        assert "data" in result

    def test_get_recent_deployments(self):
        result = self.tools.get_recent_deployments(
            service="payment-api",
            start='2026-01-01T00:00:00Z',
            end='2026-01-01T01:00:00Z'
        )

        assert result["status"] == "success"
        assert "data" in result
        assert len(result["data"]) > 0

    def test_get_related_incidents(self):
        result = self.tools.get_related_incidents(
            service="payment-api",
            symptom_keywords=["latency", "timeout"]
        )

        assert result["status"] == "success"
        assert "data" in result
        assert len(result["data"]) > 0

    def test_submit_findings(self):
        findings = [
            Finding(
                type=EvidenceType.FACT,
                content="CPU usage spiked to 95%",
                source="prometheus",
                confidence=0.9
            ),
            Finding(
                type=EvidenceType.LIKELY_CAUSE,
                content="Deployment caused resource exhaustion",
                source="investigator",
                confidence=0.75
            )
        ]

        result = self.tools.submit_findings(findings)

        assert result["status"] == "success"
        assert "findings" in result
        assert len(result["findings"]) == 2


class TestPamawasInvestigator:
    """Test main investigator orchestrator"""

    def setup_method(self):
        self.config = InvestigationConfig(
            database_url="",  # No DB for unit tests
            llm_base_url="https://api.openai.com/v1",
            llm_api_key="test-key",
            llm_model="gpt-4o-mini",
            prometheus_url="http://prometheus:9090",
            loki_url="http://loki:3100",
            deployments_url="http://deployments:8080",
            related_incidents_url="http://related:8080",
        )

    def test_investigator_initialization(self):
        investigator = PamawasInvestigator(self.config)
        assert investigator.config == self.config
        assert investigator.llm is not None
        assert investigator.tools is not None
        assert investigator.db_conn is None  # No DB URL provided

    def test_truncate_context(self):
        investigator = PamawasInvestigator(self.config)

        # Test no truncation needed
        short_text = "short"
        assert investigator._truncate_context(short_text, 100) == short_text

        # Test truncation
        long_text = "x" * 10000
        truncated = investigator._truncate_context(long_text, 1000)
        assert len(truncated) <= 1000
        assert "TRUNCATED" in truncated

    @patch('main.OpenAI')
    def test_investigate_without_db(self, mock_openai):
        # Mock the LLM response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = "Investigating..."
        mock_response.tool_calls = None
        mock_client.chat.completions.create.return_value.choices = [Mock(message=mock_response)]
        mock_openai.return_value = mock_client

        investigator = PamawasInvestigator(self.config)
        findings = investigator.investigate("test_incident_001")

        # Should return UNKNOWN finding since no DB
        assert len(findings) >= 1
        assert findings[0].type == EvidenceType.UNKNOWN


class TestSystemPrompt:
    """Test system prompt content"""

    def test_system_prompt_contains_key_instructions(self):
        # This is a documentation test - the system prompt is in the investigate method
        # We verify the key instructions are present by checking the source
        import inspect
        source = inspect.getsource(PamawasInvestigator.investigate)

        assert "Understand the symptom and blast radius" in source
        assert "Find the first abnormal signal" in source
        assert "Check recent changes near the time" in source
        assert "Check dependencies" in source
        assert "competing hypotheses" in source
        assert "UNKNOWN over a fabricated-sounding conclusion" in source
        assert "FACT" in source
        assert "LIKELY_CAUSE" in source
        assert "HYPOTHESIS" in source
        assert "UNKNOWN" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])