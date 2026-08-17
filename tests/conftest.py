import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.fake import create_fake_transports
from config import Config
from service.investigator import PamawasInvestigator


@pytest.fixture
def fake_transports():
    """Create fake transports for testing."""
    return create_fake_transports()


@pytest.fixture
def config():
    return Config(
        database_url="",
        llm_api_key="test-key",
        llm_model="test-model",
        max_tool_calls=3,
        truncation_limit=256,
    )


@pytest.fixture
def investigator(config, fake_transports):
    with patch("service.investigator.OpenAI"):
        inv = PamawasInvestigator(config)
        # Replace real adapters with fake transports
        inv.tools._prometheus_adapter = fake_transports["prometheus"]
        inv.tools._loki_adapter = fake_transports["loki"]
        inv.tools._deployment_adapter = fake_transports["deployments"]
        inv.tools._related_incidents_adapter = fake_transports["related_incidents"]
        return inv


@pytest.fixture
def cursor():
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False
    return cursor
