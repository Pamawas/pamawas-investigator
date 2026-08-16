import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from service.investigator import PamawasInvestigator


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
def investigator(config):
    with patch("service.investigator.OpenAI"):
        return PamawasInvestigator(config)


@pytest.fixture
def cursor():
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False
    return cursor
