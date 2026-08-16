import pytest

from config import Config


def test_from_env_reads_all_values(monkeypatch):
    values = {
        "DATABASE_URL": "postgres://db",
        "PORT": "9090",
        "LLM_BASE_URL": "http://llm/v1",
        "LLM_API_KEY": "secret",
        "LLM_MODEL": "model",
        "PROMETHEUS_URL": "http://prom",
        "LOKI_URL": "http://loki",
        "MAX_TOOL_CALLS": "9",
        "TRUNCATION_LIMIT": "1024",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    config = Config.from_env()

    assert config == Config(
        database_url="postgres://db",
        port=9090,
        llm_base_url="http://llm/v1",
        llm_api_key="secret",
        llm_model="model",
        prometheus_url="http://prom",
        loki_url="http://loki",
        max_tool_calls=9,
        truncation_limit=1024,
    )


def test_from_env_uses_defaults(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://db")
    for key in ["PORT", "LLM_BASE_URL", "LLM_MODEL", "MAX_TOOL_CALLS", "TRUNCATION_LIMIT"]:
        monkeypatch.delenv(key, raising=False)

    config = Config.from_env()

    assert config.port == 8080
    assert config.llm_model == "gpt-4o-mini"
    assert config.max_tool_calls == 6
    assert config.truncation_limit == 8192


def test_from_env_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="DATABASE_URL environment variable not set"):
        Config.from_env()


def test_from_env_rejects_non_numeric_port(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://db")
    monkeypatch.setenv("PORT", "invalid")
    with pytest.raises(ValueError):
        Config.from_env()
