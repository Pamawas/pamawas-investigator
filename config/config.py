"""Configuration for the Pamawas Investigator."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for the investigator service."""
    database_url: str
    port: int = 8080
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    prometheus_url: str = "http://prometheus:9090"
    loki_url: str = "http://loki:3100"
    max_tool_calls: int = 6
    truncation_limit: int = 8192

    @classmethod
    def from_env(cls) -> Config:
        """Create config from environment variables."""
        database_url = os.getenv("DATABASE_URL", "")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")

        return cls(
            database_url=database_url,
            port=int(os.getenv("PORT", "8080")),
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            prometheus_url=os.getenv("PROMETHEUS_URL", "http://prometheus:9090"),
            loki_url=os.getenv("LOKI_URL", "http://loki:3100"),
            max_tool_calls=int(os.getenv("MAX_TOOL_CALLS", "6")),
            truncation_limit=int(os.getenv("TRUNCATION_LIMIT", "8192")),
        )
