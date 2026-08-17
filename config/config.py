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
    # Adapter settings
    prometheus_timeout_seconds: float = 10.0
    prometheus_max_response_bytes: int = 1_000_000
    prometheus_max_series: int = 1000
    prometheus_max_samples_per_series: int = 10000
    prometheus_allowed_time_range_hours: int = 72
    loki_timeout_seconds: float = 10.0
    loki_max_response_bytes: int = 1_000_000
    loki_max_log_lines: int = 10000
    loki_max_log_line_length: int = 10000
    loki_allowed_time_range_hours: int = 72
    deployment_url: str = ""
    deployment_api_key: str = ""
    deployment_timeout_seconds: float = 10.0
    deployment_max_results: int = 50
    related_incidents_max_results: int = 10
    related_incidents_max_symptom_keywords: int = 20

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
            # Adapter settings from env
            prometheus_timeout_seconds=float(
                os.getenv("PROMETHEUS_TIMEOUT_SECONDS", "10.0")
            ),
            prometheus_max_response_bytes=int(
                os.getenv("PROMETHEUS_MAX_RESPONSE_BYTES", "1000000")
            ),
            prometheus_max_series=int(os.getenv("PROMETHEUS_MAX_SERIES", "1000")),
            prometheus_max_samples_per_series=int(
                os.getenv("PROMETHEUS_MAX_SAMPLES_PER_SERIES", "10000")
            ),
            prometheus_allowed_time_range_hours=int(
                os.getenv("PROMETHEUS_ALLOWED_TIME_RANGE_HOURS", "72")
            ),
            loki_timeout_seconds=float(os.getenv("LOKI_TIMEOUT_SECONDS", "10.0")),
            loki_max_response_bytes=int(
                os.getenv("LOKI_MAX_RESPONSE_BYTES", "1000000")
            ),
            loki_max_log_lines=int(os.getenv("LOKI_MAX_LOG_LINES", "10000")),
            loki_max_log_line_length=int(
                os.getenv("LOKI_MAX_LOG_LINE_LENGTH", "10000")
            ),
            loki_allowed_time_range_hours=int(
                os.getenv("LOKI_ALLOWED_TIME_RANGE_HOURS", "72")
            ),
            deployment_url=os.getenv("DEPLOYMENT_URL", ""),
            deployment_api_key=os.getenv("DEPLOYMENT_API_KEY", ""),
            deployment_timeout_seconds=float(
                os.getenv("DEPLOYMENT_TIMEOUT_SECONDS", "10.0")
            ),
            deployment_max_results=int(os.getenv("DEPLOYMENT_MAX_RESULTS", "50")),
            related_incidents_max_results=int(
                os.getenv("RELATED_INCIDENTS_MAX_RESULTS", "10")
            ),
            related_incidents_max_symptom_keywords=int(
                os.getenv("RELATED_INCIDENTS_MAX_SYMPTOM_KEYWORDS", "20")
            ),
        )
