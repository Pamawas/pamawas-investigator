"""Configuration for the Pamawas Investigator."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _parse_env_list(value: str | None) -> list[str] | None:
    """Parse a comma-separated env var into a list of non-empty stripped strings."""
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class Config:
    """Configuration for the investigator service."""

    database_url: str
    port: int = 8080
    # Primary LLM endpoint (kept for backward compatibility). llm_base_urls takes
    # precedence when set; this is the single-endpoint fallback.
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    # Redundant LLM endpoints tried in order on transient failure. Parsed from
    # LLM_BASE_URLS / LLM_API_KEYS / LLM_MODELS as comma-separated lists. When
    # unset, falls back to the single llm_base_url/llm_api_key/llm_model above.
    llm_base_urls: list[str] | None = None
    llm_api_keys: list[str] | None = None
    llm_models: list[str] | None = None
    prometheus_url: str = "http://prometheus:9090"
    loki_url: str = "http://loki:3100"
    max_tool_calls: int = 100
    truncation_limit: int = 8192
    # Query templates for evidence enrichment
    prometheus_query_template: str = '{incident="%s"}'
    loki_query_template: str = '{incident="%s"}'
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
    # Circuit breaker settings for external adapters
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: float = 30.0
    circuit_breaker_half_open_max_calls: int = 3
    # Overall investigation deadline (wall clock, seconds)
    investigation_timeout_seconds: float = 600.0
    # Auto-retry budget for a single investigation run: how many times it may be
    # (re)executed total before going failed_terminal (default: initial + 4).
    max_investigation_attempts: int = 5
    # Base back-off in seconds between investigation retries (grows exponentially,
    # capped by the retry scheduler).
    investigation_retry_backoff_seconds: int = 60
    # Wake cadence of the background retry scheduler (seconds).
    retry_scheduler_tick_seconds: float = 15.0

    @classmethod
    def from_env(cls) -> Config:
        """Create config from environment variables."""
        database_url = os.getenv("DATABASE_URL", "")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")

        # Kubernetes injects PAMAWAS_INVESTIGATOR_PORT=tcp://IP:PORT, avoid it
        port_str = os.getenv("PORT", "8080")
        # Strip any tcp:// prefix if injected by Kubernetes
        if port_str.startswith("tcp://"):
            port_str = port_str.split(":")[-1]

        cfg = cls(
            database_url=database_url,
            port=int(port_str),
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            llm_base_urls=_parse_env_list(os.getenv("LLM_BASE_URLS")),
            llm_api_keys=_parse_env_list(os.getenv("LLM_API_KEYS")),
            llm_models=_parse_env_list(os.getenv("LLM_MODELS")),
            prometheus_url=os.getenv("PROMETHEUS_URL", "http://prometheus:9090"),
            loki_url=os.getenv("LOKI_URL", "http://loki:3100"),
            prometheus_query_template=os.getenv("PROMETHEUS_QUERY_TEMPLATE", '{incident="%s"}'),
            loki_query_template=os.getenv("LOKI_QUERY_TEMPLATE", '{incident="%s"}'),
            max_tool_calls=int(os.getenv("MAX_TOOL_CALLS", "100")),
            truncation_limit=int(os.getenv("TRUNCATION_LIMIT", "8192")),
            # Adapter settings from env
            prometheus_timeout_seconds=float(os.getenv("PROMETHEUS_TIMEOUT_SECONDS", "10.0")),
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
            loki_max_response_bytes=int(os.getenv("LOKI_MAX_RESPONSE_BYTES", "1000000")),
            loki_max_log_lines=int(os.getenv("LOKI_MAX_LOG_LINES", "10000")),
            loki_max_log_line_length=int(os.getenv("LOKI_MAX_LOG_LINE_LENGTH", "10000")),
            loki_allowed_time_range_hours=int(os.getenv("LOKI_ALLOWED_TIME_RANGE_HOURS", "72")),
            deployment_url=os.getenv("DEPLOYMENT_URL", ""),
            deployment_api_key=os.getenv("DEPLOYMENT_API_KEY", ""),
            deployment_timeout_seconds=float(os.getenv("DEPLOYMENT_TIMEOUT_SECONDS", "10.0")),
            deployment_max_results=int(os.getenv("DEPLOYMENT_MAX_RESULTS", "50")),
            related_incidents_max_results=int(os.getenv("RELATED_INCIDENTS_MAX_RESULTS", "10")),
            related_incidents_max_symptom_keywords=int(
                os.getenv("RELATED_INCIDENTS_MAX_SYMPTOM_KEYWORDS", "20")
            ),
            # Circuit breaker settings
            circuit_breaker_failure_threshold=int(
                os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")
            ),
            circuit_breaker_timeout_seconds=float(
                os.getenv("CIRCUIT_BREAKER_TIMEOUT_SECONDS", "30.0")
            ),
            circuit_breaker_half_open_max_calls=int(
                os.getenv("CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS", "3")
            ),
            investigation_timeout_seconds=float(
                os.getenv("INVESTIGATION_TIMEOUT_SECONDS", "600.0")
            ),
            max_investigation_attempts=int(
                os.getenv("MAX_INVESTIGATION_ATTEMPTS", "5")
            ),
            investigation_retry_backoff_seconds=int(
                os.getenv("INVESTIGATION_RETRY_BACKOFF_SECONDS", "60")
            ),
            retry_scheduler_tick_seconds=float(
                os.getenv("RETRY_SCHEDULER_TICK_SECONDS", "15")
            ),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Fail fast on out-of-range configuration values.

        Called by from_env after parsing so a bad env var kills startup with a
        clear message instead of degrading behaviour at runtime.
        """
        # (field, env var, minimum, inclusive) — positive-duration fields use
        # a minimum just above zero.
        bounds = [
            ("port", "PORT", 1),
            ("max_tool_calls", "MAX_TOOL_CALLS", 1),
            ("truncation_limit", "TRUNCATION_LIMIT", 1),
            ("max_investigation_attempts", "MAX_INVESTIGATION_ATTEMPTS", 1),
            ("investigation_retry_backoff_seconds", "INVESTIGATION_RETRY_BACKOFF_SECONDS", 1),
            ("retry_scheduler_tick_seconds", "RETRY_SCHEDULER_TICK_SECONDS", 1),
            ("circuit_breaker_failure_threshold", "CIRCUIT_BREAKER_FAILURE_THRESHOLD", 1),
            ("circuit_breaker_half_open_max_calls", "CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS", 1),
            ("prometheus_allowed_time_range_hours", "PROMETHEUS_ALLOWED_TIME_RANGE_HOURS", 1),
            ("loki_allowed_time_range_hours", "LOKI_ALLOWED_TIME_RANGE_HOURS", 1),
            ("deployment_max_results", "DEPLOYMENT_MAX_RESULTS", 1),
            ("related_incidents_max_results", "RELATED_INCIDENTS_MAX_RESULTS", 1),
        ]
        for field, env_var, minimum in bounds:
            value = getattr(self, field)
            if value < minimum:
                raise ValueError(f"{env_var} must be >= {minimum}, got {value}")

        for field, env_var in [
            ("investigation_timeout_seconds", "INVESTIGATION_TIMEOUT_SECONDS"),
            ("circuit_breaker_timeout_seconds", "CIRCUIT_BREAKER_TIMEOUT_SECONDS"),
            ("prometheus_timeout_seconds", "PROMETHEUS_TIMEOUT_SECONDS"),
            ("loki_timeout_seconds", "LOKI_TIMEOUT_SECONDS"),
            ("deployment_timeout_seconds", "DEPLOYMENT_TIMEOUT_SECONDS"),
        ]:
            if getattr(self, field) <= 0:
                raise ValueError(f"{env_var} must be > 0, got {getattr(self, field)}")

        if self.port > 65535:
            raise ValueError(f"PORT must be <= 65535, got {self.port}")
