"""Tool implementations for the Investigator - using real bounded adapters."""

import logging
import time
from typing import Any

from adapters import (
    DeploymentAdapter,
    DeploymentConfig,
    LokiAdapter,
    LokiConfig,
    PrometheusAdapter,
    PrometheusConfig,
    RelatedIncidentsAdapter,
    RelatedIncidentsConfig,
)
from config import Config as InvestigationConfig
from metrics import increment_tool_calls, observe_tool_call_duration
from models import Finding

logger = logging.getLogger(__name__)


class InvestigatorTools:
    """Tool implementations that the LLM can call - using real adapters."""

    def __init__(self, config: InvestigationConfig):
        self.config = config

        # Initialize real adapters
        self._prometheus_adapter = PrometheusAdapter(
            PrometheusConfig(
                base_url=config.prometheus_url,
                timeout_seconds=config.prometheus_timeout_seconds,
                max_response_bytes=config.prometheus_max_response_bytes,
                max_series=config.prometheus_max_series,
                max_samples_per_series=config.prometheus_max_samples_per_series,
                allowed_time_range_hours=config.prometheus_allowed_time_range_hours,
            )
        )

        self._loki_adapter = LokiAdapter(
            LokiConfig(
                base_url=config.loki_url,
                timeout_seconds=config.loki_timeout_seconds,
                max_response_bytes=config.loki_max_response_bytes,
                max_log_lines=config.loki_max_log_lines,
                max_log_line_length=config.loki_max_log_line_length,
                allowed_time_range_hours=config.loki_allowed_time_range_hours,
            )
        )

        self._deployment_adapter = DeploymentAdapter(
            DeploymentConfig(
                base_url=config.deployment_url or None,
                timeout_seconds=config.deployment_timeout_seconds,
                max_results=config.deployment_max_results,
                api_key=config.deployment_api_key or None,
            )
        )

        self._related_incidents_adapter = RelatedIncidentsAdapter(
            RelatedIncidentsConfig(
                database_url=config.database_url,
                max_results=config.related_incidents_max_results,
                max_symptom_keywords=config.related_incidents_max_symptom_keywords,
            )
        )

    async def close(self):
        """Close all adapter connections."""
        await self._prometheus_adapter.close()
        await self._loki_adapter.close()
        await self._deployment_adapter.close()
        self._related_incidents_adapter.close()

    def query_prometheus(self, promql: str, start: str, end: str) -> dict[str, Any]:
        """Query Prometheus for metrics data (sync wrapper for async adapter)."""
        logger.info("Querying Prometheus: %s [%s to %s]", promql, start, end)
        start_time = time.time()

        try:
            import asyncio
            # Run async adapter in sync context
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                self._prometheus_adapter.query_range(promql, start, end)
            )

            duration = time.time() - start_time
            observe_tool_call_duration("prometheus", duration)
            increment_tool_calls("prometheus", "success")
            return result

        except Exception as e:  # noqa: BLE001
            duration = time.time() - start_time
            observe_tool_call_duration("prometheus", duration)
            increment_tool_calls("prometheus", "error")
            logger.error("Prometheus query failed: %s", e)
            return {"status": "error", "error": str(e)}

    def query_loki(
        self, logql: str, start: str, end: str, limit: int = 100
    ) -> dict[str, Any]:
        """Query Loki for logs data (sync wrapper for async adapter)."""
        logger.info(
            "Querying Loki: %s [%s to %s] limit=%d", logql, start, end, limit
        )
        start_time = time.time()

        try:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                self._loki_adapter.query_range(logql, start, end, limit)
            )

            duration = time.time() - start_time
            observe_tool_call_duration("loki", duration)
            increment_tool_calls("loki", "success")
            return result

        except Exception as e:  # noqa: BLE001
            duration = time.time() - start_time
            observe_tool_call_duration("loki", duration)
            increment_tool_calls("loki", "error")
            logger.error("Loki query failed: %s", e)
            return {"status": "error", "error": str(e)}

    def get_recent_deployments(
        self, service: str, start: str, end: str
    ) -> dict[str, Any]:
        """Get recent deployments for a service (sync wrapper for async adapter)."""
        logger.info(
            "Getting recent deployments for %s [%s to %s]", service, start, end
        )
        start_time = time.time()

        try:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                self._deployment_adapter.get_recent_deployments(service, start, end)
            )

            duration = time.time() - start_time
            observe_tool_call_duration("deployments", duration)
            increment_tool_calls(
                "deployments", "success" if result.get("status") == "success" else "error"
            )
            return result

        except Exception as e:  # noqa: BLE001
            duration = time.time() - start_time
            observe_tool_call_duration("deployments", duration)
            increment_tool_calls("deployments", "error")
            logger.error("Get deployments failed: %s", e)
            return {"status": "error", "error": str(e)}

    def get_related_incidents(
        self, service: str, symptom_keywords: list[str]
    ) -> dict[str, Any]:
        """Find related incidents from the database (sync wrapper for async adapter)."""
        logger.info(
            "Finding related incidents for %s with keywords %s",
            service,
            symptom_keywords,
        )
        start_time = time.time()

        try:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                self._related_incidents_adapter.find_related(service, symptom_keywords)
            )

            duration = time.time() - start_time
            observe_tool_call_duration("related_incidents", duration)
            increment_tool_calls("related_incidents", "success")
            return result

        except Exception as e:  # noqa: BLE001
            duration = time.time() - start_time
            observe_tool_call_duration("related_incidents", duration)
            increment_tool_calls("related_incidents", "error")
            logger.error("Get related incidents failed: %s", e)
            return {"status": "error", "error": str(e)}

    def submit_findings(self, findings: list[Finding]) -> dict[str, Any]:
        """Submit the final findings - this is the tool that forces structured output."""
        logger.info("Submitting %d findings", len(findings))
        start_time = time.time()

        try:
            # In a real implementation, this would persist findings to the database
            result = {
                "status": "success",
                "message": f"Submitted {len(findings)} findings",
                "findings": [f.to_dict() for f in findings],
            }
            duration = time.time() - start_time
            observe_tool_call_duration("submit_findings", duration)
            increment_tool_calls("submit_findings", "success")
            return result
        except Exception as e:  # noqa: BLE001
            duration = time.time() - start_time
            observe_tool_call_duration("submit_findings", duration)
            increment_tool_calls("submit_findings", "error")
            logger.error("Submit findings failed: %s", e)
            return {"status": "error", "error": str(e)}


class ToolRegistry:
    """Registry of available tools for the LLM."""

    def __init__(self):
        self.TOOLS = [
            {
                "type": "function",
                "function": {
                    "name": "query_prometheus",
                    "description": "Query Prometheus for metrics data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "promql": {
                                "type": "string",
                                "description": "PromQL query expression"
                            },
                            "start": {
                                "type": "string",
                                "description": "Start timestamp (ISO 8601/RFC3339)"
                            },
                            "end": {
                                "type": "string",
                                "description": "End timestamp (ISO 8601/RFC3339)"
                            }
                        },
                        "required": ["promql", "start", "end"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_loki",
                    "description": "Query Loki for logs data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "logql": {
                                "type": "string",
                                "description": "LogQL query expression"
                            },
                            "start": {
                                "type": "string",
                                "description": "Start timestamp (ISO 8601/RFC3339)"
                            },
                            "end": {
                                "type": "string",
                                "description": "End timestamp (ISO 8601/RFC3339)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": (
                                    "Maximum number of log entries to return"
                                )
                            }
                        },
                        "required": ["logql", "start", "end"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_recent_deployments",
                    "description": "Get recent deployments for a service",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service": {
                                "type": "string",
                                "description": "Service name"
                            },
                            "start": {
                                "type": "string",
                                "description": "Start timestamp (ISO 8601/RFC3339)"
                            },
                            "end": {
                                "type": "string",
                                "description": "End timestamp (ISO 8601/RFC3339)"
                            }
                        },
                        "required": ["service", "start", "end"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_related_incidents",
                    "description": "Find related incidents from the database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service": {
                                "type": "string",
                                "description": "Service name"
                            },
                            "symptom_keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Keywords to match against incident symptoms"
                                )
                            }
                        },
                        "required": ["service", "symptom_keywords"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "submit_findings",
                    "description": "Submit final findings and end the investigation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "findings": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "enum": [
                                                "fact",
                                                "likely_cause",
                                                "hypothesis",
                                                "unknown"
                                            ]
                                        },
                                        "content": {"type": "string"},
                                        "source": {"type": "string"},
                                        "confidence": {
                                            "type": "number",
                                            "minimum": 0.0,
                                            "maximum": 1.0
                                        }
                                    },
                                    "required": [
                                        "type", "content", "source", "confidence"
                                    ]
                                }
                            }
                        },
                        "required": ["findings"]
                    }
                }
            }
        ]

    @classmethod
    def get_tools(cls):
        return cls().TOOLS

    @classmethod
    def get_tool_names(cls):
        return [t["function"]["name"] for t in cls().TOOLS]
