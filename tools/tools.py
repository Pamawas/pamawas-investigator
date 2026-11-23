"""Tool implementations for the Investigator - using real bounded adapters."""

import logging
import time
from typing import Any

from adapters import (
    DeploymentAdapter,
    DeploymentConfig,
    LokiAdapter,
    LokiConfig,
    LokiError,
    PrometheusAdapter,
    PrometheusConfig,
    PrometheusError,
    RelatedIncidentsAdapter,
    RelatedIncidentsConfig,
    RelatedIncidentsError,
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
                circuit_breaker_failure_threshold=config.circuit_breaker_failure_threshold,
                circuit_breaker_timeout_seconds=config.circuit_breaker_timeout_seconds,
                circuit_breaker_half_open_max_calls=config.circuit_breaker_half_open_max_calls,
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
                circuit_breaker_failure_threshold=config.circuit_breaker_failure_threshold,
                circuit_breaker_timeout_seconds=config.circuit_breaker_timeout_seconds,
                circuit_breaker_half_open_max_calls=config.circuit_breaker_half_open_max_calls,
            )
        )

        self._deployment_adapter = DeploymentAdapter(
            DeploymentConfig(
                base_url=config.deployment_url or None,
                timeout_seconds=config.deployment_timeout_seconds,
                max_results=config.deployment_max_results,
                api_key=config.deployment_api_key or None,
                circuit_breaker_failure_threshold=config.circuit_breaker_failure_threshold,
                circuit_breaker_timeout_seconds=config.circuit_breaker_timeout_seconds,
                circuit_breaker_half_open_max_calls=config.circuit_breaker_half_open_max_calls,
            )
        )

        self._related_incidents_adapter = RelatedIncidentsAdapter(
            RelatedIncidentsConfig(
                database_url=config.database_url,
                max_results=config.related_incidents_max_results,
                max_symptom_keywords=config.related_incidents_max_symptom_keywords,
                circuit_breaker_failure_threshold=config.circuit_breaker_failure_threshold,
                circuit_breaker_timeout_seconds=config.circuit_breaker_timeout_seconds,
                circuit_breaker_half_open_max_calls=config.circuit_breaker_half_open_max_calls,
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
            if result.get("status") == "unavailable":
                increment_tool_calls("prometheus", "circuit_open")
                return result
            increment_tool_calls("prometheus", "success")
            return result

        except (PrometheusError, TimeoutError, RuntimeError, OSError) as e:
            duration = time.time() - start_time
            observe_tool_call_duration("prometheus", duration)
            increment_tool_calls("prometheus", "error")
            logger.error("Prometheus query failed: %s", e)
            return {"status": "error", "error": str(e)}

    def get_prometheus_labels(self, label: str = "") -> dict[str, Any]:
        """List Prometheus label names, or the values of one label.

        Without a label this returns every label name in the Prometheus
        instance; with a label it returns that label's values. The
        investigation uses this to build selectors against the real schema, so
        it keeps working when Prometheus is swapped for one whose labels are
        named differently.
        """
        start_time = time.time()
        try:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if label:
                result = loop.run_until_complete(self._prometheus_adapter.query_label_values(label))
            else:
                result = loop.run_until_complete(self._prometheus_adapter.query_labels())

            duration = time.time() - start_time
            observe_tool_call_duration("prometheus", duration)
            if result.get("status") == "unavailable":
                increment_tool_calls("prometheus", "circuit_open")
                return result
            increment_tool_calls("prometheus", "success")
            return result

        except (PrometheusError, TimeoutError, RuntimeError, OSError) as e:
            duration = time.time() - start_time
            observe_tool_call_duration("prometheus", duration)
            increment_tool_calls("prometheus", "error")
            logger.error("Prometheus label query failed: %s", e)
            return {"status": "error", "error": str(e)}

    def get_loki_labels(self, label: str = "") -> dict[str, Any]:
        """List Loki label names, or the values of one label.

        Without a label this returns every label name in the Loki instance.
        With a label it returns that label's values. The investigation uses
        this to build selectors against the real schema: a selector naming a
        label or value that does not exist still returns `status: success`
        with zero streams, which is indistinguishable from "no logs".
        """
        logger.info("Querying Loki labels (label=%r)", label)
        start_time = time.time()

        try:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if label:
                result = loop.run_until_complete(self._loki_adapter.query_label_values(label))
            else:
                result = loop.run_until_complete(self._loki_adapter.query_labels())

            duration = time.time() - start_time
            observe_tool_call_duration("loki", duration)
            if result.get("status") == "unavailable":
                increment_tool_calls("loki", "circuit_open")
                return result
            increment_tool_calls("loki", "success")
            return result

        except (LokiError, TimeoutError, RuntimeError, OSError) as e:
            duration = time.time() - start_time
            observe_tool_call_duration("loki", duration)
            increment_tool_calls("loki", "error")
            logger.error("Loki labels query failed: %s", e)
            return {"status": "error", "error": str(e)}

    def query_loki(self, logql: str, start: str, end: str, limit: int = 100) -> dict[str, Any]:
        """Query Loki for logs data (sync wrapper for async adapter)."""
        logger.info("Querying Loki: %s [%s to %s] limit=%d", logql, start, end, limit)
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
            if result.get("status") == "unavailable":
                increment_tool_calls("loki", "circuit_open")
                return result
            increment_tool_calls("loki", "success")
            return result

        except (LokiError, TimeoutError, RuntimeError, OSError) as e:
            duration = time.time() - start_time
            observe_tool_call_duration("loki", duration)
            increment_tool_calls("loki", "error")
            logger.error("Loki query failed: %s", e)
            return {"status": "error", "error": str(e)}

    def get_related_incidents(self, service: str, symptom_keywords: list[str]) -> dict[str, Any]:
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

        except (RelatedIncidentsError, TimeoutError, RuntimeError, OSError) as e:
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
        except (ValueError, TypeError, KeyError) as e:
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
                            "promql": {"type": "string", "description": "PromQL query expression"},
                            "start": {
                                "type": "string",
                                "description": "Start timestamp (ISO 8601/RFC3339)",
                            },
                            "end": {
                                "type": "string",
                                "description": "End timestamp (ISO 8601/RFC3339)",
                            },
                        },
                        "required": ["promql", "start", "end"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_prometheus_labels",
                    "description": (
                        "List the Prometheus label schema: with no argument, all "
                        "label names; with a label name, that label's values. Call "
                        "this before writing a PromQL selector when unsure which "
                        "labels or values exist, or when this deployment's labels "
                        "are unknown — a selector naming a label that does not "
                        "exist returns an empty result, which looks the same as "
                        "'no data'. Discovering the schema first is what lets the "
                        "investigation work against any Prometheus."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": (
                                    "Optional label name whose values to list. "
                                    "Omit to list all label names."
                                ),
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_loki_labels",
                    "description": (
                        "List the Loki label schema: with no argument, all label "
                        "names; with a label name, that label's values. Call this "
                        "before writing a LogQL selector when unsure which labels "
                        "or values exist — a selector naming a label or value that "
                        "does not exist returns zero streams, which looks the same "
                        "as 'no logs'."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": (
                                    "Optional label name returned by the catalog whose "
                                    "values to list. Omit to list all label names."
                                ),
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_loki",
                    "description": "Query Loki for logs data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "logql": {"type": "string", "description": "LogQL query expression"},
                            "start": {
                                "type": "string",
                                "description": "Start timestamp (ISO 8601/RFC3339)",
                            },
                            "end": {
                                "type": "string",
                                "description": "End timestamp (ISO 8601/RFC3339)",
                            },
                            "limit": {
                                "type": "integer",
                                "description": ("Maximum number of log entries to return"),
                            },
                        },
                        "required": ["logql", "start", "end"],
                    },
                },
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
                                "description": (
                                    "Affected target or service identifier from the incident"
                                ),
                            },
                            "symptom_keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": ("Keywords to match against incident symptoms"),
                            },
                        },
                        "required": ["service", "symptom_keywords"],
                    },
                },
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
                                                "unknown",
                                                "recommendation",
                                            ],
                                        },
                                        "content": {
                                            "type": "string",
                                            "description": (
                                                "Concrete factual finding or conclusion "
                                                "observed during investigation. Must be "
                                                "substantive text based on real evidence; "
                                                "never use placeholder text like "
                                                "'placeholder', 'TBD', 'TODO', or 'none'."
                                            ),
                                        },
                                        "source": {
                                            "type": "string",
                                            "enum": [
                                                "prometheus",
                                                "loki",
                                                "incidents",
                                                "llm",
                                                "system",
                                            ],
                                            "description": (
                                                "Where this came from: prometheus (metrics), "
                                                "loki (logs), incidents (new related-history "
                                                "or post-trigger lifecycle evidence, never "
                                                "the original alert metadata), "
                                                "llm (your own synthesis across sources), "
                                                "system (a harness limit such as a timeout)"
                                            ),
                                        },
                                        "source_detail": {
                                            "type": "string",
                                            "description": (
                                                "Exact executed query or exact incident/event "
                                                "identifier supporting this finding. For a "
                                                "recommendation, identify the finding or "
                                                "unknown that motivates the action."
                                            ),
                                        },
                                        "confidence": {
                                            "type": "number",
                                            "minimum": 0.0,
                                            "maximum": 1.0,
                                        },
                                    },
                                    "required": [
                                        "type",
                                        "content",
                                        "source",
                                        "source_detail",
                                        "confidence",
                                    ],
                                },
                            }
                        },
                        "required": ["findings"],
                    },
                },
            },
        ]

    @classmethod
    def get_tools(cls, verification: bool = False):
        tools = cls().TOOLS
        if verification:
            return [
                t
                for t in tools
                if t.get("function", {}).get("name") != "get_related_incidents"
            ]
        return tools

    @classmethod
    def get_tool_names(cls, verification: bool = False):
        return [t["function"]["name"] for t in cls.get_tools(verification=verification)]
