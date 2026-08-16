"""Tool implementations for the Investigator."""

import logging
import time
from typing import Any

from metrics import increment_tool_calls, observe_tool_call_duration
from models import Finding

logger = logging.getLogger(__name__)


class InvestigatorTools:
    """Tool implementations that the LLM can call."""

    def __init__(self, config):
        self.config = config

    def query_prometheus(self, promql: str, start: str, end: str) -> dict[str, Any]:
        """Query Prometheus for metrics data."""
        logger.info("Querying Prometheus: %s [%s to %s]", promql, start, end)
        start_time = time.time()

        try:
            # In a real implementation, this would make an HTTP request to Prometheus
            # For now, return mock data
            result = {
                "status": "success",
                "data": {
                    "resultType": "matrix",
                    "result": [
                        {
                            "metric": {
                                "__name__": "http_requests_total",
                                "job": "api-server"
                            },
                            "values": [[start, "100"], [end, "150"]]
                        }
                    ]
                }
            }
            observe_tool_call_duration("prometheus", time.time() - start_time)
            increment_tool_calls("prometheus", "success")
            return result
        except Exception as e:  # noqa: BLE001
            observe_tool_call_duration("prometheus", time.time() - start_time)
            increment_tool_calls("prometheus", "error")
            logger.error("Prometheus query failed: %s", e)
            return {"status": "error", "error": str(e)}

    def query_loki(
        self, logql: str, start: str, end: str, limit: int = 100
    ) -> dict[str, Any]:
        """Query Loki for logs data."""
        logger.info(
            "Querying Loki: %s [%s to %s] limit=%d", logql, start, end, limit
        )
        start_time = time.time()

        try:
            # In a real implementation, this would make an HTTP request to Loki
            result = {
                "status": "success",
                "data": {
                    "result": [
                        {
                            "stream": {
                                "job": "api-server",
                                "level": "error"
                            },
                            "values": [
                                [start, "Error: connection refused"],
                                [end, "Error: timeout"]
                            ]
                        }
                    ]
                }
            }
            observe_tool_call_duration("loki", time.time() - start_time)
            increment_tool_calls("loki", "success")
            return result
        except Exception as e:  # noqa: BLE001
            observe_tool_call_duration("loki", time.time() - start_time)
            increment_tool_calls("loki", "error")
            logger.error("Loki query failed: %s", e)
            return {"status": "error", "error": str(e)}

    def get_recent_deployments(
        self, service: str, start: str, end: str
    ) -> dict[str, Any]:
        """Get recent deployments for a service."""
        logger.info(
            "Getting recent deployments for %s [%s to %s]", service, start, end
        )
        start_time = time.time()

        try:
            # Mock data - in reality would query deployment service
            result = {
                "status": "success",
                "data": [
                    {
                        "service": service,
                        "version": "v1.2.3",
                        "deployed_at": start,
                        "image": f"{service}:v1.2.3"
                    }
                ]
            }
            observe_tool_call_duration("deployments", time.time() - start_time)
            increment_tool_calls("deployments", "success")
            return result
        except Exception as e:  # noqa: BLE001
            observe_tool_call_duration("deployments", time.time() - start_time)
            increment_tool_calls("deployments", "error")
            logger.error("Get deployments failed: %s", e)
            return {"status": "error", "error": str(e)}

    def get_related_incidents(
        self, service: str, symptom_keywords: list[str]
    ) -> dict[str, Any]:
        """Find related incidents from the database."""
        logger.info(
            "Finding related incidents for %s with keywords %s",
            service,
            symptom_keywords
        )
        start_time = time.time()

        try:
            # Mock data - in reality would query database
            result = {
                "status": "success",
                "data": [
                    {
                        "incident_id": "inc_123",
                        "title": f"High latency in {service}",
                        "started_at": "2026-08-13T02:00:00Z",
                        "resolved_at": "2026-08-13T03:30:00Z",
                        "symptom_keywords": symptom_keywords
                    }
                ]
            }
            observe_tool_call_duration(
                "related_incidents", time.time() - start_time
            )
            increment_tool_calls("related_incidents", "success")
            return result
        except Exception as e:  # noqa: BLE001
            observe_tool_call_duration(
                "related_incidents", time.time() - start_time
            )
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
                "findings": [f.to_dict() for f in findings]
            }
            observe_tool_call_duration(
                "submit_findings", time.time() - start_time
            )
            increment_tool_calls("submit_findings", "success")
            return result
        except Exception as e:  # noqa: BLE001
            observe_tool_call_duration(
                "submit_findings", time.time() - start_time
            )
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
                                "description": "Start timestamp (ISO 8601)"
                            },
                            "end": {
                                "type": "string",
                                "description": "End timestamp (ISO 8601)"
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
                                "description": "Start timestamp (ISO 8601)"
                            },
                            "end": {
                                "type": "string",
                                "description": "End timestamp (ISO 8601)"
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
                                "description": "Start timestamp (ISO 8601)"
                            },
                            "end": {
                                "type": "string",
                                "description": "End timestamp (ISO 8601)"
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
