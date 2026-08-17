"""Fake transport interfaces for deterministic tests.

These implementations are for TESTING ONLY and must not be used
on production paths. They provide deterministic responses without
requiring real external services.
"""

import time
from dataclasses import dataclass
from typing import Any

from metrics import increment_tool_calls, observe_tool_call_duration


@dataclass
class FakePrometheusResponse:
    """Pre-canned Prometheus response for testing."""
    status: str = "success"
    result_type: str = "matrix"
    results: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.results is None:
            self.results = [{
                "metric": {"__name__": "http_requests_total", "job": "test-service"},
                "values": [[str(int(time.time())), "100"], [str(int(time.time()) + 60), "150"]],
            }]
        return {
            "status": self.status,
            "data": {
                "resultType": self.result_type,
                "result": self.results,
            },
        }


@dataclass
class FakeLokiResponse:
    """Pre-canned Loki response for testing."""
    status: str = "success"
    results: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.results is None:
            self.results = [{
                "stream": {"job": "test-service", "level": "error"},
                "values": [[str(int(time.time()) * 1e9), "Error: connection refused"]],
            }]
        return {
            "status": self.status,
            "data": {"result": self.results},
        }


@dataclass
class FakeDeploymentResponse:
    """Pre-canned deployment response for testing."""
    status: str = "success"
    deployments: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.deployments is None:
            self.deployments = [{
                "service": "test-service",
                "version": "v1.0.0",
                "deployed_at": "2026-01-01T00:00:00Z",
                "image": "test-service:v1.0.0",
            }]
        return {
            "status": self.status,
            "data": self.deployments,
        }


@dataclass
class FakeRelatedIncidentsResponse:
    """Pre-canned related incidents response for testing."""
    status: str = "success"
    incidents: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.incidents is None:
            self.incidents = [{
                "incident_id": "inc_test123",
                "title": "High latency in test-service",
                "status": "resolved",
                "started_at": "2026-01-01T00:00:00Z",
                "resolved_at": "2026-01-01T01:00:00Z",
                "severity": "high",
                "environment": "test",
                "affected_services": ["test-service"],
                "correlation_policy": "pamawas-correlation-v1",
                "correlation_version": 1,
            }]
        return {
            "status": self.status,
            "data": self.incidents,
        }


class FakePrometheusTransport:
    """Fake Prometheus transport for deterministic tests.

    Usage:
        transport = FakePrometheusTransport()
        transport.set_response(FakePrometheusResponse(status="success", results=[...]))
        result = await transport.query_range(promql, start, end, step)

    DO NOT USE IN PRODUCTION CODE PATHS.
    """

    def __init__(self):
        self._response = FakePrometheusResponse()
        self._should_fail = False
        self._error: Exception | None = None
        self.call_count = 0
        self.last_call: dict[str, Any] | None = None

    def set_response(self, response: FakePrometheusResponse):
        """Set the response to return."""
        self._response = response
        self._should_fail = False
        self._error = None

    def set_error(self, error: Exception):
        """Configure transport to raise an error."""
        self._should_fail = True
        self._error = error

    def set_timeout(self):
        """Configure transport to simulate timeout."""
        self._should_fail = True
        self._error = TimeoutError("Simulated timeout")

    async def query_range(
        self,
        promql: str,
        start: str,
        end: str,
        step_seconds: int = 60,
    ) -> dict[str, Any]:
        """Execute fake range query."""
        self.call_count += 1
        self.last_call = {
            "promql": promql,
            "start": start,
            "end": end,
            "step_seconds": step_seconds,
        }

        observe_tool_call_duration("prometheus", 0.001)
        increment_tool_calls("prometheus", "success" if not self._should_fail else "error")

        if self._should_fail:
            raise self._error or Exception("Simulated error")

        return self._response.to_dict()

    async def query_instant(
        self,
        promql: str,
        eval_time: str | None = None,
    ) -> dict[str, Any]:
        """Execute fake instant query."""
        self.call_count += 1
        self.last_call = {
            "promql": promql,
            "eval_time": eval_time,
        }

        observe_tool_call_duration("prometheus", 0.001)
        increment_tool_calls("prometheus", "success" if not self._should_fail else "error")

        if self._should_fail:
            raise self._error or Exception("Simulated error")

        return self._response.to_dict()


class FakeLokiTransport:
    """Fake Loki transport for deterministic tests.

    DO NOT USE IN PRODUCTION CODE PATHS.
    """

    def __init__(self):
        self._response = FakeLokiResponse()
        self._should_fail = False
        self._error: Exception | None = None
        self.call_count = 0
        self.last_call: dict[str, Any] | None = None

    def set_response(self, response: FakeLokiResponse):
        self._response = response
        self._should_fail = False
        self._error = None

    def set_error(self, error: Exception):
        self._should_fail = True
        self._error = error

    def set_timeout(self):
        self._should_fail = True
        self._error = TimeoutError("Simulated timeout")

    async def query_range(
        self,
        logql: str,
        start: str,
        end: str,
        limit: int = 100,
        direction: str = "forward",
    ) -> dict[str, Any]:
        self.call_count += 1
        self.last_call = {
            "logql": logql,
            "start": start,
            "end": end,
            "limit": limit,
            "direction": direction,
        }

        observe_tool_call_duration("loki", 0.001)
        increment_tool_calls("loki", "success" if not self._should_fail else "error")

        if self._should_fail:
            raise self._error or Exception("Simulated error")

        return self._response.to_dict()

    async def query_instant(
        self,
        logql: str,
        eval_time: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        self.call_count += 1
        self.last_call = {
            "logql": logql,
            "eval_time": eval_time,
            "limit": limit,
        }

        observe_tool_call_duration("loki", 0.001)
        increment_tool_calls("loki", "success" if not self._should_fail else "error")

        if self._should_fail:
            raise self._error or Exception("Simulated error")

        return self._response.to_dict()


class FakeDeploymentTransport:
    """Fake Deployment transport for deterministic tests.

    DO NOT USE IN PRODUCTION CODE PATHS.
    """

    def __init__(self):
        self._response = FakeDeploymentResponse()
        self._should_fail = False
        self._error: Exception | None = None
        self.call_count = 0
        self.last_call: dict[str, Any] | None = None

    def set_response(self, response: FakeDeploymentResponse):
        self._response = response
        self._should_fail = False
        self._error = None

    def set_error(self, error: Exception):
        self._should_fail = True
        self._error = error

    async def get_recent_deployments(
        self,
        service: str,
        start: str,
        end: str,
        limit: int | None = None,
    ) -> dict[str, Any]:
        self.call_count += 1
        self.last_call = {
            "service": service,
            "start": start,
            "end": end,
            "limit": limit,
        }

        observe_tool_call_duration("deployments", 0.001)
        increment_tool_calls("deployments", "success" if not self._should_fail else "error")

        if self._should_fail:
            raise self._error or Exception("Simulated error")

        return self._response.to_dict()

    async def get_deployment_by_version(
        self,
        service: str,
        version: str,
    ) -> dict[str, Any]:
        self.call_count += 1
        self.last_call = {"service": service, "version": version}

        observe_tool_call_duration("deployments", 0.001)
        increment_tool_calls("deployments", "success" if not self._should_fail else "error")

        if self._should_fail:
            raise self._error or Exception("Simulated error")

        return self._response.to_dict()


class FakeRelatedIncidentsTransport:
    """Fake Related Incidents transport for deterministic tests.

    DO NOT USE IN PRODUCTION CODE PATHS.
    """

    def __init__(self):
        self._response = FakeRelatedIncidentsResponse()
        self._should_fail = False
        self._error: Exception | None = None
        self.call_count = 0
        self.last_call: dict[str, Any] | None = None

    def set_response(self, response: FakeRelatedIncidentsResponse):
        self._response = response
        self._should_fail = False
        self._error = None

    def set_error(self, error: Exception):
        self._should_fail = True
        self._error = error

    async def find_related(
        self,
        service: str,
        symptom_keywords: list[str],
        environment: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        self.call_count += 1
        self.last_call = {
            "service": service,
            "symptom_keywords": symptom_keywords,
            "environment": environment,
            "limit": limit,
        }

        observe_tool_call_duration("related_incidents", 0.001)
        increment_tool_calls("related_incidents", "success" if not self._should_fail else "error")

        if self._should_fail:
            raise self._error or Exception("Simulated error")

        return self._response.to_dict()

    async def find_by_service_and_time(
        self,
        service: str,
        start_time: str,
        end_time: str,
        environment: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        self.call_count += 1
        self.last_call = {
            "service": service,
            "start_time": start_time,
            "end_time": end_time,
            "environment": environment,
            "limit": limit,
        }

        observe_tool_call_duration("related_incidents", 0.001)
        increment_tool_calls("related_incidents", "success" if not self._should_fail else "error")

        if self._should_fail:
            raise self._error or Exception("Simulated error")

        return self._response.to_dict()


def create_fake_transports() -> dict[str, Any]:
    """Create a complete set of fake transports for testing.

    Returns dict with keys: prometheus, loki, deployments, related_incidents
    """
    return {
        "prometheus": FakePrometheusTransport(),
        "loki": FakeLokiTransport(),
        "deployments": FakeDeploymentTransport(),
        "related_incidents": FakeRelatedIncidentsTransport(),
    }
