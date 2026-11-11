"""Deployment history adapter with circuit breaker and safety bounds."""

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from adapters.circuit_breaker import CircuitBreaker
from metrics import increment_tool_calls, observe_tool_call_duration

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for deployment adapter."""

    base_url: str | None = None
    timeout_seconds: float = 10.0
    max_results: int = 50
    api_key: str | None = None
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: float = 30.0
    circuit_breaker_half_open_max_calls: int = 3

    def is_configured(self) -> bool:
        """Check if deployment adapter is configured."""
        return bool(self.base_url)


class DeploymentAdapterError(Exception):
    """Base exception for deployment adapter errors."""

    pass


class DeploymentAdapterNotConfiguredError(DeploymentAdapterError):
    """Deployment adapter is not configured."""

    pass


class DeploymentAdapter:
    """Deployment history interface with circuit breaker protection.

    Returns real adapter data only when configured.
    Otherwise returns explicit unavailable response.
    Never returns sample/fake deployments.
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self._client: Any | None = None  # httpx.AsyncClient when configured
        self._circuit_breaker = CircuitBreaker(
            name="deployments",
            failure_threshold=config.circuit_breaker_failure_threshold,
            timeout_seconds=config.circuit_breaker_timeout_seconds,
            half_open_max_calls=config.circuit_breaker_half_open_max_calls,
        )

    async def _get_client(self):
        """Get HTTP client if configured."""
        if not self.config.is_configured():
            return None

        import httpx

        if self._client is None or self._client.is_closed:
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            # base_url is guaranteed non-None here because is_configured() returned True
            base_url = self.config.base_url or ""
            self._client = httpx.AsyncClient(
                base_url=base_url.rstrip("/"),
                timeout=httpx.Timeout(self.config.timeout_seconds),
                headers=headers,
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def circuit_breaker_state(self) -> dict:
        """Return circuit breaker state for health checks."""
        return self._circuit_breaker.snapshot()

    def _unavailable_response(
        self, reason: str = "deployment_adapter_not_configured"
    ) -> dict[str, Any]:
        """Return standardized unavailable response."""
        return {
            "status": "unavailable",
            "reason_code": reason,
            "data": [],
        }

    async def get_recent_deployments(
        self,
        service: str,
        start: str,
        end: str,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Get recent deployments for a service.

        Args:
            service: Service name
            start: Start timestamp (RFC3339)
            end: End timestamp (RFC3339)
            limit: Maximum results

        Returns:
            Deployment data or unavailable response
        """
        start_time = time.time()

        if not service or not service.strip():
            observe_tool_call_duration("deployments", time.time() - start_time)
            increment_tool_calls("deployments", "unavailable")
            return self._unavailable_response("invalid_service_name")

        # If not configured, return explicit unavailable
        if not self.config.is_configured():
            observe_tool_call_duration("deployments", time.time() - start_time)
            increment_tool_calls("deployments", "unavailable")
            logger.info(
                "Deployment adapter not configured, returning unavailable for service: %s",
                service,
            )
            return self._unavailable_response("deployment_adapter_not_configured")

        # Circuit breaker: return unavailable immediately if open
        if not self._circuit_breaker.allow_call():
            observe_tool_call_duration("deployments", time.time() - start_time)
            increment_tool_calls("deployments", "circuit_open")
            logger.warning("Deployment circuit breaker open, rejecting query")
            return self._unavailable_response("circuit_breaker_open")

        client = await self._get_client()
        if not client:
            observe_tool_call_duration("deployments", time.time() - start_time)
            increment_tool_calls("deployments", "unavailable")
            return self._unavailable_response("deployment_adapter_not_configured")

        try:
            max_results = min(limit or self.config.max_results, self.config.max_results)

            params = {
                "service": service.strip(),
                "start": start,
                "end": end,
                "limit": str(max_results),
            }

            response = await client.get("/api/v1/deployments", params=params)
            response.raise_for_status()

            data = response.json()

            # Validate response structure
            if not isinstance(data, list):
                logger.warning("Unexpected deployment response format: %s", type(data))
                data = []

            self._circuit_breaker.record_success()
            observe_tool_call_duration("deployments", time.time() - start_time)
            increment_tool_calls("deployments", "success")

            return {
                "status": "success",
                "data": data,
            }

        except httpx.TimeoutException:
            self._circuit_breaker.record_failure()
            observe_tool_call_duration("deployments", time.time() - start_time)
            increment_tool_calls("deployments", "timeout")
            raise DeploymentAdapterError(
                f"Deployment query timed out after {self.config.timeout_seconds}s"
            )
        except httpx.HTTPStatusError as e:
            self._circuit_breaker.record_failure()
            observe_tool_call_duration("deployments", time.time() - start_time)
            increment_tool_calls("deployments", "error")
            raise DeploymentAdapterError(
                f"Deployment HTTP error {e.response.status_code}: {e.response.text}"
            )
        except (httpx.HTTPError, ValueError, KeyError) as e:
            self._circuit_breaker.record_failure()
            observe_tool_call_duration("deployments", time.time() - start_time)
            increment_tool_calls("deployments", "error")
            logger.error("Deployment query failed: %s", e)
            raise DeploymentAdapterError(f"Deployment query failed: {e}")

    async def get_deployment_by_version(
        self,
        service: str,
        version: str,
    ) -> dict[str, Any]:
        """
        Get specific deployment by version.

        Args:
            service: Service name
            version: Deployment version/tag

        Returns:
            Deployment data or unavailable response
        """
        if not self.config.is_configured():
            return self._unavailable_response("deployment_adapter_not_configured")

        client = await self._get_client()
        if not client:
            return self._unavailable_response("deployment_adapter_not_configured")

        # Circuit breaker: return unavailable immediately if open
        if not self._circuit_breaker.allow_call():
            logger.warning("Deployment circuit breaker open, rejecting version query")
            return self._unavailable_response("circuit_breaker_open")

        try:
            response = await client.get(f"/api/v1/deployments/{service}/{version}")
            response.raise_for_status()

            self._circuit_breaker.record_success()
            return {
                "status": "success",
                "data": response.json(),
            }

        except httpx.TimeoutException:
            self._circuit_breaker.record_failure()
            raise DeploymentAdapterError(
                f"Deployment query timed out after {self.config.timeout_seconds}s"
            )
        except httpx.HTTPStatusError as e:
            self._circuit_breaker.record_failure()
            raise DeploymentAdapterError(
                f"Deployment HTTP error {e.response.status_code}: {e.response.text}"
            )
        except (httpx.HTTPError, ValueError, KeyError) as e:
            self._circuit_breaker.record_failure()
            logger.error("Deployment version query failed: %s", e)
            raise DeploymentAdapterError(f"Deployment version query failed: {e}")
