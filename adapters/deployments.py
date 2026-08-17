"""Deployment history adapter - returns unavailable when not configured."""

import logging
from dataclasses import dataclass
from typing import Any

from metrics import increment_tool_calls, observe_tool_call_duration

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for deployment adapter."""
    base_url: str | None = None
    timeout_seconds: float = 10.0
    max_results: int = 50
    api_key: str | None = None

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
    """Deployment history interface.

    Returns real adapter data only when configured.
    Otherwise returns explicit unavailable response.
    Never returns sample/fake deployments.
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self._client: Any | None = None  # httpx.AsyncClient when configured

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
        import time
        start_time = time.time()

        if not service or not service.strip():
            return self._unavailable_response("invalid_service_name")

        # If not configured, return explicit unavailable
        if not self.config.is_configured():
            observe_tool_call_duration("deployments", time.time() - start_time)
            increment_tool_calls("deployments", "unavailable")
            logger.info(
                "Deployment adapter not configured, returning unavailable for "
                "service: %s",
                service,
            )
            return self._unavailable_response("deployment_adapter_not_configured")

        # If configured, attempt to query the deployment service
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

            observe_tool_call_duration("deployments", time.time() - start_time)
            increment_tool_calls("deployments", "success")

            return {
                "status": "success",
                "data": data,
            }

        except Exception as e:  # noqa: BLE001
            observe_tool_call_duration("deployments", time.time() - start_time)
            increment_tool_calls("deployments", "error")
            logger.error("Deployment query failed: %s", e)
            # Return error but don't fabricate data
            return {
                "status": "error",
                "reason_code": "deployment_query_failed",
                "error": str(e),
                "data": [],
            }

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

        try:
            response = await client.get(
                f"/api/v1/deployments/{service}/{version}"
            )
            response.raise_for_status()

            return {
                "status": "success",
                "data": response.json(),
            }

        except Exception as e:  # noqa: BLE001
            logger.error("Deployment version query failed: %s", e)
            return {
                "status": "error",
                "reason_code": "deployment_query_failed",
                "error": str(e),
                "data": {},
            }
