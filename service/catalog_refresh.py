"""Refresh provider catalogs independently of investigation traffic."""

import asyncio
import logging
import random
import time

import psycopg2
from prometheus_client import Counter, Gauge, Info

from adapters import LokiError, PrometheusError
from service.persistence import InvestigationPersistence
from tools import InvestigatorTools

logger = logging.getLogger(__name__)

_PROVIDERS = ("prometheus", "loki")
_POLL_SECONDS = 30
_DISCOVERY_TIMEOUT_SECONDS = 20

_last_success = Gauge(
    "pamawas_investigator_telemetry_catalog_last_success_timestamp_seconds",
    "Unix timestamp of last successful provider catalog discovery (0 if none)",
    ["provider"],
)
_age = Gauge(
    "pamawas_investigator_telemetry_catalog_age_seconds",
    "Seconds since last successful provider catalog discovery",
    ["provider"],
)
_stale = Gauge(
    "pamawas_investigator_telemetry_catalog_stale",
    "Whether the provider catalog is overdue for refresh",
    ["provider"],
)
_entries = Gauge(
    "pamawas_investigator_telemetry_catalog_entries",
    "Entries in the last successful provider catalog",
    ["provider"],
)
_revision = Info(
    "pamawas_investigator_telemetry_catalog_revision",
    "Latest published combined catalog revision",
    ["provider"],
)
_refreshes = Counter(
    "pamawas_investigator_telemetry_catalog_refresh_total",
    "Provider catalog refresh attempts by result",
    ["provider", "result"],
)


class CatalogRefreshWorker:
    def __init__(
        self, persistence: InvestigationPersistence, tools: InvestigatorTools, interval: int
    ):
        self.persistence = persistence
        self.tools = tools
        self.interval = interval
        self._wake = asyncio.Event()
        self._stopping = False
        self._next_request_at = 0.0

    def request_refresh(self) -> None:
        """Wake the worker if the normal poll is due soon."""
        if time.monotonic() >= self._next_request_at:
            self._wake.set()

    def stop(self) -> None:
        self._stopping = True
        self._wake.set()

    async def run(self) -> None:
        while not self._stopping:
            self._wake.clear()
            self._next_request_at = time.monotonic() + _POLL_SECONDS
            try:
                await self.refresh_due()
            except (psycopg2.Error, RuntimeError) as exc:
                logger.warning("Telemetry catalog poll failed with %s", type(exc).__name__)
            try:
                await asyncio.wait_for(self._wake.wait(), _POLL_SECONDS + random.uniform(0, 15))
            except TimeoutError:
                pass

    async def refresh_due(self) -> None:
        results = await asyncio.gather(
            *(self._refresh_provider(provider) for provider in _PROVIDERS),
            return_exceptions=True,
        )
        for provider, result in zip(_PROVIDERS, results, strict=True):
            if isinstance(result, (psycopg2.Error, RuntimeError)):
                logger.warning("%s catalog refresh state failed with %s", provider, type(result))
            elif isinstance(result, BaseException):
                raise result
        rows = await asyncio.to_thread(self.persistence.telemetry_catalog_status)
        by_provider = {
            provider: (timestamp, entries, rev) for provider, timestamp, entries, rev in rows
        }
        for provider in _PROVIDERS:
            timestamp, count, revision_id = by_provider.get(provider, (None, 0, None))
            age = max(0, time.time() - timestamp.timestamp()) if timestamp else float("nan")
            _last_success.labels(provider).set(timestamp.timestamp() if timestamp else 0)
            _age.labels(provider).set(age)
            _stale.labels(provider).set(timestamp is None or age >= self.interval)
            _entries.labels(provider).set(count)
            _revision.labels(provider).info({"revision": revision_id or "unpublished"})

    async def _refresh_provider(self, provider: str) -> None:
        token = await asyncio.to_thread(
            self.persistence.claim_telemetry_catalog_refresh, provider, self.interval
        )
        if token is None:
            return
        try:
            async with asyncio.timeout(_DISCOVERY_TIMEOUT_SECONDS):
                resource_kind, entries = await self._discover(provider)
            stored = await asyncio.to_thread(
                self.persistence.store_telemetry_catalog, provider, resource_kind, entries, token
            )
            result = "success" if stored else "lease_lost"
        except (
            LokiError, PrometheusError, psycopg2.Error, TimeoutError, RuntimeError, TypeError,
            ValueError,
        ) as exc:
            error_class = type(exc).__name__
            logger.warning("%s catalog refresh failed with %s", provider, error_class)
            recorded = await asyncio.to_thread(
                self.persistence.mark_telemetry_catalog_refresh_failed, provider, error_class, token
            )
            result = "failure" if recorded else "lease_lost"
        _refreshes.labels(provider, result).inc()

    async def _discover(self, provider: str) -> tuple[str, list[dict]]:
        if provider == "prometheus":
            result = await self.tools._prometheus_adapter.query_metric_catalog()
            if self._valid_discovery(result):
                return "metric", [item for item in result["data"] if isinstance(item, dict)]
        else:
            result = await self.tools._loki_adapter.query_labels()
            if self._valid_discovery(result):
                return "label", [{"name": item} for item in result["data"] if isinstance(item, str)]
        raise ValueError(f"{provider} discovery returned an invalid catalog")

    @staticmethod
    def _valid_discovery(result: dict) -> bool:
        return (
            isinstance(result, dict)
            and result.get("status") == "success"
            and isinstance(result.get("data"), list)
            and bool(result["data"])
        )
