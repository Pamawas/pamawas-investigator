#!/usr/bin/env python3
"""Main entry point for the Pamawas Investigator service."""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg2
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from pydantic import BaseModel

from config import Config as InvestigationConfig
from logging_middleware import LoggingMiddleware, get_logger
from metrics import (
    investigation_retry_errors_total,
    investigation_retry_exhausted_total,
    investigation_retry_reclaimed_total,
    investigation_retry_scheduled_total,
    set_running,
)
from otel import OTelConfig, init_tracer
from service.investigator import PamawasInvestigator

# Configure structured logging
log = get_logger()
VERSION_FILE = Path(__file__).with_name("version.txt")


def get_app_version() -> str:
    version = os.getenv("APP_VERSION")
    if version:
        return version
    if VERSION_FILE.is_file():
        return VERSION_FILE.read_text(encoding="utf-8").strip() or "1.0.0"
    return "1.0.0"


APP_VERSION = get_app_version()


def print_startup_banner(version: str, port: int) -> None:
    banner = r""" _____       /\      __  __      /\     __          __     /\      _____
|  __ \     /  \    |  \/  |    /  \    \ \        / /    /  \    / ____|
| |__) |   / /\ \   | \  / |   / /\ \    \ \  /\  / /    / /\ \   \___ \
|  ___/   / ____ \  | |\/| |  / ____ \    \ \/  \/ /    / ____ \   ___) |
|_|      /_/    \_\ |_|  |_| /_/    \_\    \__/\__/    /_/    \_\ |____/"""
    service_art = r""" _____   _   _  __      __  ______   _____   _______   _____    _____      /\      _______    ___    _____
|_   _| | \ | | \ \    / / |  ____| / ____| |__   __| |_   _|  / ____|    /  \    |__   __|  / _ \  |  __ \
  | |   |  \| |  \ \  / /  | |__    \___ \     | |      | |   | |  __    / /\ \      | |    | | | | | |__) |
  | |   | |\  |   \ \/ /   |  __|    ___) |    | |      | |   | | |_ |  / ____ \     | |    | |_| | |  _  /
 _| |_  |_| \_|    \__/    |______| |____/     |_|     _| |_   \_____| /_/    \_\    |_|     \___/  |_| \_\
""".rstrip("\n")  # noqa: E501
    print(f"\033[1;36m{banner}\n{service_art}\033[0m\nVersion: {version} | Port: :{port}\n")


# Global investigator instance
investigator: PamawasInvestigator = None
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global investigator

    # Startup
    log.info("starting_pamawas_investigator")

    # Initialize OpenTelemetry tracing
    otel_sample_ratio = float(os.getenv("OTEL_SAMPLE_RATIO", "0.1"))
    otel_config = OTelConfig(
        service_name="pamawas-investigator",
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        insecure=True,
        sample_ratio=otel_sample_ratio,
        enabled=bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")),
    )
    otel_shutdown = init_tracer(otel_config)

    try:
        config = InvestigationConfig.from_env()
        log.info(
            "config_loaded",
            llm_base_urls=config.llm_base_urls,
            llm_api_keys_count=len(config.llm_api_keys or []),
            llm_models_count=len(config.llm_models or []),
            version=APP_VERSION,
        )

        investigator = PamawasInvestigator(config)

        # Only advertise the service as running once the database dependency
        # actually answers; a process that came up with a dead DB reports
        # investigator_running=0 so alerting reflects reality.
        db_healthy, _ = _check_database_health(investigator.db_conn)
        if not db_healthy:
            raise RuntimeError("database health check failed at startup")
        set_running(True)
        log.info("investigator_ready")
    except (psycopg2.Error, RuntimeError, ValueError, OSError) as e:
        log.error("investigator_init_failed", error=str(e))
        raise

    # Start the auto-retry scheduler (reclaims due failed_retryable runs).
    retry_scheduler_task = None
    if investigator is not None and investigator.persistence:
        retry_scheduler_task = asyncio.create_task(
            _investigation_retry_scheduler(config.retry_scheduler_tick_seconds)
        )

    yield

    # Shutdown
    if retry_scheduler_task is not None:
        retry_scheduler_task.cancel()
        try:
            await retry_scheduler_task
        except asyncio.CancelledError:
            pass
    if otel_shutdown:
        otel_shutdown()
    # Close investigator
    if investigator is not None:
        await investigator.close()
    log.info("shutting_down_pamawas_investigator")
    set_running(False)


app = FastAPI(
    title="Pamawas Investigator",
    description="Bounded tool-calling LLM investigation engine for infrastructure incidents",
    version=APP_VERSION,
    lifespan=lifespan,
)

# Add structured logging middleware
app.add_middleware(LoggingMiddleware, service_name="pamawas-investigator")

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/healthz")
async def healthz():
    """Health check endpoint."""
    if investigator is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "Investigator not initialized"},
        )

    # Check database connectivity
    if investigator.db_conn is None:
        return JSONResponse(
            status_code=503, content={"status": "unhealthy", "error": "Database not connected"}
        )

    # Gather dependency health
    dependencies = {}
    overall_healthy = True

    db_healthy, db_deps = _check_database_health(investigator.db_conn)
    dependencies.update(db_deps)
    overall_healthy = overall_healthy and db_healthy

    adapter_healthy, adapter_deps = _check_adapter_health(getattr(investigator, "tools", None))
    dependencies.update(adapter_deps)
    overall_healthy = overall_healthy and adapter_healthy

    if overall_healthy:
        return {"status": "healthy", "timestamp": time.time(), "dependencies": dependencies}

    # Top-level error for clients that read it directly (e.g. database down).
    top_level_error = None
    db = dependencies.get("database", {})
    if db.get("status") == "unhealthy":
        top_level_error = db.get("error")
    return JSONResponse(
        status_code=503,
        content={
            "status": "unhealthy",
            "timestamp": time.time(),
            "dependencies": dependencies,
            "error": top_level_error,
        },
    )


def _check_database_health(db_conn) -> tuple[bool, dict]:
    """Check database connectivity; return (healthy, dependency entries)."""
    try:
        with db_conn.cursor() as cursor:
            cursor.execute("SELECT 1")
        return True, {"database": {"status": "healthy"}}
    except (psycopg2.Error, RuntimeError) as e:
        deps = {"database": {"status": "unhealthy", "error": str(e)}}
        try:
            db_conn.rollback()
        except psycopg2.Error:
            pass
        return False, deps


def _check_adapter_health(tools) -> tuple[bool, dict]:
    """Check external adapter circuit breaker states; return (healthy, dependency entries)."""
    deps: dict = {}
    healthy = True

    for attr, name in (
        ("_prometheus_adapter", "prometheus"),
        ("_loki_adapter", "loki"),
        ("_deployment_adapter", "deployments"),
        ("_related_incidents_adapter", "related_incidents"),
    ):
        adapter = getattr(tools, attr, None) if tools is not None else None
        if adapter is None:
            deps[name] = {"status": "unknown", "error": "adapter not initialized"}
            continue
        cb_state = getattr(adapter, "circuit_breaker_state", None)
        if cb_state:
            state = cb_state()
            deps[name] = {
                "status": "unhealthy" if state.get("is_open") else "healthy",
                "circuit_breaker": state,
            }
            if state.get("is_open"):
                healthy = False
        else:
            deps[name] = {"status": "unknown"}

    return healthy, deps


@app.get("/ready")
async def ready():
    """Readiness check endpoint."""
    if investigator is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "error": "Investigator not initialized"},
        )

    # Check database connectivity
    if investigator.db_conn is None:
        return JSONResponse(
            status_code=503, content={"status": "not ready", "error": "Database not ready"}
        )

    try:
        with investigator.db_conn.cursor() as cursor:
            cursor.execute("SELECT 1")
        return {"status": "ready"}
    except (psycopg2.Error, RuntimeError) as e:
        log.error("readiness_check_failed", error=str(e))
        try:
            investigator.db_conn.rollback()
        except psycopg2.Error:
            pass
        return JSONResponse(status_code=503, content={"status": "not ready", "error": str(e)})


@app.post("/api/v1/investigate")
async def investigate(incident_id: str):
    """Legacy transitional endpoint (deprecated).

    The v1 frozen contract for investigation creation is
    POST /api/v1/investigations (async, idempotent, service-token protected).
    This unversioned endpoint is transitional and MUST be removed after
    clients migrate. See pamawas-docs/system-design/api-workflow-contracts.md
    §12 Compatibility.

    To preserve the "do not maintain two paths with different authorization
    or idempotency semantics" rule, this shim mirrors the async v1 creation
    contract: it persists a queued run and returns 202, never inline findings.
    """
    if investigator is None:
        return JSONResponse(status_code=503, content={"error": "Investigator not initialized"})

    log.info("legacy_investigate_deprecated", incident_id=incident_id)

    import hashlib

    request_key = f"{incident_id}:incident_created:1"
    request_key_hash = hashlib.sha256(request_key.encode()).hexdigest()

    run_id, existing_status = await _ensure_investigation_run(incident_id, request_key_hash)

    if run_id is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )

    if existing_status is not None:
        return JSONResponse(
            status_code=409,
            content={
                "error": "Investigation already exists",
                "request_id": f"req_{request_key_hash[:26]}",
                "data": {
                    "run_id": run_id,
                    "incident_id": incident_id,
                    "status": existing_status,
                },
            },
        )

    return JSONResponse(
        status_code=202,
        content={
            "request_id": f"req_{request_key_hash[:26]}",
            "data": {
                "run_id": run_id,
                "incident_id": incident_id,
                "status": "queued",
            },
        },
        headers={
            "Deprecation": "true",
            "Link": '</api/v1/investigations>; rel="successor-version"',
        },
    )


class InvestigationRequest(BaseModel):
    """Typed request body for POST /api/v1/investigations (frozen v1 contract)."""

    contract_version: int
    incident_id: str
    reason: str
    correlation_version: int


async def _verify_service_token(x_service_token: str | None = Header(None)) -> str:
    """Verify the service token for internal mutation endpoints."""
    expected_token = os.getenv("PAMAWAS_SERVICE_TOKEN", "")
    if not expected_token:
        log.error("service_token_not_configured")
        raise HTTPException(status_code=503, detail="Service token not configured")
    if not x_service_token or x_service_token != expected_token:
        log.warning("service_token_invalid")
        raise HTTPException(status_code=401, detail="Invalid service token")
    return x_service_token


async def _ensure_investigation_run(
    incident_id: str, request_key_hash: str
) -> tuple[str | None, str | None]:
    """Persist (or fetch) an investigation run. Returns (run_id, existing_status).

    run_id is the DB-generated id returned by create_investigation_run — never a
    locally fabricated `irun_...`. existing_status is set only when the run
    already exists (idempotent retry), else None for a newly created run.
    """
    if investigator is None:
        return None, None

    if not investigator.persistence:
        log.error("persistence_not_available")
        return None, None

    try:
        run_id, created_new = investigator.persistence.create_investigation_run(
            incident_id=incident_id,
            request_key_hash=request_key_hash,
            model_provider=investigator.config.llm_base_url,
            model_name=investigator.config.llm_model,
            prompt_version="pamawas-investigator-v1",
            tool_contract=1,
            max_tool_calls=investigator.config.max_tool_calls,
        )
        if created_new:
            return run_id, None
        # Existing run - fetch its current status
        existing_status = _get_run_status(run_id)
        return run_id, existing_status
    except (psycopg2.Error, RuntimeError) as e:
        log.error("failed_to_persist_investigation_run", error=str(e))
        return None, None


def _get_run_status(run_id: str) -> str | None:
    """Fetch a run's status, or None on failure."""
    if not investigator or not investigator.persistence:
        return None
    conn = None
    try:
        conn = investigator.persistence._get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT status FROM investigation_runs WHERE id = %s", (run_id,))
            row = cursor.fetchone()
        conn.commit()
        return row[0] if row else None
    except (psycopg2.Error, RuntimeError) as e:
        log.error("failed_to_get_run_status", error=str(e))
        if conn is not None:
            try:
                conn.rollback()
            except psycopg2.Error, AttributeError:
                pass
        return None
    finally:
        if conn is not None:
            investigator.persistence._put_connection(conn)


@app.post("/api/v1/investigations")
async def create_investigation(
    request: InvestigationRequest,
    x_service_token: str | None = Header(None),
):
    """Create an investigation run (idempotent on request_key_hash).

    Frozen v1 contract (api-workflow-contracts.md §5):
    - Caller: correlator, audience investigator, scope investigator:investigate
    - Body: contract_version, incident_id, reason, correlation_version
    - Idempotency key is deterministic from incident_id:reason:correlation_version
    - Accepted response: 202 with request_id, data.run_id, data.status == "queued"
    - Idempotent retry returns the same run_id; a changed request under the
      same key returns 409.
    """
    # Verify service token
    await _verify_service_token(x_service_token)

    if investigator is None:
        return JSONResponse(status_code=503, content={"error": "Investigator not initialized"})

    incident_id = request.incident_id
    reason = request.reason
    correlation_version = request.correlation_version

    # Generate deterministic request key hash
    import hashlib

    request_key = f"{incident_id}:{reason}:{correlation_version}"
    request_key_hash = hashlib.sha256(request_key.encode()).hexdigest()

    # Persist (or fetch) the run first, capturing the real DB run_id. This
    # closes the race where the async kick-off and the idempotency check
    # previously both fabricated a run_id locally.
    run_id, existing_status = await _ensure_investigation_run(incident_id, request_key_hash)

    if run_id is None:
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

    if existing_status is not None:
        # Run already exists - return conflict per contract. The status is
        # included so the correlator can decide whether to auto-retry a
        # failed_terminal run via POST /api/v1/investigations/{run_id}/retry.
        log.info(
            "investigation_already_exists",
            incident_id=incident_id,
            run_id=run_id,
        )
        return JSONResponse(
            status_code=409,
            content={
                "error": "Investigation already exists",
                "request_id": f"req_{request_key_hash[:26]}",
                "data": {
                    "run_id": run_id,
                    "incident_id": incident_id,
                    "status": existing_status,
                },
            },
        )

    # New run persisted. Kick off the investigation asynchronously in the
    # background; return 202 with status "queued" per the frozen contract.
    log.info("investigation_started", incident_id=incident_id, reason=reason)
    import asyncio

    asyncio.get_running_loop().create_task(
        _run_investigation_task(run_id, incident_id, request_key_hash)
    )

    return JSONResponse(
        status_code=202,
        content={
            "request_id": f"req_{request_key_hash[:26]}",
            "data": {
                "run_id": run_id,
                "incident_id": incident_id,
                "status": "queued",
            },
        },
    )


@app.get("/api/v1/status")
async def status():
    """Status endpoint."""
    if investigator is None:
        return JSONResponse(status_code=503, content={"error": "Investigator not initialized"})

    return {
        "running": True,
        "uptime_seconds": time.time() - start_time,
        "version": APP_VERSION,
    }


@app.get("/api/v1/investigations/{run_id}")
async def get_investigation_status(run_id: str):
    """Fetch the current status of an investigation run.

    Frozen v1 contract (api-workflow-contracts.md §6): callers poll this
    endpoint after receiving 202 from POST /api/v1/investigations. The
    investigation runs asynchronously, so the status transitions
    queued -> running -> completed | failed.
    """
    if investigator is None or not investigator.persistence:
        return JSONResponse(status_code=503, content={"error": "Investigator not initialized"})

    current_status = _get_run_status(run_id)
    if current_status is None:
        return JSONResponse(
            status_code=404,
            content={"error": "Investigation not found", "run_id": run_id},
        )

    return {
        "run_id": run_id,
        "status": current_status,
    }


def _retry_backoff_seconds(attempts: int, base: int) -> float:
    """Exponential back-off for investigation retries: base * 2^attempts, capped at 1h."""
    if attempts > 5:
        return 3600.0
    delay = base * (2 ** attempts)
    return float(min(delay, 3600))


async def _run_investigation_task(run_id: str, incident_id: str, request_key_hash: str):
    """Background wrapper around investigate_async.

    The investigation runs in a fire-and-forget task spawned from the HTTP
    endpoint. investigate_async persists the run (terminal state + evidence)
    on every return path, so a normal return leaves the run completed. If it
    raises, the run is retried under the auto-retry budget rather than failed
    outright: schedule_investigation_retry increments the run's `attempts` and,
    while attempts < max_attempts, leaves it `failed_retryable` with a back-off
    `next_retry_at` that the retry scheduler later reclaims and re-runs. Once
    the budget is exhausted the run goes `failed_terminal`, which lets the
    scheduler dispatch a (possibly partial) report instead of a stuck queued run.
    """
    try:
        await investigator.investigate_async(incident_id, request_key_hash)
    except Exception as e:  # noqa: BLE001
        log.error(
            "investigation_task_failed",
            run_id=run_id,
            incident_id=incident_id,
            error=str(e),
        )
        # Record the failure under the auto-retry budget. If attempts remain,
        # schedule_investigation_retry leaves the run `failed_retryable` with a
        # back-off `next_retry_at` that the retry scheduler later reclaims and
        # re-runs; if the budget is exhausted it goes `failed_terminal`. Swallow
        # the exception either way: the run's status is the source of truth, and
        # the failed_retryable case already has a re-run scheduled.
        if not investigator.persistence:
            log.warning(
                "investigation_persistence_unavailable",
                run_id=run_id,
                incident_id=incident_id,
            )
        else:
            from datetime import UTC, datetime, timedelta

            state = investigator.persistence.get_run_retry_state(run_id) or {}
            attempts = state.get("attempts", 0)
            base = investigator.config.investigation_retry_backoff_seconds
            backoff = _retry_backoff_seconds(attempts, base)
            next_retry_at = datetime.now(UTC) + timedelta(seconds=backoff)
            scheduled = investigator.persistence.schedule_investigation_retry(
                run_id,
                error=str(e),
                next_retry_at=next_retry_at,
            )
            if scheduled:
                investigation_retry_scheduled_total.inc()
            else:
                investigation_retry_exhausted_total.inc()
                log.warning(
                    "investigation_retry_budget_exhausted",
                    run_id=run_id,
                    incident_id=incident_id,
                    attempts=attempts,
                )


async def _investigation_retry_scheduler(tick_seconds: float):
    """Reclaim due `failed_retryable` runs and re-execute them.

    Runs as a background asyncio task for the lifetime of the process. It wakes
    every `tick_seconds` (from config, RETRY_SCHEDULER_TICK_SECONDS, default
    15), atomically reclaims any run whose retry back-off has elapsed, and
    spawns a fresh _run_investigation_task for each. This both drives
    in-process retries and self-heals runs that were mid-retry when the process
    crashed (a failed_retryable run is simply reclaimed again).
    """
    while True:
        try:
            if investigator is not None and investigator.persistence:
                due = investigator.persistence.list_due_investigation_retries(limit=10)
                for run_id in due:
                    if investigator.persistence.reclaim_due_investigation_retry(run_id):
                        investigation_retry_reclaimed_total.inc()
                        # reclaim moved it to running; re-run the investigation.
                        # We don't have incident_id here, so look it up from the run.
                        incident_id = _get_run_incident_id(run_id)
                        if incident_id:
                            loop = asyncio.get_running_loop()
                            loop.create_task(
                                _run_investigation_task(
                                    run_id, incident_id, _get_run_request_key(run_id)
                                )
                            )
        except Exception as e:  # noqa: BLE001
            investigation_retry_errors_total.inc()
            log.error("investigation_retry_scheduler_error", error=str(e))
        await asyncio.sleep(tick_seconds)


def _get_run_incident_id(run_id: str) -> str | None:
    """Return a run's incident_id, or None on failure."""
    if not investigator or not investigator.persistence:
        return None
    conn = None
    try:
        conn = investigator.persistence._get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT incident_id FROM investigation_runs WHERE id = %s", (run_id,))
            row = cursor.fetchone()
        conn.commit()
        return row[0] if row else None
    except (psycopg2.Error, RuntimeError):
        if conn is not None:
            try:
                conn.rollback()
            except (psycopg2.Error, AttributeError):
                pass
        return None
    finally:
        if conn is not None:
            investigator.persistence._put_connection(conn)


_GET_RUN_REQUEST_KEY_SQL = "SELECT request_key_hash FROM investigation_runs WHERE id = %s"


def _get_run_request_key(run_id: str) -> str:
    """Return a run's request_key_hash (used for idempotent re-execution)."""
    if not investigator or not investigator.persistence:
        return ""
    conn = None
    try:
        conn = investigator.persistence._get_connection()
        with conn.cursor() as cursor:
            cursor.execute(_GET_RUN_REQUEST_KEY_SQL, (run_id,))
            row = cursor.fetchone()
        conn.commit()
        return row[0] if row else ""
    except (psycopg2.Error, RuntimeError):
        if conn is not None:
            try:
                conn.rollback()
            except (psycopg2.Error, AttributeError):
                pass
        return ""
    finally:
        if conn is not None:
            investigator.persistence._put_connection(conn)


def main():
    """Main entry point for running as a standalone service."""
    import uvicorn

    config = InvestigationConfig.from_env()
    print_startup_banner(APP_VERSION, config.port)
    log.info("starting_server", port=config.port)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
