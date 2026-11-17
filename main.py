#!/usr/bin/env python3
"""Main entry point for the Pamawas Investigator service."""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import psycopg2
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from pydantic import BaseModel

from config import Config as InvestigationConfig
from config import rate_limit_from_env
from logging_middleware import LoggingMiddleware, configure_uvicorn_logging, get_logger
from metrics import (
    increment_stale_verification_claimed,
    increment_stale_verification_completed,
    increment_stale_verification_failed,
    increment_stale_verification_requeued,
    increment_verify,
    investigation_retry_errors_total,
    investigation_retry_exhausted_total,
    investigation_retry_reclaimed_total,
    investigation_retry_scheduled_total,
    set_running,
)
from otel import OTelConfig, init_tracer
from ratelimit_middleware import RateLimitMiddleware
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
    import sys

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
    # Write to stderr so the banner doesn't contaminate the structured JSON stream on stdout.
    sys.stderr.write(
        f"\033[1;36m{banner}\n{service_art}\033[0m\nVersion: {version} | Port: :{port}\n\n"
    )


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
    otel_sample_ratio = float(os.getenv("OTEL_SAMPLE_RATIO", "1.0"))
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

    # Start the investigation worker pool. Each worker claims one queued run at
    # a time, so the pool size is the concurrency limit on the LLM endpoint —
    # the scarce resource. A free endpoint serves a handful of concurrent
    # requests; starting a loop per incident turned a burst into a storm of
    # timeouts where every investigation failed for lack of capacity rather
    # than lack of evidence.
    workers = []
    retry_scheduler_task = None
    stale_verification_worker_task = None
    if investigator is not None and investigator.persistence:
        for worker_id in range(max(1, config.max_concurrent_investigations)):
            workers.append(asyncio.create_task(_investigation_worker(worker_id, config)))
        # Reclaims due failed_retryable runs and stranded running ones, both of
        # which then flow back through the same queue the workers claim from.
        retry_scheduler_task = asyncio.create_task(
            _investigation_retry_scheduler(config.retry_scheduler_tick_seconds)
        )
        # Async worker for stale-incident verification jobs enqueued by the
        # Scheduler into stale_verifications.  A single worker is sufficient
        # for the typical verification load; it shares the same LLM concurrency
        # pool via verify_incident().
        stale_verification_worker_task = asyncio.create_task(_stale_verification_worker())
        log.info("investigation_workers_started", count=len(workers))

    yield

    # Shutdown
    for task in [*workers, retry_scheduler_task, stale_verification_worker_task]:
        if task is None:
            continue
        task.cancel()
        try:
            await task
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
    # FastAPI's own OpenTelemetry integration activates itself as soon as a
    # tracer provider exists, and it cannot be told to skip liveness probes.
    # Left on, every request produced two server spans — one from FastAPI
    # (scope "fastapi") and one from LoggingMiddleware — and /healthz, /ready
    # and /metrics filled Tempo with spans nobody opens. LoggingMiddleware is
    # the single span owner here, matching the four Go services, so the access
    # log lines and the trace share one set of IDs and probes stay untraced.
    telemetry={"tracing": False},
)

# Per-route rate limiting (design review P5). Installed at app construction,
# so its knobs come from rate_limit_from_env() — validated there, failing
# startup on a malformed value. /healthz, /ready and /metrics are exempt.
# Added before LoggingMiddleware so logging wraps it: rejected requests are
# still logged with trace context, matching the Go services' ordering.
_rate_rps, _rate_burst = rate_limit_from_env()
app.add_middleware(
    RateLimitMiddleware,
    service_name="pamawas-investigator",
    limits={
        "/api/v1/investigations": (_rate_rps, _rate_burst),
        "/api/v1/verifications": (_rate_rps, _rate_burst),
        "/api/v1/status": (_rate_rps, _rate_burst),
        # The daily summary calls the LLM, so it carries a budget like the
        # investigation endpoints rather than being unmetered.
        "/api/v1/summaries": (_rate_rps, _rate_burst),
    },
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


class InvestigationRequest(BaseModel):
    """Typed request body for POST /api/v1/investigations (frozen v1 contract)."""

    contract_version: int
    incident_id: str
    reason: str
    correlation_version: int


class DailySummaryRequest(BaseModel):
    """Facts for the daily report's prose summary (POST /api/v1/summaries).

    Structured facts only, no transcripts: the summary is only as good as what
    it is given, and sending whole investigations invites invention.
    """

    period_start: str
    period_end: str
    timezone: str
    incidents: list[dict[str, Any]]


class VerificationRequest(BaseModel):
    """Request body for POST /api/v1/verifications (asynchronous verification run).

    Matches the contract in stale-incident-verification.md §6.1.
    """

    contract_version: int = 1
    incident_id: str


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

    # The run row IS the queue entry. A worker claims it from the database, so
    # the work survives a restart and concurrency is bounded by the pool size
    # rather than by how many webhooks happened to arrive at once.
    log.info("investigation_queued", incident_id=incident_id, reason=reason, run_id=run_id)

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


@app.post("/api/v1/summaries")
async def summarize_day(
    body: DailySummaryRequest,
    x_service_token: str | None = Header(None),
):
    """Write a short prose summary of a day's incidents.

    The reporter calls this while rendering the daily digest, so the summary is
    part of the message that goes out rather than a follow-up. All LLM access
    lives here: the reporter never holds model credentials or prompts.

    Service-token protected like every other mutation endpoint: this call costs
    an LLM request, so it must not be reachable by anything in the cluster.

    Degrades instead of failing: any model problem returns an empty summary and
    a 200, and the reporter renders its deterministic incident lists alone.
    """
    await _verify_service_token(x_service_token)

    if investigator is None:
        return {"summary": "", "reason": "investigator_not_initialized"}
    if not body.incidents:
        return {"summary": "", "reason": "no_incidents"}
    return {"summary": investigator.write_day_summary(body.model_dump())}


@app.post("/api/v1/verifications", status_code=202)
async def request_verification(
    body: VerificationRequest,
    x_service_token: str | None = Header(None),
):
    """Queue an asynchronous stale-incident verification run.

    Contract: stale-incident-verification.md §6.1.
    Allows the scheduler or operator CLI to explicitly request an asynchronous verification run.
    """
    await _verify_service_token(x_service_token)

    if investigator is None or not investigator.persistence:
        return JSONResponse(status_code=503, content={"error": "Investigator not initialized"})

    record = await asyncio.to_thread(
        investigator.persistence.enqueue_stale_verification,
        body.incident_id,
    )
    if not record:
        raise HTTPException(status_code=500, detail="Failed to enqueue verification")

    return {
        "request_id": f"req_verify_{body.incident_id[:16]}",
        "data": record,
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
    delay = base * (2**attempts)
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


async def _investigation_worker(worker_id: int, config: InvestigationConfig):
    """Execute queued investigation runs, one at a time.

    The pool of these is the concurrency limit for the LLM endpoint. Each
    iteration claims a single run from the queue in the database and executes
    it; if nothing is queued it waits briefly and polls again. Nothing is
    spawned per request and nothing waits in process memory, so a restart loses
    no work — the row is still `queued` and the pool claims it on the way back
    up.

    Claiming is atomic (`FOR UPDATE SKIP LOCKED`), so two workers polling at the
    same moment take different runs rather than racing for one.
    """
    poll_seconds = 1.0
    logger_prefix = f"worker-{worker_id}"
    while True:
        if investigator is None or not investigator.persistence:
            await asyncio.sleep(poll_seconds)
            continue

        try:
            claimed = await asyncio.to_thread(investigator.persistence.claim_next_queued_run)
            if claimed is None:
                await asyncio.sleep(poll_seconds)
                continue

            run_id, incident_id, request_key_hash = claimed
            log.info(
                "investigation_claimed",
                worker=logger_prefix,
                run_id=run_id,
                incident_id=incident_id,
            )
            # Run in this task: the worker is the concurrency slot, so spawning
            # here would reintroduce unbounded parallelism.
            await _run_investigation_task(run_id, incident_id, request_key_hash)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            # A worker must never die: it is one of a fixed pool, and losing one
            # silently reduces capacity for the life of the process.
            log.error("investigation_worker_error", worker=logger_prefix, error=str(e))
            await asyncio.sleep(poll_seconds)


async def _investigation_retry_scheduler(tick_seconds: float):
    """Sweep abandoned and retryable runs back into the queue.

    Runs as a background asyncio task for the lifetime of the process, waking
    every `tick_seconds` (RETRY_SCHEDULER_TICK_SECONDS, default 15). It does no
    execution itself: it moves two kinds of stranded run back to `queued` and
    the worker pool claims them.

    - `running` runs whose worker died mid-investigation (a pod restart). Without
      this they stay `running` forever, reporting progress that never comes.
    - `failed_retryable` runs whose back-off has elapsed.
    """
    while True:
        try:
            if investigator is not None and investigator.persistence:
                # Runs abandoned by a worker that died mid-investigation go back
                # to the queue before retries are considered, so a restart
                # recovers in-flight work instead of stranding it in `running`.
                stale_after = (
                    investigator.config.investigation_timeout_seconds * 2
                    + investigator.config.retry_scheduler_tick_seconds
                )
                investigator.persistence.requeue_stale_running(stale_after)

                due = investigator.persistence.list_due_investigation_retries(limit=10)
                for run_id in due:
                    if investigator.persistence.reclaim_due_investigation_retry(run_id):
                        investigation_retry_reclaimed_total.inc()
                        # The reclaim put it back in the queue; a worker picks
                        # it up on its next poll. Nothing is spawned here.
                        log.info("investigation_retry_requeued", run_id=run_id)

                # Reclaim stale_verifications rows whose worker died mid-verify
                # (pod restart, OOM kill).  Without this they stay 'running'
                # forever and the scheduler never sees a verdict.
                requeued = investigator.persistence.requeue_expired_stale_running()
                if requeued:
                    increment_stale_verification_requeued(requeued)
                    log.info("stale_verifications_requeued", count=requeued)
        except Exception as e:  # noqa: BLE001
            investigation_retry_errors_total.inc()
            log.error("investigation_retry_scheduler_error", error=str(e))
        await asyncio.sleep(tick_seconds)


async def _stale_verification_worker():
    """Claim queued stale_verifications rows, run verify_incident(), write verdict back.

    A single background worker is sufficient for the typical stale-verification
    load: the Scheduler enqueues at most one row per open incident, gated by
    the anti-flood back-off, so the queue is never deep.  The worker shares the
    investigator's LLM concurrency implicitly through verify_incident().

    Design invariants:
    - Claims exactly one row per iteration (FOR UPDATE SKIP LOCKED) — no race
      with the investigation workers or a second stale-verification worker.
    - On success  → complete_stale_verification(verdict, reason) + metric.
    - On failure  → fail_stale_verification(error) + metric.  Conservative:
      the Scheduler treats 'failed' as 'still_down' and schedules a re-verify.
    - On CancelledError → re-raise so the lifespan shutdown loop can await it.
    - Any other exception → log and sleep; a worker must never die silently.
    """
    poll_seconds = 5.0
    while True:
        if investigator is None or not investigator.persistence:
            await asyncio.sleep(poll_seconds)
            continue

        try:
            incident_id = await asyncio.to_thread(
                investigator.persistence.claim_next_stale_verification
            )
            if incident_id is None:
                await asyncio.sleep(poll_seconds)
                continue

            increment_stale_verification_claimed()
            log.info("stale_verification_claimed", incident_id=incident_id)

            import time as _time

            start = _time.monotonic()
            try:
                result = await investigator.verify_incident(incident_id)
                verdict = result.get("verdict", "still_down") if result else "still_down"
                reason = result.get("reason", "") if result else "verify_incident returned None"
                duration = _time.monotonic() - start
                await asyncio.to_thread(
                    investigator.persistence.complete_stale_verification,
                    incident_id,
                    verdict,
                    reason,
                )
                increment_stale_verification_completed(verdict)
                increment_verify(verdict, duration)
                log.info(
                    "stale_verification_completed",
                    incident_id=incident_id,
                    verdict=verdict,
                    duration_seconds=round(duration, 2),
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.error(
                    "stale_verification_failed",
                    incident_id=incident_id,
                    error=str(e),
                )
                await asyncio.to_thread(
                    investigator.persistence.fail_stale_verification,
                    incident_id,
                    str(e)[:256],
                )
                increment_stale_verification_failed()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.error("stale_verification_worker_error", error=str(e))
            await asyncio.sleep(poll_seconds)


def main():
    """Main entry point for running as a standalone service."""
    import uvicorn

    config = InvestigationConfig.from_env()
    print_startup_banner(APP_VERSION, config.port)
    log.info("starting_server", port=config.port)

    # Must be called BEFORE uvicorn.run() so our JSON handler is in place.
    # log_config=None tells uvicorn NOT to call dictConfig(), which would
    # overwrite our formatters. access_log=False suppresses uvicorn's own
    # access handler; our LoggingMiddleware logs HTTP at the application level
    # and uvicorn.access is re-routed to our JSON formatter anyway.
    configure_uvicorn_logging()

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.port,
        log_config=None,
        access_log=True,
    )


if __name__ == "__main__":
    main()
