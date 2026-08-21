#!/usr/bin/env python3
"""Main entry point for the Pamawas Investigator service."""

import os
import time
from contextlib import asynccontextmanager

import psycopg2
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from config import Config as InvestigationConfig
from logging_middleware import LoggingMiddleware, get_logger
from metrics import set_running
from otel import OTelConfig, init_tracer
from service.investigator import PamawasInvestigator

# Configure structured logging
log = get_logger()

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
    otel_config = OTelConfig(
        service_name="pamawas-investigator",
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        insecure=True,
        sample_ratio=1.0,
        enabled=bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")),
    )
    otel_shutdown = init_tracer(otel_config)

    try:
        config = InvestigationConfig.from_env()
        log.info("config_loaded", llm_model=config.llm_model, llm_base_url=config.llm_base_url)

        investigator = PamawasInvestigator(config)
        set_running(True)
        log.info("investigator_ready")
    except Exception as e:
        log.error("investigator_init_failed", error=str(e))
        raise

    yield

    # Shutdown
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
    version="1.0.0",
    lifespan=lifespan,
)

# Add structured logging middleware
app.add_middleware(LoggingMiddleware, service_name="pamawas-investigator")

# Mount Prometheus metrics endpoint
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

    try:
        with investigator.db_conn.cursor() as cursor:
            cursor.execute("SELECT 1")
        return {"status": "healthy", "timestamp": time.time()}
    except Exception as e:  # noqa: BLE001
        log.error("health_check_failed", error=str(e))
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})


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
    except Exception as e:  # noqa: BLE001
        log.error("readiness_check_failed", error=str(e))
        return JSONResponse(status_code=503, content={"status": "not ready", "error": str(e)})


@app.post("/investigate")
async def investigate(incident_id: str):
    """Trigger investigation for an incident."""
    if investigator is None:
        return JSONResponse(status_code=503, content={"error": "Investigator not initialized"})

    log.info("investigation_started", incident_id=incident_id)
    findings = investigator.investigate(incident_id)

    return {
        "incident_id": incident_id,
        "findings": [f.to_dict() for f in findings],
        "completed": True,
    }


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


@app.post("/v1/investigations")
async def create_investigation(
    contract_version: int,
    incident_id: str,
    reason: str,
    correlation_version: int,
    x_service_token: str | None = Header(None),
):
    """Create an investigation run (idempotent on request_key_hash)."""
    # Verify service token
    await _verify_service_token(x_service_token)

    if investigator is None:
        return JSONResponse(status_code=503, content={"error": "Investigator not initialized"})

    # Generate deterministic request key hash
    import hashlib
    request_key = f"{incident_id}:{reason}:{correlation_version}"
    request_key_hash = hashlib.sha256(request_key.encode()).hexdigest()

    # Check if investigation run already exists (idempotency)
    if investigator.persistence:
        existing_run_id = None
        existing_status = None
        try:
            conn = investigator.persistence._get_connection()
            with conn.cursor() as cursor:
                query = (
                    "SELECT id, status FROM investigation_runs "
                    "WHERE incident_id = %s AND request_key_hash = %s"
                )
                cursor.execute(query, (incident_id, request_key_hash))
                existing = cursor.fetchone()
                if existing:
                    existing_run_id, existing_status = existing
        except (psycopg2.Error, RuntimeError) as e:
            log.error("failed_to_check_existing_run", error=str(e))
            return JSONResponse(status_code=500, content={"error": "Internal server error"})

        if existing_run_id:
            # Run already exists - return conflict per contract
            log.info(
                "investigation_already_exists",
                incident_id=incident_id,
                run_id=existing_run_id,
            )
            return JSONResponse(
                status_code=409,
                content={
                    "error": "Investigation already exists",
                    "request_id": f"req_{request_key_hash[:26]}",
                    "data": {
                        "run_id": existing_run_id,
                        "incident_id": incident_id,
                        "status": existing_status or "unknown",
                    },
                },
            )

    # Run investigation asynchronously
    log.info("investigation_started", incident_id=incident_id, reason=reason)
    investigator.investigate(incident_id, request_key_hash=request_key_hash)

    run_id = f"irun_{request_key_hash[:26]}"

    return {
        "request_id": f"req_{request_key_hash[:26]}",
        "data": {
            "run_id": run_id,
            "incident_id": incident_id,
            "status": "completed"
        }
    }


@app.get("/status")
async def status():
    """Status endpoint."""
    if investigator is None:
        return JSONResponse(status_code=503, content={"error": "Investigator not initialized"})

    return {
        "running": True,
        "uptime_seconds": time.time() - start_time,
        "version": "1.0.0",
    }


def main():
    """Main entry point for running as a standalone service."""
    import uvicorn

    config = InvestigationConfig.from_env()
    log.info("starting_server", port=config.port)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
