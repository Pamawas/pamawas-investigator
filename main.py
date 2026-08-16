#!/usr/bin/env python3
"""Main entry point for the Pamawas Investigator service."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from config import Config
from metrics import set_running
from service.investigator import PamawasInvestigator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global investigator instance
investigator: PamawasInvestigator = None
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global investigator

    # Startup
    logger.info("Starting Pamawas Investigator")
    try:
        config = Config.from_env()
        logger.info(f"Loaded config: LLM={config.llm_model} via {config.llm_base_url}")

        investigator = PamawasInvestigator(config)
        set_running(True)
        logger.info("Investigator service is ready to process incidents")
    except Exception as e:
        logger.error(f"Failed to initialize investigator: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Pamawas Investigator")
    set_running(False)


app = FastAPI(
    title="Pamawas Investigator",
    description="Bounded tool-calling LLM investigation engine for infrastructure incidents",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/healthz")
async def healthz():
    """Health check endpoint."""
    if investigator is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "Investigator not initialized"}
        )

    # Check database connectivity
    if investigator.db_conn is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "Database not connected"}
        )

    try:
        with investigator.db_conn.cursor() as cursor:
            cursor.execute("SELECT 1")
        return {"status": "healthy", "timestamp": time.time()}
    except Exception as e:  # noqa: BLE001
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/ready")
async def ready():
    """Readiness check endpoint."""
    if investigator is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "error": "Investigator not initialized"}
        )

    # Check database connectivity
    if investigator.db_conn is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "error": "Database not ready"}
        )

    try:
        with investigator.db_conn.cursor() as cursor:
            cursor.execute("SELECT 1")
        return {"status": "ready"}
    except Exception as e:  # noqa: BLE001
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "error": str(e)}
        )


@app.post("/investigate")
async def investigate(incident_id: str):
    """Trigger investigation for an incident."""
    if investigator is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Investigator not initialized"}
        )

    logger.info(f"Starting investigation for incident {incident_id}")
    findings = investigator.investigate(incident_id)

    return {
        "incident_id": incident_id,
        "findings": [f.to_dict() for f in findings],
        "completed": True
    }


@app.get("/status")
async def status():
    """Status endpoint."""
    if investigator is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Investigator not initialized"}
        )

    return {
        "running": True,
        "uptime_seconds": time.time() - start_time,
        "version": "1.0.0",
    }


def main():
    """Main entry point for running as a standalone service."""
    import uvicorn

    config = Config.from_env()
    logger.info(f"Starting Pamawas Investigator on port {config.port}")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
