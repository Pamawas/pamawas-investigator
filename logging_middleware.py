"""Structured logging middleware for FastAPI with OpenTelemetry integration."""

import time
import uuid
from typing import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from opentelemetry import trace
from opentelemetry.propagate import extract, inject
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

# Get logger
log = structlog.get_logger()

# Trace context propagator
trace_propagator = TraceContextTextMapPropagator()


def get_trace_context() -> dict[str, str]:
    """Extract trace_id and span_id from current span."""
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        ctx = span.get_span_context()
        return {
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
        }
    return {"trace_id": "0" * 32, "span_id": "0" * 16}


class LoggingMiddleware(BaseHTTPMiddleware):
    """Structured request/response logging with trace context."""
    
    def __init__(self, app: ASGIApp, service_name: str):
        super().__init__(app)
        self.service_name = service_name
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        
        # Extract trace context from incoming headers
        carrier = dict(request.headers)
        ctx = extract(carrier)
        
        # Generate or extract trace/span IDs
        trace_id = carrier.get("x-trace-id") or str(uuid.uuid4())[:8]
        span_id = str(uuid.uuid4())[:8]
        
        # Bind context
        structlog.contextvars.bind_contextvars(
            service=self.service_name,
            component="http",
            trace_id=trace_id,
            span_id=span_id,
        )
        
        # Add trace headers to request state for downstream
        request.state.trace_id = trace_id
        request.state.span_id = span_id
        
        # Log request
        log.info(
            "http_request_started",
            method=request.method,
            path=request.url.path,
            query=str(request.query_params),
            client_host=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        
        # Process request with trace context
        with trace.use_span(trace.get_current_span(ctx), end_on_exit=False):
            try:
                response = await call_next(request)
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                log.error(
                    "http_request_failed",
                    method=request.method,
                    path=request.url.path,
                    error=str(e),
                    duration_ms=round(duration_ms, 2),
                    **get_trace_context(),
                )
                raise
        
        duration_ms = (time.perf_counter() - start) * 1000
        
        # Inject trace context into response headers
        inject(response.headers)
        
        # Log response
        log.info(
            "http_request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            **get_trace_context(),
        )
        
        # Add trace headers to response
        response.headers["x-trace-id"] = trace_id
        response.headers["x-span-id"] = span_id
        
        return response


def get_logger(name: str = None) -> structlog.BoundLogger:
    """Get a structured logger with service context."""
    if name:
        return structlog.get_logger(name)
    return log