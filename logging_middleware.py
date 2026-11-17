"""Structured logging middleware for FastAPI with OpenTelemetry integration."""

import logging
import re
import time
from collections.abc import Callable
from typing import Any

import structlog
from fastapi import Request, Response
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# ---------------------------------------------------------------------------
# Uvicorn access-log → structured fields processor.
#
# Uvicorn's default access handler emits a flat string like:
#   10.0.0.1:12345 - "GET /metrics/ HTTP/1.1" 200
# We intercept that at the structlog `foreign_pre_chain` level and break it
# into discrete fields so Loki can filter by status_code, method, path, etc.
# ---------------------------------------------------------------------------
_ACCESS_RE = re.compile(
    r'^(?P<client_addr>\S+)\s+-\s+"(?P<method>[A-Z]+)\s+(?P<path>\S+)\s+(?P<proto>[^"]+)"\s+(?P<status>\d+)$'
)


def _parse_uvicorn_access(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """If this record came from uvicorn.access, parse the flat line into fields."""
    record: logging.LogRecord | None = event_dict.get("_record")
    if record is None or getattr(record, "name", "") != "uvicorn.access":
        return event_dict

    msg = event_dict.get("event", "")
    m = _ACCESS_RE.match(msg)
    if m:
        event_dict["event"] = "http_access"
        event_dict["client_addr"] = m.group("client_addr")
        event_dict["method"] = m.group("method")
        event_dict["path"] = m.group("path")
        event_dict["proto"] = m.group("proto")
        event_dict["status_code"] = int(m.group("status"))
        event_dict["component"] = "http"
    return event_dict


# ---------------------------------------------------------------------------
# Single shared structlog / stdlib formatter — every logger in the process
# (structlog, stdlib, uvicorn) writes through this.
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

_formatter = structlog.stdlib.ProcessorFormatter(
    # Applied to records that did NOT originate from structlog (stdlib / uvicorn).
    foreign_pre_chain=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Must come after TimeStamper so _record is still on the dict.
        _parse_uvicorn_access,
    ],
    processors=[
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
        structlog.processors.JSONRenderer(),
    ],
)

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_formatter)

_root_logger = logging.getLogger()
_root_logger.handlers.clear()
_root_logger.addHandler(_stream_handler)
_root_logger.setLevel(logging.INFO)


def configure_uvicorn_logging() -> None:
    """Route all uvicorn loggers through the shared JSON handler.

    Call this BEFORE uvicorn.run().  Passing log_config=None to uvicorn.run()
    prevents uvicorn from reinstalling its default plaintext formatters, so
    whatever we set here survives the full server startup.
    """
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi"):
        logger = logging.getLogger(name)
        logger.handlers = [_stream_handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)

# Get logger
log = structlog.get_logger()

# Trace context propagator.
#
# This is used directly rather than through `opentelemetry.propagate`, whose
# global is a no-op until `init_tracer` installs one. Trace propagation is a
# wire contract with the Go services, so it must not depend on whether this
# process has finished initialising: a request that arrived with a traceparent
# continues that trace, or the chain breaks at this hop.
trace_propagator = TraceContextTextMapPropagator()

# Liveness, readiness and scrape endpoints. They run every few seconds forever
# and have no incident behind them.
PROBE_PATHS = frozenset({"/health", "/healthz", "/livez", "/ready", "/readyz", "/metrics"})


def is_probe_path(path: str) -> bool:
    """True for a probe path, with or without a trailing slash.

    FastAPI redirects /metrics to /metrics/, and the redirect target is a second
    request that would otherwise be traced. Normalizing here keeps both halves
    of that exchange out of the trace backend.
    """
    return path.rstrip("/") in PROBE_PATHS or path in PROBE_PATHS


def get_trace_context() -> dict[str, str]:
    """Return the trace and span IDs of the current span, for structured logs.

    Returns empty strings when no valid span is active, so a log line says
    "no trace" instead of carrying an identifier that correlates with nothing.
    """
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        ctx = span.get_span_context()
        return {
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
        }
    return {"trace_id": "", "span_id": ""}


class LoggingMiddleware(BaseHTTPMiddleware):
    """Structured request/response logging with trace context."""

    def __init__(self, app: ASGIApp, service_name: str):
        super().__init__(app)
        self.service_name = service_name

    def _tracer(self):
        """Resolve the tracer per request, not at construction.

        Middleware is added when the app object is built, which is before the
        lifespan runs `init_tracer`. A tracer cached at construction time would
        be bound to the no-op provider and every span would be discarded, which
        is exactly the "instrumented but invisible" state this replaced.
        """
        return trace.get_tracer(self.service_name)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()

        # Probes and scrapes are not requests anyone will debug. Tracing them
        # fills the backend with spans that explain nothing and, at a sample
        # ratio below 1, competes with the traffic that does matter. The Go
        # services make the same exception, so Tempo holds only real work.
        if is_probe_path(request.url.path):
            # Still logged, so "did the probe reach us" stays answerable, but
            # without a span and without a trace ID it does not have.
            response = await call_next(request)
            log.info(
                "http_probe",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
            )
            return response

        # Continue an existing trace when the caller supplied one. Extract
        # before starting the span, or every request starts a second trace and
        # the correlator -> investigator -> reporter chain breaks at this hop.
        parent_ctx = trace_propagator.extract(dict(request.headers))

        span_name = f"{request.method} {request.url.path}"
        with self._tracer().start_as_current_span(
            span_name,
            context=parent_ctx,
            kind=SpanKind.SERVER,
            attributes={
                "http.request.method": request.method,
                "url.path": request.url.path,
                "server.address": request.headers.get("host", ""),
            },
        ) as span:
            trace_id = format(span.get_span_context().trace_id, "032x")
            span_id = format(span.get_span_context().span_id, "016x")

            structlog.contextvars.bind_contextvars(
                service=self.service_name,
                component="http",
                trace_id=trace_id,
                span_id=span_id,
            )
            request.state.trace_id = trace_id
            request.state.span_id = span_id

            log.info(
                "http_request_started",
                method=request.method,
                path=request.url.path,
                query=str(request.query_params),
                client_host=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )

            try:
                response = await call_next(request)

                duration_ms = (time.perf_counter() - start) * 1000
                span.set_attribute("http.response.status_code", response.status_code)
                if response.status_code >= 500:
                    span.set_status(Status(StatusCode.ERROR))

                # Inject trace context into response headers so a caller can
                # link its own records to this span.
                trace_propagator.inject(response.headers)
                response.headers["x-trace-id"] = trace_id
                response.headers["x-span-id"] = span_id

                log.info(
                    "http_request_completed",
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    duration_ms=round(duration_ms, 2),
                )

                return response
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                log.error(
                    "http_request_failed",
                    method=request.method,
                    path=request.url.path,
                    error=str(e),
                    duration_ms=round(duration_ms, 2),
                )
                raise
            finally:
                # Bindings are per-request; clearing them keeps a finished
                # request's trace ID off anything logged outside its span.
                structlog.contextvars.clear_contextvars()


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a structured logger with service context."""
    if name:
        return structlog.get_logger(name)
    return log
