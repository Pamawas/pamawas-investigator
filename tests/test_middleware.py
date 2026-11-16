from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

import logging_middleware
from logging_middleware import LoggingMiddleware, get_logger, get_trace_context

TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


@pytest.fixture(autouse=True)
def real_tracer():
    """Give the middleware a real provider.

    The default global provider is a no-op, so without this every span would be
    non-recording and the assertions below could not tell a working tracer from
    a broken one.
    """
    provider = TracerProvider()
    with patch.object(logging_middleware.trace, "get_tracer", provider.get_tracer):
        yield


def test_get_trace_context_reports_no_trace_without_a_span():
    """A log line must not carry an identifier that correlates with nothing."""
    with patch(
        "logging_middleware.trace.get_current_span",
        return_value=NonRecordingSpan(SpanContext(0, 0, False)),
    ):
        assert get_trace_context() == {"trace_id": "", "span_id": ""}


def test_get_trace_context_reports_the_real_span_ids():
    context = SpanContext(0x123, 0x456, False, TraceFlags(1))
    with patch("logging_middleware.trace.get_current_span", return_value=NonRecordingSpan(context)):
        assert get_trace_context() == {"trace_id": f"{0x123:032x}", "span_id": f"{0x456:016x}"}


def make_app():
    app = FastAPI()
    app.add_middleware(LoggingMiddleware, service_name="test-service")

    @app.get("/ok")
    async def ok():
        return {"ok": True}

    @app.get("/fail")
    async def fail():
        raise RuntimeError("boom")

    return app


def test_logging_middleware_continues_the_callers_trace():
    """An inbound traceparent must be adopted, not replaced.

    The middleware used to mint its own eight-character IDs and ignore the
    caller entirely, which broke the chain at this hop: correlator started a
    trace, the investigator started a different one, and Tempo showed two
    unrelated single-service traces instead of one incident's path.
    """
    with TestClient(make_app(), raise_server_exceptions=False) as client:
        result = client.get("/ok", headers={"traceparent": TRACEPARENT})

    assert result.status_code == 200
    assert result.headers["x-trace-id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    # The span is this service's own child, so it differs from the parent's.
    assert result.headers["x-span-id"] != "00f067aa0ba902b7"
    assert len(result.headers["x-span-id"]) == 16


def test_logging_middleware_starts_a_trace_when_the_caller_sent_none():
    with TestClient(make_app(), raise_server_exceptions=False) as client:
        result = client.get("/ok")

    assert result.status_code == 200
    trace_id = result.headers["x-trace-id"]
    assert len(trace_id) == 32 and trace_id != "0" * 32


def test_logging_middleware_logs_and_reraises_errors():
    logger = MagicMock()
    with (
        patch.object(logging_middleware, "log", logger),
        TestClient(make_app(), raise_server_exceptions=False) as client,
    ):
        result = client.get("/fail")
    assert result.status_code == 500
    assert logger.error.call_args.args[0] == "http_request_failed"
    assert logger.error.call_args.kwargs["error"] == "boom"


def test_get_logger_named_and_default():
    named = SimpleNamespace()
    with patch("logging_middleware.structlog.get_logger", return_value=named) as factory:
        assert get_logger("component") is named
        factory.assert_called_once_with("component")
    assert get_logger() is logging_middleware.log
