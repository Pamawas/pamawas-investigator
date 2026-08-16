from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

import logging_middleware
from logging_middleware import LoggingMiddleware, get_logger, get_trace_context


def test_get_trace_context_for_invalid_and_valid_spans():
    with patch(
        "logging_middleware.trace.get_current_span",
        return_value=NonRecordingSpan(SpanContext(0, 0, False)),
    ):
        assert get_trace_context() == {"trace_id": "0" * 32, "span_id": "0" * 16}
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


def test_logging_middleware_propagates_trace_headers():
    with TestClient(make_app(), raise_server_exceptions=False) as client:
        result = client.get("/ok", headers={"x-trace-id": "trace-1"})
    assert result.status_code == 200
    assert result.headers["x-trace-id"] == "trace-1"
    assert len(result.headers["x-span-id"]) == 8


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
