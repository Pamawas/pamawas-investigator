from unittest.mock import MagicMock, patch

from otel import OTelConfig, get_tracer, init_tracer, trace_span


def test_otel_config_uses_environment_and_enablement(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "collector:4317")
    config = OTelConfig(service_name="svc")
    assert config.otlp_endpoint == "collector:4317"
    assert config.enabled is True
    assert OTelConfig(otlp_endpoint="", enabled=True).enabled is False


def test_init_tracer_disabled_returns_none():
    assert init_tracer(OTelConfig(enabled=False)) is None


def test_init_tracer_configures_provider_and_shutdown():
    exporter = MagicMock()
    provider = MagicMock()
    with (
        patch("otel.OTLPSpanExporter", return_value=exporter),
        patch("otel.TracerProvider", return_value=provider),
        patch("otel.BatchSpanProcessor", return_value="processor") as processor,
        patch("otel.trace.set_tracer_provider") as set_provider,
        patch("otel.set_global_textmap") as set_textmap,
    ):
        shutdown = init_tracer(
            OTelConfig(service_name="svc", otlp_endpoint="collector:4317", sample_ratio=0.5)
        )
    assert shutdown is not None
    processor.assert_called_once_with(exporter)
    provider.add_span_processor.assert_called_once_with("processor")
    set_provider.assert_called_once_with(provider)
    set_textmap.assert_called_once()
    shutdown()
    provider.shutdown.assert_called_once_with()


def test_init_tracer_returns_none_on_exporter_error():
    with patch("otel.OTLPSpanExporter", side_effect=RuntimeError("bad endpoint")):
        assert init_tracer(OTelConfig(otlp_endpoint="collector:4317")) is None


def test_get_tracer_delegates():
    tracer = MagicMock()
    with patch("otel.trace.get_tracer", return_value=tracer) as factory:
        assert get_tracer("component") is tracer
        factory.assert_called_once_with("component")


def test_trace_span_sets_attributes_and_yields_span():
    tracer = MagicMock()
    span = MagicMock()
    tracer.start_as_current_span.return_value.__enter__.return_value = span
    with trace_span(tracer, "work", incident="inc-1", attempt=2) as yielded:
        assert yielded is span
    tracer.start_as_current_span.assert_called_once_with("work")
    span.set_attribute.assert_any_call("incident", "inc-1")
    span.set_attribute.assert_any_call("attempt", 2)
