"""OpenTelemetry initialization for pamawas-investigator."""

import logging
import os
from collections.abc import Callable
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.resources import (
    DEPLOYMENT_ENVIRONMENT,
    SERVICE_NAME,
    SERVICE_VERSION,
    Resource,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)


class OTelConfig:
    """OpenTelemetry configuration."""

    def __init__(
        self,
        service_name: str = "pamawas-investigator",
        otlp_endpoint: str | None = None,
        insecure: bool = True,
        sample_ratio: float = 1.0,
        enabled: bool = True,
    ):
        self.service_name = service_name
        self.otlp_endpoint = (
            os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") if otlp_endpoint is None else otlp_endpoint
        )
        self.insecure = insecure
        self.sample_ratio = sample_ratio
        self.enabled = enabled and bool(self.otlp_endpoint)


def init_tracer(config: OTelConfig) -> Callable | None:
    """Initialize OpenTelemetry tracer provider with OTLP gRPC exporter.

    Returns a shutdown function if tracing is enabled, None otherwise.
    """
    if not config.enabled:
        logger.info("[otel] tracing disabled for %s", config.service_name)
        return None

    try:
        # Create OTLP gRPC exporter
        exporter = OTLPSpanExporter(
            endpoint=config.otlp_endpoint,
            insecure=config.insecure,
        )

        # Create resource with service info
        resource = Resource.create(
            {
                SERVICE_NAME: config.service_name,
                SERVICE_VERSION: "1.0.0",
                DEPLOYMENT_ENVIRONMENT: "development",
            }
        )

        # Create trace provider with batch span processor
        provider = TracerProvider(
            resource=resource,
        )

        # Add batch span processor
        provider.add_span_processor(BatchSpanProcessor(exporter))

        # Set sampler
        sampler = ParentBased(root=TraceIdRatioBased(config.sample_ratio))
        provider._sampler = sampler  # type: ignore

        # Set global tracer provider
        trace.set_tracer_provider(provider)

        # Set global propagator for W3C trace context
        set_global_textmap(
            CompositePropagator(
                [
                    TraceContextTextMapPropagator(),
                    W3CBaggagePropagator(),
                ]
            )
        )

        logger.info(
            "[otel] tracer initialized for %s, exporting to %s",
            config.service_name,
            config.otlp_endpoint,
        )

        # Return shutdown function
        def shutdown():
            provider.shutdown()

        return shutdown

    except Exception as e:  # noqa: BLE001 - tracing must not prevent service startup
        logger.error("[otel] failed to initialize tracer: %s", e)
        return None


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer for the given name."""
    return trace.get_tracer(name)


@contextmanager
def trace_span(tracer: trace.Tracer, name: str, **attributes):
    """Context manager for creating a traced span."""
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield span
