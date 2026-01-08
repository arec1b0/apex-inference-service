"""OpenTelemetry distributed tracing setup."""
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.trace import Status, StatusCode
from fastapi import FastAPI
from loguru import logger

from src.app.core.config import settings


def setup_tracing(app: FastAPI):
    """
    Setup OpenTelemetry distributed tracing.
    
    Exports traces to OTLP endpoint (Jaeger).
    """
    otlp_endpoint = os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT", 
        "http://localhost:4317"
    )
    
    # Create resource with service metadata
    resource = Resource.create(
        attributes={
            "service.name": settings.PROJECT_NAME,
            "service.version": settings.VERSION,
            "deployment.environment": settings.ENV,
        }
    )
    
    # Setup tracer provider
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    
    # Add OTLP exporter
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)
    
    # Auto-instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)
    
    logger.info(f"✅ Tracing configured: exporting to {otlp_endpoint}")


# Global tracer instance
tracer = trace.get_tracer(__name__, settings.VERSION)


def get_current_span():
    """Get current active span."""
    return trace.get_current_span()


def set_span_attributes(span, **attributes):
    """Set multiple attributes on a span."""
    if span and span.is_recording():
        for key, value in attributes.items():
            span.set_attribute(key, value)


def record_exception(span, exception: Exception):
    """Record exception in span."""
    if span and span.is_recording():
        span.record_exception(exception)
        span.set_status(Status(StatusCode.ERROR, str(exception)))
