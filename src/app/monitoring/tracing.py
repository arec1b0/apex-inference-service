import os
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from fastapi import FastAPI
from src.app.core.config import settings

def setup_tracing(app: FastAPI):
    """Configure OpenTelemetry Tracing."""
    
    # Check if tracing is disabled (e.g. during tests)
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    
    if otlp_endpoint.lower() == "none":
        return

    resource = Resource.create(attributes={
        "service.name": settings.PROJECT_NAME,
        "service.version": settings.VERSION,
        "deployment.environment": settings.ENV
    })

    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    
    # Configure Exporter
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)

    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)