"""Request context middleware for correlation and logging."""
import time
import uuid
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from src.app.core.logging import trace_id as trace_id_var
from src.app.monitoring.tracing import get_current_span, set_span_attributes


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to add correlation ID and request context."""
    
    async def dispatch(self, request: Request, call_next):
        # Check if service is shutting down
        if getattr(request.app.state, "shutting_down", False):
            logger.warning("⚠️  Rejecting request during shutdown")
            return JSONResponse(
                status_code=503,
                content={"detail": "Service is shutting down"},
            )
        
        start_time = time.time()
        
        # Get or generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        
        # Set trace_id in context for logging
        trace_id_var.set(correlation_id)
        
        # Link correlation ID to OpenTelemetry span
        span = get_current_span()
        if span:
            set_span_attributes(
                span,
                correlation_id=correlation_id,
                http_method=request.method,
                http_url=str(request.url),
            )
        
        # Store in request state
        request.state.correlation_id = correlation_id
        request.state.start_time = start_time
        
        logger.info(
            f"➡️  {request.method} {request.url.path}",
            extra={"correlation_id": correlation_id}
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Add span attributes for response
            if span:
                set_span_attributes(
                    span,
                    http_status_code=response.status_code,
                    http_response_time_ms=round(process_time * 1000, 2),
                )
            
            logger.info(
                f"✅ {request.method} {request.url.path} → {response.status_code}",
                extra={
                    "correlation_id": correlation_id,
                    "latency_ms": round(process_time * 1000, 2),
                    "status_code": response.status_code,
                }
            )
            
            # Add headers
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Process-Time"] = f"{process_time:.4f}"
            
            return response
            
        except Exception as e:
            logger.error(
                f"❌ {request.method} {request.url.path} failed: {e}",
                extra={"correlation_id": correlation_id}
            )
            # Record exception in span
            if span:
                from src.app.monitoring.tracing import record_exception
                record_exception(span, e)
            raise
