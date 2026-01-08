import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from fastapi import Request, Response, JSONResponse
from loguru import logger
from src.app.core.logging import trace_id as trace_id_var, logger as structured_logger

class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        # Check shutdown state
        if getattr(request.app.state, "shutting_down", False):
            structured_logger.warning("⚠️  Rejecting request during shutdown")
            return JSONResponse(
                status_code=503,
                content={"detail": "Service is shutting down"},
            )
        
        start_time = time.time()
        
        # Get or generate Correlation ID
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        
        # Set trace_id in context
        trace_id_var.set(correlation_id)
        request.state.correlation_id = correlation_id
        
        structured_logger.info(f"Request started: {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            structured_logger.info(
                f"Request completed: {response.status_code}",
                extra={"latency_ms": round(process_time * 1000, 2)},
            )
            
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Correlation-ID"] = correlation_id
            return response
        except Exception as e:
            structured_logger.error(f"Request failed: {e}")
            raise