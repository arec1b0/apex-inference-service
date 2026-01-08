import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from fastapi import Request, Response

# Define Metrics
REQUEST_COUNT = Counter(
    "http_requests_total", 
    "Total number of HTTP requests", 
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", 
    "HTTP request latency in seconds", 
    ["method", "endpoint"]
)

MODEL_PREDICTION_COUNT = Counter(
    "model_prediction_total",
    "Total number of model predictions",
    ["version"]
)

class PrometheusMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            raise
        finally:
            process_time = time.time() - start_time
            
            # Record metrics (skip health checks to reduce noise)
            if "/health" not in request.url.path and "/metrics" not in request.url.path:
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=status_code
                ).inc()
                
                REQUEST_LATENCY.labels(
                    method=request.method,
                    endpoint=request.url.path
                ).observe(process_time)

        return response

def metrics_endpoint():
    """Endpoint for Prometheus scraping."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)