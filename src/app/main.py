from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.app.core.config import settings
from src.app.core.logging import setup_logging
from src.app.core.middleware import RequestContextMiddleware
from src.app.api import api_router
from src.app.services.model_wrapper import model_service
from src.app.monitoring.metrics import PrometheusMiddleware, metrics_endpoint
from src.app.monitoring.tracing import setup_tracing

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging()
    
    # Pre-load model
    try:
        model_service.load()
    except Exception:
        pass
        
    yield
    # Shutdown logic here

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        lifespan=lifespan
    )

    # 1. Tracing (Instrument first to capture everything)
    setup_tracing(app)

    # 2. Middleware
    app.add_middleware(RequestContextMiddleware) # Correlation IDs
    app.add_middleware(PrometheusMiddleware)     # Metrics

    # 3. Routes
    app.include_router(api_router, prefix="/api")
    
    # 4. Metrics Endpoint (Standard /metrics path for Prometheus)
    app.add_route("/metrics", metrics_endpoint)

    return app

app = create_app()