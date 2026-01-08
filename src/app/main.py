import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from loguru import logger

from src.app.core.config import settings
from src.app.core.logging import setup_logging
from src.app.core.middleware import RequestContextMiddleware
from src.app.core.limiter import limiter # Import limiter
from src.app.api import api_router
from src.app.services.model_wrapper import model_service
from src.app.monitoring.metrics import PrometheusMiddleware, metrics_endpoint
from src.app.monitoring.tracing import setup_tracing

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging()
    logger.info("Starting up Apex Inference Service...")
    try:
        model_service.load()
    except Exception:
        pass
    yield
    # Shutdown
    logger.info("Shutdown signal received. Cleanup initiated...")
    await asyncio.sleep(1) 
    logger.info("Cleanup complete.")

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        lifespan=lifespan
    )

    # 1. Rate Limiting State
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # 2. Tracing
    setup_tracing(app)

    # 3. Middleware
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(PrometheusMiddleware)

    # 4. Routes
    app.include_router(api_router, prefix="/api")
    app.add_route("/metrics", metrics_endpoint)

    return app

app = create_app()