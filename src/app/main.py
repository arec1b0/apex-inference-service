"""Main FastAPI application."""
import asyncio
import signal
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger

from src.app.core.config import settings
from src.app.core.middleware import RequestContextMiddleware
from src.app.monitoring.metrics import PrometheusMiddleware, metrics_endpoint
from src.app.monitoring.tracing import setup_tracing
from src.app.api import api_router
from src.app.api.health import router as health_router
from src.app.services.model_wrapper import model_service
from src.app.core.limiter import limiter
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from src.app.core.logging import setup_logging

# Global shutdown event
shutdown_event = asyncio.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle: startup and graceful shutdown."""
    logger.info("🚀 Starting Apex Inference Service v{}", settings.VERSION)
    
    # Load model
    model_service.load()
    logger.info("✅ Model loaded successfully")
    
    # Register signal handlers for graceful shutdown
    def shutdown_handler(signum, frame):
        logger.warning(f"⚠️  Received signal {signum}, initiating graceful shutdown...")
        app.state.shutting_down = True
        shutdown_event.set()
    
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
    
    logger.info("✅ Service is ready to accept requests")
    
    yield
    
    # Graceful shutdown sequence
    logger.info("🛑 Shutting down gracefully...")
    await asyncio.sleep(5)  # Allow inflight requests to complete
    logger.info("✅ Shutdown complete")


def create_app() -> FastAPI:
    """Create FastAPI application with all middleware and routes."""
    
    # Setup structured logging
    setup_logging()
    
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        lifespan=lifespan,
    )
    
    # Initialize state
    app.state.shutting_down = False
    
    # Rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Add middleware (order matters: last added = first executed)
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(PrometheusMiddleware)
    
    # Setup distributed tracing
    setup_tracing(app)
    
    # Include routers
    app.include_router(health_router, prefix="/api", tags=["health"])
    app.include_router(api_router, prefix=settings.API_V1_STR, tags=["inference"])
    
    # Prometheus metrics endpoint
    app.add_route("/metrics", metrics_endpoint)
    
    logger.info("📦 Application configured successfully")
    
    return app


app = create_app()
