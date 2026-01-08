"""Health check endpoints."""
from fastapi import APIRouter, Response, status
from prometheus_client import Gauge
from loguru import logger

from src.app.core.config import settings
from src.app.services.model_wrapper import model_service
from src.app.core.circuit_breaker import model_breaker

router = APIRouter()

SERVICE_READY = Gauge("service_ready", "Service readiness: 1=ready, 0=not ready")


@router.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz():
    """Liveness probe: is the process alive?"""
    return {
        "status": "ok",
        "version": settings.VERSION,
        "service": settings.PROJECT_NAME,
    }


@router.get("/ready", status_code=status.HTTP_200_OK)
async def ready(response: Response):
    """
    Readiness probe: is the service ready to receive traffic?
    
    Checks:
    - Model loaded
    - Circuit breaker not open
    """
    checks = {
        "model_loaded": model_service.model is not None,
        "circuit_breaker_closed": model_breaker.current_state == "closed",
    }
    
    is_ready = all(checks.values())
    SERVICE_READY.set(1 if is_ready else 0)
    
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        logger.warning(f"Service NOT READY: {checks}")
        return {
            "status": "not_ready",
            "checks": checks,
            "circuit_state": model_breaker.current_state,
        }
    
    return {
        "status": "ready",
        "checks": checks,
    }
