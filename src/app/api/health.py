from fastapi import APIRouter, Response, status
from src.app.models.schemas import HealthResponse
from src.app.services.model_wrapper import model_service
from src.app.core.config import settings
from src.app.core.circuit_breaker import model_breaker
from prometheus_client import Gauge

router = APIRouter()

SERVICE_READY = Gauge("service_ready", "Service readiness: 1=ready, 0=not ready")

@router.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz():
    """Liveness check: is the service alive?"""
    return {"status": "ok", "version": settings.VERSION}

@router.get("/ready", status_code=status.HTTP_200_OK)
async def ready(response: Response):
    """Readiness check: is the service ready to receive traffic?"""
    checks = {
        "model_loaded": model_service.model is not None,
        "circuit_breaker": model_breaker.current_state == "closed",
    }
    
    is_ready = all(checks.values())
    SERVICE_READY.set(1 if is_ready else 0)
    
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "checks": checks,
            "circuit_state": model_breaker.current_state,
        }
    
    return {"status": "ready", "checks": checks}