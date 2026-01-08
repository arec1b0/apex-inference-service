from fastapi import APIRouter, Response, status
from src.app.models.schemas import HealthResponse
from src.app.services.model_wrapper import model_service
from src.app.core.config import settings

router = APIRouter()

@router.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz():
    """Liveness probe: Service is running."""
    return {"status": "ok"}

@router.get("/ready", status_code=status.HTTP_200_OK)
async def ready(response: Response):
    """Readiness probe: Model is loaded and ready to serve."""
    if model_service.model is None:
        try:
            model_service.load()
        except Exception:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "not ready", "reason": "model not loaded"}
            
    return HealthResponse(status="ready", version=settings.VERSION)