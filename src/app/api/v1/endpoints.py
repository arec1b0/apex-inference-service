from fastapi import APIRouter, HTTPException, Request, JSONResponse
from loguru import logger
from slowapi import Limiter
from slowapi.util import get_remote_address
import pybreaker
from src.app.models.schemas import PredictionRequest, PredictionResponse
from src.app.services.model_wrapper import model_service
from src.app.core.config import settings
from src.app.core.limiter import limiter
from src.app.core.circuit_breaker import model_breaker

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
@limiter.limit("100/minute")
async def predict(request: Request, body: PredictionRequest):
    """Predict endpoint with rate limiting and circuit breaker."""
    logger.info(f"Prediction request received", extra={"request_id": body.id})
    
    try:
        # Unpack tuple: (result, probability)
        result, probability = await model_service.predict(body.features)
        
        return PredictionResponse(
            id=body.id,
            prediction=result,
            probability=probability,
            model_version=settings.VERSION
        )
    except pybreaker.CircuitBreakerError:
        # Circuit is open
        logger.warning(f"Circuit breaker open, state={model_breaker.current_state}", extra={"request_id": body.id})
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Model service temporarily unavailable",
                "circuit_state": model_breaker.current_state,
                "retry_after": 60,
            },
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}", extra={"request_id": body.id})
        raise HTTPException(status_code=500, detail="Internal prediction error")