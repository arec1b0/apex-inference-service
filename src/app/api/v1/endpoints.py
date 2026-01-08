"""Inference endpoints."""
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
import pybreaker

from src.app.models.schemas import PredictionRequest, PredictionResponse
from src.app.services.model_wrapper import model_service
from src.app.core.limiter import limiter
from src.app.core.circuit_breaker import model_breaker

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
@limiter.limit("100/minute")
async def predict(request: Request, body: PredictionRequest):
    """
    Predict endpoint with rate limiting and circuit breaker protection.
    
    Returns prediction with confidence score.
    """
    try:
        # Call model service (includes circuit breaker)
        result, confidence = await model_service.predict(body.features)
        
        logger.info(
            f"Prediction successful: {result}",
            extra={
                "prediction": result,
                "confidence": confidence,
                "input_length": len(body.features),
            }
        )
        
        return PredictionResponse(
            id=body.id,
            prediction=result,
            probability=confidence,
            model_version="v1",
        )
        
    except pybreaker.CircuitBreakerError:
        # Circuit is open - service degraded
        logger.warning(
            f"Circuit breaker OPEN, returning 503",
            extra={"circuit_state": model_breaker.current_state}
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Model service temporarily unavailable",
                "circuit_state": model_breaker.current_state,
                "retry_after": 60,
            },
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
