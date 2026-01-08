"""Inference endpoints with distributed tracing."""
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
import pybreaker

from src.app.models.schemas import PredictionRequest, PredictionResponse
from src.app.services.model_wrapper import model_service
from src.app.core.limiter import limiter
from src.app.core.circuit_breaker import model_breaker
from src.app.monitoring.tracing import tracer, set_span_attributes, record_exception

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
@limiter.limit("100/minute")
async def predict(request: Request, body: PredictionRequest):
    """
    Predict endpoint with rate limiting, circuit breaker, and tracing.
    
    Returns prediction with confidence score.
    """
    # Create manual span for endpoint logic
    with tracer.start_as_current_span("predict_endpoint") as span:
        try:
            # Add input metadata to span
            set_span_attributes(
                span,
                request_id=body.id or "unknown",
                input_feature_count=len(body.features),
                input_features_sample=str(body.features[:3]),  # First 3 features
            )
            
            # Call model service (will create nested span)
            result, confidence = await model_service.predict(body.features)
            
            # Add output metadata to span
            set_span_attributes(
                span,
                prediction=str(result),
                confidence=confidence or 0.0,
                model_version="v1",
            )
            
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
            
        except pybreaker.CircuitBreakerError as e:
            # Circuit is open - service degraded
            logger.warning(
                f"Circuit breaker OPEN, returning 503",
                extra={"circuit_state": model_breaker.current_state}
            )
            
            # Record in span
            span.add_event(
                "circuit_breaker_open",
                attributes={
                    "circuit_state": model_breaker.current_state,
                    "failure_count": model_breaker.fail_counter,
                }
            )
            record_exception(span, e)
            
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
            record_exception(span, e)
            raise HTTPException(
                status_code=500,
                detail=f"Prediction error: {str(e)}"
            )
