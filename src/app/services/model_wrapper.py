"""Model wrapper service with circuit breaker and tracing."""
import asyncio
from typing import Any, Optional, Tuple, List
from loguru import logger
import numpy as np
import pickle
import os
import pybreaker

from src.app.services.base import ModelService
from src.app.core.config import settings
from src.app.core.circuit_breaker import model_breaker
from src.app.monitoring.metrics import MODEL_PREDICTION_COUNT, MODEL_CONFIDENCE
from src.app.monitoring.tracing import tracer, set_span_attributes, record_exception


class MLModelService(ModelService):
    """ML Model wrapper with async prediction, circuit breaker, and tracing."""
    
    def __init__(self):
        self.model = None
        self.model_version = "v1"
    
    def load(self) -> None:
        """Load model from disk."""
        if not os.path.exists(settings.MODEL_PATH):
            logger.warning(f"⚠️  Model file not found: {settings.MODEL_PATH}, using dummy model")
            self.model = None
            return
        
        try:
            with open(settings.MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"✅ Model loaded from {settings.MODEL_PATH}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.model = None
    
    async def predict(self, input_data: List[float]) -> Tuple[Any, Optional[float]]:
        """
        Async prediction with circuit breaker protection and tracing.
        Returns (prediction, confidence).
        """
        # Create span for model service call
        with tracer.start_as_current_span("model_service.predict") as span:
            try:
                set_span_attributes(
                    span,
                    model_version=self.model_version,
                    model_type="sklearn" if self.model else "dummy",
                    input_size=len(input_data),
                )
                
                # Call sync predict via circuit breaker in thread
                prediction, confidence = await asyncio.to_thread(
                    model_breaker.call, self._predict_sync, input_data
                )
                
                # Record success
                set_span_attributes(
                    span,
                    prediction=str(prediction),
                    confidence=confidence or 0.0,
                )
                
                return prediction, confidence
                
            except pybreaker.CircuitBreakerError as e:
                logger.error(f"⚠️  Circuit breaker OPEN: {e}")
                
                # Record circuit breaker event
                span.add_event(
                    "circuit_breaker_triggered",
                    attributes={
                        "state": model_breaker.current_state,
                        "fail_counter": model_breaker.fail_counter,
                    }
                )
                
                # Return fallback prediction
                MODEL_PREDICTION_COUNT.labels(
                    model_version=self.model_version,
                    prediction="fallback",
                ).inc()
                
                return 0, 0.1  # Fallback with low confidence
                
            except Exception as e:
                logger.error(f"❌ Prediction error: {e}")
                record_exception(span, e)
                raise
    
    def _predict_sync(self, input_data: List[float]) -> Tuple[Any, Optional[float]]:
        """
        Synchronous prediction logic (called by circuit breaker).
        This runs in a separate thread pool.
        """
        # Create nested span for actual inference
        with tracer.start_as_current_span("model_inference") as span:
            prediction = None
            confidence = None
            
            try:
                set_span_attributes(
                    span,
                    inference_type="dummy" if self.model is None else "sklearn",
                )
                
                # 1. Dummy model if no model loaded
                if self.model is None:
                    score = sum(input_data)
                    prediction = 1 if score > 0.5 else 0
                    confidence = min(max(abs(score - 0.5) + 0.5, 0.5), 1.0)
                    
                    span.add_event("using_dummy_model")
                    
                else:
                    # 2. Real sklearn model
                    pred_array = self.model.predict([input_data])
                    prediction = int(pred_array[0])
                    
                    # Get probability if available
                    if hasattr(self.model, "predict_proba"):
                        proba = self.model.predict_proba([input_data])
                        confidence = float(np.max(proba))
                    else:
                        confidence = 1.0
                
                # 3. Record metrics
                pred_label = str(prediction)
                MODEL_PREDICTION_COUNT.labels(
                    model_version=self.model_version,
                    prediction=pred_label,
                ).inc()
                
                if confidence is not None:
                    MODEL_CONFIDENCE.observe(confidence)
                
                # Add output to span
                set_span_attributes(
                    span,
                    output_prediction=pred_label,
                    output_confidence=confidence or 0.0,
                )
                
                return prediction, confidence
                
            except Exception as e:
                logger.error(f"Model inference failed: {e}")
                record_exception(span, e)
                raise


# Global instance
model_service = MLModelService()
