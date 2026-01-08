import joblib
import os
import asyncio
import numpy as np
import pybreaker
from typing import Any, List, Tuple, Optional
from loguru import logger
from src.app.core.config import settings
from src.app.services.base import ModelService
from src.app.monitoring.metrics import MODEL_PREDICTION_COUNT, MODEL_CONFIDENCE
from src.app.core.circuit_breaker import model_breaker

class MLModelService(ModelService):
    def __init__(self):
        self.model = None
        self.version = settings.VERSION
        self.semaphore = asyncio.Semaphore(5)

    def load(self) -> None:
        logger.info(f"Loading model from {settings.MODEL_PATH}...")
        if os.path.exists(settings.MODEL_PATH):
            try:
                self.model = joblib.load(settings.MODEL_PATH)
                logger.info("Model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise RuntimeError(f"Could not load model at {settings.MODEL_PATH}")
        else:
            if settings.DEBUG:
                logger.warning(f"Model file not found at {settings.MODEL_PATH}. Using DUMMY model.")
                self.model = "DUMMY"
            else:
                logger.critical(f"Model file missing at {settings.MODEL_PATH}.")
                raise FileNotFoundError(f"Model not found at {settings.MODEL_PATH}")

    async def predict(self, input_data: List[float]) -> Tuple[Any, Optional[float]]:
        """Returns (prediction, confidence) with circuit breaker protection."""
        async with self.semaphore:
            if self.model is None:
                self.load()
            
            try:
                # Wrap in circuit breaker
                prediction, confidence = await asyncio.to_thread(
                    model_breaker.call, self._predict_sync, input_data
                )
                return prediction, confidence
            except pybreaker.CircuitBreakerError:
                logger.error("⚠️  Circuit breaker is OPEN, returning fallback")
                # Record fallback metric
                MODEL_PREDICTION_COUNT.labels(
                    version=self.version,
                    predicted_class="fallback",
                ).inc()
                # Return fallback prediction
                return 0, 0.1  # Low confidence fallback

    def _predict_sync(self, input_data: List[float]) -> Tuple[Any, Optional[float]]:
        prediction = None
        confidence = None
        
        # 1. Dummy Model Logic
        if self.model == "DUMMY":
            score = sum(input_data)
            prediction = 1 if score > 0.5 else 0
            # Fake confidence for dummy model
            confidence = min(max(abs(score - 0.5) + 0.5, 0.5), 1.0) 
        
        # 2. Sklearn Model Logic
        else:
            try:
                # Get the class prediction
                pred_array = self.model.predict([input_data])
                prediction = pred_array[0]
                
                # Get confidence (probability) if available
                if hasattr(self.model, "predict_proba"):
                    proba = self.model.predict_proba([input_data])
                    # Take the max probability as "confidence"
                    confidence = float(np.max(proba))
                else:
                    confidence = 1.0 # Fallback if model doesn't support probabilities
                    
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise

        # 3. Record Metrics
        try:
            # Cast prediction to string for label
            pred_label = str(prediction)
            
            MODEL_PREDICTION_COUNT.labels(
                version=self.version, 
                predicted_class=pred_label
            ).inc()
            
            if confidence is not None:
                MODEL_CONFIDENCE.labels(
                    version=self.version
                ).observe(confidence)
                
        except Exception as e:
            logger.error(f"Error recording metrics: {e}")
            # Don't fail the request just because metrics failed

        return prediction, confidence

model_service = MLModelService()