import joblib
import os
import random
from typing import Any, List
from loguru import logger
from src.app.core.config import settings
from src.app.services.base import ModelService

class MLModelService(ModelService):
    def __init__(self):
        self.model = None
        self.version = settings.VERSION

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
                logger.warning(f"Model file not found at {settings.MODEL_PATH}. Using DUMMY model for dev/testing.")
                self.model = "DUMMY"
            else:
                logger.critical(f"Model file missing at {settings.MODEL_PATH} in non-debug mode.")
                raise FileNotFoundError(f"Model not found at {settings.MODEL_PATH}")

    def predict(self, input_data: List[float]) -> Any:
        if self.model is None:
            self.load()
        
        if self.model == "DUMMY":
            # Simulate prediction for testing infrastructure before training
            return 1 if sum(input_data) > 0.5 else 0

        # Scikit-learn prediction
        try:
            # Reshape for single sample prediction: [n_features] -> [1, n_features]
            prediction = self.model.predict([input_data])
            return prediction[0]
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

# Global instance
model_service = MLModelService()