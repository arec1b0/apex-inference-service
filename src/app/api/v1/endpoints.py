from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger
from src.app.models.schemas import PredictionRequest, PredictionResponse
from src.app.services.model_wrapper import model_service
from src.app.core.config import settings

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    logger.info(f"Received prediction request: {request.id}")
    
    try:
        result = model_service.predict(request.features)
        
        # In a real scenario, we might return probability too if the model supports it
        return PredictionResponse(
            id=request.id,
            prediction=result,
            probability=None, # Update if using predict_proba
            model_version=settings.VERSION
        )
    except Exception as e:
        logger.error(f"Error processing request {request.id}: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")