from fastapi import APIRouter
from src.app.api.v1.endpoints import router as v1_router
from src.app.api.health import router as health_router

api_router = APIRouter()

api_router.include_router(health_router, tags=["health"])
api_router.include_router(v1_router, prefix="/v1", tags=["prediction"])