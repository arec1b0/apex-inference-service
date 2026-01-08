import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "apex-inference-service"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    # Environment
    ENV: str = "dev"  # dev, staging, prod
    DEBUG: bool = False
    
    # Model Configuration
    MODEL_PATH: str = "model_store/model.pkl"
    MODEL_TYPE: str = "sklearn" # sklearn or router
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True
    )

settings = Settings()