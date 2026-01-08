from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Union

class PredictionRequest(BaseModel):
    id: Optional[str] = Field(default=None, description="Optional request identifier")
    features: List[float] = Field(..., description="List of numerical features for the model")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "123-abc",
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }
    )

class PredictionResponse(BaseModel):
    id: Optional[str]
    prediction: Union[int, float, str]
    probability: Optional[float] = None
    model_version: str

class HealthResponse(BaseModel):
    status: str
    version: str