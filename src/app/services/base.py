from abc import ABC, abstractmethod
from typing import Any, List

class ModelService(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load the model artifacts into memory."""
        pass

    @abstractmethod
    def predict(self, input_data: List[float]) -> Any:
        """Make a prediction based on input data."""
        pass