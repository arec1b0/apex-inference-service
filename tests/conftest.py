import os
import pytest
from fastapi.testclient import TestClient

# Disable OTLP export during tests to prevent connection errors
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "none"

# Import app AFTER setting the environment variable
from src.app.main import app

@pytest.fixture(scope="module")
def client():
    # Context manager triggers the lifespan events (startup/shutdown)
    with TestClient(app) as c:
        yield c