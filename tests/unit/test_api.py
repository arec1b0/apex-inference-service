from fastapi.testclient import TestClient

def test_healthz(client: TestClient):
    response = client.get("/api/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint_success(client: TestClient):
    payload = {
        "id": "test-req-1",
        "features": [0.5, 0.2, 0.1, 0.9]
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "test-req-1"
    assert "prediction" in data
    assert "model_version" in data

def test_predict_endpoint_validation_error(client: TestClient):
    # Missing features
    payload = {"id": "bad-req"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422