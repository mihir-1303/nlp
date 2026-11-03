from pathlib import Path

from app.core.model_loader import ModelLoader


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["models"] == {"nlp_model_loaded": True, "delivery_model_loaded": True}


def test_valid_incident_classification(client):
    response = client.post("/api/incidents/classify", json={"description": "Heavy rainfall caused flooding and delayed transportation."})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_category"] in {"Accident", "Administrative Issue", "Cyber Attack", "Human Error", "Others", "Terrorism", "Weather", "Worker Strike"}
    assert 0 <= body["model_confidence"] <= 1
    assert "not disruption probability" in body["confidence_message"]


def test_empty_incident_description(client):
    response = client.post("/api/incidents/classify", json={"description": "  "})
    assert response.status_code == 422


def test_valid_delivery_prediction(client, delivery_payload):
    response = client.post("/api/delivery/predict", json=delivery_payload)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["estimated_late_delivery_probability"] <= 1
    assert body["prediction"] in {0, 1}
    assert body["warning"] == (body["prediction"] == 1)
    assert body["threshold_used"] == 0.4


def test_invalid_delivery_input(client, delivery_payload):
    del delivery_payload["Market"]
    response = client.post("/api/delivery/predict", json=delivery_payload)
    assert response.status_code == 422


def test_threshold_validation(client, delivery_payload):
    delivery_payload["threshold"] = 1.1
    response = client.post("/api/delivery/predict", json=delivery_payload)
    assert response.status_code == 422


def test_missing_model_handling():
    loader = ModelLoader(Path("does-not-exist-nlp.joblib"), Path("does-not-exist-delivery.joblib"))
    loader.load_all()
    assert loader.status() == {"nlp_model_loaded": False, "delivery_model_loaded": False}


def test_missing_delivery_model_returns_503(client, delivery_payload):
    loader = client.app.state.model_loader
    original_model = loader.delivery_model
    loader.delivery_model = None
    try:
        response = client.post("/api/delivery/predict", json=delivery_payload)
    finally:
        loader.delivery_model = original_model
    assert response.status_code == 503
