"""Tests de l'API — /health, /predict (valide + invalide), /info."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_payload(client: TestClient, valid_payload: dict) -> None:
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ("yes", "no")
    assert 0.0 <= data["probability"] <= 1.0
    assert "request_id" in data
    assert "model_version" in data


def test_predict_missing_field_returns_422(client: TestClient, valid_payload: dict) -> None:
    invalid = {k: v for k, v in valid_payload.items() if k != "campaign"}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422
    assert "campaign" in response.text


def test_predict_invalid_category_returns_422(client: TestClient, valid_payload: dict) -> None:
    invalid = {**valid_payload, "month": "jan"}  # jan n'a jamais été observé en entraînement
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_out_of_range_returns_422(client: TestClient, valid_payload: dict) -> None:
    invalid = {**valid_payload, "campaign": -1}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_is_deterministic(client: TestClient, valid_payload: dict) -> None:
    """Même entrée -> même sortie, à chaque appel."""
    r1 = client.post("/predict", json=valid_payload).json()
    r2 = client.post("/predict", json=valid_payload).json()
    assert r1["probability"] == r2["probability"]


def test_info_exposes_required_keys(client: TestClient) -> None:
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    required_keys = ["api_version", "model_name", "model_version", "created_at",
                      "sklearn_version", "dataset_sha256", "metrics_holdout"]
    for key in required_keys:
        assert key in data, f"clé manquante : {key}"
        assert data[key] is not None, f"clé nulle : {key}"


def test_request_id_header_present(client: TestClient) -> None:
    """Le middleware ajoute X-Request-ID à toutes les réponses, même /health."""
    response = client.get("/health")
    assert "x-request-id" in response.headers
