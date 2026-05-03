"""Testes para endpoints de monitoramento de drift."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churn_prediction.api.main import app


@pytest.fixture
def client():
    """Cliente de teste FastAPI."""
    return TestClient(app)


def test_drift_check_with_empty_list(client):
    """POST /drift/check com lista vazia deve retornar 400."""
    response = client.post("/drift/check", json=[])
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_drift_check_with_valid_data(client):
    """POST /drift/check com dados válidos retorna recomendação."""
    payload = [
        {
            "age": 35,
            "gender": "male",
            "region": "Sudeste",
            "plan_type": "pos",
            "tenure_months": 24,
        }
    ]
    response = client.post("/drift/check", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "recommendation" in data
    assert data["recommendation"] in ["monitor", "investigate", "retrain", "insufficient_data"]
    assert "drift_ratio" in data
    assert "timestamp" in data


def test_drift_report_endpoint(client):
    """GET /drift/report retorna relatório estruturado."""
    response = client.get("/drift/report?sample_size=50")
    assert response.status_code == 200

    data = response.json()
    assert "timestamp" in data
    assert "total_features" in data
    assert "drift_alerts" in data
    assert "drift_ratio" in data
    assert "features" in data
    assert isinstance(data["features"], dict)
