"""Testes para endpoints de gerenciamento de modelo."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churn_prediction.api.main import app


@pytest.fixture
def client():
    """Cliente de teste FastAPI."""
    return TestClient(app)


def test_list_model_versions(client):
    """GET /model/versions retorna lista de versões."""
    response = client.get("/model/versions")
    assert response.status_code == 200

    data = response.json()
    assert "total_versions" in data
    assert "versions" in data
    assert isinstance(data["versions"], list)
    assert data["total_versions"] >= 0


def test_retrain_recommendation_no_drift(client):
    """POST /model/retrain-recommendation com drift_ratio=0.05 -> não retrain."""
    response = client.post("/model/retrain-recommendation?drift_ratio=0.05")
    assert response.status_code == 200

    data = response.json()
    assert "should_retrain" in data
    assert isinstance(data["should_retrain"], bool)
    assert "reason" in data


def test_retrain_recommendation_high_drift(client):
    """POST /model/retrain-recommendation com drift_ratio=0.35 -> retrain."""
    response = client.post("/model/retrain-recommendation?drift_ratio=0.35")
    assert response.status_code == 200

    data = response.json()
    assert "should_retrain" in data
    assert isinstance(data["should_retrain"], bool)
    assert "reason" in data


def test_retrain_recommendation_invalid(client):
    """POST /model/retrain-recommendation com drift_ratio > 1 -> 400."""
    response = client.post("/model/retrain-recommendation?drift_ratio=1.5")
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_retrain_recommendation_negative_drift(client):
    """POST /model/retrain-recommendation com drift_ratio < 0 -> 400."""
    response = client.post("/model/retrain-recommendation?drift_ratio=-0.1")
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_retrain_recommendation_boundary_zero(client):
    """POST /model/retrain-recommendation com drift_ratio=0.0 -> 200."""
    response = client.post("/model/retrain-recommendation?drift_ratio=0.0")
    assert response.status_code == 200
    data = response.json()
    assert "should_retrain" in data


def test_retrain_recommendation_boundary_one(client):
    """POST /model/retrain-recommendation com drift_ratio=1.0 -> 200."""
    response = client.post("/model/retrain-recommendation?drift_ratio=1.0")
    assert response.status_code == 200
    data = response.json()
    assert "should_retrain" in data
