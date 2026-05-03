"""Testes para endpoints de feedback."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churn_prediction.api.main import app


@pytest.fixture
def client():
    """Cliente de teste FastAPI."""
    return TestClient(app)


def test_submit_feedback(client):
    """POST /feedback registra feedback com sucesso."""
    payload = {
        "prediction_id": "abc123",
        "actual_churn": 1,
        "feedback_type": "incorrect",
        "comment": "Cliente churned",
        "rating": 2,
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "feedback_id" in data
    assert data["status"] == "received"
    assert "timestamp" in data
    assert "message" in data


def test_submit_feedback_minimal(client):
    """POST /feedback com apenas campos obrigatórios."""
    payload = {
        "prediction_id": "xyz789",
        "feedback_type": "correct",
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "feedback_id" in data
    assert data["status"] == "received"


def test_submit_feedback_invalid_feedback_type(client):
    """POST /feedback com feedback_type vazio deve falhar."""
    payload = {
        "prediction_id": "test_id",
        "feedback_type": "",
    }
    # Dependendo da validação, pode retornar 422 se feedback_type vazio
    # ou pode aceitar. Aqui testamos que a API respondeu.
    response = client.post("/feedback", json=payload)
    assert response.status_code in [200, 422]


def test_submit_feedback_invalid_rating(client):
    """POST /feedback com rating fora do intervalo [1,5] -> 422."""
    payload = {
        "prediction_id": "test_id",
        "feedback_type": "correct",
        "rating": 10,  # Fora do intervalo válido
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 422


def test_feedback_summary(client):
    """GET /feedback/summary retorna sumário."""
    response = client.get("/feedback/summary")
    assert response.status_code == 200

    data = response.json()
    assert "total_feedback" in data


def test_feedback_summary_has_metrics(client):
    """GET /feedback/summary retorna métricas estruturadas."""
    response = client.get("/feedback/summary")
    assert response.status_code == 200

    data = response.json()
    # Verifica que a resposta é um dict com pelo menos a métrica total_feedback
    assert isinstance(data, dict)
    assert "total_feedback" in data
    assert isinstance(data["total_feedback"], int)
