"""Teste end-to-end: workflow completo predict → drift → feedback → retrain recommendation.

Valida que:
1. Health check funciona
2. Predições são realizadas corretamente
3. Detecção de drift é ativada
4. Recomendações de retreinamento são geradas
5. Feedback é registrado e resumido
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churn_prediction.api.main import MODEL_STATE, app


@pytest.fixture
def client():
    """Cliente de teste FastAPI."""
    return TestClient(app)


@pytest.fixture
def mock_pipeline():
    """Mock de um pipeline sklearn treinado."""

    class MockPipeline:
        """Pipeline mock com predict_proba para testes."""

        def predict_proba(self, X):
            """Retorna probabilidades fake: [~não-churn, ~churn]."""
            import numpy as np

            return np.array([[0.35, 0.65]])

        def predict(self, X):
            """Retorna predição binária."""
            import numpy as np

            return np.array([1])

    return MockPipeline()


@pytest.fixture(autouse=True)
def setup_mock_model(mock_pipeline):
    """Setup: carrega mock do modelo antes de cada teste."""
    MODEL_STATE["pipeline"] = mock_pipeline
    MODEL_STATE["model_version"] = "mock_v1.0"
    yield
    # Cleanup após teste
    MODEL_STATE["pipeline"] = None
    MODEL_STATE["model_version"] = None


class TestFullWorkflow:
    """Testes do workflow completo end-to-end."""

    def test_workflow_step_1_health_check(self, client):
        """Passo 1: Health check retorna status ok."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["model_version"] == "mock_v1.0"

    def test_workflow_step_2_single_prediction(self, client):
        """Passo 2: Predição individual funciona e retorna probabilidade + risco."""
        pred_request = {
            "age": 35,
            "gender": "male",
            "region": "Sudeste",
            "plan_type": "pos",
            "tenure_months": 24,
            "nps_score": 7,
            "monthly_charges": 120.0,
        }
        response = client.post("/predict", json=pred_request)
        assert response.status_code == 200

        prediction = response.json()
        assert "churn_probability" in prediction
        assert "churn_prediction" in prediction
        assert "risk_level" in prediction
        assert "model_version" in prediction

        # Validações de tipos e ranges
        assert 0 <= prediction["churn_probability"] <= 1
        assert prediction["churn_prediction"] in [0, 1]
        assert prediction["risk_level"] in ["baixo", "medio", "alto"]

    def test_workflow_step_3_batch_drift_check(self, client):
        """Passo 3: Detecção de drift em batch de dados de produção."""
        # Simula um batch de dados de produção
        batch_data = [
            {
                "age": 35,
                "gender": "male",
                "region": "Sudeste",
                "plan_type": "pos",
                "tenure_months": 24,
                "nps_score": 7,
                "monthly_charges": 120.0,
            },
            {
                "age": 42,
                "gender": "female",
                "region": "Sul",
                "plan_type": "pre",
                "tenure_months": 36,
                "nps_score": 8,
                "monthly_charges": 95.0,
            },
            {
                "age": 28,
                "gender": "male",
                "region": "Norte",
                "plan_type": "pos",
                "tenure_months": 12,
                "nps_score": 5,
                "monthly_charges": 150.0,
            },
        ]

        response = client.post("/drift/check", json=batch_data)
        assert response.status_code == 200

        drift_result = response.json()
        assert "drift_ratio" in drift_result
        assert "recommendation" in drift_result
        assert "drift_alerts" in drift_result

        # drift_ratio deve estar entre 0 e 1
        assert 0 <= drift_result["drift_ratio"] <= 1

    def test_workflow_step_4_retrain_recommendation(self, client):
        """Passo 4: Recomendação de retreinamento com base em drift."""
        # Simula drift leve
        low_drift_response = client.post(
            "/model/retrain-recommendation",
            params={"drift_ratio": 0.1},
        )
        assert low_drift_response.status_code == 200
        assert "should_retrain" in low_drift_response.json()
        assert "reason" in low_drift_response.json()

        # Simula drift severo
        high_drift_response = client.post(
            "/model/retrain-recommendation",
            params={"drift_ratio": 0.8},
        )
        assert high_drift_response.status_code == 200
        assert "should_retrain" in high_drift_response.json()

    def test_workflow_step_5_feedback_submission(self, client):
        """Passo 5: Submissão de feedback sobre predição."""
        feedback_payload = {
            "prediction_id": "test_pred_001",
            "actual_churn": 1,
            "feedback_type": "correct",
            "comment": "Cliente de fato churned",
            "rating": 5,
        }

        response = client.post("/feedback", json=feedback_payload)
        assert response.status_code == 200

        feedback_response = response.json()
        assert "feedback_id" in feedback_response
        assert "timestamp" in feedback_response
        assert feedback_response["status"] == "received"
        assert "message" in feedback_response

    def test_workflow_step_6_feedback_summary(self, client):
        """Passo 6: Sumário de feedback coletado."""
        # Primeiro submete alguns feedbacks
        for i in range(3):
            client.post("/feedback", json={
                "prediction_id": f"pred_{i}",
                "actual_churn": i % 2,
                "feedback_type": "correct" if i % 2 == 0 else "incorrect",
                "rating": 4,
            })

        # Busca sumário
        summary_response = client.get("/feedback/summary")
        assert summary_response.status_code == 200

        summary = summary_response.json()
        assert "total_feedback" in summary
        assert summary["total_feedback"] >= 0

    def test_workflow_complete_sequence(self, client):
        """Teste completo: sequência integrada de operações."""
        # 1. Health
        health = client.get("/health")
        assert health.status_code == 200

        # 2. Predição
        pred = client.post("/predict", json={
            "age": 45,
            "gender": "female",
            "region": "Sudeste",
            "plan_type": "pos",
            "tenure_months": 24,
            "nps_score": 6,
        })
        assert pred.status_code == 200
        pred_data = pred.json()
        churn_prob = pred_data["churn_probability"]

        # 3. Drift check com batch
        batch = [{"age": 35 + i, "gender": "male", "tenure_months": 20 + i} for i in range(5)]
        drift = client.post("/drift/check", json=batch)
        assert drift.status_code == 200
        drift_data = drift.json()
        drift_ratio = drift_data["drift_ratio"]

        # 4. Retrain recommendation
        retrain = client.post(
            "/model/retrain-recommendation",
            params={"drift_ratio": drift_ratio},
        )
        assert retrain.status_code == 200
        retrain_data = retrain.json()
        assert "should_retrain" in retrain_data

        # 5. Feedback
        feedback = client.post("/feedback", json={
            "prediction_id": f"pred_churn_{churn_prob:.2f}",
            "actual_churn": 1,
            "feedback_type": "correct",
            "rating": 5,
        })
        assert feedback.status_code == 200

        # 6. Summary
        summary = client.get("/feedback/summary")
        assert summary.status_code == 200

        # Todos os passos completados com sucesso
        assert health.status_code == 200
        assert pred.status_code == 200
        assert drift.status_code == 200
        assert retrain.status_code == 200
        assert feedback.status_code == 200
        assert summary.status_code == 200


class TestWorkflowEdgeCases:
    """Testes de casos extremos no workflow."""

    def test_invalid_drift_ratio_parameter(self, client):
        """drift_ratio fora do intervalo [0, 1] retorna 400."""
        response = client.post(
            "/model/retrain-recommendation",
            params={"drift_ratio": 1.5},
        )
        assert response.status_code == 400

        response = client.post(
            "/model/retrain-recommendation",
            params={"drift_ratio": -0.1},
        )
        assert response.status_code == 400

    def test_empty_batch_for_drift_check(self, client):
        """Batch vazio para drift retorna 400."""
        response = client.post("/drift/check", json=[])
        assert response.status_code == 400

    def test_prediction_with_minimal_fields(self, client):
        """Predição com apenas age mínimo."""
        response = client.post("/predict", json={"age": 25})
        assert response.status_code == 200
        assert "churn_probability" in response.json()

    def test_feedback_without_optional_fields(self, client):
        """Feedback com apenas campos obrigatórios."""
        response = client.post("/feedback", json={
            "prediction_id": "min_feedback",
            "feedback_type": "correct",
        })
        assert response.status_code == 200
        assert response.json()["status"] == "received"

    def test_multiple_predictions_sequence(self, client):
        """Múltiplas predições sequenciais."""
        for i in range(5):
            response = client.post("/predict", json={
                "age": 25 + (i * 5),
                "gender": "male" if i % 2 == 0 else "female",
                "tenure_months": 10 + i,
            })
            assert response.status_code == 200
            assert "churn_probability" in response.json()


class TestWorkflowErrorHandling:
    """Testes de tratamento de erros no workflow."""

    def test_predict_with_invalid_schema(self, client):
        """Schema inválido retorna 422."""
        response = client.post("/predict", json={"age": "not_a_number"})
        assert response.status_code == 422

    def test_feedback_with_invalid_rating(self, client):
        """Rating fora do intervalo válido retorna 422."""
        response = client.post("/feedback", json={
            "prediction_id": "test",
            "feedback_type": "correct",
            "rating": 99,
        })
        assert response.status_code == 422

    def test_model_not_loaded_returns_503(self, client):
        """Sem modelo carregado, /predict retorna 503."""
        MODEL_STATE["pipeline"] = None
        response = client.post("/predict", json={"age": 35})
        assert response.status_code == 503
        assert "Modelo não carregado" in response.json()["detail"]


class TestWorkflowMetricsAndLogging:
    """Testes de métricas e logging no workflow."""

    def test_prediction_includes_model_version(self, client):
        """Resposta de predição inclui model_version."""
        response = client.post("/predict", json={"age": 35})
        assert response.status_code == 200
        data = response.json()
        assert "model_version" in data
        assert data["model_version"] == "mock_v1.0"

    def test_health_includes_model_version(self, client):
        """Health check inclui model_version."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "model_version" in data
        assert data["model_version"] == "mock_v1.0"

    def test_drift_check_response_structure(self, client):
        """Resposta de drift check possui estrutura esperada."""
        batch = [{"age": 35, "tenure_months": 24} for _ in range(3)]
        response = client.post("/drift/check", json=batch)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, dict)
        assert "drift_ratio" in data
        assert "recommendation" in data
        assert "drift_alerts" in data

    def test_retrain_recommendation_response_structure(self, client):
        """Recomendação de retrain possui estrutura esperada."""
        response = client.post(
            "/model/retrain-recommendation",
            params={"drift_ratio": 0.5},
        )
        assert response.status_code == 200

        data = response.json()
        assert "should_retrain" in data
        assert "reason" in data
        assert isinstance(data["should_retrain"], bool)
        assert isinstance(data["reason"], str)
