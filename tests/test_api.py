"""Testes de integração da API FastAPI (sem modelo real carregado)."""

from fastapi.testclient import TestClient

from churn_prediction.api.main import MODEL_STATE, app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data

    def test_health_model_not_loaded(self):
        """Sem modelo, model_loaded deve ser False."""
        MODEL_STATE["pipeline"] = None
        MODEL_STATE["model_version"] = None
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is False


class TestRootEndpoint:
    def test_root_returns_info(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "Churn Prediction API"


class TestPredictEndpoint:
    def test_predict_no_model_returns_503(self):
        """Sem modelo carregado, /predict retorna 503."""
        MODEL_STATE["pipeline"] = None
        response = client.post("/predict", json={"age": 35, "gender": "male"})
        assert response.status_code == 503

    def test_predict_with_mock_model(self):
        """Com modelo mock, /predict retorna 200 com schema correto."""

        class MockPipeline:
            def predict_proba(self, X):
                import numpy as np
                return np.array([[0.3, 0.7]])

            def predict(self, X):
                import numpy as np
                return np.array([1])

        MODEL_STATE["pipeline"] = MockPipeline()
        MODEL_STATE["model_version"] = "mock_v1"

        response = client.post("/predict", json={
            "age": 45,
            "gender": "female",
            "region": "Sudeste",
            "plan_type": "pos",
            "monthly_charges": 120.0,
            "nps_score": 3,
        })
        assert response.status_code == 200
        data = response.json()
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "risk_level" in data
        assert data["churn_probability"] == 0.7
        assert data["churn_prediction"] == 1
        assert data["risk_level"] == "alto"

    def test_predict_low_risk(self):
        """Probabilidade baixa → risk_level='baixo'."""

        class MockPipeline:
            def predict_proba(self, X):
                import numpy as np
                return np.array([[0.8, 0.2]])

            def predict(self, X):
                import numpy as np
                return np.array([0])

        MODEL_STATE["pipeline"] = MockPipeline()
        MODEL_STATE["model_version"] = "mock_v1"

        response = client.post("/predict", json={"age": 25})
        data = response.json()
        assert data["risk_level"] == "baixo"
        assert data["churn_prediction"] == 0

    def test_predict_invalid_input(self):
        """Campo com tipo errado retorna 422."""
        MODEL_STATE["pipeline"] = None
        response = client.post("/predict", json={"age": "not_a_number"})
        assert response.status_code == 422

    def test_predict_empty_body(self):
        """Body vazio é aceito (todos campos opcionais), mas sem modelo → 503."""
        MODEL_STATE["pipeline"] = None
        response = client.post("/predict", json={})
        assert response.status_code == 503


class TestLatencyMiddleware:
    def test_health_has_process_time_header(self):
        """LatencyMiddleware injeta X-Process-Time-Ms em /health."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "x-process-time-ms" in response.headers

    def test_root_has_process_time_header(self):
        """LatencyMiddleware injeta X-Process-Time-Ms em /."""
        response = client.get("/")
        assert response.status_code == 200
        assert "x-process-time-ms" in response.headers

    def test_predict_error_response_has_process_time_header(self):
        """LatencyMiddleware injeta X-Process-Time-Ms mesmo em respostas de erro (503)."""
        MODEL_STATE["pipeline"] = None
        response = client.post("/predict", json={"age": 35})
        assert response.status_code == 503
        assert "x-process-time-ms" in response.headers

    def test_process_time_value_is_non_negative_float(self):
        """Valor do header X-Process-Time-Ms é um float >= 0."""
        response = client.get("/health")
        latency = float(response.headers["x-process-time-ms"])
        assert latency >= 0
