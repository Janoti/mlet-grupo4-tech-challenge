"""Testes para integração de métricas Prometheus na API."""

from fastapi.testclient import TestClient

from churn_prediction.api.main import MODEL_STATE, app

client = TestClient(app)


class TestMetricsEndpoint:
    def test_metrics_returns_200(self):
        """Endpoint /metrics deve estar acessível."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type_prometheus(self):
        """/metrics retorna content-type compatível com Prometheus."""
        response = client.get("/metrics")
        content_type = response.headers.get("content-type", "")
        assert "text/plain" in content_type

    def test_metrics_contains_http_metrics(self):
        """Métricas HTTP (requests_total, duration) devem estar presentes."""
        client.get("/health")
        response = client.get("/metrics")
        text = response.text
        assert "churn_api_http_requests_total" in text
        assert "churn_api_http_request_duration_seconds" in text

    def test_metrics_contains_model_gauge(self):
        """Gauge do modelo carregado deve estar presente."""
        response = client.get("/metrics")
        assert "churn_api_model_loaded" in response.text

    def test_metrics_contains_prediction_counter(self):
        """Counter de predições deve estar registrado."""
        response = client.get("/metrics")
        assert "churn_api_predictions_total" in response.text

    def test_metrics_format_prometheus_valid(self):
        """Saída deve seguir formato texto do Prometheus (# HELP, # TYPE)."""
        response = client.get("/metrics")
        lines = response.text.strip().split("\n")
        help_lines = [line for line in lines if line.startswith("# HELP")]
        type_lines = [line for line in lines if line.startswith("# TYPE")]
        assert len(help_lines) > 0, "Deve conter linhas # HELP"
        assert len(type_lines) > 0, "Deve conter linhas # TYPE"

    def test_health_request_recorded_in_metrics(self):
        """Requisição ao /health deve aparecer nas métricas."""
        client.get("/health")
        response = client.get("/metrics")
        assert 'endpoint="/health"' in response.text

    def test_predict_records_risk_level_metric(self):
        """Predição com mock deve registrar métrica por risk_level."""

        class MockPipeline:
            def predict_proba(self, X):
                import numpy as np
                return np.array([[0.3, 0.7]])

            def predict(self, X):
                import numpy as np
                return np.array([1])

        MODEL_STATE["pipeline"] = MockPipeline()
        MODEL_STATE["model_version"] = "mock_test"

        client.post("/predict", json={"age": 45, "gender": "female"})

        response = client.get("/metrics")
        assert 'risk_level="alto"' in response.text

        MODEL_STATE["pipeline"] = None
        MODEL_STATE["model_version"] = None
