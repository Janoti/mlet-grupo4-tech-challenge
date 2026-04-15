"""Testes de validação dos schemas Pydantic e do pipeline de dados."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from churn_prediction.api.schemas import CustomerFeatures, HealthResponse, PredictionResponse
from churn_prediction.data_cleaning import (
    clip_numeric_features,
    create_age_group,
    drop_leakage_columns,
    remove_duplicates,
    standardize_categoricals,
)
from churn_prediction.evaluation import compute_business_metrics, compute_classification_metrics

# ---------------------------------------------------------------------------
# Schemas Pydantic
# ---------------------------------------------------------------------------

class TestCustomerFeatures:
    def test_valid_minimal_input(self):
        """Aceita input com apenas campos opcionais (None)."""
        customer = CustomerFeatures()
        assert customer.age is None

    def test_valid_full_input(self):
        """Aceita input completo com valores válidos."""
        customer = CustomerFeatures(
            age=35, gender="male", region="Sudeste",
            plan_type="pos", monthly_charges=95.50, nps_score=7,
        )
        assert customer.age == 35
        assert customer.gender == "male"

    def test_invalid_age_negative(self):
        """Rejeita idade negativa."""
        with pytest.raises(ValidationError):
            CustomerFeatures(age=-5)

    def test_invalid_nps_out_of_range(self):
        """Rejeita NPS fora do range 0-10."""
        with pytest.raises(ValidationError):
            CustomerFeatures(nps_score=15)


class TestPredictionResponse:
    def test_valid_response(self):
        resp = PredictionResponse(
            churn_probability=0.85,
            churn_prediction=1,
            risk_level="alto",
            model_version="v1",
        )
        assert resp.risk_level == "alto"

    def test_invalid_probability(self):
        with pytest.raises(ValidationError):
            PredictionResponse(
                churn_probability=1.5,
                churn_prediction=1,
                risk_level="alto",
                model_version="v1",
            )


class TestHealthResponse:
    def test_valid(self):
        resp = HealthResponse(status="ok", model_loaded=True, model_version="v1")
        assert resp.model_loaded is True


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

class TestDataCleaning:
    def test_remove_duplicates(self):
        df = pd.DataFrame({
            "customer_id": [1, 1, 2],
            "age": [30, 30, 25],
        })
        result = remove_duplicates(df)
        assert len(result) == 2
        assert result["customer_id"].nunique() == 2

    def test_standardize_gender(self):
        df = pd.DataFrame({"gender": ["Male", " FEMALE ", "", np.nan]})
        result = standardize_categoricals(df)
        assert list(result["gender"]) == ["male", "female", "unknown", "unknown"]

    def test_clip_numeric(self):
        df = pd.DataFrame({"nps_score": [-1, 5, 15], "age": [10, 50, 200]})
        result = clip_numeric_features(df)
        assert result["nps_score"].min() == 0
        assert result["nps_score"].max() == 10
        assert result["age"].min() == 18
        assert result["age"].max() == 100

    def test_create_age_group(self):
        df = pd.DataFrame({"age": [20, 30, 45, 60]})
        result = create_age_group(df)
        assert "age_group" in result.columns
        assert result["age_group"].iloc[0] == "<25"
        assert result["age_group"].iloc[-1] == "55+"

    def test_drop_leakage(self):
        df = pd.DataFrame({
            "customer_id": [1],
            "churn_probability": [0.5],
            "age": [30],
            "churn": [1],
        })
        result = drop_leakage_columns(df)
        assert "customer_id" not in result.columns
        assert "churn_probability" not in result.columns
        assert "age" in result.columns


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_classification_metrics(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.4, 0.2, 0.8])
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_business_metrics(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        bm = compute_business_metrics(y_true, y_pred)
        assert bm["tp"] == 1
        assert bm["fp"] == 1
        assert bm["clientes_abordados"] == 2
        # valor_liquido = 1*500 - 2*50 = 400
        assert bm["valor_liquido"] == 400.0

    def test_business_metrics_all_correct(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        bm = compute_business_metrics(y_true, y_pred)
        assert bm["tp"] == 2
        assert bm["fp"] == 0
        assert bm["valor_liquido"] == 2 * 500 - 2 * 50  # 900
