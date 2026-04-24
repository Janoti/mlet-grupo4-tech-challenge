"""Testes do módulo de explainability (SHAP top-N).

Os testes verificam contrato, erros e importabilidade. Cálculo real de
SHAP é testado apenas quando a lib está instalada (dep opcional).
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from churn_prediction.explainability import (
    compute_shap_top_features,
    compute_shap_values,
)


def _shap_available() -> bool:
    try:
        importlib.import_module("shap")
        return True
    except ImportError:
        return False


class TestImportContract:
    def test_functions_importable(self):
        assert callable(compute_shap_values)
        assert callable(compute_shap_top_features)

    def test_helpful_error_when_shap_missing(self, monkeypatch):
        """Quando shap não está disponível, erro tem mensagem orientativa."""
        import churn_prediction.explainability as mod

        def fake_import(*args, **kwargs):
            raise ImportError("No module named 'shap'")

        monkeypatch.setattr("builtins.__import__", fake_import)
        with pytest.raises(ImportError, match="poetry install --extras explainability"):
            mod._require_shap()


@pytest.mark.skipif(not _shap_available(), reason="shap not installed (optional dep)")
class TestSHAPWithLib:
    @pytest.fixture
    def trained_logreg(self):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(200, 5))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = LogisticRegression().fit(X, y)
        return model, X

    def test_returns_dataframe_with_expected_columns(self, trained_logreg):
        model, X = trained_logreg
        X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
        top = compute_shap_top_features(model, X_df, n_top=3, max_samples=100)
        assert set(top.columns) == {"feature", "mean_abs_shap", "mean_shap", "rank"}
        assert len(top) == 3

    def test_top_features_ranked_correctly(self, trained_logreg):
        model, X = trained_logreg
        X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
        top = compute_shap_top_features(model, X_df, n_top=5, max_samples=100)
        # As primeiras duas features geraram y → devem estar no topo
        top_features = set(top.head(2)["feature"])
        assert top_features.issubset({"feat_0", "feat_1"})
        # Rank monotônico decrescente em importância
        assert top["mean_abs_shap"].is_monotonic_decreasing

    def test_shap_values_shape(self, trained_logreg):
        model, X = trained_logreg
        shap_vals, X_sample = compute_shap_values(model, X, max_samples=50)
        assert shap_vals.shape == (50, 5)
        assert len(X_sample) == 50
