"""Testes para bootstrap CI 95% em métricas técnicas."""

import numpy as np
import pytest

from churn_prediction.evaluation import (
    bootstrap_confidence_interval,
    compute_metrics_with_ci,
)


@pytest.fixture
def synthetic_predictions():
    """Dataset sintético com classes balanceadas e predições realistas."""
    rng = np.random.default_rng(42)
    n = 500
    y_true = rng.integers(0, 2, size=n)
    # Probabilidades correlacionadas com y_true (modelo "razoável")
    y_prob = np.clip(y_true * 0.7 + rng.normal(0.15, 0.2, size=n), 0.01, 0.99)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


class TestBootstrapCI:
    def test_returns_expected_keys(self, synthetic_predictions):
        y_true, y_pred, y_prob = synthetic_predictions
        result = bootstrap_confidence_interval(
            y_true, y_pred, y_prob, metric="roc_auc", n_bootstrap=200
        )
        for key in (
            "point_estimate", "ci_lower", "ci_upper",
            "mean_bootstrap", "std_bootstrap", "n_bootstrap_valid", "confidence",
        ):
            assert key in result

    def test_ci_contains_point_estimate(self, synthetic_predictions):
        y_true, y_pred, y_prob = synthetic_predictions
        result = bootstrap_confidence_interval(
            y_true, y_pred, y_prob, metric="roc_auc", n_bootstrap=500
        )
        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]

    def test_ci_bounds_within_valid_range(self, synthetic_predictions):
        y_true, y_pred, y_prob = synthetic_predictions
        for metric in ("accuracy", "f1", "roc_auc", "pr_auc"):
            result = bootstrap_confidence_interval(
                y_true, y_pred, y_prob, metric=metric, n_bootstrap=200
            )
            assert 0.0 <= result["ci_lower"] <= 1.0
            assert 0.0 <= result["ci_upper"] <= 1.0
            assert result["ci_lower"] <= result["ci_upper"]

    def test_reproducibility_with_seed(self, synthetic_predictions):
        y_true, y_pred, y_prob = synthetic_predictions
        r1 = bootstrap_confidence_interval(
            y_true, y_pred, y_prob, metric="roc_auc", n_bootstrap=200, random_state=42
        )
        r2 = bootstrap_confidence_interval(
            y_true, y_pred, y_prob, metric="roc_auc", n_bootstrap=200, random_state=42
        )
        assert r1["ci_lower"] == r2["ci_lower"]
        assert r1["ci_upper"] == r2["ci_upper"]

    def test_invalid_metric_raises(self, synthetic_predictions):
        y_true, y_pred, y_prob = synthetic_predictions
        with pytest.raises(ValueError, match="não suportada"):
            bootstrap_confidence_interval(
                y_true, y_pred, y_prob, metric="invalid_metric"
            )

    def test_low_n_bootstrap_raises(self, synthetic_predictions):
        y_true, y_pred, y_prob = synthetic_predictions
        with pytest.raises(ValueError, match=">= 100"):
            bootstrap_confidence_interval(
                y_true, y_pred, y_prob, metric="f1", n_bootstrap=50
            )

    def test_confidence_level_affects_width(self, synthetic_predictions):
        y_true, y_pred, y_prob = synthetic_predictions
        narrow = bootstrap_confidence_interval(
            y_true, y_pred, y_prob, metric="roc_auc",
            n_bootstrap=500, confidence=0.80,
        )
        wide = bootstrap_confidence_interval(
            y_true, y_pred, y_prob, metric="roc_auc",
            n_bootstrap=500, confidence=0.99,
        )
        narrow_width = narrow["ci_upper"] - narrow["ci_lower"]
        wide_width = wide["ci_upper"] - wide["ci_lower"]
        assert wide_width >= narrow_width


class TestComputeMetricsWithCI:
    def test_returns_all_metrics(self, synthetic_predictions):
        y_true, y_pred, y_prob = synthetic_predictions
        results = compute_metrics_with_ci(
            y_true, y_pred, y_prob, n_bootstrap=200
        )
        assert set(results.keys()) == {"accuracy", "f1", "roc_auc", "pr_auc"}

    def test_without_probabilities(self, synthetic_predictions):
        y_true, y_pred, _ = synthetic_predictions
        results = compute_metrics_with_ci(y_true, y_pred, n_bootstrap=200)
        assert set(results.keys()) == {"accuracy", "f1"}
        assert "roc_auc" not in results
