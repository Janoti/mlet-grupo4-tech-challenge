"""Testes para métricas de calibração (Brier, ECE, reliability diagram)."""

import numpy as np
import pytest

from churn_prediction.evaluation import (
    brier_score,
    compute_calibration_metrics,
    expected_calibration_error,
    reliability_diagram_data,
)


class TestBrierScore:
    def test_perfect_calibration(self):
        """Modelo perfeito (prob = label) → Brier = 0."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        assert brier_score(y_true, y_prob) == 0.0

    def test_worst_case(self):
        """Modelo completamente errado (prob oposta ao label) → Brier = 1."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])
        assert brier_score(y_true, y_prob) == 1.0

    def test_constant_0_5_on_balanced(self):
        """Probabilidade constante 0.5 em classes balanceadas → Brier = 0.25."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        assert brier_score(y_true, y_prob) == 0.25

    def test_returns_float(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.3, 0.7])
        assert isinstance(brier_score(y_true, y_prob), float)


class TestECE:
    def test_perfect_calibration(self):
        """Probabilidades alinhadas com taxa real → ECE = 0."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        y_prob = np.array([0.0] * 4 + [1.0] * 6)
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_miscalibrated(self):
        """Modelo sobreconfiante (prob sempre 0.9 mas acurácia real 0.5) → ECE alto."""
        n = 100
        y_true = np.array([1] * 50 + [0] * 50)
        y_prob = np.full(n, 0.9)
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        # Diferença entre confiança média 0.9 e acurácia real 0.5 ≈ 0.4
        assert ece == pytest.approx(0.4, abs=0.05)

    def test_returns_range_0_to_1(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200)
        y_prob = rng.uniform(0, 1, size=200)
        ece = expected_calibration_error(y_true, y_prob)
        assert 0.0 <= ece <= 1.0


class TestReliabilityDiagram:
    def test_returns_expected_keys(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])
        data = reliability_diagram_data(y_true, y_prob, n_bins=10)
        assert set(data.keys()) == {
            "bin_centers", "mean_predicted", "fraction_positives", "bin_counts",
        }

    def test_bin_counts_sum_to_n(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=100)
        y_prob = rng.uniform(0, 1, size=100)
        data = reliability_diagram_data(y_true, y_prob, n_bins=10)
        assert data["bin_counts"].sum() == 100

    def test_array_shapes(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.2, 0.8])
        data = reliability_diagram_data(y_true, y_prob, n_bins=5)
        for key in ("bin_centers", "mean_predicted", "fraction_positives", "bin_counts"):
            assert len(data[key]) == 5


class TestComputeCalibrationMetrics:
    def test_returns_both_metrics(self):
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.7, 0.2])
        result = compute_calibration_metrics(y_true, y_prob)
        assert set(result.keys()) == {"brier_score", "ece"}
        assert all(isinstance(v, float) for v in result.values())
