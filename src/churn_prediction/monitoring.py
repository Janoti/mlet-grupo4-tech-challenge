"""Monitoramento de data drift e performance do modelo em produção.

Implementa:
- Detecção de drift via teste Kolmogorov-Smirnov (numéricas)
- Detecção de drift via teste Qui-Quadrado (categóricas)
- Population Stability Index (PSI) para features numéricas
- Logging estruturado de inferências para auditoria
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("churn_monitoring")


# ---------------------------------------------------------------------------
# Data Drift — Kolmogorov-Smirnov (numéricas)
# ---------------------------------------------------------------------------

def ks_drift_test(
    reference: pd.Series,
    production: pd.Series,
    alpha: float = 0.05,
) -> dict:
    """Testa drift em feature numérica via KS 2-sample test.

    Args:
        reference: Distribuição de referência (treino).
        production: Distribuição de produção (inferência).
        alpha: Nível de significância.

    Returns:
        Dicionário com statistic, p_value, drift_detected.
    """
    ref = reference.dropna().values
    prod = production.dropna().values
    stat, p_value = stats.ks_2samp(ref, prod)
    return {
        "test": "kolmogorov_smirnov",
        "statistic": float(stat),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < alpha),
    }


# ---------------------------------------------------------------------------
# Data Drift — Qui-Quadrado (categóricas)
# ---------------------------------------------------------------------------

def chi2_drift_test(
    reference: pd.Series,
    production: pd.Series,
    alpha: float = 0.05,
) -> dict:
    """Testa drift em feature categórica via Chi-Squared test.

    Args:
        reference: Distribuição de referência (treino).
        production: Distribuição de produção (inferência).
        alpha: Nível de significância.

    Returns:
        Dicionário com statistic, p_value, drift_detected.
    """
    ref_counts = reference.value_counts(normalize=True)
    prod_counts = production.value_counts(normalize=True)

    # Alinha categorias
    all_cats = sorted(set(ref_counts.index) | set(prod_counts.index))
    ref_freq = np.array([ref_counts.get(c, 0) for c in all_cats])
    prod_freq = np.array([prod_counts.get(c, 0) for c in all_cats])

    # Evita divisão por zero
    ref_freq = np.clip(ref_freq, 1e-10, None)

    n_prod = len(production)
    observed = prod_freq * n_prod
    expected = ref_freq * n_prod

    stat, p_value = stats.chisquare(observed, f_exp=expected)
    return {
        "test": "chi_squared",
        "statistic": float(stat),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < alpha),
    }


# ---------------------------------------------------------------------------
# Population Stability Index (PSI)
# ---------------------------------------------------------------------------

def compute_psi(
    reference: np.ndarray,
    production: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calcula PSI entre distribuição de referência e produção.

    PSI < 0.10 → sem drift significativo
    PSI 0.10-0.20 → drift moderado (investigar)
    PSI > 0.20 → drift significativo (retreinar)

    Returns:
        Valor escalar do PSI.
    """
    ref = np.asarray(reference)
    prod = np.asarray(production)

    # Bins baseados na distribuição de referência
    breakpoints = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    ref_hist = np.histogram(ref, bins=breakpoints)[0] / len(ref)
    prod_hist = np.histogram(prod, bins=breakpoints)[0] / len(prod)

    # Evita log(0)
    ref_hist = np.clip(ref_hist, 1e-10, None)
    prod_hist = np.clip(prod_hist, 1e-10, None)

    psi = float(np.sum((prod_hist - ref_hist) * np.log(prod_hist / ref_hist)))
    return psi


# ---------------------------------------------------------------------------
# Drift report completo
# ---------------------------------------------------------------------------

def generate_drift_report(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    alpha: float = 0.05,
) -> dict:
    """Gera relatório completo de drift entre dados de referência e produção.

    Returns:
        Dicionário com drift por feature, contagem de alertas e PSI.
    """
    if numeric_cols is None:
        numeric_cols = reference_df.select_dtypes(include=["number"]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = [c for c in reference_df.columns if c not in numeric_cols]

    # Filtra apenas colunas presentes em ambos
    common_num = [c for c in numeric_cols if c in production_df.columns]
    common_cat = [c for c in categorical_cols if c in production_df.columns]

    report: dict = {"timestamp": datetime.now(UTC).isoformat(), "features": {}}
    alerts = 0

    for col in common_num:
        ks = ks_drift_test(reference_df[col], production_df[col], alpha)
        psi = compute_psi(reference_df[col].dropna().values, production_df[col].dropna().values)
        report["features"][col] = {**ks, "psi": psi, "type": "numeric"}
        if ks["drift_detected"] or psi > 0.20:
            alerts += 1

    for col in common_cat:
        chi2 = chi2_drift_test(reference_df[col], production_df[col], alpha)
        report["features"][col] = {**chi2, "type": "categorical"}
        if chi2["drift_detected"]:
            alerts += 1

    report["total_features"] = len(common_num) + len(common_cat)
    report["drift_alerts"] = alerts
    report["drift_ratio"] = alerts / max(report["total_features"], 1)

    return report


# ---------------------------------------------------------------------------
# Inference logger (para análise posterior)
# ---------------------------------------------------------------------------

class InferenceLogger:
    """Logger estruturado de inferências para monitoramento."""

    def __init__(self, log_path: str = "logs/inference.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        input_features: dict,
        prediction: float,
        probability: float,
        model_version: str,
        latency_ms: float,
    ) -> None:
        """Registra uma inferência em formato JSONL."""
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "model_version": model_version,
            "input_features": input_features,
            "prediction": prediction,
            "probability": probability,
            "latency_ms": round(latency_ms, 2),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
