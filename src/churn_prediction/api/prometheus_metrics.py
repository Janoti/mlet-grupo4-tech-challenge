"""Métricas Prometheus específicas de domínio para a API de Churn.

Métricas expostas:
- churn_api_predictions_total — Counter de predições por risk_level, gender,
  plan_type e churn_prediction (cardinalidade controlada via normalização)
- churn_api_model_loaded — Gauge indicando se o modelo está carregado (0 ou 1)
- churn_api_model_info — Info com versão do modelo ativo
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

VALID_GENDERS = {"male", "female", "other"}
VALID_PLAN_TYPES = {"pre", "pos", "controle", "empresarial"}


def normalize_gender(value: str | None) -> str:
    """Normaliza gênero para um conjunto fechado (controla cardinalidade)."""
    if not value:
        return "unknown"
    v = str(value).strip().lower()
    return v if v in VALID_GENDERS else "other"


def normalize_plan_type(value: str | None) -> str:
    """Normaliza tipo de plano para um conjunto fechado."""
    if not value:
        return "unknown"
    v = str(value).strip().lower()
    return v if v in VALID_PLAN_TYPES else "outro"


predictions_total = Counter(
    "churn_api_predictions_total",
    "Total de predições realizadas",
    labelnames=["risk_level", "gender", "plan_type", "churn_prediction"],
)

model_loaded = Gauge(
    "churn_api_model_loaded",
    "Indica se o modelo está carregado (1=sim, 0=não)",
)

model_info = Info(
    "churn_api_model",
    "Informações sobre o modelo ativo",
)

prediction_probability = Histogram(
    "churn_api_prediction_probability",
    "Distribuição da probabilidade de churn predita (suporta heatmap de drift)",
    buckets=(
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    ),
)
