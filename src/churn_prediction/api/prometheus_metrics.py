"""Métricas Prometheus específicas de domínio para a API de Churn.

Métricas expostas:
- churn_api_predictions_total — Counter de predições por risk_level
- churn_api_model_loaded — Gauge indicando se o modelo está carregado (0 ou 1)
- churn_api_model_info — Info com versão do modelo ativo
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Info

predictions_total = Counter(
    "churn_api_predictions_total",
    "Total de predições realizadas",
    labelnames=["risk_level"],
)

model_loaded = Gauge(
    "churn_api_model_loaded",
    "Indica se o modelo está carregado (1=sim, 0=não)",
)

model_info = Info(
    "churn_api_model",
    "Informações sobre o modelo ativo",
)
