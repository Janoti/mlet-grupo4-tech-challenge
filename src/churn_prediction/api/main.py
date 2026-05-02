"""FastAPI application para inferência de churn em telecom.

Endpoints:
    GET  /health  — health check e status do modelo
    POST /predict — predição de churn para um cliente
    GET  /metrics — métricas Prometheus para monitoramento
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from churn_prediction.api.middleware import observability_middleware
from churn_prediction.api.prometheus_metrics import (
    model_info,
    model_loaded,
    normalize_gender,
    normalize_plan_type,
    prediction_probability,
    predictions_total,
)
from churn_prediction.api.schemas import (
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
    DriftCheckResponse,
    DriftReportResponse,
    ModelVersionsResponse,
    RetargetRecommendation,
    FeedbackRequest,
    FeedbackResponse,
)
from churn_prediction.api.drift_service import DriftService
from churn_prediction.api.model_service import ModelService
from churn_prediction.api.feedback_service import FeedbackService
from churn_prediction.config import LEAKAGE_COLS
from churn_prediction.data_cleaning import clip_numeric_features, create_age_group, standardize_categoricals

UTC = timezone.utc

logger = logging.getLogger("churn_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Estado global da aplicação
# ---------------------------------------------------------------------------
MODEL_STATE: dict = {
    "pipeline": None,
    "model_version": None,
}

# Serviços
DRIFT_SERVICE = DriftService()
MODEL_SERVICE = ModelService()
FEEDBACK_SERVICE = FeedbackService()

MODEL_PATH = os.getenv(
    "CHURN_MODEL_PATH",
    str(Path(__file__).resolve().parents[3] / "models" / "churn_pipeline.joblib"),
)


def load_model(path: str | None = None) -> None:
    """Carrega pipeline serializado e metadata do champion."""
    path = path or MODEL_PATH
    if not Path(path).exists():
        logger.warning("Modelo não encontrado em %s", path)
        model_loaded.set(0)
        return
    MODEL_STATE["pipeline"] = joblib.load(path)

    # Tenta carregar metadata do champion para identificar o modelo real
    metadata_path = Path(path).parent / "champion_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        MODEL_STATE["model_version"] = metadata.get("champion_run_name", Path(path).stem)
        logger.info(
            "Champion carregado: %s (run_id=%s, valor_liquido=%.0f)",
            metadata.get("champion_run_name"),
            metadata.get("champion_run_id", "?")[:8],
            metadata.get("metrics", {}).get("valor_liquido", 0),
        )
    else:
        MODEL_STATE["model_version"] = Path(path).stem
        logger.info("Modelo carregado: %s (sem champion_metadata.json)", path)

    model_loaded.set(1)
    model_info.info({"version": MODEL_STATE["model_version"] or "unknown"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega modelo na inicialização da aplicação."""
    load_model()
    yield


# ---------------------------------------------------------------------------
# Instância FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Churn Prediction API",
    description=(
        "API para predição de churn em telecom. "
        "Recebe dados de um cliente e retorna probabilidade de churn, "
        "predição binária e faixa de risco."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Observability middleware: mede latência uma vez e propaga para header HTTP,
# log estruturado e métricas Prometheus.
app.middleware("http")(observability_middleware)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Monitoramento"])
async def health_check():
    """Verifica se a API está operacional e se o modelo está carregado."""
    return HealthResponse(
        status="ok",
        model_loaded=MODEL_STATE["pipeline"] is not None,
        model_version=MODEL_STATE["model_version"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inferência"])
async def predict(customer: CustomerFeatures):
    """Recebe dados de um cliente e retorna predição de churn.

    O pipeline aplica as mesmas transformações de pré-processamento
    usadas no treinamento (imputação, scaling, OHE), garantindo
    consistência entre treino e inferência (sem training-serving skew).
    """
    if MODEL_STATE["pipeline"] is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Verifique CHURN_MODEL_PATH.",
        )

    start = time.time()

    # Converte input Pydantic → DataFrame
    data = customer.model_dump()
    df = pd.DataFrame([data])

    # Aplica mesmas limpezas do pipeline de treino
    df = standardize_categoricals(df)
    df = clip_numeric_features(df)
    df = create_age_group(df)

    # Remove colunas de leakage caso presentes
    cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Predição
    pipeline = MODEL_STATE["pipeline"]
    try:
        proba = pipeline.predict_proba(df)[:, 1][0]
    except Exception:
        # Fallback para modelos sem predict_proba (ex: DummyClassifier)
        pred = pipeline.predict(df)[0]
        proba = float(pred)

    prediction = int(proba >= 0.5)

    # Faixa de risco
    if proba >= 0.70:
        risk = "alto"
    elif proba >= 0.40:
        risk = "medio"
    else:
        risk = "baixo"

    # Registra métrica Prometheus segmentada por risco, perfil e decisão
    predictions_total.labels(
        risk_level=risk,
        gender=normalize_gender(customer.gender),
        plan_type=normalize_plan_type(customer.plan_type),
        churn_prediction=str(prediction),
    ).inc()
    # Histogram da probabilidade — base para detecção de drift no dashboard de saúde
    prediction_probability.observe(float(proba))

    latency_ms = (time.time() - start) * 1000

    # Log estruturado da requisição (para monitoramento)
    logger.info(
        '{"event":"prediction","churn_prob":%.4f,"risk":"%s","latency_ms":%.1f}',
        proba,
        risk,
        latency_ms,
    )

    return PredictionResponse(
        churn_probability=round(float(proba), 4),
        churn_prediction=prediction,
        risk_level=risk,
        model_version=MODEL_STATE["model_version"] or "unknown",
    )


# ========== DRIFT MONITORING ENDPOINTS ==========

@app.post("/drift/check", response_model=DriftCheckResponse, tags=["Drift Monitoring"])
async def drift_check(production_data: list[CustomerFeatures]):
    """Verifica presença de data drift em batch de dados de produção."""
    if not production_data:
        raise HTTPException(status_code=400, detail="Lista de dados vazia")

    data_dicts = [d.model_dump() for d in production_data]
    df_production = pd.DataFrame(data_dicts)

    df_production = standardize_categoricals(df_production)
    df_production = clip_numeric_features(df_production)
    df_production = create_age_group(df_production)

    cols_to_drop = [c for c in LEAKAGE_COLS if c in df_production.columns]
    if cols_to_drop:
        df_production = df_production.drop(columns=cols_to_drop)

    result = DRIFT_SERVICE.check_drift(df_production)

    logger.info(
        '{"event":"drift_check","drift_ratio":%.3f,"recommendation":"%s"}',
        result["drift_ratio"],
        result["recommendation"],
    )

    return DriftCheckResponse(**result)


@app.get("/drift/report", response_model=DriftReportResponse, tags=["Drift Monitoring"])
async def drift_report_detailed(sample_size: int = 100):
    """Gera relatório completo de drift com detalhes por feature."""
    logger.warning("drift_report_detailed: usando dados de demonstração")

    dummy_production = pd.DataFrame({
        "age": [35, 42, 28, 55, 31] * (sample_size // 5),
        "tenure_months": [24, 36, 12, 48, 18] * (sample_size // 5),
    })

    result = DRIFT_SERVICE.get_detailed_report(dummy_production)

    logger.info(
        '{"event":"drift_report","total_features":%d,"alerts":%d}',
        result["total_features"],
        result["drift_alerts"],
    )

    return DriftReportResponse(**result)


# ========== MODEL MANAGEMENT ENDPOINTS ==========

@app.get("/model/versions", response_model=ModelVersionsResponse, tags=["Model Management"])
async def list_model_versions():
    """Lista histórico de versões de modelo treinadas."""
    result = MODEL_SERVICE.get_model_versions()

    logger.info(
        '{"event":"list_versions","total_versions":%d,"champion":"%s"}',
        result["total_versions"],
        result.get("champion_version", "unknown"),
    )

    return ModelVersionsResponse(**result)


@app.post("/model/retrain-recommendation", response_model=RetargetRecommendation, tags=["Model Management"])
async def get_retrain_recommendation(drift_ratio: float = 0.0, days_since_retrain: int | None = None):
    """Recomenda se o modelo deve ser retreinado."""
    if not 0 <= drift_ratio <= 1:
        raise HTTPException(status_code=400, detail="drift_ratio deve estar entre 0 e 1")

    result = MODEL_SERVICE.recommend_retrain(
        drift_ratio=drift_ratio,
        days_since_last_retrain=days_since_retrain,
    )

    logger.info(
        '{"event":"retrain_recommendation","should_retrain":%s,"reason":"%s"}',
        result["should_retrain"],
        result["reason"],
    )

    return RetargetRecommendation(**result)


# ========== FEEDBACK ENDPOINTS ==========

@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(feedback: FeedbackRequest):
    """Registra feedback do usuário sobre uma predição."""
    try:
        feedback_id = FEEDBACK_SERVICE.log_feedback(
            prediction_id=feedback.prediction_id,
            actual_churn=feedback.actual_churn,
            feedback_type=feedback.feedback_type,
            comment=feedback.comment,
            rating=feedback.rating,
        )

        logger.info(
            '{"event":"feedback_received","feedback_id":"%s","prediction_id":"%s"}',
            feedback_id,
            feedback.prediction_id,
        )

        return FeedbackResponse(
            feedback_id=feedback_id,
            timestamp=datetime.now(UTC).isoformat(),
            status="received",
            message="Feedback registrado com sucesso",
        )
    except Exception as e:
        logger.error(f"Erro ao registrar feedback: {e}")
        raise HTTPException(status_code=500, detail="Erro ao registrar feedback")


@app.get("/feedback/summary", tags=["Feedback"])
async def feedback_summary():
    """Retorna sumário de feedback recebido."""
    summary = FEEDBACK_SERVICE.get_feedback_summary()

    logger.info(
        '{"event":"feedback_summary","total":%d,"accuracy":%.3f}',
        summary.get("total_feedback", 0),
        summary.get("accuracy") or 0,
    )

    return summary


@app.get("/metrics", include_in_schema=False, tags=["Monitoramento"])
async def metrics():
    """Endpoint Prometheus — expõe métricas em formato texto para scraping."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/", tags=["Info"])
async def root():
    """Informações sobre a API."""
    return {
        "name": "Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)",
        "metrics": "/metrics",
    }
