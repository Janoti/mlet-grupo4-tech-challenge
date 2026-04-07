"""FastAPI application para inferência de churn em telecom.

Endpoints:
    GET  /health  — health check e status do modelo
    POST /predict — predição de churn para um cliente
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from churn_prediction.api.schemas import (
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
)
from churn_prediction.config import LEAKAGE_COLS
from churn_prediction.data_cleaning import clip_numeric_features, standardize_categoricals

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

MODEL_PATH = os.getenv(
    "CHURN_MODEL_PATH",
    str(Path(__file__).resolve().parents[3] / "models" / "churn_pipeline.joblib"),
)


def load_model(path: str | None = None) -> None:
    """Carrega pipeline sklearn serializado."""
    path = path or MODEL_PATH
    if not Path(path).exists():
        logger.warning("Modelo não encontrado em %s", path)
        return
    MODEL_STATE["pipeline"] = joblib.load(path)
    MODEL_STATE["model_version"] = Path(path).stem
    logger.info("Modelo carregado: %s", path)


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


@app.get("/", tags=["Info"])
async def root():
    """Informações sobre a API."""
    return {
        "name": "Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)",
    }
