"""Exporta pipeline sklearn treinado para uso pela API FastAPI.

Uso:
    poetry run python scripts/export_model.py

Carrega a base bruta, treina o melhor baseline (LogisticRegression)
e salva o pipeline completo (preprocessing + modelo) em models/churn_pipeline.joblib.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression

from churn_prediction.config import LOG_REG_KWARGS
from churn_prediction.pipelines import prepare_data
from churn_prediction.preprocessing import build_sklearn_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DATA = "data/raw/telecom_churn_base_extended.csv"
OUT_DIR = Path("models")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Preparando dados...")
    data = prepare_data(RAW_DATA)

    logger.info("Treinando LogisticRegression...")
    model = LogisticRegression(**LOG_REG_KWARGS)
    pipe = build_sklearn_pipeline(data["X_train"], model)
    pipe.fit(data["X_train"], data["y_train"])

    out_path = OUT_DIR / "churn_pipeline.joblib"
    joblib.dump(pipe, out_path)
    logger.info("Pipeline exportado em %s", out_path)


if __name__ == "__main__":
    main()
