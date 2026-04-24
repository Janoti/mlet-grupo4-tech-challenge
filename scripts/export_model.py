"""Exporta o champion do MLflow para uso pela API FastAPI.

Uso:
    poetry run python scripts/export_model.py

Consulta todos os experimentos no MLflow, seleciona o melhor modelo por
valor_liquido (métrica de negócio) e exporta o pipeline + metadata para
models/churn_pipeline.joblib e models/champion_metadata.json.
"""

from __future__ import annotations

import logging

from churn_prediction.pipelines import prepare_data
from churn_prediction.registry import export_champion, find_champion, register_champion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DATA = "data/raw/telecom_churn_base_extended.csv"
MLFLOW_URI = "file:./mlruns"


def main():
    logger.info("Buscando champion no MLflow...")
    champion = find_champion(tracking_uri=MLFLOW_URI)

    logger.info("Registrando champion no MLflow Model Registry...")
    version = register_champion(
        run_id=champion["run_id"],
        tracking_uri=MLFLOW_URI,
    )

    logger.info("Preparando dados para exportação...")
    data = prepare_data(RAW_DATA)

    logger.info("Exportando champion...")
    export_champion(
        champion=champion,
        version=version,
        data=data,
        tracking_uri=MLFLOW_URI,
    )

    # Log resumo dos candidatos
    logger.info("=" * 60)
    logger.info("CHAMPION: %s", champion["run_name"])
    logger.info("  run_id: %s", champion["run_id"][:8])
    for metric, value in champion["metrics"].items():
        logger.info("  %s: %.4f", metric, value)
    logger.info("-" * 60)
    logger.info("Candidatos avaliados:")
    for i, c in enumerate(champion["all_candidates"], 1):
        vl = c.get("valor_liquido", 0)
        roc = c.get("roc_auc", 0)
        logger.info("  %d. %s | valor_liquido=%.0f | roc_auc=%.4f", i, c["run_name"], vl, roc)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
