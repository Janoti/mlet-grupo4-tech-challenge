"""Treino remoto — executa notebooks e exporta champion com MLflow tracking direto."""

from __future__ import annotations

import logging
import os
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("train_remote")

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"


def run_notebook(path: str) -> None:
    """Executa um notebook via nbconvert."""
    logger.info("Executando %s...", path)
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=600",
            path,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "MLFLOW_TRACKING_URI": MLFLOW_URI},
    )
    if result.returncode != 0:
        logger.error("Falha em %s:\n%s", path, result.stderr[-2000:])
        raise RuntimeError(f"Notebook {path} falhou")
    logger.info("Concluído: %s", path)


def main() -> None:
    logger.info("=== TREINO REMOTO ===")
    logger.info("MLflow: %s", MLFLOW_URI)

    # Gerar dados se não existem
    data_path = "data/raw/telecom_churn_base_extended.csv"
    if not os.path.exists(data_path):
        logger.info("Gerando dataset sintético...")
        subprocess.run(
            [sys.executable, "scripts/generate_synthetic.py",
             "--n-rows", "50000", "--seed", "42", "--out-dir", "data/raw"],
            check=True,
        )

    # Executar notebooks (pula EDA — só visualização, não afeta treino)
    run_notebook("notebooks/02_baselines.ipynb")
    run_notebook("notebooks/03_mlp_pytorch.ipynb")

    # Exportar champion
    logger.info("Exportando champion...")
    subprocess.run(
        [sys.executable, "scripts/export_model.py"],
        check=True,
        env={**os.environ, "MLFLOW_TRACKING_URI": MLFLOW_URI},
    )

    logger.info("=== TREINO REMOTO CONCLUÍDO ===")


if __name__ == "__main__":
    main()
