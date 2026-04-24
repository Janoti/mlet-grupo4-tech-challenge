"""Analisa drift entre dados de treino e logs de produção.

Uso:
    poetry run python scripts/check_drift.py \
        --reference data/raw/telecom_churn_base_extended.csv \
        --production logs/drift_simulation.jsonl

Gera relatório de drift com KS test, Chi-Squared e PSI para cada feature.
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from churn_prediction.monitoring import generate_drift_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_production_logs(path: str) -> pd.DataFrame:
    """Carrega logs JSONL de inferência e extrai features de input."""
    records = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            if "input" in data:
                records.append(data["input"])
            elif "input_features" in data:
                records.append(data["input_features"])
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Verifica data drift")
    parser.add_argument(
        "--reference", default="data/raw/telecom_churn_base_extended.csv",
        help="CSV de referência (dados de treino)",
    )
    parser.add_argument(
        "--production", default="logs/drift_simulation.jsonl",
        help="JSONL com logs de produção",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Nível de significância")
    args = parser.parse_args()

    logger.info("Carregando dados de referência: %s", args.reference)
    ref_df = pd.read_csv(args.reference)

    logger.info("Carregando dados de produção: %s", args.production)
    prod_df = load_production_logs(args.production)

    logger.info("Referência: %d linhas, Produção: %d linhas", len(ref_df), len(prod_df))

    report = generate_drift_report(ref_df, prod_df, alpha=args.alpha)

    sep = "=" * 70
    logger.info(sep)
    logger.info("RELATORIO DE DATA DRIFT")
    logger.info(sep)
    logger.info("Timestamp: %s", report["timestamp"])
    logger.info("Features analisadas: %d", report["total_features"])
    logger.info("Alertas de drift: %d", report["drift_alerts"])
    logger.info("Razao de drift: %.1f%%", report["drift_ratio"] * 100)
    logger.info("-" * 70)

    for feat, info in sorted(report["features"].items()):
        drift_flag = "[DRIFT]" if info["drift_detected"] else "[OK]   "
        if info["type"] == "numeric":
            psi_str = f" PSI={info['psi']:.4f}"
            psi_alert = " [PSI>0.20]" if info["psi"] > 0.20 else ""
        else:
            psi_str = ""
            psi_alert = ""
        logger.info(
            "  %s | %-30s | %-20s | p=%.4f%s%s",
            drift_flag,
            feat,
            info["test"],
            info["p_value"],
            psi_str,
            psi_alert,
        )

    logger.info(sep)

    if report["drift_alerts"] > 0:
        logger.warning(
            "AÇÃO RECOMENDADA: %d features com drift detectado. "
            "Considere retreinar o modelo.",
            report["drift_alerts"],
        )
    else:
        logger.info("Nenhum drift significativo detectado.")


if __name__ == "__main__":
    main()
