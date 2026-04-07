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

    # Exibe resultados
    print("\n" + "=" * 70)
    print("RELATÓRIO DE DATA DRIFT")
    print("=" * 70)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Features analisadas: {report['total_features']}")
    print(f"Alertas de drift: {report['drift_alerts']}")
    print(f"Razão de drift: {report['drift_ratio']:.1%}")
    print("-" * 70)

    for feat, info in sorted(report["features"].items()):
        drift = "⚠ DRIFT" if info["drift_detected"] else "  OK"
        if info["type"] == "numeric":
            psi_str = f" PSI={info['psi']:.4f}"
            psi_flag = " ⚠ PSI>0.20" if info["psi"] > 0.20 else ""
        else:
            psi_str = ""
            psi_flag = ""
        print(
            f"  {drift} | {feat:30s} | {info['test']:20s} | "
            f"p={info['p_value']:.4f}{psi_str}{psi_flag}"
        )

    print("=" * 70)

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
