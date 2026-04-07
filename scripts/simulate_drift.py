"""Simula data drift enviando dados modificados para a API.

Uso:
    poetry run python scripts/simulate_drift.py [--url http://localhost:8000] [--n-requests 100]

Modifica a distribuição de features (age +10 anos, monthly_charges +30%)
para demonstrar detecção de drift no monitoramento.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys

import httpx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def generate_drifted_customer(seed: int | None = None) -> dict:
    """Gera dados de um cliente com drift artificial."""
    rng = np.random.default_rng(seed)

    return {
        "age": float(rng.normal(55, 12)),       # Drift: média 45→55
        "gender": random.choice(["male", "female"]),
        "region": random.choice(["Sudeste", "Sul", "Nordeste", "Norte", "Centro-Oeste"]),
        "plan_type": random.choice(["pre", "pos", "controle"]),
        "plan_price": float(rng.uniform(30, 200)),
        "tenure_months": float(rng.integers(1, 120)),
        "monthly_charges": float(rng.normal(130, 40)),  # Drift: média 100→130
        "nps_score": float(rng.integers(0, 11)),
        "support_calls_30d": float(rng.integers(0, 10)),
        "late_payments_6m": float(rng.integers(0, 6)),
        "avg_signal_quality": float(rng.uniform(1, 5)),
        "call_drop_rate": float(rng.uniform(0, 0.3)),
        "days_since_last_usage": float(rng.integers(0, 90)),
        "complaints_30d": float(rng.integers(0, 5)),
        "portability_request_flag": int(rng.choice([0, 1], p=[0.7, 0.3])),  # Drift: 10%→30%
    }


def main():
    parser = argparse.ArgumentParser(description="Simula drift enviando dados para a API")
    parser.add_argument("--url", default="http://localhost:8000", help="URL base da API")
    parser.add_argument("--n-requests", type=int, default=100, help="Número de requisições")
    parser.add_argument("--output", default="logs/drift_simulation.jsonl", help="Arquivo de saída")
    args = parser.parse_args()

    logger.info("Enviando %d requisições com drift para %s", args.n_requests, args.url)

    results = []
    errors = 0

    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        # Verifica health
        try:
            health = client.get("/health")
            logger.info("Health check: %s", health.json())
        except httpx.ConnectError:
            logger.error("API não disponível em %s", args.url)
            sys.exit(1)

        for i in range(args.n_requests):
            customer = generate_drifted_customer(seed=i)
            try:
                resp = client.post("/predict", json=customer)
                if resp.status_code == 200:
                    result = {
                        "request_id": i,
                        "input": customer,
                        "response": resp.json(),
                        "status_code": resp.status_code,
                    }
                    results.append(result)
                else:
                    errors += 1
                    logger.warning("Request %d: status %d", i, resp.status_code)
            except Exception as e:
                errors += 1
                logger.error("Request %d: %s", i, e)

    # Salva resultados
    from pathlib import Path

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(
        "Concluído: %d sucesso, %d erros. Resultados em %s",
        len(results), errors, args.output,
    )

    # Resumo das predições
    if results:
        probs = [r["response"]["churn_probability"] for r in results]
        risks = [r["response"]["risk_level"] for r in results]
        logger.info(
            "Distribuição de risco: alto=%d, medio=%d, baixo=%d",
            risks.count("alto"), risks.count("medio"), risks.count("baixo"),
        )
        logger.info("Probabilidade média: %.3f (std: %.3f)", np.mean(probs), np.std(probs))


if __name__ == "__main__":
    main()
