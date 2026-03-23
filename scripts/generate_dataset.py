from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from logging_utils import get_logger, log_kv, setup_logging


logger = get_logger("generate_dataset")


def synth_features(n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Gera features do cliente conforme a Seção 5 (perfil, produtos, uso, financeiro, atendimento)."""
    customer_id = [f"C{idx:06d}" for idx in range(1, n_rows + 1)]

    # Perfil
    age = rng.integers(18, 80, size=n_rows)
    gender = rng.choice(["male", "female", "other", None], size=n_rows, p=[0.48, 0.48, 0.02, 0.02])
    region = rng.choice(
        ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"],
        size=n_rows,
        p=[0.08, 0.25, 0.06, 0.45, 0.16],
    )
    contract = rng.choice(["mensal", "anual_1", "anual_2"], size=n_rows, p=[0.55, 0.25, 0.20])

    # Produtos / Serviços
    internet_service = rng.choice(["dsl", "fiber", "none"], size=n_rows, p=[0.25, 0.60, 0.15])
    has_tv = rng.choice([0, 1], size=n_rows, p=[0.7, 0.3])
    has_fixed = rng.choice([0, 1], size=n_rows, p=[0.6, 0.4])
    tenure_months = rng.integers(0, 121, size=n_rows)  # até 10 anos

    # Uso (último mês)
    minutes_monthly = rng.integers(0, 3000, size=n_rows)
    data_gb_monthly = (rng.exponential(scale=5.0, size=n_rows)).round(2)
    sms_monthly = rng.integers(0, 500, size=n_rows)
    # variação de uso (%) nos últimos 3 meses (pode indicar queda brusca)
    usage_delta_pct = (rng.normal(loc=0.0, scale=0.2, size=n_rows)).round(3)

    # Financeiro
    monthly_charges = (20 + data_gb_monthly * 2 + rng.uniform(0, 80, size=n_rows)).round(2)
    total_charges = (monthly_charges * tenure_months + rng.uniform(0, 500, size=n_rows)).round(2)
    default_flag = rng.choice([0, 1], size=n_rows, p=[0.92, 0.08])  # inadimplência
    days_past_due = rng.integers(0, 120, size=n_rows) * default_flag

    # Atendimento
    support_calls_30d = rng.integers(0, 6, size=n_rows)
    support_calls_90d = support_calls_30d + rng.integers(0, 6, size=n_rows)
    complaints_30d = rng.choice([0, 1], size=n_rows, p=[0.95, 0.05])
    tickets_open_30d = rng.integers(0, 3, size=n_rows)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            # Perfil
            "age": age,
            "gender": gender,
            "region": region,
            "contract": contract,
            # Produtos / Serviços
            "internet_service": internet_service,
            "has_tv": has_tv,
            "has_fixed": has_fixed,
            "tenure_months": tenure_months,
            # Uso
            "minutes_monthly": minutes_monthly,
            "data_gb_monthly": data_gb_monthly,
            "sms_monthly": sms_monthly,
            "usage_delta_pct": usage_delta_pct,
            # Financeiro
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "default_flag": default_flag,
            "days_past_due": days_past_due,
            # Atendimento
            "support_calls_30d": support_calls_30d,
            "support_calls_90d": support_calls_90d,
            "complaints_30d": complaints_30d,
            "tickets_open_30d": tickets_open_30d,
        }
    )
    return df


def add_churn_label(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Gera rótulo `churn` com lógica baseada em fatores de risco realistas (~20-25% de churn)."""
    # Score de risco acumulado (quanto maior, mais chance de churnar)
    risk = np.zeros(len(df))

    # Contrato mensal → maior risco
    risk += (df["contract"] == "mensal").astype(float) * 1.5

    # Pouco tempo de casa → mais instável
    risk += (df["tenure_months"] < 12).astype(float) * 1.2

    # Queda brusca de uso
    risk += (df["usage_delta_pct"] < -0.15).astype(float) * 1.0

    # Inadimplência
    risk += df["default_flag"].astype(float) * 1.3

    # Muitas chamadas no suporte (insatisfação)
    risk += (df["support_calls_30d"] >= 3).astype(float) * 0.8

    # Reclamações recentes
    risk += df["complaints_30d"].astype(float) * 1.0

    # Tickets abertos
    risk += (df["tickets_open_30d"] >= 2).astype(float) * 0.6

    # Fatura alta com pouco tempo de casa (potencial choque de preço)
    risk += ((df["monthly_charges"] > 100) & (df["tenure_months"] < 6)).astype(float) * 0.8

    # Converte score em probabilidade via sigmoide e aplica limiar com ruído
    prob = 1 / (1 + np.exp(-(risk - 3.45)))
    noise = rng.uniform(0, 0.05, size=len(df))
    df["churn"] = (rng.random(size=len(df)) < (prob + noise)).astype(int)
    return df


def main(
    n_rows: int = 50_000,
    seed: int = 42,
    out_dir: str | Path = "data/raw",
    log_level: str | None = None,
) -> None:
    setup_logging(log_level)
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_kv(logger, "generate.start", n_rows=n_rows, seed=seed, out_dir=out_dir)

    df = synth_features(n_rows, rng)
    df = add_churn_label(df, rng)

    churn_rate = df["churn"].mean()
    out_path = out_dir / "telecom_churn_base.csv"
    df.to_csv(out_path, index=False)
    log_kv(
        logger,
        "generate.wrote",
        path=out_path,
        rows=len(df),
        churn_rate=f"{churn_rate:.1%}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera base sintética telecom com label de churn.")
    parser.add_argument("--n-rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="data/raw", help="Diretório de saída para features (raw).")
    parser.add_argument("--log-level", type=str, default=None, help="Nível de log (DEBUG, INFO, WARNING, ERROR).")
    args = parser.parse_args()
    main(n_rows=args.n_rows, seed=args.seed, out_dir=args.out_dir, log_level=args.log_level)

