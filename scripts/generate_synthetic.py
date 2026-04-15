"""
generate_synthetic.py

Gera base sintética de telecom (versão estendida) com muitos campos adicionais
e cálculo de NPS por cliente (nps_score + categoria promoter/passive/detractor).
Substitui prints por logging estruturado (structlog).

Uso:
    python generate_synthetic.py --n-rows 50000 --seed 42 --out-dir data/raw
    python generate_synthetic.py --n-rows 50000 --seed 42 --duplicate-row-rate 0.03 --duplicate-id-rate 0.02

Saída:
    {out_dir}/telecom_churn_base_extended.csv
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if importlib.util.find_spec("structlog") is not None:
    structlog = importlib.import_module("structlog")
else:
    structlog = None

# -------------------------
# Logging (structlog JSON)
# -------------------------
def configure_logging(level: str | None = None) -> None:
    level = (level or "INFO").upper()
    logging.basicConfig(level=getattr(logging, level), format="%(message)s")
    if structlog is None:
        return
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


class _FallbackLogger:
    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def info(self, event: str, **kwargs: Any) -> None:
        if kwargs:
            self._logger.info("%s %s", event, json.dumps(kwargs, ensure_ascii=False, sort_keys=True))
            return
        self._logger.info(event)


if structlog is not None:
    logger: Any = structlog.get_logger("mlet_churn_generator")
else:
    logger = _FallbackLogger("mlet_churn_generator")


# -------------------------
# Helper utilities
# -------------------------
def _clip_round(arr: np.ndarray, low: float, high: float, decimals: int = 0) -> np.ndarray:
    return np.round(np.clip(arr, low, high), decimals)


# -------------------------
# Feature generation
# -------------------------
def synth_features_extended(n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Gera features estendidas conforme a especificação do usuário."""
    customer_id = [f"C{idx:06d}" for idx in range(1, n_rows + 1)]

    # -----------------
    # Perfil básico
    # -----------------
    age = rng.integers(18, 80, size=n_rows)
    gender = rng.choice(["male", "female", "other", None], size=n_rows, p=[0.48, 0.48, 0.02, 0.02])
    region = rng.choice(
        ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"],
        size=n_rows,
        p=[0.08, 0.25, 0.06, 0.45, 0.16],
    )

    # -----------------
    # Relacionamento e contrato
    # -----------------
    plan_type = rng.choice(["pre", "controle", "pos", "empresarial"], size=n_rows, p=[0.2, 0.25, 0.45, 0.10])
    # preço base do plano por tipo (distribuição simples)
    base_price_map = {"pre": 15.0, "controle": 30.0, "pos": 60.0, "empresarial": 200.0}
    plan_price = np.array([base_price_map[t] for t in plan_type]) + rng.normal(0, 5.0, size=n_rows)
    is_promotional_plan = rng.choice([0, 1], size=n_rows, p=[0.8, 0.2])
    discount_amount = (is_promotional_plan * rng.uniform(5, 30, size=n_rows)).round(2)
    # reajuste recente (flag boolean)
    price_increase_last_3m = rng.choice([0, 1], size=n_rows, p=[0.9, 0.1])

    # contrato/renovação: meses até término
    months_to_contract_end = rng.integers(0, 36, size=n_rows)
    # datas derivadas (usamos timestamp hoje + months)
    today = pd.Timestamp.now().normalize()
    contract_renewal_date = [ (today + pd.DateOffset(months=int(m))).date() for m in months_to_contract_end ]
    # fidelização (loyalty) — se tem fidelidade, geramos meses até o fim; senão NaN
    has_loyalty = rng.choice([0, 1], size=n_rows, p=[0.6, 0.4])
    months_to_loyalty_end = has_loyalty * rng.integers(0, 24, size=n_rows)
    loyalty_end_date = [ (today + pd.DateOffset(months=int(m))).date() if has_loyalty[i] else pd.NaT
                         for i, m in enumerate(months_to_loyalty_end) ]

    # -----------------
    # Produtos / Serviços (existentes)
    # -----------------
    internet_service = rng.choice(["dsl", "fiber", "none"], size=n_rows, p=[0.25, 0.60, 0.15])
    has_tv = rng.choice([0, 1], size=n_rows, p=[0.7, 0.3])
    has_fixed = rng.choice([0, 1], size=n_rows, p=[0.6, 0.4])
    tenure_months = rng.integers(0, 121, size=n_rows)

    # -----------------
    # Qualidade do serviço
    # -----------------
    network_outages_30d = rng.poisson(0.1, size=n_rows) + rng.choice([0,1], size=n_rows, p=[0.95,0.05]) * rng.integers(0,3,size=n_rows)
    avg_signal_quality = _clip_round(rng.normal(4.0, 0.8, size=n_rows), 0.5, 5.0, 2)  # 1-5 scale
    call_drop_rate = _clip_round(rng.beta(1, 50, size=n_rows) * 0.2, 0.0, 1.0, 4)  # baixa em geral
    avg_internet_speed = _clip_round(rng.normal(25.0, 15.0, size=n_rows), 0.1, 500.0, 2)  # Mbps
    service_failures_30d = rng.poisson(0.05, size=n_rows) + (network_outages_30d > 0).astype(int)
    installation_delay_days = rng.integers(0, 15, size=n_rows) * rng.choice([0, 1], size=n_rows, p=[0.9, 0.1])
    repair_visits_90d = rng.poisson(0.1, size=n_rows)

    # -----------------
    # Uso / comportamento
    # -----------------
    minutes_monthly = rng.integers(0, 3000, size=n_rows)
    data_gb_monthly = (rng.exponential(scale=5.0, size=n_rows)).round(2)
    sms_monthly = rng.integers(0, 500, size=n_rows)
    usage_delta_pct = (rng.normal(loc=0.0, scale=0.2, size=n_rows)).round(3)

    days_since_last_usage = rng.integers(0, 60, size=n_rows)
    active_days_30d = rng.integers(0, 30, size=n_rows)
    night_usage_pct = _clip_round(rng.beta(2, 5, size=n_rows) * 100, 0.0, 100.0, 1)
    roaming_usage = rng.choice([0, 1], size=n_rows, p=[0.95, 0.05])
    topup_frequency = rng.poisson(1.2, size=n_rows)  # relevante para pré
    avg_usage_last_3m = (data_gb_monthly * rng.uniform(0.7, 1.3, size=n_rows)).round(2)
    usage_trend_3m = _clip_round(rng.normal(0.0, 0.3, size=n_rows), -1.0, 1.0, 3)

    # -----------------
    # Financeiro
    # -----------------
    payment_method = rng.choice(["card", "boleto", "debito", "pix"], size=n_rows, p=[0.5, 0.2, 0.15, 0.15])
    billing_type = rng.choice(["digital", "paper", "auto"], size=n_rows, p=[0.7, 0.1, 0.2])

    monthly_charges = (plan_price - discount_amount + data_gb_monthly * 2 + rng.uniform(0, 20, size=n_rows)).round(2)
    avg_bill_last_6m = (monthly_charges * rng.uniform(0.9, 1.1, size=n_rows)).round(2)
    bill_variation_pct = _clip_round(rng.normal(0.0, 0.08, size=n_rows), -1.0, 1.0, 3)
    invoice_shock_flag = ((bill_variation_pct > 0.25) | (price_increase_last_3m == 1)).astype(int)
    late_payments_6m = rng.poisson(0.1, size=n_rows) + rng.choice([0,1], size=n_rows, p=[0.95,0.05]) * rng.integers(1,3,size=n_rows)
    collection_attempts_30d = rng.poisson(0.05, size=n_rows)

    default_flag = (late_payments_6m > 0).astype(int)
    days_past_due = rng.integers(0, 120, size=n_rows) * default_flag

    # -----------------
    # Atendimento e experiência
    # -----------------
    support_calls_30d = rng.integers(0, 6, size=n_rows)
    support_calls_90d = support_calls_30d + rng.integers(0, 6, size=n_rows)
    complaints_30d = rng.choice([0, 1], size=n_rows, p=[0.95, 0.05])
    tickets_open_30d = rng.integers(0, 3, size=n_rows)
    last_complaint_reason = rng.choice(
        ["cobrança", "qualidade", "atraso_instalacao", "atendimento", "nenhum"],
        size=n_rows,
        p=[0.07, 0.15, 0.05, 0.08, 0.65],
    )
    resolution_time_avg = _clip_round(rng.exponential(scale=1.5, size=n_rows), 0.1, 30.0, 2)  # dias
    first_call_resolution_flag = rng.choice([0, 1], size=n_rows, p=[0.7, 0.3])
    transferred_calls_count = rng.poisson(0.2, size=n_rows)
    last_contact_channel = rng.choice(["whatsapp", "telefone", "app", "loja", "email"], size=n_rows, p=[0.4, 0.25, 0.2, 0.05, 0.1])

    # -----------------
    # Engajamento digital
    # -----------------
    app_login_30d = rng.integers(0, 60, size=n_rows)
    self_service_usage_30d = rng.integers(0, 30, size=n_rows)
    marketing_open_rate = _clip_round(rng.beta(2, 5, size=n_rows) * 100, 0.0, 100.0, 1)
    campaign_response_flag = rng.choice([0, 1], size=n_rows, p=[0.85, 0.15])
    last_offer_accepted = rng.choice([0, 1], size=n_rows, p=[0.9, 0.1])
    days_since_last_interaction = rng.integers(0, 180, size=n_rows)

    # -----------------
    # Portabilidade e concorrência / retenção
    # -----------------
    portability_request_flag = rng.choice([0, 1], size=n_rows, p=[0.98, 0.02])
    sim_swap_count = rng.poisson(0.05, size=n_rows)
    competitor_offer_contact_flag = rng.choice([0, 1], size=n_rows, p=[0.96, 0.04])
    retention_offer_made = rng.choice([0, 1], size=n_rows, p=[0.95, 0.05])
    retention_offer_accepted = (retention_offer_made * rng.choice([0, 1], size=n_rows, p=[0.8, 0.2]))

    # -----------------
    # NPS e CSAT (geração correlacionada)
    # -----------------
    # base de satisfação: influenciada por service quality, bills, suporte, resolução, last_offer acceptance
    sat_base = (
        (avg_signal_quality / 5.0) * 2.5
        + (1 - call_drop_rate) * 2.0
        + (1 - invoice_shock_flag) * 1.0
        + (1 - complaints_30d) * 1.0
        + first_call_resolution_flag * 0.8
        - (days_since_last_interaction > 90).astype(float) * 0.5
    )
    # transforma em escala 0-10 com ruído
    nps_score_raw = sat_base + rng.normal(0.0, 1.5, size=n_rows)
    nps_score = np.clip(np.round((nps_score_raw / sat_base.max()) * 10.0), 0, 10).astype(int)
    # pequena correção: se sat_base é muito baixo, forçar scores menores
    nps_score = np.where(sat_base < 1.0, rng.integers(0, 6, size=n_rows), nps_score)

    # CSAT em 1-5
    csat_score = np.clip(np.round((sat_base / sat_base.max()) * 4.0 + 1.0 + rng.normal(0, 0.6, size=n_rows)), 1.0, 5.0,).round(1)

    # categorização NPS por pessoa
    def nps_category_from_score(s: int) -> str:
        if s >= 9:
            return "promoter"
        if s >= 7:
            return "passive"
        return "detractor"

    nps_category = np.array([nps_category_from_score(int(s)) for s in nps_score])
    nps_promoter_flag = (nps_score >= 9).astype(int)
    nps_detractor_flag = (nps_score <= 6).astype(int)

    # -----------------
    # Monta DataFrame
    # -----------------
    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            # perfil
            "age": age,
            "gender": gender,
            "region": region,
            # relacionamento / contrato
            "plan_type": plan_type,
            "plan_price": plan_price.round(2),
            "is_promotional_plan": is_promotional_plan,
            "discount_amount": discount_amount,
            "price_increase_last_3m": price_increase_last_3m,
            "months_to_contract_end": months_to_contract_end,
            "contract_renewal_date": contract_renewal_date,
            "has_loyalty": has_loyalty,
            "months_to_loyalty_end": months_to_loyalty_end,
            "loyalty_end_date": loyalty_end_date,
            # produtos/serviços
            "internet_service": internet_service,
            "has_tv": has_tv,
            "has_fixed": has_fixed,
            "tenure_months": tenure_months,
            # qualidade do serviço
            "network_outages_30d": network_outages_30d,
            "avg_signal_quality": avg_signal_quality,
            "call_drop_rate": call_drop_rate,
            "avg_internet_speed": avg_internet_speed,
            "service_failures_30d": service_failures_30d,
            "installation_delay_days": installation_delay_days,
            "repair_visits_90d": repair_visits_90d,
            # uso
            "minutes_monthly": minutes_monthly,
            "data_gb_monthly": data_gb_monthly,
            "sms_monthly": sms_monthly,
            "usage_delta_pct": usage_delta_pct,
            "days_since_last_usage": days_since_last_usage,
            "active_days_30d": active_days_30d,
            "night_usage_pct": night_usage_pct,
            "roaming_usage": roaming_usage,
            "topup_frequency": topup_frequency,
            "avg_usage_last_3m": avg_usage_last_3m,
            "usage_trend_3m": usage_trend_3m,
            # financeiro
            "payment_method": payment_method,
            "billing_type": billing_type,
            "late_payments_6m": late_payments_6m,
            "invoice_shock_flag": invoice_shock_flag,
            "avg_bill_last_6m": avg_bill_last_6m,
            "bill_variation_pct": bill_variation_pct,
            "collection_attempts_30d": collection_attempts_30d,
            "monthly_charges": monthly_charges,
            "default_flag": default_flag,
            "days_past_due": days_past_due,
            # atendimento
            "support_calls_30d": support_calls_30d,
            "support_calls_90d": support_calls_90d,
            "complaints_30d": complaints_30d,
            "tickets_open_30d": tickets_open_30d,
            "last_complaint_reason": last_complaint_reason,
            "resolution_time_avg": resolution_time_avg,
            "first_call_resolution_flag": first_call_resolution_flag,
            "transferred_calls_count": transferred_calls_count,
            "last_contact_channel": last_contact_channel,
            # engajamento digital
            "app_login_30d": app_login_30d,
            "self_service_usage_30d": self_service_usage_30d,
            "marketing_open_rate": marketing_open_rate,
            "campaign_response_flag": campaign_response_flag,
            "last_offer_accepted": last_offer_accepted,
            "days_since_last_interaction": days_since_last_interaction,
            # portabilidade / concorrência / retenção
            "portability_request_flag": portability_request_flag,
            "sim_swap_count": sim_swap_count,
            "competitor_offer_contact_flag": competitor_offer_contact_flag,
            "retention_offer_made": retention_offer_made,
            "retention_offer_accepted": retention_offer_accepted,
            # NPS / CSAT
            "nps_score": nps_score,
            "nps_category": nps_category,
            "nps_promoter_flag": nps_promoter_flag,
            "nps_detractor_flag": nps_detractor_flag,
            "csat_score": csat_score,
        }
    )

    return df


# -------------------------
# Label generation (churn)
# -------------------------
def add_churn_label_extended(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Gera rótulo `churn` usando lógica baseada em múltiplos fatores (maior realismo).
    Retorna df modificado (coluna 'churn' adicionada).
    """
    n = len(df)
    risk = np.zeros(n)

    # contrato: mensal / pré-pago / curto prazo aumentam risco
    risk += (df["plan_type"] == "pre").astype(float) * 1.6
    risk += (df["plan_type"] == "controle").astype(float) * 0.8
    risk += (df["months_to_contract_end"] < 2).astype(float) * 1.0

    # finance
    risk += (df["late_payments_6m"] >= 1).astype(float) * 1.5
    risk += df["invoice_shock_flag"].astype(float) * 1.6
    risk += (df["collection_attempts_30d"] >= 1).astype(float) * 1.0
    # default flag (inadimplência)
    risk += df["default_flag"].astype(float) * 1.8

    # qualidade / experiência
    risk += (df["avg_signal_quality"] < 3.0).astype(float) * 1.2
    risk += (df["call_drop_rate"] > 0.05).astype(float) * 1.0
    risk += (df["service_failures_30d"] >= 1).astype(float) * 0.9
    risk += (df["support_calls_30d"] >= 3).astype(float) * 0.7
    risk += df["complaints_30d"].astype(float) * 1.0
    risk += (df["resolution_time_avg"] > 3.0).astype(float) * 0.6
    risk += (df["first_call_resolution_flag"] == 0).astype(float) * 0.5

    # engajamento
    risk += (df["days_since_last_usage"] > 30).astype(float) * 0.9
    risk += (df["app_login_30d"] == 0).astype(float) * 0.4
    risk += (df["days_since_last_interaction"] > 90).astype(float) * 0.5

    # concorrência / portabilidade
    risk += df["portability_request_flag"].astype(float) * 1.8
    risk += df["competitor_offer_contact_flag"].astype(float) * 1.2

    # retenção: se aceitou oferta de retenção, reduz risco
    risk -= df["retention_offer_accepted"].astype(float) * 1.8
    risk -= df["last_offer_accepted"].astype(float) * 0.8

    # NPS/Csat: detractors aumentam muito o risco, promoters reduzem
    risk += df["nps_detractor_flag"].astype(float) * 1.8
    risk -= df["nps_promoter_flag"].astype(float) * 1.2
    # CSAT scaling (1-5): lower csat increases risk
    risk += ((5.0 - df["csat_score"]) / 5.0) * 1.2

    # tenure and loyalty
    risk -= (df["tenure_months"] > 24).astype(float) * 0.9
    risk -= (df["has_loyalty"] == 1).astype(float) * 0.8

    # price sensitivity
    risk += (df["price_increase_last_3m"] == 1).astype(float) * 1.0
    risk += ((df["monthly_charges"] / (df["plan_price"] + 1e-6)) > 1.5).astype(float) * 0.8

    # converte score em probabilidade via sigmoide
    # calibramos o offset para atingir churn ~20-25% em média
    offset = 3.5
    prob = 1 / (1 + np.exp(-(risk - offset)))
    noise = rng.uniform(0, 0.06, size=n)
    churn = (rng.random(size=n) < (prob + noise)).astype(int)
    df["churn"] = churn
    df["churn_probability"] = prob.round(4)
    return df


def inject_quality_issues(
    df: pd.DataFrame,
    rng: np.random.Generator,
    duplicate_row_rate: float = 0.0,
    duplicate_id_rate: float = 0.0,
    missing_noise_rate: float = 0.0,
    invalid_value_rate: float = 0.0,
) -> pd.DataFrame:
    """Injeta problemas de qualidade para simular ambiente real (dirty data)."""
    n_rows_original = len(df)

    def _n_from_rate(rate: float, n: int) -> int:
        if rate <= 0:
            return 0
        return int(np.ceil(rate * n))

    # 1) Duplica linhas inteiras (duplicidade de registro)
    n_dup_rows = _n_from_rate(duplicate_row_rate, n_rows_original)
    if n_dup_rows > 0:
        sampled_idx = rng.choice(df.index.to_numpy(), size=n_dup_rows, replace=True)
        duplicated_chunk = df.loc[sampled_idx].copy()
        df = pd.concat([df, duplicated_chunk], ignore_index=True)

    # 2) Duplica customer_id em linhas diferentes
    n_dup_ids = _n_from_rate(duplicate_id_rate, len(df))
    if n_dup_ids > 0 and "customer_id" in df.columns:
        target_idx = rng.choice(df.index.to_numpy(), size=n_dup_ids, replace=False)
        source_idx = rng.choice(df.index.to_numpy(), size=n_dup_ids, replace=True)
        df.loc[target_idx, "customer_id"] = df.loc[source_idx, "customer_id"].to_numpy()

    # 3) Missing aleatorio em colunas-chave (exceto id/target)
    n_missing = _n_from_rate(missing_noise_rate, len(df))
    missing_cols = [
        "gender",
        "region",
        "internet_service",
        "payment_method",
        "monthly_charges",
        "avg_signal_quality",
        "nps_score",
        "csat_score",
    ]
    missing_cols = [col for col in missing_cols if col in df.columns]
    if n_missing > 0 and missing_cols:
        for col in missing_cols:
            idx = rng.choice(df.index.to_numpy(), size=n_missing, replace=False)
            df.loc[idx, col] = np.nan

    # 4) Valores invalidos (fora de faixa, categorias inesperadas, inconsistencias logicas)
    n_invalid = _n_from_rate(invalid_value_rate, len(df))
    if n_invalid > 0:
        idx = rng.choice(df.index.to_numpy(), size=n_invalid, replace=False)

        if "age" in df.columns:
            df.loc[idx, "age"] = rng.choice([-10, 0, 130], size=n_invalid)
        if "monthly_charges" in df.columns:
            df.loc[idx, "monthly_charges"] = rng.uniform(-50, 0, size=n_invalid).round(2)
        if "avg_signal_quality" in df.columns:
            df.loc[idx, "avg_signal_quality"] = rng.uniform(5.5, 8.0, size=n_invalid).round(2)
        if "call_drop_rate" in df.columns:
            df.loc[idx, "call_drop_rate"] = rng.uniform(1.1, 1.8, size=n_invalid).round(4)
        if "nps_score" in df.columns:
            df.loc[idx, "nps_score"] = rng.choice([-2, 11, 15], size=n_invalid)

        if "plan_type" in df.columns:
            df.loc[idx, "plan_type"] = rng.choice(["desconhecido", "??", ""], size=n_invalid)
        if "payment_method" in df.columns:
            df.loc[idx, "payment_method"] = rng.choice(["crypto", "", "na"], size=n_invalid)

        # Inconsistencia: sem fidelidade, mas com meses para fim da fidelidade
        if "has_loyalty" in df.columns and "months_to_loyalty_end" in df.columns:
            df.loc[idx, "has_loyalty"] = 0
            df.loc[idx, "months_to_loyalty_end"] = rng.integers(1, 24, size=n_invalid)

        # Inconsistencia: nao inadimplente, mas com atraso > 0
        if "default_flag" in df.columns and "days_past_due" in df.columns:
            df.loc[idx, "default_flag"] = 0
            df.loc[idx, "days_past_due"] = rng.integers(1, 90, size=n_invalid)

    return df


# -------------------------
# Main
# -------------------------
def main(
    n_rows: int = 50_000,
    seed: int = 42,
    out_dir: str | Path = "data/raw",
    duplicate_row_rate: float = 0.03,
    duplicate_id_rate: float = 0.02,
    missing_noise_rate: float = 0.01,
    invalid_value_rate: float = 0.01,
) -> None:
    configure_logging()
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("generate.start", n_rows=n_rows, seed=seed, out_dir=str(out_dir))

    df = synth_features_extended(n_rows, rng)
    logger.info("features.generated", n_rows=len(df), sample_customer=df["customer_id"].iloc[0])

    df = add_churn_label_extended(df, rng)
    df = inject_quality_issues(
        df,
        rng,
        duplicate_row_rate=duplicate_row_rate,
        duplicate_id_rate=duplicate_id_rate,
        missing_noise_rate=missing_noise_rate,
        invalid_value_rate=invalid_value_rate,
    )

    churn_rate = df["churn"].mean()
    n_promoters = int(df["nps_promoter_flag"].sum())
    n_detractors = int(df["nps_detractor_flag"].sum())
    n_passives = int(((df["nps_category"] == "passive")).sum())
    global_nps = (n_promoters / n_rows - n_detractors / n_rows) * 100.0
    duplicated_rows = int(df.duplicated().sum())
    duplicated_ids = int(df["customer_id"].duplicated().sum()) if "customer_id" in df.columns else 0
    missing_cells = int(df.isna().sum().sum())

    out_path = out_dir / "telecom_churn_base_extended.csv"
    df.to_csv(out_path, index=False)

    logger.info(
        "generate.wrote",
        path=str(out_path),
        rows=len(df),
        churn_rate=f"{churn_rate:.3f}",
        duplicate_rows=duplicated_rows,
        duplicate_customer_ids=duplicated_ids,
        missing_cells=missing_cells,
        n_promoters=n_promoters,
        n_passives=n_passives,
        n_detractors=n_detractors,
        global_nps=f"{global_nps:.1f}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera base sintética telecom estendida com NPS por cliente.")
    parser.add_argument("--n-rows", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="data/raw", help="Diretório de saída para features (raw).")
    parser.add_argument("--duplicate-row-rate", type=float, default=0.03)
    parser.add_argument("--duplicate-id-rate", type=float, default=0.02)
    parser.add_argument("--missing-noise-rate", type=float, default=0.01)
    parser.add_argument("--invalid-value-rate", type=float, default=0.01)
    args = parser.parse_args()
    main(
        n_rows=args.n_rows,
        seed=args.seed,
        out_dir=args.out_dir,
        duplicate_row_rate=args.duplicate_row_rate,
        duplicate_id_rate=args.duplicate_id_rate,
        missing_noise_rate=args.missing_noise_rate,
        invalid_value_rate=args.invalid_value_rate,
    )