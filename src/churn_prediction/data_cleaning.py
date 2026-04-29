"""Limpeza e saneamento da base bruta de churn.

Replica as transformações do notebook 01_eda.ipynb / 02_baselines.ipynb
de forma reutilizável e testável.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn_prediction.config import (
    AGE_BINS,
    AGE_LABELS,
    BUSINESS_CLIP_RULES,
    LEAKAGE_COLS,
    VALID_PAYMENT_METHODS,
)


def load_raw_data(path: str) -> pd.DataFrame:
    """Carrega CSV bruto e converte colunas de data."""
    df = pd.read_csv(path)
    for col in ("contract_renewal_date", "loyalty_end_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicatas exatas e duplicatas por customer_id (mantém primeira)."""
    df = df.drop_duplicates().copy()
    if "customer_id" in df.columns:
        df = df.drop_duplicates(subset=["customer_id"], keep="first")
    return df.reset_index(drop=True)


def standardize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza colunas categóricas (strip, lowercase, valores inválidos → unknown)."""
    df = df.copy()

    if "gender" in df.columns:
        df["gender"] = (
            df["gender"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"": np.nan, "nan": np.nan})
            .fillna("unknown")
        )

    if "plan_type" in df.columns:
        df["plan_type"] = (
            df["plan_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"desconhecido": "unknown", "??": "unknown", "nan": "unknown"})
        )

    if "payment_method" in df.columns:
        df["payment_method"] = df["payment_method"].apply(
            lambda x: x if str(x).strip().lower() in VALID_PAYMENT_METHODS else "unknown"
        )

    return df


def clip_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica regras de clipping para valores fora de faixa."""
    df = df.copy()
    for col, (low, high) in BUSINESS_CLIP_RULES.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=low, upper=high)
    return df


def create_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Cria coluna age_group a partir de age."""
    df = df.copy()
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True
        )
        df["age_group"] = df["age_group"].astype(str).replace({"nan": "unknown"})
    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas com risco de leakage."""
    cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def extract_sensitive_features(
    df: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Extrai atributos sensíveis para fairness antes de remover do DataFrame."""
    sensitive = {}
    for col in ("gender", "age_group", "region", "plan_type"):
        if col in df.columns:
            sensitive[col] = df[col].copy()
    return sensitive


def clean_dataset(path: str) -> pd.DataFrame:
    """Pipeline completo de limpeza: load → dedup → categoricals → clip → age_group."""
    df = load_raw_data(path)
    df = remove_duplicates(df)
    df = standardize_categoricals(df)
    df = create_age_group(df)
    df = clip_numeric_features(df)
    return df
