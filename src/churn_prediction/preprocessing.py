"""Pipeline de pré-processamento sklearn para churn prediction.

Constrói ColumnTransformer com:
- Numéricas: mediana + StandardScaler
- Categóricas: moda + OneHotEncoder

Evita training-serving skew ao encapsular todas as transformações
em um único objeto Pipeline serializável.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def detect_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Retorna listas de colunas numéricas e categóricas."""
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Cria ColumnTransformer baseado nos tipos de coluna do DataFrame.

    Retorna um transformador não fitado — deve ser usado dentro de
    um Pipeline completo com o modelo.
    """
    num_cols, cat_cols = detect_column_types(X)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    return preprocessor


def build_sklearn_pipeline(X: pd.DataFrame, model) -> Pipeline:
    """Retorna Pipeline completo: preprocessamento + modelo."""
    preprocessor = build_preprocessor(X)
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])
