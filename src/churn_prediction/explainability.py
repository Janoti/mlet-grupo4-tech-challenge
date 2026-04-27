"""Explainability via SHAP: top-N features mais importantes com valores SHAP.

Requer `shap` instalado como dependência opcional:
    poetry install --extras explainability

Uso típico nos notebooks:
    from churn_prediction.explainability import compute_shap_top_features

    top10 = compute_shap_top_features(
        model=trained_pipeline,
        X=X_test,
        n_top=10,
    )
    print(top10)  # DataFrame: feature, mean_abs_shap, direction
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _require_shap():
    """Importa shap ou levanta ImportError com mensagem orientativa."""
    try:
        import shap  # noqa: F401
        return shap
    except ImportError as exc:
        raise ImportError(
            "shap não está instalado. Rode: poetry install --extras explainability"
        ) from exc


def compute_shap_values(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    max_samples: int = 500,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Calcula valores SHAP para um modelo (sklearn, pipeline ou tree).

    Detecta automaticamente o tipo de explainer apropriado:
    - TreeExplainer para RandomForest / GradientBoosting
    - LinearExplainer para LogisticRegression / LinearRegression
    - KernelExplainer como fallback (mais lento)

    Args:
        model: Modelo treinado ou Pipeline sklearn (a última etapa é o estimador).
        X: Dados para calcular SHAP values.
        max_samples: Limita quantas amostras usar (SHAP é custoso). Default 500.
        random_state: Seed para amostragem reprodutível.

    Returns:
        Tupla (shap_values_array, X_sample_as_df).
        shap_values_array tem shape (n_samples, n_features).
    """
    shap = _require_shap()

    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    if len(X_df) > max_samples:
        X_df = X_df.sample(n=max_samples, random_state=random_state)

    # Se for pipeline sklearn, separa preprocessor do estimador final
    if hasattr(model, "steps") and len(model.steps) > 1:
        preprocessor = model[:-1]
        estimator = model.steps[-1][1]
        X_transformed = preprocessor.transform(X_df)
    else:
        estimator = model
        X_transformed = X_df.values if isinstance(X_df, pd.DataFrame) else X_df

    # Escolhe explainer conforme o tipo do estimador
    estimator_class = estimator.__class__.__name__

    if estimator_class in ("RandomForestClassifier", "GradientBoostingClassifier",
                            "XGBClassifier", "LGBMClassifier"):
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_transformed)
        # Para classificadores binários, pega valores da classe positiva
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    elif estimator_class in ("LogisticRegression", "LinearRegression", "Ridge", "Lasso"):
        explainer = shap.LinearExplainer(estimator, X_transformed)
        shap_values = explainer.shap_values(X_transformed)
    else:
        # Fallback lento mas genérico
        def predict_fn(data):
            if hasattr(estimator, "predict_proba"):
                return estimator.predict_proba(data)[:, 1]
            return estimator.predict(data)

        background = shap.sample(X_transformed, min(100, len(X_transformed)),
                                  random_state=random_state)
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_transformed, silent=True)

    return np.asarray(shap_values), X_df


def compute_shap_top_features(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    n_top: int = 10,
    max_samples: int = 500,
    random_state: int = 42,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Retorna top-N features por importância SHAP (mean |SHAP|).

    Args:
        model: Modelo ou Pipeline sklearn treinado.
        X: Dados para análise.
        n_top: Quantas features retornar (default 10).
        max_samples: Máximo de amostras para cálculo (SHAP é custoso).
        random_state: Seed.
        feature_names: Override dos nomes (se X for ndarray sem colunas).

    Returns:
        DataFrame com colunas:
        - feature: Nome da feature
        - mean_abs_shap: Importância (média do valor absoluto do SHAP)
        - mean_shap: Direção média (positivo = aumenta churn, negativo = reduz)
        - rank: Posição no top-N (1 = mais importante)
    """
    shap_values, X_sample = compute_shap_values(
        model=model, X=X, max_samples=max_samples, random_state=random_state,
    )

    if feature_names is None:
        if hasattr(X_sample, "columns"):
            feature_names = list(X_sample.columns)
        else:
            feature_names = [f"f{i}" for i in range(shap_values.shape[1])]

    if len(feature_names) != shap_values.shape[1]:
        # Fallback se pipeline gerou novas colunas (OHE) e não temos names
        feature_names = [f"f{i}" for i in range(shap_values.shape[1])]

    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_signed,
    })
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df.head(n_top)


__all__ = [
    "compute_shap_top_features",
    "compute_shap_values",
]
