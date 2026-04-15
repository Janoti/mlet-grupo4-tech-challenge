"""Pipeline orquestrado de treinamento e avaliação de churn."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn_prediction.config import RANDOM_STATE, SENSITIVE_FEATURES, TEST_SIZE
from churn_prediction.data_cleaning import (
    clean_dataset,
    drop_leakage_columns,
    extract_sensitive_features,
)
from churn_prediction.evaluation import (
    compute_business_metrics,
    compute_classification_metrics,
)
from churn_prediction.preprocessing import build_sklearn_pipeline

logger = logging.getLogger(__name__)


def dataset_hash(df: pd.DataFrame) -> str:
    """Gera hash SHA-256 do DataFrame para versionamento."""
    return hashlib.sha256(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:12]


def prepare_data(
    raw_path: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Executa limpeza, split e extração de features sensíveis.

    Returns:
        Dicionário com X_train, X_test, y_train, y_test,
        sensitive_train, sensitive_test, dataset_version.
    """
    df = clean_dataset(raw_path)
    ds_hash = dataset_hash(df)
    logger.info("Dataset carregado: %d linhas, hash=%s", len(df), ds_hash)

    sensitive = extract_sensitive_features(df)
    df = drop_leakage_columns(df)

    target_col = "churn"
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # Remove age_group do X (usado apenas como sensível)
    if "age_group" in X.columns:
        X = X.drop(columns=["age_group"])

    split_arrays = [X, y] + [sensitive[f] for f in SENSITIVE_FEATURES if f in sensitive]

    results = train_test_split(
        *split_arrays,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Desempacota resultados do split
    X_train, X_test = results[0], results[1]
    y_train, y_test = results[2], results[3]

    sensitive_train = {}
    sensitive_test = {}
    idx = 4
    for feat in SENSITIVE_FEATURES:
        if feat in sensitive:
            sensitive_train[feat] = results[idx]
            sensitive_test[feat] = results[idx + 1]
            idx += 2

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "sensitive_train": sensitive_train,
        "sensitive_test": sensitive_test,
        "dataset_version": ds_hash,
    }


def train_and_evaluate(
    model,
    model_name: str,
    data: dict,
    experiment_name: str = "churn-baselines",
    mlflow_uri: str = "file:./mlruns",
) -> dict:
    """Treina modelo sklearn, avalia e registra no MLflow.

    Args:
        model: Instância de estimador sklearn (não fitado).
        model_name: Nome para o run no MLflow.
        data: Saída de prepare_data().
        experiment_name: Nome do experimento MLflow.
        mlflow_uri: URI do MLflow tracking.

    Returns:
        Dicionário com métricas técnicas e de negócio.
    """
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    pipe = build_sklearn_pipeline(X_train, model)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = None
    if hasattr(pipe, "predict_proba"):
        y_prob = pipe.predict_proba(X_test)[:, 1]

    tech_metrics = compute_classification_metrics(
        np.asarray(y_test), y_pred, y_prob
    )
    biz_metrics = compute_business_metrics(np.asarray(y_test), y_pred)
    all_metrics = {**tech_metrics, **biz_metrics}

    # Log no MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        mlflow.log_param("dataset_version", data["dataset_version"])
        mlflow.log_metrics(all_metrics)
        mlflow.sklearn.log_model(pipe, "model")

    logger.info(
        "[%s] f1=%.4f roc_auc=%.4f valor_liquido=R$%.0f",
        model_name,
        tech_metrics.get("f1", 0),
        tech_metrics.get("roc_auc", 0),
        biz_metrics["valor_liquido"],
    )
    return all_metrics


def save_processed_splits(data: dict, out_dir: str = "data/processed") -> Path:
    """Salva splits processados para uso pelo MLP PyTorch."""
    import joblib

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    artifact = {
        "X_train": data["X_train"],
        "X_test": data["X_test"],
        "y_train": data["y_train"],
        "y_test": data["y_test"],
        "dataset_version": data["dataset_version"],
    }
    file_path = out_path / "baseline_splits.joblib"
    joblib.dump(artifact, file_path)
    logger.info("Splits salvos em %s", file_path)
    return file_path
