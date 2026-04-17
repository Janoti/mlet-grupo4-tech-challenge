"""Testes do módulo registry: seleção, registro e exportação do champion."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Smoke test: importação
# ---------------------------------------------------------------------------

def test_import_registry():
    from churn_prediction.registry import (
        PyTorchChurnWrapper,
        export_champion,
        find_champion,
        register_champion,
    )

    assert callable(find_champion)
    assert callable(register_champion)
    assert callable(export_champion)
    assert callable(PyTorchChurnWrapper)


# ---------------------------------------------------------------------------
# find_champion
# ---------------------------------------------------------------------------

def _make_search_runs_df(rows: list[dict]) -> pd.DataFrame:
    """Helper: cria DataFrame no formato retornado por mlflow.search_runs()."""
    return pd.DataFrame(rows)


@patch("churn_prediction.registry.MlflowClient")
@patch("churn_prediction.registry.mlflow")
def test_find_champion_returns_best(mock_mlflow, mock_client_cls):
    """Dado 2 runs, retorna o de maior valor_liquido."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    exp1 = MagicMock()
    exp1.experiment_id = "1"
    mock_client.search_experiments.return_value = [exp1]

    mock_mlflow.search_runs.return_value = _make_search_runs_df([
        {
            "run_id": "aaa",
            "experiment_id": "1",
            "tags.mlflow.runName": "mlp_pytorch_v1",
            "artifact_uri": "file:./mlruns/1/aaa/artifacts",
            "metrics.valor_liquido": 1194000.0,
            "metrics.roc_auc": 0.8772,
            "metrics.f1": 0.74,
            "metrics.pr_auc": 0.84,
            "metrics.accuracy": 0.80,
            "params.dataset_version": "sha256:abc",
        },
        {
            "run_id": "bbb",
            "experiment_id": "1",
            "tags.mlflow.runName": "log_reg",
            "artifact_uri": "file:./mlruns/1/bbb/artifacts",
            "metrics.valor_liquido": 1190800.0,
            "metrics.roc_auc": 0.8783,
            "metrics.f1": 0.74,
            "metrics.pr_auc": 0.84,
            "metrics.accuracy": 0.80,
            "params.dataset_version": "sha256:abc",
        },
    ])

    from churn_prediction.registry import find_champion

    result = find_champion(tracking_uri="file:./mlruns")

    assert result["run_id"] == "aaa"
    assert result["run_name"] == "mlp_pytorch_v1"
    assert result["metrics"]["valor_liquido"] == 1194000.0


@patch("churn_prediction.registry.MlflowClient")
@patch("churn_prediction.registry.mlflow")
def test_find_champion_tiebreaker(mock_mlflow, mock_client_cls):
    """Se valor_liquido empata, desempata por roc_auc."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    exp1 = MagicMock()
    exp1.experiment_id = "1"
    mock_client.search_experiments.return_value = [exp1]

    # Mesmo valor_liquido, roc_auc diferente — search_runs já vem ordenado
    mock_mlflow.search_runs.return_value = _make_search_runs_df([
        {
            "run_id": "ccc",
            "experiment_id": "1",
            "tags.mlflow.runName": "gradient_boosting",
            "artifact_uri": "uri_c",
            "metrics.valor_liquido": 1190000.0,
            "metrics.roc_auc": 0.8815,
            "metrics.f1": 0.74,
            "metrics.pr_auc": 0.84,
            "metrics.accuracy": 0.81,
            "params.dataset_version": "sha256:abc",
        },
        {
            "run_id": "ddd",
            "experiment_id": "1",
            "tags.mlflow.runName": "log_reg",
            "artifact_uri": "uri_d",
            "metrics.valor_liquido": 1190000.0,
            "metrics.roc_auc": 0.8783,
            "metrics.f1": 0.74,
            "metrics.pr_auc": 0.84,
            "metrics.accuracy": 0.80,
            "params.dataset_version": "sha256:abc",
        },
    ])

    from churn_prediction.registry import find_champion

    result = find_champion(tracking_uri="file:./mlruns")

    assert result["run_id"] == "ccc"
    assert result["run_name"] == "gradient_boosting"


@patch("churn_prediction.registry.MlflowClient")
@patch("churn_prediction.registry.mlflow")
def test_find_champion_no_runs_raises(mock_mlflow, mock_client_cls):
    """Se não há runs, levanta ValueError."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    exp1 = MagicMock()
    exp1.experiment_id = "1"
    mock_client.search_experiments.return_value = [exp1]

    mock_mlflow.search_runs.return_value = pd.DataFrame()

    from churn_prediction.registry import find_champion

    with pytest.raises(ValueError, match="Nenhum run encontrado"):
        find_champion(tracking_uri="file:./mlruns")


# ---------------------------------------------------------------------------
# export_champion
# ---------------------------------------------------------------------------

def _make_fake_pipeline():
    """Cria um pipeline sklearn real e serializável para testes."""
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", DummyClassifier(strategy="constant", constant=0)),
    ]).fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))


def test_export_creates_joblib_and_metadata(tmp_path):
    """Verifica que export_champion cria joblib e champion_metadata.json."""
    from churn_prediction.registry import export_champion

    champion = {
        "run_id": "abc123",
        "run_name": "log_reg",
        "experiment_id": "1",
        "artifact_uri": "",
        "metrics": {"valor_liquido": 1190800.0, "roc_auc": 0.8783},
        "params": {"dataset_version": "sha256:test"},
        "all_candidates": [
            {"run_name": "log_reg", "valor_liquido": 1190800.0, "roc_auc": 0.8783},
        ],
    }

    n_features = 5
    rng = np.random.RandomState(42)
    X_train = pd.DataFrame(rng.randn(100, n_features), columns=[f"f{i}" for i in range(n_features)])
    y_train = pd.Series(rng.randint(0, 2, 100))
    data = {"X_train": X_train, "y_train": y_train}

    out_dir = str(tmp_path / "models")

    with patch("churn_prediction.registry._detect_flavor", return_value="sklearn_retrain"), \
         patch("churn_prediction.registry._load_champion_sklearn") as mock_load:
        mock_load.return_value = _make_fake_pipeline()

        result = export_champion(
            champion=champion,
            version=1,
            data=data,
            out_dir=out_dir,
        )

    assert result.exists()
    assert result.name == "churn_pipeline.joblib"

    metadata_path = Path(out_dir) / "champion_metadata.json"
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text())
    assert metadata["champion_run_name"] == "log_reg"
    assert metadata["champion_run_id"] == "abc123"
    assert metadata["registry_version"] == 1
    assert metadata["selection_metric"] == "valor_liquido"


def test_metadata_contains_all_candidates(tmp_path):
    """O JSON deve conter a lista rankeada de todos os candidatos."""
    from churn_prediction.registry import export_champion

    candidates = [
        {"run_name": "mlp_pytorch_v1", "valor_liquido": 1194000.0, "roc_auc": 0.8772},
        {"run_name": "log_reg", "valor_liquido": 1190800.0, "roc_auc": 0.8783},
        {"run_name": "gradient_boosting", "valor_liquido": 1183300.0, "roc_auc": 0.8815},
    ]
    champion = {
        "run_id": "xyz",
        "run_name": "mlp_pytorch_v1",
        "experiment_id": "1",
        "artifact_uri": "",
        "metrics": {"valor_liquido": 1194000.0, "roc_auc": 0.8772},
        "params": {"dataset_version": "sha256:test"},
        "all_candidates": candidates,
    }

    data = {
        "X_train": pd.DataFrame({"a": [1, 2]}),
        "y_train": pd.Series([0, 1]),
    }
    out_dir = str(tmp_path / "models")

    with patch("churn_prediction.registry._detect_flavor", return_value="sklearn_retrain"), \
         patch("churn_prediction.registry._load_champion_sklearn", return_value=_make_fake_pipeline()):
        export_champion(champion=champion, version=None, data=data, out_dir=out_dir)

    metadata = json.loads((Path(out_dir) / "champion_metadata.json").read_text())

    assert len(metadata["all_candidates"]) == 3
    assert metadata["all_candidates"][0]["run_name"] == "mlp_pytorch_v1"
    assert metadata["all_candidates"][1]["run_name"] == "log_reg"
    assert metadata["all_candidates"][2]["run_name"] == "gradient_boosting"
