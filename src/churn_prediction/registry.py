"""Model Registry: seleção, registro e exportação do champion de churn.

Fluxo:
    1. find_champion()      — consulta todos os runs no MLflow, rankeia por métrica de negócio
    2. register_champion()  — registra o modelo vencedor no MLflow Model Registry
    3. export_champion()    — exporta joblib + champion_metadata.json para servir na API
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from torch import nn

from churn_prediction.preprocessing import build_preprocessor

logger = logging.getLogger(__name__)

# Métricas registradas nos runs que serão incluídas no metadata
CANDIDATE_METRICS = ["valor_liquido", "roc_auc", "f1", "pr_auc", "accuracy"]


class PyTorchChurnWrapper:
    """Adapta MLP PyTorch à interface sklearn (predict / predict_proba).

    Empacota o modelo PyTorch + preprocessor sklearn fitado em um único
    objeto serializável com joblib, permitindo que a API FastAPI use a
    mesma interface independente do flavor do champion.

    Usa __reduce__ para contornar problemas de pickling do modelo PyTorch
    salvando apenas state_dict + arquitectura, não a instância direta.
    """

    def __init__(self, model: nn.Module, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self._model_state_dict = None
        self._model_config = None

    def _save_state(self):
        """Guarda state_dict e config para serialização."""
        self._model_state_dict = self.model.state_dict()
        # Extrair dimensões do modelo
        first_layer = self.model.net[0]
        input_dim = first_layer.in_features
        self._model_config = {
            "input_dim": input_dim,
        }

    def _load_state(self):
        """Reconstrói o modelo a partir do state_dict e config."""
        from churn_prediction.model import MLP

        if self._model_state_dict is None:
            return  # Modelo já carregado (pós-unpickle)

        self.model = MLP(**self._model_config)
        self.model.load_state_dict(self._model_state_dict)
        self._model_state_dict = None
        self._model_config = None

    def __reduce__(self):
        """Serialização customizada para joblib/pickle.

        Salva state_dict + config em vez da instância do modelo PyTorch.
        """
        self._save_state()
        return (
            _rebuild_pytorch_wrapper,
            (self._model_state_dict, self._model_config, self.preprocessor),
        )

    def predict_proba(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_processed = self.preprocessor.transform(X).astype("float32")
        else:
            X_processed = np.asarray(X, dtype="float32")
        tensor = torch.tensor(X_processed, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            proba = torch.sigmoid(logits).numpy().flatten()
        return np.column_stack([1 - proba, proba])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _rebuild_pytorch_wrapper(state_dict, config, preprocessor):
    """Função helper para desserializar PyTorchChurnWrapper."""
    from churn_prediction.model import MLP

    model = MLP(**config)
    model.load_state_dict(state_dict)
    wrapper = PyTorchChurnWrapper(model, preprocessor)
    return wrapper


def find_champion(
    tracking_uri: str = "file:./mlruns",
    metric: str = "valor_liquido",
    tiebreaker: str = "roc_auc",
    experiment_names: list[str] | None = None,
) -> dict:
    """Consulta todos os runs no MLflow e retorna o melhor por métrica de negócio.

    Args:
        tracking_uri: URI do MLflow tracking store.
        metric: Métrica primária para seleção (maior = melhor).
        tiebreaker: Métrica de desempate (maior = melhor).
        experiment_names: Lista de nomes de experimentos a consultar.
            Se None, consulta todos os experimentos.

    Returns:
        Dict com run_id, run_name, experiment_id, artifact_uri, metrics, params.

    Raises:
        ValueError: Se não encontrar nenhum run com a métrica principal.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)

    if experiment_names:
        experiments = [
            client.get_experiment_by_name(name)
            for name in experiment_names
        ]
        experiment_ids = [e.experiment_id for e in experiments if e is not None]
    else:
        experiments = client.search_experiments()
        experiment_ids = [
            e.experiment_id
            for e in experiments
            if e.experiment_id != "0"
        ]

    if not experiment_ids:
        raise ValueError("Nenhum experimento encontrado no MLflow.")

    runs_df = mlflow.search_runs(
        experiment_ids=experiment_ids,
        filter_string=f"metrics.{metric} > 0",
        order_by=[f"metrics.{metric} DESC", f"metrics.{tiebreaker} DESC"],
    )

    if runs_df.empty:
        raise ValueError(
            f"Nenhum run encontrado com métrica '{metric}' > 0. "
            "Execute os notebooks de treinamento primeiro."
        )

    best = runs_df.iloc[0]

    # Coleta métricas do champion
    metrics = {}
    for m in CANDIDATE_METRICS:
        col = f"metrics.{m}"
        if col in best.index and pd.notna(best[col]):
            metrics[m] = float(best[col])

    # Coleta métricas de todos os candidatos para o metadata
    all_candidates = []
    for _, row in runs_df.iterrows():
        candidate = {"run_name": row.get("tags.mlflow.runName", "unknown")}
        for m in CANDIDATE_METRICS:
            col = f"metrics.{m}"
            if col in row.index and pd.notna(row[col]):
                candidate[m] = float(row[col])
        all_candidates.append(candidate)

    champion = {
        "run_id": best["run_id"],
        "run_name": best.get("tags.mlflow.runName", "unknown"),
        "experiment_id": best["experiment_id"],
        "artifact_uri": best.get("artifact_uri", ""),
        "metrics": metrics,
        "params": {
            k.replace("params.", ""): v
            for k, v in best.items()
            if isinstance(k, str) and k.startswith("params.") and pd.notna(v)
        },
        "all_candidates": all_candidates,
    }

    logger.info(
        "Champion selecionado: %s (run_id=%s) | %s=%.2f | %s=%.4f",
        champion["run_name"],
        champion["run_id"][:8],
        metric,
        metrics.get(metric, 0),
        tiebreaker,
        metrics.get(tiebreaker, 0),
    )
    return champion


def register_champion(
    run_id: str,
    model_name: str = "churn-champion",
    tracking_uri: str = "file:./mlruns",
) -> int | None:
    """Registra o modelo champion no MLflow Model Registry.

    Args:
        run_id: ID do run vencedor.
        model_name: Nome do modelo no registry.
        tracking_uri: URI do MLflow tracking store.

    Returns:
        Número da versão registrada, ou None se o registro falhar
        (ex: file store sem suporte completo ao registry).
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)

    # Detecta o nome do artefato (sklearn usa "model", pytorch usa "mlp_model")
    artifacts = client.list_artifacts(run_id)
    artifact_names = [a.path for a in artifacts]

    model_artifact = None
    for name in ["model", "mlp_model"]:
        if name in artifact_names:
            model_artifact = name
            break

    if model_artifact is None:
        logger.warning(
            "Run %s não possui artefato de modelo no MLflow. "
            "O export usará retreino.",
            run_id[:8],
        )
        return None

    model_uri = f"runs:/{run_id}/{model_artifact}"

    try:
        result = mlflow.register_model(model_uri, model_name)
        version = int(result.version)
        logger.info(
            "Modelo registrado: %s v%d (run_id=%s)",
            model_name,
            version,
            run_id[:8],
        )
        return version
    except Exception as exc:
        logger.warning(
            "Registro no MLflow Model Registry falhou (esperado com file store): %s. "
            "champion_metadata.json será o fallback.",
            exc,
        )
        return None


def _load_champion_sklearn(champion: dict, data: dict) -> object:
    """Retreina o pipeline sklearn do champion para exportação."""
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    from churn_prediction.config import GB_KWARGS, LOG_REG_KWARGS, RF_KWARGS
    from churn_prediction.preprocessing import build_sklearn_pipeline

    model_map = {
        "log_reg": lambda: LogisticRegression(**LOG_REG_KWARGS),
        "dummy_stratified": lambda: DummyClassifier(strategy="stratified", random_state=42),
        "random_forest": lambda: RandomForestClassifier(**RF_KWARGS),
        "gradient_boosting": lambda: GradientBoostingClassifier(**GB_KWARGS),
    }

    run_name = champion["run_name"]
    factory = model_map.get(run_name)
    if factory is None:
        raise ValueError(
            f"Modelo '{run_name}' não encontrado no model_map. "
            f"Modelos disponíveis: {list(model_map.keys())}"
        )

    estimator = factory()
    pipe = build_sklearn_pipeline(data["X_train"], estimator)
    pipe.fit(data["X_train"], data["y_train"])

    logger.info("Pipeline sklearn retreinado para champion '%s'.", run_name)
    return pipe


def _load_champion_pytorch(champion: dict, data: dict) -> PyTorchChurnWrapper:
    """Carrega MLP PyTorch do MLflow e empacota com preprocessor sklearn."""
    # Importar MLP para garantir que a classe está disponível para desserialização
    from churn_prediction.model import MLP  # noqa: F401

    model_uri = f"runs:/{champion['run_id']}/mlp_model"
    pytorch_model = mlflow.pytorch.load_model(model_uri)

    preprocessor = build_preprocessor(data["X_train"])
    preprocessor.fit(data["X_train"])

    wrapper = PyTorchChurnWrapper(pytorch_model, preprocessor)
    logger.info("MLP PyTorch carregado e empacotado com PyTorchChurnWrapper.")
    return wrapper


def _detect_flavor(champion: dict, tracking_uri: str) -> str:
    """Detecta se o champion é sklearn ou pytorch pelo artefato no MLflow."""
    client = MlflowClient(tracking_uri)
    artifacts = client.list_artifacts(champion["run_id"])
    artifact_names = [a.path for a in artifacts]

    if "mlp_model" in artifact_names:
        return "pytorch"
    if "model" in artifact_names:
        return "sklearn"
    # Sem artefato — precisa retreinar (baselines do notebook)
    return "sklearn_retrain"


def export_champion(
    champion: dict,
    version: int | None,
    data: dict,
    out_dir: str = "models",
    tracking_uri: str = "file:./mlruns",
) -> Path:
    """Exporta o champion como joblib + champion_metadata.json.

    Args:
        champion: Saída de find_champion().
        version: Versão do registry (saída de register_champion()), ou None.
        data: Saída de prepare_data() — necessário para retreino ou fit do preprocessor.
        out_dir: Diretório de saída.
        tracking_uri: URI do MLflow tracking store.

    Returns:
        Path do arquivo joblib exportado.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    flavor = _detect_flavor(champion, tracking_uri)
    logger.info("Flavor detectado para champion '%s': %s", champion["run_name"], flavor)

    if flavor == "pytorch":
        mlflow.set_tracking_uri(tracking_uri)
        pipeline = _load_champion_pytorch(champion, data)
    elif flavor == "sklearn":
        mlflow.set_tracking_uri(tracking_uri)
        model_uri = f"runs:/{champion['run_id']}/model"
        pipeline = mlflow.sklearn.load_model(model_uri)
        logger.info("Pipeline sklearn carregado do MLflow (run_id=%s).", champion["run_id"][:8])
    else:
        pipeline = _load_champion_sklearn(champion, data)

    # Salva pipeline
    joblib_path = out_path / "churn_pipeline.joblib"
    joblib.dump(pipeline, joblib_path)
    logger.info("Pipeline exportado em %s", joblib_path)

    # Salva metadata
    metadata = {
        "champion_run_id": champion["run_id"],
        "champion_run_name": champion["run_name"],
        "registry_version": version,
        "selection_metric": "valor_liquido",
        "selection_tiebreaker": "roc_auc",
        "metrics": champion["metrics"],
        "dataset_version": champion["params"].get("dataset_version", "unknown"),
        "exported_at": datetime.now(UTC).isoformat(),
        "all_candidates": champion["all_candidates"],
    }
    metadata_path = out_path / "champion_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    logger.info("Metadata exportado em %s", metadata_path)

    return joblib_path
