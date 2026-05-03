"""Exporta o champion do MLflow para uso pela API FastAPI.

Uso:
    poetry run python scripts/export_model.py

Consulta todos os experimentos no MLflow, seleciona o melhor modelo por
valor_liquido (métrica de negócio) e exporta o pipeline + metadata para
models/churn_pipeline.joblib e models/champion_metadata.json.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from churn_prediction.pipelines import prepare_data
from churn_prediction.registry import export_champion, find_champion, register_champion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DATA = "data/raw/telecom_churn_base_extended.csv"
MLFLOW_REMOTE = os.environ.get("MLFLOW_TRACKING_URI", "")
MLFLOW_LOCAL = "file:./mlruns"

# Se MLFLOW_TRACKING_URI aponta para um server remoto, os notebooks gravam lá
# Nesse caso, usamos o remoto como fonte principal (treino remoto na EC2)
IS_REMOTE_TRAINING = MLFLOW_REMOTE.startswith("http")
MLFLOW_SOURCE = MLFLOW_REMOTE if IS_REMOTE_TRAINING else MLFLOW_LOCAL


def _sync_runs_to_remote(champion, local_uri, remote_uri):
    """Replica métricas e parâmetros dos runs locais para o MLflow remoto."""
    import mlflow

    local_client = mlflow.MlflowClient(local_uri)
    mlflow.set_tracking_uri(remote_uri)

    champion_name = champion.get("run_name", "")

    # Buscar todos os runs locais por nome
    local_runs = {}
    for exp in local_client.search_experiments():
        for run in local_client.search_runs(exp.experiment_id):
            local_runs[run.info.run_name] = run

    for candidate in champion.get("all_candidates", []):
        run_name = candidate.get("run_name", "unknown")
        local_run = local_runs.get(run_name)
        if not local_run:
            logger.warning("  Run local não encontrado: %s", run_name)
            continue
        try:
            exp_name = local_client.get_experiment(local_run.info.experiment_id).name
            mlflow.set_experiment(exp_name)
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(local_run.data.params)
                mlflow.log_metrics(local_run.data.metrics)
                # Tags de rastreabilidade
                is_champion = run_name == champion_name
                mlflow.set_tag("champion", str(is_champion).lower())
                mlflow.set_tag("dataset_version", champion.get("params", {}).get("dataset_version", "unknown"))
                mlflow.set_tag("deploy_timestamp", datetime.now(UTC).isoformat())
                if is_champion:
                    mlflow.set_tag("champion_metric", f"valor_liquido={candidate.get('valor_liquido', 0):.0f}")
            logger.info("  Sincronizado: %s%s", run_name, " ★ champion" if is_champion else "")
        except Exception as e:
            logger.warning("  Falha ao sincronizar %s: %s", run_name, e)

    mlflow.set_tracking_uri(local_uri)


def _tag_champion_remote(champion, tracking_uri):
    """Adiciona tags de champion diretamente nos runs do MLflow remoto."""
    import mlflow

    client = mlflow.MlflowClient(tracking_uri)
    champion_name = champion.get("run_name", "")

    for exp in client.search_experiments():
        for run in client.search_runs(exp.experiment_id):
            is_champ = run.info.run_name == champion_name
            client.set_tag(run.info.run_id, "champion", str(is_champ).lower())
            client.set_tag(run.info.run_id, "deploy_timestamp", datetime.now(UTC).isoformat())
            client.set_tag(run.info.run_id, "dataset_version",
                           champion.get("params", {}).get("dataset_version", "unknown"))
            if is_champ:
                vl = champion.get("metrics", {}).get("valor_liquido", 0)
                client.set_tag(run.info.run_id, "champion_metric", f"valor_liquido={vl:.0f}")
            logger.info("  Tagged: %s%s", run.info.run_name, " ★ champion" if is_champ else "")


def main():
    logger.info("Buscando champion no MLflow (%s)...", MLFLOW_SOURCE)
    champion = find_champion(tracking_uri=MLFLOW_SOURCE)

    logger.info("Registrando champion no MLflow Model Registry...")
    version = register_champion(
        run_id=champion["run_id"],
        tracking_uri=MLFLOW_SOURCE,
    )

    # Se treino local + MLflow remoto disponível, sincroniza métricas
    if not IS_REMOTE_TRAINING and MLFLOW_REMOTE:
        logger.info("Sincronizando experimentos para %s...", MLFLOW_REMOTE)
        _sync_runs_to_remote(champion, MLFLOW_LOCAL, MLFLOW_REMOTE)
    elif IS_REMOTE_TRAINING:
        # Treino remoto: adicionar tags de champion diretamente
        _tag_champion_remote(champion, MLFLOW_SOURCE)
        logger.info("Sincronizando experimentos para %s...", MLFLOW_REMOTE)
        _sync_runs_to_remote(champion, MLFLOW_LOCAL, MLFLOW_REMOTE)

    logger.info("Preparando dados para exportação...")
    data = prepare_data(RAW_DATA)

    logger.info("Exportando champion...")
    export_champion(
        champion=champion,
        version=version,
        data=data,
        tracking_uri=MLFLOW_SOURCE,
    )

    # Log resumo dos candidatos
    logger.info("=" * 60)
    logger.info("CHAMPION: %s", champion["run_name"])
    logger.info("  run_id: %s", champion["run_id"][:8])
    for metric, value in champion["metrics"].items():
        logger.info("  %s: %.4f", metric, value)
    logger.info("-" * 60)
    logger.info("Candidatos avaliados:")
    for i, c in enumerate(champion["all_candidates"], 1):
        vl = c.get("valor_liquido", 0)
        roc = c.get("roc_auc", 0)
        logger.info("  %d. %s | valor_liquido=%.0f | roc_auc=%.4f", i, c["run_name"], vl, roc)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
