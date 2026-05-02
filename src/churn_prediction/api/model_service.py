"""Serviço de gerenciamento de versões de modelo e recomendações de retreinamento.

Integra com MLflow para acessar histórico de treinamentos e recomendar
quando retreinar baseado em critérios de performance e drift.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow

logger = logging.getLogger("churn_model_service")


class ModelService:
    """Gerenciador de versões e recomendações de modelo."""

    def __init__(self, mlflow_tracking_uri: str | None = None):
        """
        Args:
            mlflow_tracking_uri: URI do MLflow (padrão: ./mlruns)
        """
        self.mlflow_uri = mlflow_tracking_uri or str(
            Path(__file__).resolve().parents[3] / "mlruns"
        )
        mlflow.set_tracking_uri(self.mlflow_uri)

    def get_champion_metadata(self) -> dict | None:
        """Carrega metadata do champion (modelo em produção)."""
        metadata_path = Path(__file__).resolve().parents[3] / "models" / "champion_metadata.json"
        if not metadata_path.exists():
            logger.warning("champion_metadata.json não encontrado")
            return None

        with open(metadata_path) as f:
            return json.load(f)

    def list_recent_runs(self, limit: int = 10) -> list[dict]:
        """Lista últimos N runs do MLflow.

        Args:
            limit: Número máximo de runs a retornar.

        Returns:
            Lista de dicts com run_id, metrics, params, status.
        """
        try:
            experiment = mlflow.get_experiment_by_name("churn_prediction")
            if not experiment:
                logger.warning("Experimento 'churn_prediction' não encontrado")
                return []

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=limit,
            )

            results = []
            for _, run in runs.iterrows():
                results.append({
                    "run_id": run["run_id"][:8],  # primeiros 8 chars
                    "status": run["status"],
                    "metrics": dict(
                        (k.replace("metrics.", ""), v)
                        for k, v in run.items()
                        if k.startswith("metrics.")
                    ),
                    "params": dict(
                        (k.replace("params.", ""), v)
                        for k, v in run.items()
                        if k.startswith("params.")
                    ),
                })

            return results
        except Exception as e:
            logger.error(f"Erro ao listar runs: {e}")
            return []

    def get_model_versions(self) -> dict:
        """Retorna histórico de versões de modelo com métricas.

        Returns:
            Dict com total_versions, champion_version e lista de versions.
        """
        champion = self.get_champion_metadata()
        champion_id = champion.get("champion_run_id", "")[:8] if champion else None

        recent_runs = self.list_recent_runs(limit=20)

        return {
            "total_versions": len(recent_runs),
            "champion_version": champion_id,
            "versions": [
                {
                    "version_id": run["run_id"],
                    "model_name": f"mlp_pytorch_{run['run_id']}",
                    "metrics": run.get("metrics", {}),
                    "params": run.get("params", {}),
                    "registered_at": None,  # MLflow não rastreia isso por padrão
                    "is_champion": run["run_id"] == champion_id,
                }
                for run in recent_runs
            ],
        }

    def recommend_retrain(
        self,
        drift_ratio: float,
        days_since_last_retrain: int | None = None,
    ) -> dict:
        """Recomenda retreinamento baseado em drift e idade do modelo.

        Args:
            drift_ratio: Proporção de features com drift (0-1).
            days_since_last_retrain: Dias desde último retreinamento (None = desconhecido).

        Returns:
            Dict com should_retrain, reason e detalhes de degradação.
        """
        should_retrain = False
        reason = ""
        estimated_cost = "low"

        # Critério 1: Drift significativo
        if drift_ratio > 0.3:
            should_retrain = True
            reason = f"Drift significativo detectado ({drift_ratio:.1%} das features)"
            estimated_cost = "high"
        elif drift_ratio > 0.1:
            should_retrain = True
            reason = f"Drift moderado detectado ({drift_ratio:.1%} das features) — investigar"
            estimated_cost = "medium"

        # Critério 2: Idade do modelo
        if days_since_last_retrain and days_since_last_retrain > 90:
            should_retrain = True
            reason = f"Modelo com {days_since_last_retrain} dias — retrainamento periódico recomendado"
            estimated_cost = "medium"

        if not should_retrain:
            reason = "Modelo em bom estado — continue monitorando"

        return {
            "should_retrain": should_retrain,
            "reason": reason,
            "metrics_degradation": None,
            "last_retrain_days_ago": days_since_last_retrain,
            "estimated_retrain_cost": estimated_cost,
        }
