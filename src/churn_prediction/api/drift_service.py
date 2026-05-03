"""Serviço de detecção e análise de data drift.

Encapsula a lógica de comparação entre distribuição de treino (referência)
e produção, expondo uma interface simples para os endpoints.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from churn_prediction.monitoring import generate_drift_report

logger = logging.getLogger("churn_drift_service")


class DriftService:
    """Gerenciador de detecção de drift."""

    def __init__(self, reference_data_path: str | None = None):
        """
        Args:
            reference_data_path: Caminho para CSV com dados de referência (treino).
                                 Se None, tenta carregar de data/processed/train_final.csv
        """
        self.reference_data_path = reference_data_path or (
            Path(__file__).resolve().parents[3] / "data" / "processed" / "train_final.csv"
        )
        self._reference_df: pd.DataFrame | None = None

    def load_reference_data(self) -> pd.DataFrame:
        """Carrega dados de referência (distribuição de treino)."""
        if self._reference_df is not None:
            return self._reference_df

        if not Path(self.reference_data_path).exists():
            logger.warning(f"Arquivo de referência não encontrado: {self.reference_data_path}")
            return pd.DataFrame()

        self._reference_df = pd.read_csv(self.reference_data_path)
        logger.info(f"Dados de referência carregados: {self._reference_df.shape[0]} linhas")
        return self._reference_df

    def check_drift(
        self,
        production_df: pd.DataFrame,
        alpha: float = 0.05,
    ) -> dict:
        """Faz check rápido de drift e retorna recomendação.

        Args:
            production_df: DataFrame com dados de produção (inferências recentes).
            alpha: Nível de significância para testes estatísticos.

        Returns:
            Dict com timestamp, contagem de alertas, features com drift e recomendação.
        """
        reference = self.load_reference_data()

        if reference.empty or production_df.empty:
            logger.warning("Referência ou produção vazios — impossível calcular drift")
            return {
                "timestamp": datetime.now(UTC).isoformat(),
                "total_features_checked": 0,
                "drift_alerts": 0,
                "drift_ratio": 0.0,
                "features_with_drift": [],
                "recommendation": "insufficient_data",
            }

        report = generate_drift_report(
            reference_df=reference,
            production_df=production_df,
            alpha=alpha,
        )

        # Extrai features com drift
        features_with_drift = [
            fname
            for fname, fdata in report.get("features", {}).items()
            if fdata.get("drift_detected", False)
        ]

        # Define recomendação baseada em drift_ratio
        drift_ratio = report.get("drift_ratio", 0.0)
        if drift_ratio < 0.1:
            recommendation = "monitor"
        elif drift_ratio < 0.3:
            recommendation = "investigate"
        else:
            recommendation = "retrain"

        return {
            "timestamp": report["timestamp"],
            "total_features_checked": report.get("total_features", 0),
            "drift_alerts": report.get("drift_alerts", 0),
            "drift_ratio": drift_ratio,
            "features_with_drift": features_with_drift,
            "recommendation": recommendation,
        }

    def get_detailed_report(
        self,
        production_df: pd.DataFrame,
        alpha: float = 0.05,
    ) -> dict:
        """Gera relatório completo com detalhes por feature.

        Args:
            production_df: DataFrame com dados de produção.
            alpha: Nível de significância.

        Returns:
            Dict com timestamp, métricas agregadas e dict de features com testes detalhados.
        """
        reference = self.load_reference_data()

        if reference.empty or production_df.empty:
            logger.warning("Referência ou produção vazios")
            return {
                "timestamp": datetime.now(UTC).isoformat(),
                "total_features": 0,
                "drift_alerts": 0,
                "drift_ratio": 0.0,
                "features": {},
            }

        report = generate_drift_report(
            reference_df=reference,
            production_df=production_df,
            alpha=alpha,
        )

        return {
            "timestamp": report["timestamp"],
            "total_features": report.get("total_features", 0),
            "drift_alerts": report.get("drift_alerts", 0),
            "drift_ratio": report.get("drift_ratio", 0.0),
            "features": report.get("features", {}),
        }
