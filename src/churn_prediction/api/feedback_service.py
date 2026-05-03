"""Serviço de coleta e persistência de feedback de predições."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("churn_feedback_service")


class FeedbackService:
    """Gerenciador de feedback de predições."""

    def __init__(self, feedback_dir: str | None = None):
        """
        Args:
            feedback_dir: Diretório para armazenar feedback em JSONL.
                          Padrão: logs/feedback/
        """
        self.feedback_dir = Path(feedback_dir or os.environ.get("FEEDBACK_DIR", "/tmp/churn-feedback"))
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_dir / "feedback.jsonl"

    def log_feedback(
        self,
        prediction_id: str,
        actual_churn: int | None,
        feedback_type: str,
        comment: str | None = None,
        rating: int | None = None,
    ) -> str:
        """Registra feedback em arquivo JSONL.

        Args:
            prediction_id: ID da predição.
            actual_churn: Resultado real (0/1).
            feedback_type: 'correct', 'incorrect', ou 'uncertain'.
            comment: Comentário.
            rating: Avaliação 1-5.

        Returns:
            feedback_id gerado (UUID[:8]).
        """
        feedback_id = str(uuid.uuid4())[:8]

        record = {
            "feedback_id": feedback_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "prediction_id": prediction_id,
            "actual_churn": actual_churn,
            "feedback_type": feedback_type,
            "comment": comment,
            "rating": rating,
        }

        try:
            with open(self.feedback_file, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(f"Feedback registrado: {feedback_id}")
            return feedback_id
        except Exception as e:
            logger.error(f"Erro ao registrar feedback: {e}")
            raise

    def get_feedback_summary(self) -> dict:
        """Retorna sumário de feedback.

        Returns:
            Dict com total_feedback, accuracy, avg_rating, feedback_by_type.
        """
        if not self.feedback_file.exists():
            return {
                "total_feedback": 0,
                "accuracy": None,
                "avg_rating": None,
                "feedback_by_type": {},
            }

        feedback_list = []
        with open(self.feedback_file) as f:
            for line in f:
                try:
                    feedback_list.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not feedback_list:
            return {
                "total_feedback": 0,
                "accuracy": None,
                "avg_rating": None,
                "feedback_by_type": {},
            }

        # Métricas
        correct_count = sum(1 for fb in feedback_list if fb.get("feedback_type") == "correct")
        total_with_result = sum(1 for fb in feedback_list if fb.get("actual_churn") is not None)
        accuracy = correct_count / total_with_result if total_with_result > 0 else None

        ratings = [fb["rating"] for fb in feedback_list if fb.get("rating")]
        avg_rating = sum(ratings) / len(ratings) if ratings else None

        feedback_by_type = {}
        for fb in feedback_list:
            fb_type = fb.get("feedback_type", "unknown")
            feedback_by_type[fb_type] = feedback_by_type.get(fb_type, 0) + 1

        return {
            "total_feedback": len(feedback_list),
            "accuracy": round(accuracy, 3) if accuracy else None,
            "avg_rating": round(avg_rating, 2) if avg_rating else None,
            "feedback_by_type": feedback_by_type,
        }
