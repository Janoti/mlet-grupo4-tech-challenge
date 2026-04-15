"""Métricas de avaliação técnica e de negócio para modelos de churn."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from churn_prediction.config import C_ACAO, V_RETIDO

# ---------------------------------------------------------------------------
# Métricas técnicas
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Calcula métricas de classificação padrão.

    Args:
        y_true: Rótulos reais (0/1).
        y_pred: Predições binárias (0/1).
        y_prob: Probabilidades da classe positiva (opcional, para AUC).

    Returns:
        Dicionário com accuracy, f1, roc_auc, pr_auc.
    """
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return metrics


# ---------------------------------------------------------------------------
# Métricas de negócio
# ---------------------------------------------------------------------------

def compute_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    value_retained: float = V_RETIDO,
    action_cost: float = C_ACAO,
) -> dict[str, float]:
    """Calcula métricas de negócio para campanha de retenção.

    Fórmula: valor_liquido = TP × V_RETIDO − (TP + FP) × C_ACAO

    Args:
        y_true: Rótulos reais (0/1).
        y_pred: Predições binárias (0/1).
        value_retained: Valor retido por churn evitado (R$).
        action_cost: Custo da ação por cliente abordado (R$).

    Returns:
        Dicionário com tp, fp, clientes_abordados, valor_bruto,
        custo_total_acao, valor_liquido, valor_por_cliente.
    """
    y_true_np = np.asarray(y_true).astype(int)
    y_pred_np = np.asarray(y_pred).astype(int)

    tp = int(((y_true_np == 1) & (y_pred_np == 1)).sum())
    fp = int(((y_true_np == 0) & (y_pred_np == 1)).sum())
    contacted = int((y_pred_np == 1).sum())
    n = int(len(y_true_np))

    gross_value = float(tp * value_retained)
    action_total_cost = float((tp + fp) * action_cost)
    net_value = float(gross_value - action_total_cost)
    value_per_customer = float(net_value / n) if n > 0 else float("nan")

    return {
        "tp": float(tp),
        "fp": float(fp),
        "clientes_abordados": float(contacted),
        "valor_bruto": gross_value,
        "custo_total_acao": action_total_cost,
        "valor_liquido": net_value,
        "valor_por_cliente": value_per_customer,
    }


# ---------------------------------------------------------------------------
# Otimização de threshold
# ---------------------------------------------------------------------------

def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
    value_retained: float = V_RETIDO,
    action_cost: float = C_ACAO,
) -> dict:
    """Varre thresholds e retorna o que maximiza valor_liquido.

    Returns:
        Dicionário com best_threshold, best_valor_liquido e tabela completa.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)

    y_true_np = np.asarray(y_true).astype(int)
    results = []

    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        bm = compute_business_metrics(y_true_np, preds, value_retained, action_cost)
        bm["threshold"] = round(float(thr), 2)
        bm["f1"] = float(f1_score(y_true_np, preds, zero_division=0))
        results.append(bm)

    best = max(results, key=lambda r: r["valor_liquido"])
    return {
        "best_threshold": best["threshold"],
        "best_valor_liquido": best["valor_liquido"],
        "results": results,
    }
