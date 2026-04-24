"""Métricas de avaliação técnica e de negócio para modelos de churn."""

from __future__ import annotations

from collections.abc import Callable

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


# ---------------------------------------------------------------------------
# Intervalos de confiança via bootstrap
# ---------------------------------------------------------------------------

_METRIC_FUNCTIONS: dict[str, Callable[..., float]] = {
    "accuracy": lambda y, p, prob: float(accuracy_score(y, p)),
    "f1": lambda y, p, prob: float(f1_score(y, p, zero_division=0)),
    "roc_auc": lambda y, p, prob: float(roc_auc_score(y, prob)),
    "pr_auc": lambda y, p, prob: float(average_precision_score(y, prob)),
}


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    metric: str = "roc_auc",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Calcula intervalo de confiança via bootstrap com reamostragem com reposição.

    Referência metodológica: Efron & Tibshirani (1993) — "An Introduction to the
    Bootstrap". Recomenda-se n_bootstrap ≥ 1000 para IC 95% estável.

    Args:
        y_true: Rótulos reais (0/1).
        y_pred: Predições binárias (necessário para accuracy/f1).
        y_prob: Probabilidades (necessário para roc_auc/pr_auc).
        metric: Uma de {"accuracy", "f1", "roc_auc", "pr_auc"}.
        n_bootstrap: Número de reamostragens (default 1000, mínimo 100).
        confidence: Nível de confiança (default 0.95).
        random_state: Seed para reprodutibilidade.

    Returns:
        Dict com point_estimate, ci_lower, ci_upper, mean_bootstrap, std_bootstrap.

    Raises:
        ValueError: se a métrica não é suportada ou n_bootstrap < 100.
    """
    if metric not in _METRIC_FUNCTIONS:
        raise ValueError(
            f"Métrica '{metric}' não suportada. Opções: {list(_METRIC_FUNCTIONS)}"
        )
    if n_bootstrap < 100:
        raise ValueError(f"n_bootstrap deve ser >= 100 (recebido {n_bootstrap})")

    metric_fn = _METRIC_FUNCTIONS[metric]

    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    y_prob_np = np.asarray(y_prob) if y_prob is not None else None

    point_estimate = metric_fn(y_true_np, y_pred_np, y_prob_np)

    rng = np.random.default_rng(random_state)
    n = len(y_true_np)
    bootstrap_scores: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_true_boot = y_true_np[idx]
        y_pred_boot = y_pred_np[idx]
        y_prob_boot = y_prob_np[idx] if y_prob_np is not None else None

        # Pula amostras degeneradas (apenas uma classe) para métricas que exigem ambas
        if metric in ("roc_auc", "pr_auc", "f1") and len(np.unique(y_true_boot)) < 2:
            continue

        try:
            score = metric_fn(y_true_boot, y_pred_boot, y_prob_boot)
            bootstrap_scores.append(score)
        except ValueError:
            continue

    if len(bootstrap_scores) < n_bootstrap * 0.5:
        raise RuntimeError(
            f"Muitas reamostragens inválidas para métrica '{metric}' "
            f"({len(bootstrap_scores)}/{n_bootstrap}). Verifique o balanceamento."
        )

    alpha = 1.0 - confidence
    lower_pct = (alpha / 2) * 100
    upper_pct = (1 - alpha / 2) * 100
    ci_lower = float(np.percentile(bootstrap_scores, lower_pct))
    ci_upper = float(np.percentile(bootstrap_scores, upper_pct))

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "mean_bootstrap": float(np.mean(bootstrap_scores)),
        "std_bootstrap": float(np.std(bootstrap_scores)),
        "n_bootstrap_valid": len(bootstrap_scores),
        "confidence": confidence,
    }


def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Retorna todas as métricas técnicas com IC 95% via bootstrap.

    Útil para registro no MLflow ou relatórios do Model Card.

    Returns:
        Dict por métrica, cada uma com point_estimate, ci_lower, ci_upper.
    """
    metrics_to_compute = ["accuracy", "f1"]
    if y_prob is not None:
        metrics_to_compute.extend(["roc_auc", "pr_auc"])

    return {
        name: bootstrap_confidence_interval(
            y_true, y_pred, y_prob,
            metric=name,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        for name in metrics_to_compute
    }


# ---------------------------------------------------------------------------
# Calibração de probabilidades
# ---------------------------------------------------------------------------

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score: erro quadrático médio entre probabilidade prevista e label.

    Fórmula: BS = (1/n) * Σ(y_prob_i - y_true_i)²

    Interpretação:
    - BS = 0 → calibração perfeita
    - BS = 0.25 → modelo aleatório (probabilidade constante 0.5 com base 50/50)
    - Quanto menor, melhor.

    Args:
        y_true: Rótulos reais (0/1).
        y_prob: Probabilidades previstas da classe positiva.

    Returns:
        Brier score (float).
    """
    y_true_np = np.asarray(y_true).astype(float)
    y_prob_np = np.asarray(y_prob).astype(float)
    return float(np.mean((y_prob_np - y_true_np) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) via binning uniforme.

    Divide [0, 1] em n_bins intervalos e calcula a diferença entre confiança
    média prevista e acurácia real em cada bin, ponderada pelo tamanho do bin.

    Referência: Guo et al. (2017) — "On Calibration of Modern Neural Networks".

    Interpretação:
    - ECE = 0 → calibração perfeita
    - ECE < 0.05 → bem calibrado
    - ECE > 0.10 → calibração pobre, considerar Platt scaling ou isotonic regression

    Args:
        y_true: Rótulos reais (0/1).
        y_prob: Probabilidades previstas.
        n_bins: Número de bins uniformes em [0, 1].

    Returns:
        ECE (float, entre 0 e 1).
    """
    y_true_np = np.asarray(y_true).astype(float)
    y_prob_np = np.asarray(y_prob).astype(float)
    n = len(y_true_np)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob_np >= lo) & (y_prob_np < hi) if i < n_bins - 1 else (y_prob_np >= lo) & (y_prob_np <= hi)
        bin_size = int(mask.sum())
        if bin_size == 0:
            continue
        avg_confidence = float(y_prob_np[mask].mean())
        avg_accuracy = float(y_true_np[mask].mean())
        weight = bin_size / n
        ece += weight * abs(avg_confidence - avg_accuracy)

    return float(ece)


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Dados para plotar diagrama de confiabilidade (reliability diagram).

    O diagrama plota probabilidade média prevista (x) vs. taxa observada (y) por bin.
    Calibração perfeita → pontos sobre a diagonal y=x.

    Args:
        y_true: Rótulos reais (0/1).
        y_prob: Probabilidades previstas.
        n_bins: Número de bins.

    Returns:
        Dict com bin_centers, mean_predicted, fraction_positives, bin_counts.
    """
    y_true_np = np.asarray(y_true).astype(float)
    y_prob_np = np.asarray(y_prob).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mean_predicted = np.zeros(n_bins)
    fraction_positives = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob_np >= lo) & (y_prob_np < hi) if i < n_bins - 1 else (y_prob_np >= lo) & (y_prob_np <= hi)
        count = int(mask.sum())
        bin_counts[i] = count
        if count > 0:
            mean_predicted[i] = float(y_prob_np[mask].mean())
            fraction_positives[i] = float(y_true_np[mask].mean())

    return {
        "bin_centers": bin_centers,
        "mean_predicted": mean_predicted,
        "fraction_positives": fraction_positives,
        "bin_counts": bin_counts,
    }


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Conveniência: retorna Brier + ECE num dicionário.

    Pronto para registro no MLflow:
        mlflow.log_metrics(compute_calibration_metrics(y_true, y_prob))
    """
    return {
        "brier_score": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
    }
