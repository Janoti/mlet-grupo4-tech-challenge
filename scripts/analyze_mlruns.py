#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from logging_utils import get_logger, log_kv, setup_logging

RUN_NAMES = [
    "dummy_stratified",
    "log_reg",
    "log_reg_mitigated_equalized_odds",
    "random_forest",
    "gradient_boosting",
    "mlp_pytorch_v1",
]

# Runs opcionais: ausência não interrompe a análise
OPTIONAL_RUN_NAMES = {"mlp_pytorch_v1", "random_forest", "gradient_boosting"}

METRICS = [
    "accuracy",
    "f1",
    "roc_auc",
    "pr_auc",
    "positive_rate",
    "best_val_loss",
    "dp_diff_gender",
    "eo_diff_gender",
    "tp",
    "fp",
    "clientes_abordados",
    "valor_bruto",
    "custo_total_acao",
    "valor_liquido",
    "valor_por_cliente",
]

logger = get_logger("analyze_mlruns")


@dataclass
class RunData:
    run_id: str
    run_name: str
    start_time: int
    params: dict[str, str]
    metrics: dict[str, float]


def parse_meta(meta_path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        data[k.strip()] = v.strip().strip('"')
    return data


def read_param(run_dir: Path, name: str) -> str | None:
    p = run_dir / "params" / name
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


def read_metric(run_dir: Path, name: str) -> float | None:
    p = run_dir / "metrics" / name
    if not p.exists():
        return None
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1].split()
    if len(last) < 2:
        return None
    try:
        return float(last[1])
    except ValueError:
        return None


def collect_runs(mlruns_root: Path) -> dict[str, RunData]:
    latest_by_name: dict[str, RunData] = {}

    for exp_dir in mlruns_root.iterdir():
        if not exp_dir.is_dir() or exp_dir.name in {"models", ".trash"}:
            continue
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            meta_file = run_dir / "meta.yaml"
            if not meta_file.exists():
                continue
            meta = parse_meta(meta_file)
            run_id = meta.get("run_id", run_dir.name)
            run_name = (run_dir / "tags" / "mlflow.runName").read_text(encoding="utf-8").strip() if (run_dir / "tags" / "mlflow.runName").exists() else meta.get("run_name", "")
            if run_name not in RUN_NAMES:
                continue

            try:
                start_time = int(meta.get("start_time", "0"))
            except ValueError:
                start_time = 0

            params: dict[str, str] = {}
            for p in [
                "model",
                "base_model",
                "constraint",
                "mitigation_eps",
                "mitigation_max_iter",
                "mitigation_sample_size",
                "test_size",
                "dataset_version",
            ]:
                v = read_param(run_dir, p)
                if v is not None:
                    params[p] = v

            metrics: dict[str, float] = {}
            for m in METRICS:
                v = read_metric(run_dir, m)
                if v is not None:
                    metrics[m] = v

            candidate = RunData(run_id=run_id, run_name=run_name, start_time=start_time, params=params, metrics=metrics)
            prev = latest_by_name.get(run_name)
            if prev is None or candidate.start_time > prev.start_time:
                latest_by_name[run_name] = candidate

    return latest_by_name


def fmt(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def fmt_intlike(x: float | None) -> str:
    if x is None:
        return "n/a"
    return str(int(round(x)))


def main(log_level: str | None = None) -> int:
    setup_logging(log_level)
    mlruns_root = Path("mlruns")
    if not mlruns_root.exists():
        logger.error("[analysis] mlruns/ nao encontrado.")
        return 1

    runs = collect_runs(mlruns_root)
    required = [n for n in RUN_NAMES if n not in OPTIONAL_RUN_NAMES]
    missing = [n for n in required if n not in runs]
    if missing:
        log_kv(logger, "analysis.missing_runs", missing=",".join(missing))
        logger.info("[analysis] Rode: make notebooks")
        return 1

    missing_optional = [n for n in OPTIONAL_RUN_NAMES if n not in runs]
    if missing_optional:
        log_kv(logger, "analysis.missing_optional_runs", missing=",".join(missing_optional))
        logger.info("[analysis] Para incluir MLP: make notebooks-mlp")

    dummy  = runs["dummy_stratified"]
    logreg = runs["log_reg"]
    mitig  = runs["log_reg_mitigated_equalized_odds"]
    rf     = runs.get("random_forest")
    gb     = runs.get("gradient_boosting")
    mlp    = runs.get("mlp_pytorch_v1")

    logger.info("[analysis] Resumo automatico da run")
    log_kv(logger, "analysis.dataset", dataset_version=logreg.params.get("dataset_version", "n/a"))

    logger.info("[analysis] Performance")
    log_kv(
        logger,
        "analysis.performance.dummy",
        accuracy=fmt(dummy.metrics.get("accuracy")),
        f1=fmt(dummy.metrics.get("f1")),
        roc_auc=fmt(dummy.metrics.get("roc_auc")),
        pr_auc=fmt(dummy.metrics.get("pr_auc")),
    )
    log_kv(
        logger,
        "analysis.performance.log_reg",
        accuracy=fmt(logreg.metrics.get("accuracy")),
        f1=fmt(logreg.metrics.get("f1")),
        roc_auc=fmt(logreg.metrics.get("roc_auc")),
        pr_auc=fmt(logreg.metrics.get("pr_auc")),
    )

    d_acc = logreg.metrics.get("accuracy", 0.0) - dummy.metrics.get("accuracy", 0.0)
    d_f1 = logreg.metrics.get("f1", 0.0) - dummy.metrics.get("f1", 0.0)
    d_roc = logreg.metrics.get("roc_auc", 0.0) - dummy.metrics.get("roc_auc", 0.0)
    d_pr = logreg.metrics.get("pr_auc", 0.0) - dummy.metrics.get("pr_auc", 0.0)

    logger.info("[analysis] Ganho log_reg vs dummy")
    log_kv(
        logger,
        "analysis.delta.log_reg_vs_dummy",
        delta_accuracy=f"{d_acc:.4f}",
        delta_f1=f"{d_f1:.4f}",
        delta_roc_auc=f"{d_roc:.4f}",
        delta_pr_auc=f"{d_pr:.4f}",
    )

    for ensemble_run, label in [(rf, "random_forest"), (gb, "gradient_boosting")]:
        if ensemble_run is not None:
            log_kv(
                logger,
                f"analysis.performance.{label}",
                accuracy=fmt(ensemble_run.metrics.get("accuracy")),
                f1=fmt(ensemble_run.metrics.get("f1")),
                roc_auc=fmt(ensemble_run.metrics.get("roc_auc")),
                pr_auc=fmt(ensemble_run.metrics.get("pr_auc")),
                valor_liquido=fmt(ensemble_run.metrics.get("valor_liquido"), digits=2),
            )
            d_roc_ens = ensemble_run.metrics.get("roc_auc", 0.0) - logreg.metrics.get("roc_auc", 0.0)
            log_kv(
                logger,
                f"analysis.delta.{label}_vs_log_reg",
                delta_roc_auc=f"{d_roc_ens:+.4f}",
            )

    logger.info("[analysis] Mitigacao (trade-off)")
    log_kv(
        logger,
        "analysis.performance.mitigated",
        accuracy=fmt(mitig.metrics.get("accuracy")),
        f1=fmt(mitig.metrics.get("f1")),
        dp_diff_gender=fmt(mitig.metrics.get("dp_diff_gender")),
        eo_diff_gender=fmt(mitig.metrics.get("eo_diff_gender")),
    )

    m_acc = mitig.metrics.get("accuracy", 0.0) - logreg.metrics.get("accuracy", 0.0)
    m_f1 = mitig.metrics.get("f1", 0.0) - logreg.metrics.get("f1", 0.0)
    log_kv(
        logger,
        "analysis.delta.mitigated_vs_log_reg",
        delta_accuracy_vs_log_reg=f"{m_acc:.4f}",
        delta_f1_vs_log_reg=f"{m_f1:.4f}",
    )

    logger.info("[analysis] Metrica de negocio")
    log_kv(
        logger,
        "analysis.business.log_reg",
        tp=fmt_intlike(logreg.metrics.get("tp")),
        fp=fmt_intlike(logreg.metrics.get("fp")),
        clientes_abordados=fmt_intlike(logreg.metrics.get("clientes_abordados")),
        valor_liquido=fmt(logreg.metrics.get("valor_liquido"), digits=2),
        valor_por_cliente=fmt(logreg.metrics.get("valor_por_cliente"), digits=4),
    )
    log_kv(
        logger,
        "analysis.business.mitigated",
        tp=fmt_intlike(mitig.metrics.get("tp")),
        fp=fmt_intlike(mitig.metrics.get("fp")),
        clientes_abordados=fmt_intlike(mitig.metrics.get("clientes_abordados")),
        valor_liquido=fmt(mitig.metrics.get("valor_liquido"), digits=2),
        valor_por_cliente=fmt(mitig.metrics.get("valor_por_cliente"), digits=4),
    )

    d_value = mitig.metrics.get("valor_liquido", 0.0) - logreg.metrics.get("valor_liquido", 0.0)
    d_value_per_customer = mitig.metrics.get("valor_por_cliente", 0.0) - logreg.metrics.get("valor_por_cliente", 0.0)
    log_kv(
        logger,
        "analysis.delta.business.mitigated_vs_log_reg",
        delta_valor_liquido_vs_log_reg=f"{d_value:.2f}",
        delta_valor_por_cliente_vs_log_reg=f"{d_value_per_customer:.4f}",
    )

    if mlp is not None:
        logger.info("[analysis] MLP PyTorch")
        log_kv(
            logger,
            "analysis.performance.mlp_pytorch",
            accuracy=fmt(mlp.metrics.get("accuracy")),
            f1=fmt(mlp.metrics.get("f1")),
            roc_auc=fmt(mlp.metrics.get("roc_auc")),
            pr_auc=fmt(mlp.metrics.get("pr_auc")),
            best_val_loss=fmt(mlp.metrics.get("best_val_loss")),
        )
        log_kv(
            logger,
            "analysis.business.mlp_pytorch",
            tp=fmt_intlike(mlp.metrics.get("tp")),
            fp=fmt_intlike(mlp.metrics.get("fp")),
            clientes_abordados=fmt_intlike(mlp.metrics.get("clientes_abordados")),
            valor_liquido=fmt(mlp.metrics.get("valor_liquido"), digits=2),
            valor_por_cliente=fmt(mlp.metrics.get("valor_por_cliente"), digits=4),
        )
        d_roc_mlp = (mlp.metrics.get("roc_auc", 0.0) - logreg.metrics.get("roc_auc", 0.0))
        d_f1_mlp  = (mlp.metrics.get("f1", 0.0) - logreg.metrics.get("f1", 0.0))
        d_vl_mlp  = (mlp.metrics.get("valor_liquido", 0.0) - logreg.metrics.get("valor_liquido", 0.0))
        log_kv(
            logger,
            "analysis.delta.mlp_vs_log_reg",
            delta_roc_auc=f"{d_roc_mlp:+.4f}",
            delta_f1=f"{d_f1_mlp:+.4f}",
            delta_valor_liquido=f"{d_vl_mlp:+.2f}",
        )

    logger.info("[analysis] Leitura sugerida")
    logger.info("1) log_reg supera dummy com folga em metricas de classificacao.")
    logger.info("2) mitigacao reduz disparidade (DP/EO) com pequena perda de performance.")
    logger.info("3) decisao final deve equilibrar desempenho e equidade.")
    if mlp is not None:
        logger.info("4) MLP PyTorch disponivel — compare com log_reg antes de decidir pelo modelo de producao.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisa os runs mais recentes no MLflow local.")
    parser.add_argument("--log-level", type=str, default=None, help="Nível de log (DEBUG, INFO, WARNING, ERROR).")
    args = parser.parse_args()
    raise SystemExit(main(log_level=args.log_level))
