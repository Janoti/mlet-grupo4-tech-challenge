#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


RUN_NAMES = [
    "dummy_stratified",
    "log_reg",
    "log_reg_mitigated_equalized_odds",
]

METRICS = [
    "accuracy",
    "f1",
    "roc_auc",
    "pr_auc",
    "positive_rate",
    "dp_diff_gender",
    "eo_diff_gender",
]


@dataclass
class RunData:
    run_id: str
    run_name: str
    start_time: int
    params: Dict[str, str]
    metrics: Dict[str, float]


def parse_meta(meta_path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        data[k.strip()] = v.strip().strip('"')
    return data


def read_param(run_dir: Path, name: str) -> Optional[str]:
    p = run_dir / "params" / name
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


def read_metric(run_dir: Path, name: str) -> Optional[float]:
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


def collect_runs(mlruns_root: Path) -> Dict[str, RunData]:
    latest_by_name: Dict[str, RunData] = {}

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

            params: Dict[str, str] = {}
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

            metrics: Dict[str, float] = {}
            for m in METRICS:
                v = read_metric(run_dir, m)
                if v is not None:
                    metrics[m] = v

            candidate = RunData(run_id=run_id, run_name=run_name, start_time=start_time, params=params, metrics=metrics)
            prev = latest_by_name.get(run_name)
            if prev is None or candidate.start_time > prev.start_time:
                latest_by_name[run_name] = candidate

    return latest_by_name


def fmt(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def main() -> int:
    mlruns_root = Path("mlruns")
    if not mlruns_root.exists():
        print("[analysis] mlruns/ nao encontrado.")
        return 1

    runs = collect_runs(mlruns_root)
    missing = [n for n in RUN_NAMES if n not in runs]
    if missing:
        print("[analysis] Runs esperados ausentes:", ", ".join(missing))
        print("[analysis] Rode: make notebooks")
        return 1

    dummy = runs["dummy_stratified"]
    logreg = runs["log_reg"]
    mitig = runs["log_reg_mitigated_equalized_odds"]

    print("[analysis] Resumo automatico da run")
    print(f"[analysis] dataset_version: {logreg.params.get('dataset_version', 'n/a')}")

    print("\n[analysis] Performance")
    print(
        "  dummy_stratified: "
        f"accuracy={fmt(dummy.metrics.get('accuracy'))}, "
        f"f1={fmt(dummy.metrics.get('f1'))}, "
        f"roc_auc={fmt(dummy.metrics.get('roc_auc'))}, "
        f"pr_auc={fmt(dummy.metrics.get('pr_auc'))}"
    )
    print(
        "  log_reg:          "
        f"accuracy={fmt(logreg.metrics.get('accuracy'))}, "
        f"f1={fmt(logreg.metrics.get('f1'))}, "
        f"roc_auc={fmt(logreg.metrics.get('roc_auc'))}, "
        f"pr_auc={fmt(logreg.metrics.get('pr_auc'))}"
    )

    d_acc = logreg.metrics.get("accuracy", 0.0) - dummy.metrics.get("accuracy", 0.0)
    d_f1 = logreg.metrics.get("f1", 0.0) - dummy.metrics.get("f1", 0.0)
    d_roc = logreg.metrics.get("roc_auc", 0.0) - dummy.metrics.get("roc_auc", 0.0)
    d_pr = logreg.metrics.get("pr_auc", 0.0) - dummy.metrics.get("pr_auc", 0.0)

    print("\n[analysis] Ganho log_reg vs dummy")
    print(f"  delta_accuracy={d_acc:.4f}")
    print(f"  delta_f1={d_f1:.4f}")
    print(f"  delta_roc_auc={d_roc:.4f}")
    print(f"  delta_pr_auc={d_pr:.4f}")

    print("\n[analysis] Mitigacao (trade-off)")
    print(
        "  log_reg_mitigated_equalized_odds: "
        f"accuracy={fmt(mitig.metrics.get('accuracy'))}, "
        f"f1={fmt(mitig.metrics.get('f1'))}, "
        f"dp_diff_gender={fmt(mitig.metrics.get('dp_diff_gender'))}, "
        f"eo_diff_gender={fmt(mitig.metrics.get('eo_diff_gender'))}"
    )

    m_acc = mitig.metrics.get("accuracy", 0.0) - logreg.metrics.get("accuracy", 0.0)
    m_f1 = mitig.metrics.get("f1", 0.0) - logreg.metrics.get("f1", 0.0)
    print(f"  delta_accuracy_vs_log_reg={m_acc:.4f}")
    print(f"  delta_f1_vs_log_reg={m_f1:.4f}")

    print("\n[analysis] Leitura sugerida")
    print("  1) log_reg supera dummy com folga em metricas de classificacao.")
    print("  2) mitigacao reduz disparidade (DP/EO) com pequena perda de performance.")
    print("  3) decisao final deve equilibrar desempenho e equidade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
