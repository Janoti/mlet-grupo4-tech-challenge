"""Helpers para rastreamento rico no MLflow seguindo padrão FIAP z-fab.

Fornece:
- Tags padronizadas (git_sha, author, dataset_version, run_phase)
- Parâmetros hierárquicos (model.*, train.*, data.*)
- mlflow.note.content formatado automaticamente
- Wrapper start_run_with_context(...) para setup completo em uma chamada

Uso típico nos notebooks:
    from churn_prediction.mlflow_utils import start_run_with_context

    with start_run_with_context(
        experiment="churn-baselines",
        run_name="log_reg",
        phase="baseline",
        dataset_path="data/raw/telecom_churn_base_extended.csv",
        note="Regressão Logística com GridSearchCV sobre C em {0.01, 0.1, 1, 10}.",
    ):
        mlflow.log_params(flatten_params({"model": {...}, "train": {...}}))
        mlflow.log_metrics({...})
"""

from __future__ import annotations

import getpass
import hashlib
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow


# ---------------------------------------------------------------------------
# Tags e metadados
# ---------------------------------------------------------------------------

def get_git_sha(short: bool = True) -> str:
    """Retorna o SHA do commit HEAD, ou 'unknown' se não estiver num git repo."""
    try:
        cmd = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        if short:
            cmd = ["git", "rev-parse", "--short", "HEAD"]
        else:
            cmd = ["git", "rev-parse", "HEAD"]
        sha = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_author() -> str:
    """Retorna autor: git user.email se configurado, senão usuário do sistema."""
    try:
        email = subprocess.check_output(
            ["git", "config", "user.email"], stderr=subprocess.DEVNULL
        ).decode().strip()
        if email:
            return email
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return getpass.getuser()


def compute_dataset_version(path: str | Path) -> str:
    """Hash SHA-256 (primeiros 12 chars) de um arquivo de dataset."""
    p = Path(path)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def build_standard_tags(
    phase: str,
    dataset_path: str | Path | None = None,
    model_type: str | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Monta o dict de tags padronizadas para um run MLflow.

    Tags geradas:
    - git_sha: commit atual (ou 'unknown')
    - author: git user.email ou usuário do sistema
    - run_phase: fase lógica (baseline / mlp / tuning / production)
    - dataset_version: hash do CSV se fornecido
    - model_type: tipo do modelo (se fornecido)
    - (mais qualquer chave em `extra`)

    Args:
        phase: Fase lógica do run (ex: "baseline", "mlp", "tuning").
        dataset_path: Caminho do CSV usado (para hash).
        model_type: Nome curto do modelo (ex: "log_reg", "mlp_pytorch").
        extra: Tags extras.

    Returns:
        Dict pronto para `mlflow.set_tags(...)`.
    """
    tags: dict[str, str] = {
        "git_sha": get_git_sha(short=True),
        "author": get_author(),
        "run_phase": phase,
    }
    if dataset_path is not None:
        tags["dataset_version"] = compute_dataset_version(dataset_path)
    if model_type is not None:
        tags["model_type"] = model_type
    if extra:
        tags.update({k: str(v) for k, v in extra.items()})
    return tags


# ---------------------------------------------------------------------------
# Parâmetros hierárquicos
# ---------------------------------------------------------------------------

def flatten_params(
    nested: dict[str, Any],
    separator: str = ".",
    prefix: str = "",
) -> dict[str, Any]:
    """Achata um dict aninhado em chaves com dot notation.

    Exemplo:
        >>> flatten_params({"model": {"hidden1": 128}, "train": {"lr": 1e-3}})
        {"model.hidden1": 128, "train.lr": 0.001}

    Útil para organizar `mlflow.log_params(...)` em grupos lógicos:
    - `model.*` — arquitetura e hiperparâmetros
    - `train.*` — batch_size, lr, epochs, otimizador
    - `data.*` — split, seed, dataset_version

    Args:
        nested: Dict possivelmente aninhado.
        separator: Separador entre níveis (default ".").
        prefix: Prefixo interno (uso recursivo).

    Returns:
        Dict flat com chaves compostas.
    """
    flat: dict[str, Any] = {}
    for key, value in nested.items():
        full_key = f"{prefix}{separator}{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_params(value, separator=separator, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


# ---------------------------------------------------------------------------
# Notas ricas em MLflow
# ---------------------------------------------------------------------------

def format_note_content(
    objective: str,
    approach: str,
    dataset_info: str | None = None,
    expected_outcome: str | None = None,
    caveats: str | None = None,
) -> str:
    """Formata um bloco markdown para `mlflow.note.content` de um run.

    Aparece na UI do MLflow como descrição rica do experimento.

    Args:
        objective: Objetivo do run em 1-2 linhas.
        approach: Abordagem técnica (algoritmo, hiperparâmetros principais).
        dataset_info: Informação do dataset (versão, split).
        expected_outcome: Resultado esperado (baseline, métrica-alvo).
        caveats: Limitações conhecidas.

    Returns:
        String markdown pronta para `mlflow.set_tag("mlflow.note.content", ...)`.
    """
    sections = [f"### Objetivo\n{objective}", f"### Abordagem\n{approach}"]
    if dataset_info:
        sections.append(f"### Dataset\n{dataset_info}")
    if expected_outcome:
        sections.append(f"### Resultado esperado\n{expected_outcome}")
    if caveats:
        sections.append(f"### Limitações\n{caveats}")
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Context manager conveniente
# ---------------------------------------------------------------------------

@contextmanager
def start_run_with_context(
    experiment: str,
    run_name: str,
    phase: str,
    dataset_path: str | Path | None = None,
    model_type: str | None = None,
    note: str | None = None,
    extra_tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
):
    """Context manager que abre um run MLflow já com tags padronizadas e nota.

    Substitui o boilerplate de mlflow.set_experiment + start_run + set_tags.

    Exemplo:
        with start_run_with_context(
            experiment="churn-baselines",
            run_name="log_reg",
            phase="baseline",
            dataset_path="data/raw/telecom_churn_base_extended.csv",
            model_type="log_reg",
            note="Regressão Logística com C=0.1, L2.",
        ) as run:
            mlflow.log_params({"model.C": 0.1})
            mlflow.log_metrics({"roc_auc": 0.87})

    Args:
        experiment: Nome do experimento MLflow.
        run_name: Nome do run.
        phase: Fase lógica (ex: "baseline", "mlp", "tuning").
        dataset_path: Caminho do CSV (para hash de versão).
        model_type: Tipo do modelo (tag).
        note: Descrição markdown do run (vai para mlflow.note.content).
        extra_tags: Tags adicionais.
        tracking_uri: Override do tracking URI (ex: "file:./mlruns").

    Yields:
        Objeto Run do MLflow ativo.
    """
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as run:
        tags = build_standard_tags(
            phase=phase,
            dataset_path=dataset_path,
            model_type=model_type,
            extra=extra_tags,
        )
        mlflow.set_tags(tags)
        if note:
            mlflow.set_tag("mlflow.note.content", note)
        yield run


__all__ = [
    "build_standard_tags",
    "compute_dataset_version",
    "flatten_params",
    "format_note_content",
    "get_author",
    "get_git_sha",
    "start_run_with_context",
]
