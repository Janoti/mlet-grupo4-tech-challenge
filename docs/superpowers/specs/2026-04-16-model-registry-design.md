# Design: MLflow Model Registry com Seleção Inteligente de Champion

**Data:** 2026-04-16
**Status:** Aprovado
**Contexto:** Tech Challenge Fase 1 — FIAP Pós-graduação em ML Engineering

## Problema

O `scripts/export_model.py` exporta sempre `LogisticRegression` hardcoded para produção,
ignorando os resultados reais dos experimentos no MLflow. Não há rastreabilidade de qual
modelo está em produção, por que foi escolhido, nem como voltar para uma versão anterior.

## Decisões de Design

### Métrica de seleção

- **Primária:** `valor_liquido` (TP × R$500 − (TP+FP) × R$50)
- **Desempate:** `roc_auc`
- **Justificativa:** O PDF do Tech Challenge exige "métrica de negócio (custo de churn evitado)"
  e "análise de trade-off de custo (FP vs FN)". O `valor_liquido` captura ambos.

### Comparação cross-experiment

Todos os modelos (baselines sklearn + MLP PyTorch) competem na mesma seleção.
O PDF exige "Comparar MLP vs. baselines usando ≥ 4 métricas".

### Abordagem escolhida (Abordagem 3)

Script de seleção inteligente + registro no MLflow + metadata JSON.
Funciona com o file store local existente sem migração de backend.

## Arquitetura

```
Notebooks treinam → MLflow grava runs
                         ↓
             export_model.py (refatorado)
                         ↓
             registry.find_champion()
             ├─ mlflow.search_runs() cross-experiment
             ├─ rankeia por valor_liquido → roc_auc
             └─ retorna melhor run
                         ↓
             registry.register_champion()
             └─ mlflow.register_model("churn-champion")
                         ↓
             registry.export_champion()
             ├─ se sklearn: retreina via pipelines/
             ├─ se pytorch: load MLflow + PyTorchChurnWrapper
             ├─ salva models/churn_pipeline.joblib
             └─ salva models/champion_metadata.json
                         ↓
             API (main.py)
             ├─ carrega joblib (como antes)
             ├─ lê metadata → model_version real
             └─ /health e /predict retornam nome do champion
```

## Arquivos

### Novo: `src/churn_prediction/registry.py`

Módulo com 3 funções públicas:

- `find_champion(tracking_uri, metric, tiebreaker, experiment_names)` → dict com melhor run
- `register_champion(run_id, model_name, tracking_uri)` → version number
- `export_champion(champion, version, out_dir)` → Path do joblib exportado

Inclui classe `PyTorchChurnWrapper` para adaptar MLP PyTorch à interface sklearn
(`predict`, `predict_proba`).

### Modificado: `scripts/export_model.py`

De hardcoded LogisticRegression para orquestrador que delega ao registry.py.
Mantém a mesma interface de chamada: `poetry run python scripts/export_model.py`.

### Modificado: `src/churn_prediction/api/main.py`

`load_model()` lê `champion_metadata.json` para popular `model_version` com o nome
real do champion em vez de "churn_pipeline".

### Modificado: `docker-compose.yml`

Imagem MLflow atualizada de v2.14.0 para v2.22.4 (consistência com poetry.lock).

### Novo: `tests/test_registry.py`

5 testes: find_champion (best, tiebreaker, no_runs), export (joblib+metadata, candidates).

### Modificado: `README.md`

Documentação do Model Registry, champion_metadata.json e comandos de uso.

## champion_metadata.json

```json
{
  "champion_run_id": "<run_id>",
  "champion_run_name": "<model_name>",
  "registry_version": 1,
  "selection_metric": "valor_liquido",
  "selection_tiebreaker": "roc_auc",
  "metrics": {
    "valor_liquido": 1194000.0,
    "roc_auc": 0.8772,
    "f1": 0.7431,
    "pr_auc": 0.8465
  },
  "dataset_version": "sha256:75960d3ae8af",
  "exported_at": "2026-04-16T14:30:00Z",
  "all_candidates": [
    {"run_name": "...", "valor_liquido": ..., "roc_auc": ...}
  ]
}
```

## Problema do MLP PyTorch na API

O MLP é um `nn.Module` sem `predict_proba()`. A API espera interface sklearn.

Solução: `PyTorchChurnWrapper` empacota modelo + preprocessor fitado, expondo
`predict()` e `predict_proba()`. Serializado com joblib como pipeline único.

O preprocessor é obtido via `build_preprocessor(X_train).fit(X_train)` usando os
mesmos dados de `prepare_data()`.

## Descoberta: Baselines sem artefatos no MLflow

Os baselines do notebook 02 não chamam `mlflow.sklearn.log_model()` — apenas logam
métricas e parâmetros. Quando um baseline vence, o export retreina o modelo usando
`train_and_evaluate()` do `pipelines/__init__.py`.

O MLP tem artefato salvo (flavor pytorch) e pode ser carregado diretamente.

## Compatibilidade

- Docker/API: lê `churn_pipeline.joblib` como antes
- CI/CD: chama `poetry run python scripts/export_model.py` como antes
- Notebooks: sem alteração
- MLflow UI: modelos registrados aparecem na aba Models
