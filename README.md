# Predição de Churn em Telecom — Tech Challenge Grupo 4

Pipeline completo de Machine Learning para prever churn em telecom: do dado bruto ao modelo servido via API, com rastreabilidade, testes automatizados e monitoramento de drift.

---

## Índice

1. [Visão geral](#1-visão-geral)
2. [Arquitetura do projeto](#2-arquitetura-do-projeto)
3. [Estrutura de arquivos](#3-estrutura-de-arquivos)
4. [Setup do ambiente](#4-setup-do-ambiente)
5. [Fluxo completo: do treino à API](#5-fluxo-completo-do-treino-à-api)
6. [Executando os notebooks](#6-executando-os-notebooks)
7. [Exportando o modelo para a API](#7-exportando-o-modelo-para-a-api)
8. [API de inferência (FastAPI)](#8-api-de-inferência-fastapi)
9. [Docker](#9-docker)
10. [MLflow — rastreamento de experimentos](#10-mlflow--rastreamento-de-experimentos)
11. [Testes automatizados](#11-testes-automatizados)
12. [Qualidade de código (ruff)](#12-qualidade-de-código-ruff)
13. [Monitoramento de drift](#13-monitoramento-de-drift)
14. [CI/CD](#14-cicd)
15. [Resultados](#15-resultados)
16. [Documentação complementar](#16-documentação-complementar)

---

## 1. Visão geral

Uma operadora de telecomunicações precisa identificar clientes com risco de cancelamento para acionar campanhas de retenção proativas. Este projeto constrói um pipeline end-to-end:

- **EDA** completa sobre base sintética de 50.000 clientes
- **Baselines** com Scikit-Learn: Dummy, LogisticRegression, RandomForest, GradientBoosting
- **Rede neural MLP** em PyTorch com early stopping, mini-batches e otimização de threshold
- **API FastAPI** para inferência em tempo real
- **MLflow** para rastrear experimentos, parâmetros e métricas
- **Docker** para containerização da API
- **Monitoramento de drift** com KS test, Chi² e PSI

### Modelos e resultados (resumo)

| Modelo | ROC-AUC | F1 | Valor Líquido |
|---|---|---|---|
| Dummy (stratified) | 0.505 | 0.394 | R$ 561.850 |
| LogisticRegression | 0.878 | 0.743 | R$ 1.190.800 |
| RandomForest | 0.861 | 0.715 | R$ 1.103.750 |
| GradientBoosting | 0.882 | 0.747 | R$ 1.183.300 |
| **MLP PyTorch** | **0.877** | **0.743** | **R$ 1.192.700** |

> Fórmula de negócio: `valor_liquido = TP × R$500 − (TP + FP) × R$50`

---

## 2. Arquitetura do projeto

```
Dataset (CSV)
     │
     ▼
notebooks/01_eda.ipynb          ← Análise exploratória
     │
     ▼
notebooks/02_baselines.ipynb    ← Treina Dummy, LogReg, RF, GB → MLflow
     │  └─ exporta splits preprocessados → data/processed/
     ▼
notebooks/03_mlp_pytorch.ipynb  ← Treina MLP → MLflow
     │
     ▼
scripts/export_model.py         ← Serializa pipeline → models/churn_pipeline.joblib
     │
     ▼
src/churn_prediction/api/       ← FastAPI carrega o .joblib
     │
     ▼
Docker (churn-api:8000)         ← API containerizada para produção
     │
     ▼
scripts/simulate_drift.py       ← Simula requisições com drift
scripts/check_drift.py          ← Detecta drift (KS, PSI)
```

**Notebooks vs. `src/`:**
- Os notebooks são o ambiente de experimentação e treinamento (MLflow, visualizações, análises)
- O `src/churn_prediction/` contém o código modularizado e reutilizável que a API e os testes consomem
- O `export_model.py` é a ponte: serializa o pipeline treinado nos notebooks para que a API use

---

## 3. Estrutura de arquivos

```text
mlet-grupo4-tech-challenge/
├── .github/workflows/
│   └── ci_ml_pipeline.yml        # CI/CD: lint → testes → treino → Docker
├── data/
│   ├── raw/                      # CSVs brutos (gerados por scripts/generate_synthetic.py)
│   ├── interim/                  # Artefatos intermediários
│   └── processed/                # Splits pré-processados (gerados pelo notebook 02)
├── docs/
│   ├── ml_canvas.md              # ML Canvas (padrão Dorard)
│   ├── model_card.md             # Model Card (padrão Google Mitchell et al.)
│   ├── eda_metodologia.md        # Metodologia da EDA
│   ├── baseline_metodologia.md   # Metodologia dos baselines
│   └── fase1_doc_tecnica.md      # Documento técnico consolidado
├── models/
│   └── churn_pipeline.joblib     # Pipeline serializado (gerado por export_model.py)
├── notebooks/
│   ├── 01_eda.ipynb              # EDA: volume, qualidade, distribuição, correlações
│   ├── 02_baselines.ipynb        # Baselines + fairness + MLflow + export splits
│   └── 03_mlp_pytorch.ipynb      # MLP PyTorch + early stopping + trade-off custo
├── scripts/
│   ├── analyze_mlruns.py         # Lê MLflow e imprime resumo de todos os modelos
│   ├── export_model.py           # Treina pipeline sklearn e salva .joblib para a API
│   ├── simulate_drift.py         # Envia requisições com distribuição alterada à API
│   ├── check_drift.py            # Compara treino vs produção (KS, Chi², PSI)
│   ├── generate_synthetic.py     # Gera dataset sintético estendido
│   ├── generate_dataset.py       # Gera dataset simplificado (legado)
│   └── logging_utils.py          # Logger padronizado para scripts
├── src/churn_prediction/
│   ├── config.py                 # Constantes: LEAKAGE_COLS, V_RETIDO, C_ACAO, seeds
│   ├── data_cleaning.py          # Limpeza: duplicados, clip, padronização, age_group
│   ├── preprocessing.py          # Pipeline sklearn: imputer + scaler + OHE
│   ├── evaluation.py             # Métricas técnicas (ROC-AUC, F1, PR-AUC) e de negócio
│   ├── model.py                  # Classe MLP PyTorch (BatchNorm, Dropout)
│   ├── monitoring.py             # Drift detection: KS test, Chi², PSI, InferenceLogger
│   ├── api/
│   │   ├── main.py               # FastAPI: /predict, /health, /
│   │   └── schemas.py            # Pydantic schemas: CustomerFeatures, PredictionResponse
│   └── pipelines/
│       └── __init__.py           # Orquestração: prepare_data() → train_and_evaluate()
├── tests/
│   ├── test_smoke.py             # Smoke tests: imports de todos os módulos src/
│   ├── test_schema.py            # Testes de schema Pydantic + data cleaning + evaluation
│   └── test_api.py               # Testes de integração: /health, /predict, /
├── Dockerfile                    # Imagem Docker da API
├── docker-compose.yml            # Serviços: churn-api + mlflow
├── Makefile                      # Automação: run-all, test, lint, mlflow, docker
└── pyproject.toml                # Dependências, linting (ruff), pytest — single source of truth
```

---

## 4. Setup do ambiente

**Pré-requisitos:**
- Python 3.11+ (recomendado 3.12)
- [Poetry](https://python-poetry.org/docs/#installation)
- Docker Desktop (opcional, para containerização)

**Instalação:**

```bash
git clone https://github.com/Janoti/mlet-grupo4-tech-challenge.git
cd mlet-grupo4-tech-challenge
poetry install
```

Ou via Makefile:

```bash
make install
```

**Gerar o dataset sintético** (se `data/raw/telecom_churn_base_extended.csv` não existir):

```bash
poetry run python scripts/generate_synthetic.py --n-rows 50000 --seed 42 --out-dir data/raw
```

O `make run-all` faz isso automaticamente se o arquivo não existir.

---

## 5. Fluxo completo: do treino à API

Este é o caminho completo do zero à API rodando:

```bash
# Passo 1: instalar dependências
make install

# Passo 2: treinar todos os modelos (EDA + baselines + MLP → MLflow)
make run-all

# Passo 3: exportar o pipeline treinado para a API
PYTHONPATH=src poetry run python scripts/export_model.py
# → gera: models/churn_pipeline.joblib

# Passo 4a: rodar a API localmente
PYTHONPATH=src poetry run uvicorn churn_prediction.api.main:app --reload --port 8000

# Passo 4b: OU rodar via Docker
docker compose up --build churn-api

# Passo 5: verificar se o modelo foi carregado
curl http://localhost:8000/health
# esperado: {"status":"ok","model_loaded":true,"model_version":"churn_pipeline"}

# Passo 6: fazer uma predição
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "gender": "male", "monthly_charges": 120, "nps_score": 3}'
```

> **Atenção:** se o `/health` retornar `"model_loaded": false`, o arquivo `models/churn_pipeline.joblib` não foi gerado. Rode o Passo 3 antes de subir a API.

---

## 6. Executando os notebooks

### Todos de uma vez (recomendado)

```bash
make run-all
```

Executa em sequência: gera dados → EDA → baselines → MLP → análise → sobe MLflow.

### Individualmente

```bash
make notebooks-eda        # executa notebooks/01_eda.ipynb
make notebooks-baselines  # executa notebooks/02_baselines.ipynb
make notebooks-mlp        # executa notebooks/03_mlp_pytorch.ipynb
make analyze              # lê MLflow e imprime resumo comparativo
```

### O que cada notebook faz

**`01_eda.ipynb`** — Análise exploratória:
- Volume, qualidade, missing, outliers
- Distribuições e correlações (14 pares com |r| > 0.7 identificados)
- Data readiness e diagnóstico de leakage

**`02_baselines.ipynb`** — Baselines e fairness:
- Treina: `DummyClassifier`, `LogisticRegression` (GridSearchCV), `RandomForest`, `GradientBoosting`
- Pipeline sklearn completo: limpeza → imputação → scaling → OHE
- Avalia fairness por `gender`, `age_group`, `region`, `plan_type` (Fairlearn)
- Mitigação com `ExponentiatedGradient` + `EqualizedOdds`
- Registra tudo no MLflow (`churn-baselines`)
- **Exporta splits pré-processados** para `data/processed/` (usado pelo notebook 03)

**`03_mlp_pytorch.ipynb`** — MLP PyTorch:
- Arquitetura: `input → 128 → 64 → 1` com BatchNorm e Dropout=0.3
- Loss: `BCEWithLogitsLoss`, otimizador: `Adam`
- Treinamento com mini-batches (BATCH_SIZE=512) e early stopping (patience=10)
- `ReduceLROnPlateau` para ajuste dinâmico da taxa de aprendizado
- Seeds globais fixadas: `random=42`, `numpy=42`, `torch=42`
- Comparação dinâmica com todos os baselines via MLflow
- Varredura de threshold (0.10–0.90) para maximização de valor líquido
- Feature Importance via RF e GB
- Registra no MLflow (`churn-mlp-pytorch`) incluindo `mlflow.pytorch.log_model`

### Saída esperada no terminal

```
[notebooks-eda] Iniciando execucao de notebooks/01_eda.ipynb...
[notebooks-eda] Concluido.
[notebooks-baselines] Iniciando execucao de notebooks/02_baselines.ipynb...
[notebooks-baselines] Concluido.
[notebooks-mlp] Iniciando execucao de notebooks/03_mlp_pytorch.ipynb...
[notebooks-mlp] Concluido.
[analysis] Resumo automatico da run...
  dummy_stratified  | roc_auc=0.505 | f1=0.394 | valor_liquido=561850
  log_reg           | roc_auc=0.878 | f1=0.743 | valor_liquido=1190800
  random_forest     | roc_auc=0.861 | f1=0.715 | valor_liquido=1103750
  gradient_boosting | roc_auc=0.882 | f1=0.747 | valor_liquido=1183300
  mlp_pytorch_v1    | roc_auc=0.877 | f1=0.743 | valor_liquido=1192700
```

---

## 7. Exportando o modelo para a API

Após o `make run-all`, o pipeline treinado precisa ser serializado para que a API use:

```bash
PYTHONPATH=src poetry run python scripts/export_model.py
```

O script `scripts/export_model.py`:
1. Carrega os dados brutos
2. Aplica o pipeline de limpeza e pré-processamento (`src/churn_prediction/`)
3. Treina a `LogisticRegression` (modelo de referência para produção)
4. Serializa com `joblib.dump` em `models/churn_pipeline.joblib`

> O modelo exportado é o `LogisticRegression` por ser o melhor equilíbrio entre performance, interpretabilidade e simplicidade operacional. Para trocar para outro modelo, edite `scripts/export_model.py`.

---

## 8. API de inferência (FastAPI)

### Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Informações gerais da API |
| `GET` | `/health` | Status da API e do modelo carregado |
| `POST` | `/predict` | Predição de churn para um cliente |
| `GET` | `/docs` | Swagger UI (documentação interativa) |
| `GET` | `/redoc` | ReDoc |

### Rodar localmente (sem Docker)

```bash
# Pré-requisito: models/churn_pipeline.joblib deve existir (seção 7)
PYTHONPATH=src poetry run uvicorn churn_prediction.api.main:app --reload --port 8000
```

### Testar os endpoints

```bash
# Verificar se o modelo foi carregado
curl http://localhost:8000/health
```

Resposta esperada:
```json
{"status": "ok", "model_loaded": true, "model_version": "churn_pipeline"}
```

Se `model_loaded` for `false`, rode `scripts/export_model.py` (seção 7) e reinicie.

```bash
# Predição de churn
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "gender": "male",
    "region": "Sudeste",
    "plan_type": "pos",
    "monthly_charges": 120.0,
    "nps_score": 3,
    "support_calls_30d": 5,
    "late_payments_6m": 2,
    "months_to_contract_end": 1
  }'
```

Resposta esperada:
```json
{
  "churn_probability": 0.7234,
  "churn_prediction": 1,
  "risk_level": "alto",
  "model_version": "churn_pipeline"
}
```

### Campos de entrada (`/predict`)

Todos os campos são **opcionais** — o pipeline imputa valores ausentes automaticamente com mediana/moda, exatamente como no treinamento.

| Campo | Tipo | Descrição |
|---|---|---|
| `age` | int (18–100) | Idade do cliente |
| `gender` | string | `male` / `female` |
| `region` | string | Região geográfica |
| `plan_type` | string | `pre` / `pos` / `controle` |
| `monthly_charges` | float | Valor mensal do plano |
| `nps_score` | int (0–10) | NPS do cliente |
| `support_calls_30d` | int | Ligações ao suporte nos últimos 30 dias |
| `late_payments_6m` | int | Pagamentos atrasados nos últimos 6 meses |
| `months_to_contract_end` | int | Meses restantes de contrato |
| *(demais campos)* | | Ver `/docs` para lista completa |

**Faixas de risco retornadas:**
- `alto`: probabilidade ≥ 0.70
- `medio`: 0.40 ≤ probabilidade < 0.70
- `baixo`: probabilidade < 0.40

**Erros possíveis:**
- `HTTP 422`: campo com tipo inválido (ex.: `age: "abc"`) — validação Pydantic
- `HTTP 503`: modelo não carregado — rode `export_model.py` e reinicie a API

### Variável de ambiente

```bash
# Apontar para outro modelo serializado
CHURN_MODEL_PATH=/caminho/para/outro_modelo.joblib uvicorn churn_prediction.api.main:app
```

### Documentação interativa

Acesse `http://localhost:8000/docs` para testar todos os endpoints pelo navegador via Swagger UI.

---

## 9. Docker

### Subir a API com Docker Compose

```bash
# Pré-requisito: models/churn_pipeline.joblib deve existir (seção 7)
docker compose up --build churn-api
```

A API fica disponível em `http://localhost:8000`.

### Build manual da imagem

```bash
docker build -t churn-api:latest .
docker run -p 8000:8000 -v $(pwd)/models:/app/models churn-api:latest
```

### Subir API + MLflow juntos

```bash
docker compose up --build
# API:    http://localhost:8000
# MLflow: http://localhost:5000
```

---

## 10. MLflow — rastreamento de experimentos

### Subir a UI

```bash
make mlflow-up
# Acesse: http://127.0.0.1:5000
```

Ou manualmente:
```bash
poetry run mlflow ui --backend-store-uri ./mlruns
```

Se a porta 5000 estiver ocupada:
```bash
MLFLOW_PORT=5001 make mlflow-up
```

### Experimentos registrados

| Experimento | Modelos |
|---|---|
| `churn-baselines` | dummy_stratified, log_reg, log_reg_mitigated, random_forest, gradient_boosting |
| `churn-mlp-pytorch` | mlp_pytorch_v1 |

### O que é registrado

- **Parâmetros**: arquitetura, hiperparâmetros, seed, batch_size, threshold
- **Métricas técnicas**: accuracy, f1, roc_auc, pr_auc, best_val_loss
- **Métricas de negócio**: tp, fp, clientes_abordados, valor_liquido, valor_por_cliente
- **Artefatos**: modelo serializado (`.joblib` nos baselines, `.pt` no MLP via `mlflow.pytorch.log_model`)
- **Dataset version**: hash SHA-256 do CSV como tag `dataset_version`

### Ver resumo no terminal (sem abrir o browser)

```bash
make analyze
# ou
PYTHONPATH=src poetry run python scripts/analyze_mlruns.py
```

---

## 11. Testes automatizados

```bash
# Rodar todos (30 testes)
make test
# ou
poetry run pytest tests/ -v

# Por arquivo
poetry run pytest tests/test_smoke.py -v     # imports dos módulos src/
poetry run pytest tests/test_schema.py -v    # Pydantic + data cleaning + métricas
poetry run pytest tests/test_api.py -v       # endpoints /health, /predict, /

# Classe específica
poetry run pytest tests/test_schema.py::TestDataCleaning -v
```

### Cobertura dos testes

| Arquivo | O que testa |
|---|---|
| `test_smoke.py` | Todos os módulos `src/` importam sem erro; classes e funções existem |
| `test_schema.py` | Schemas Pydantic (campos válidos/inválidos); limpeza de dados; métricas técnicas e de negócio |
| `test_api.py` | `/health` retorna 200; `/predict` retorna 503 sem modelo, 200 com mock; validação de input inválido (422) |

---

## 12. Qualidade de código (ruff)

```bash
# Verificar
poetry run ruff check src/ scripts/ tests/

# Corrigir automaticamente
poetry run ruff check src/ scripts/ tests/ --fix
```

Configuração em `pyproject.toml`: `line-length=100`, rules `E, F, I, B, UP`, target Python 3.11.

---

## 13. Monitoramento de drift

### Simular drift

Com a API rodando (`http://localhost:8000`):

```bash
poetry run python scripts/simulate_drift.py --url http://localhost:8000 --n-requests 100
# Salva logs em: logs/drift_simulation.jsonl
```

O script envia requisições com distribuição alterada (ex.: `age +10`, `charges +30%`) para simular mudança de perfil da base.

### Analisar drift

```bash
PYTHONPATH=src poetry run python scripts/check_drift.py \
  --reference data/raw/telecom_churn_base_extended.csv \
  --production logs/drift_simulation.jsonl
```

Saída esperada:
```
======================================================================
RELATÓRIO DE DATA DRIFT
======================================================================
Features analisadas: 15
Alertas de drift: 3
----------------------------------------------------------------------
  ⚠ DRIFT | age             | kolmogorov_smirnov | p=0.0001 PSI=0.2541
  ⚠ DRIFT | monthly_charges | kolmogorov_smirnov | p=0.0003 PSI=0.1872
    OK     | nps_score       | kolmogorov_smirnov | p=0.4521 PSI=0.0123
======================================================================
```

### Testes estatísticos

| Teste | Tipo de feature | Alerta |
|---|---|---|
| KS (Kolmogorov-Smirnov) | Numéricas | p < 0.05 |
| Chi² (Qui-Quadrado) | Categóricas | p < 0.05 |
| PSI | Numéricas | < 0.10 OK · 0.10–0.20 investigar · > 0.20 retreinar |

---

## 14. CI/CD

Pipeline em `.github/workflows/ci_ml_pipeline.yml`, executado em todo push e PR:

1. **Quality** (todos os branches): `ruff check` + `pytest` com 30 testes
2. **Train** (apenas `main`): gera dados → treina baselines → exporta modelo
3. **Docker** (apenas `main`): valida build da imagem `churn-api`

---

## 15. Resultados

### Comparação completa de modelos

| Modelo | Accuracy | F1 | ROC-AUC | PR-AUC | Valor Líquido |
|---|---|---|---|---|---|
| Dummy (stratified) | 0.529 | 0.394 | 0.505 | 0.396 | R$ 561.850 |
| LogisticRegression (C=0.1) | 0.807 | 0.743 | 0.878 | 0.848 | R$ 1.190.800 |
| RandomForest (200 est.) | 0.795 | 0.715 | 0.861 | 0.823 | R$ 1.103.750 |
| GradientBoosting (100 est.) | 0.814 | 0.747 | 0.882 | 0.854 | R$ 1.183.300 |
| **MLP PyTorch** | **0.807** | **0.743** | **0.877** | **0.846** | **R$ 1.192.700** |
| LogReg + EqualizedOdds | 0.801 | 0.735 | — | — | R$ 1.180.950 |

### Fairness (LogisticRegression, sem mitigação)

| Atributo sensível | dp_diff | eo_diff |
|---|---|---|
| `gender` | 0.0552 | 0.0741 |
| `age_group` | 0.1101 | 0.0758 |
| `region` | 0.1565 | 0.1672 |
| `plan_type` | 0.2960 | 0.1840 |

### Feature Importance — top features (consenso RF + GB)

| Feature | Importância RF | Importância GB |
|---|---|---|
| `nps_detractor_flag` | 0.098 | 0.195 |
| `late_payments_6m` | 0.060 | 0.215 |
| `nps_category_detractor` | 0.080 | 0.090 |
| `invoice_shock_flag` | 0.054 | 0.087 |
| `plan_price` | 0.036 | 0.103 |

Satisfação (`nps_detractor_flag`) e inadimplência (`late_payments_6m`) são os sinais mais fortes de churn.

---

## 16. Documentação complementar

| Documento | Descrição |
|---|---|
| [`docs/ml_canvas.md`](docs/ml_canvas.md) | ML Canvas: problema, dados, pipeline, métricas, SLOs |
| [`docs/model_card.md`](docs/model_card.md) | Model Card: performance, fairness, limitações, deploy, monitoramento |
| [`docs/eda_metodologia.md`](docs/eda_metodologia.md) | Metodologia detalhada da EDA |
| [`docs/baseline_metodologia.md`](docs/baseline_metodologia.md) | Metodologia dos baselines |
| [`docs/fase1_doc_tecnica.md`](docs/fase1_doc_tecnica.md) | Documento técnico consolidado |

---

## Contato

Grupo 4 — Tech Challenge FIAP · Machine Learning Engineering
