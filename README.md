# Predição de Churn em Telecom — Tech Challenge (Grupo 4)

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.14-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Ruff](https://img.shields.io/badge/linted-ruff-261230?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![Tests](https://img.shields.io/badge/tests-81%20passed-brightgreen)](tests/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

Projeto de Machine Learning para prever churn em telecom, com pipeline robusto, automação, rastreabilidade e foco em métricas de negócio.

## Arquitetura do pipeline

```mermaid
flowchart TD
    A[data/raw/<br>telecom_churn_base_extended.csv] --> B[notebooks/01_eda.ipynb<br>EDA completa]
    B --> C[notebooks/02_baselines.ipynb<br>Dummy + LogReg + RF + GB]
    C --> D[notebooks/03_mlp_pytorch.ipynb<br>MLP PyTorch]
    C -- splits pré-processados --> D
    C --> E[(MLflow<br>churn-baselines)]
    D --> F[(MLflow<br>churn-mlp-pytorch)]
    E --> G[scripts/export_model.py<br>+ Model Registry]
    F --> G
    G --> H[models/churn_pipeline.joblib<br>champion automático]
    H --> I[FastAPI<br>/predict /health]
    I --> J[Docker<br>churn-api:8000]
    J --> K[scripts/simulate_drift.py<br>+ check_drift.py<br>KS / Chi² / PSI]

    style E fill:#0194E2,color:#fff
    style F fill:#0194E2,color:#fff
    style I fill:#009688,color:#fff
    style J fill:#2496ED,color:#fff
```

## Fluxo de requisição na API

```mermaid
sequenceDiagram
    participant Client
    participant Middleware as LatencyMiddleware
    participant API as FastAPI
    participant Pipeline as sklearn Pipeline
    participant Model as LogisticRegression

    Client->>Middleware: POST /predict (JSON)
    Middleware->>Middleware: start = perf_counter()
    Middleware->>API: forward request

    API->>API: valida schema (Pydantic)
    alt schema inválido
        API-->>Middleware: HTTP 422
    else schema ok
        API->>Pipeline: transform features
        Pipeline->>Model: predict_proba(X)
        Model-->>Pipeline: probability
        Pipeline-->>API: churn_probability
        API->>API: classifica risk_level
        API-->>Middleware: HTTP 200 + JSON
    end

    Middleware->>Middleware: latency_ms = perf_counter() - start
    Middleware->>Middleware: inject X-Process-Time-Ms header
    Middleware-->>Client: response
```

## Status atual

- EDA executável e documentada
- Baselines: `DummyClassifier`, `LogisticRegression`, `RandomForest`, `GradientBoosting` com fairness (Fairlearn)
- Pipeline MLP em PyTorch com rastreabilidade no MLflow
- Métricas de negócio integradas (clientes abordados, valor líquido, ROI)
- Automação fim a fim via Makefile (`run-all`, `analyze`, `mlflow-up/down`)
- **Pipeline reutilizável** em `src/churn_prediction/` (refatorado dos notebooks)
- **API FastAPI** com endpoints `/predict` e `/health` (Pydantic schemas)
- **Dockerfile + docker-compose** para deploy containerizado
- **Model Registry** com seleção automática do champion por métrica de negócio (`valor_liquido`)
- **Testes automatizados** (81 testes: smoke, schema, API, integration, registry)
- **CI/CD** com GitHub Actions (lint, testes, treinamento, build Docker)
- **Monitoramento de drift** (KS test, Chi², PSI) com simulação
- **Deploy na AWS** com CloudFormation (VPC, EC2, ECR, Elastic IP, Prometheus, Grafana)
- **Treino remoto** na EC2 com MLflow tracking (métricas, artifacts, system metrics, tags)
- **Versionamento automático** de imagens Docker (git hash + timestamp)
- **Segurança** — iptables hardening, fail2ban, SSH desabilitado, anti-exfiltração
- **Teste end-to-end** (workflow completo: predict → drift → feedback → retrain recommendation)
- Documentação técnica e de negócio atualizada

## Quick Start (sem Makefile)

Caminho mais curto para clonar o repositório e subir a API containerizada com o modelo treinado:

```bash
# 1. Clone e instale as dependências
git clone https://github.com/Janoti/mlet-grupo4-tech-challenge.git
cd mlet-grupo4-tech-challenge
poetry install

# 2. Gere o dataset sintético
poetry run python scripts/generate_synthetic.py --n-rows 50000 --seed 42 --out-dir data/raw

# 3. Treine e exporte o pipeline (gera models/churn_pipeline.joblib)
#    Pré-requisito: rodar os notebooks (make notebooks) ou ter runs no MLflow.
PYTHONPATH=src poetry run python scripts/export_model.py

# 4. Suba a stack (API + MLflow + Prometheus + Grafana) em background
docker compose up -d --build

# 5. Valide
curl http://localhost:8000/health
```

URLs disponíveis após o passo 4: API em http://localhost:8000/docs, MLflow em
http://localhost:5000, Prometheus em http://localhost:9090, Grafana em
http://localhost:3000 (`mletg4` / `mletg4`). Para apenas a API:
`docker compose up -d --build churn-api`.

> Para rodar a API direto na máquina (sem Docker), pule os passos 4–5 e use
> `PYTHONPATH=src poetry run uvicorn churn_prediction.api.main:app --reload`.
> Para o pipeline completo automatizado (notebooks + análise + MLflow), use
> `make run-all` — ver [seção 7.1](#71-atalhos-com-makefile).

## Modos de execução

Escolha o caminho conforme o objetivo. Todos os modos partem de `git clone` + `poetry install`.

| Modo | Comando principal | Quando usar | Detalhes |
|---|---|---|---|
| **Pipeline completo (Makefile)** | `make run-all` | Reproduzir tudo do zero: EDA → baselines → MLP → MLflow UI | [§7.1](#71-atalhos-com-makefile) e [§9](#9-fluxo-recomendado) |
| **Quick Start manual (Docker)** | `docker compose up -d --build` | Subir API + observabilidade sem Makefile (precisa de `mlruns/` populado antes) | [Quick Start](#quick-start-sem-makefile) |
| **API local sem Docker** | `uvicorn churn_prediction.api.main:app --reload` | Desenvolver/depurar a API com hot-reload | [§13.1](#131-passo-a-passo-para-rodar-localmente) |
| **API containerizada (só API)** | `docker compose up -d --build churn-api` | Servir só a API sem Prometheus/Grafana | [§13.2](#132-passo-a-passo-via-docker) |
| **Stack de observabilidade** | `docker compose up -d --build` | API + MLflow + Prometheus + Grafana com dashboards provisionados | [§14](#14-observabilidade--prometheus--grafana) |
| **Só MLflow UI** | `make mlflow` ou `mlflow ui --backend-store-uri ./mlruns` | Inspecionar runs já registrados | [§11](#11-mlflow-opcional-local) |
| **Notebooks individuais** | `make notebooks-eda` / `notebooks-baselines` / `notebooks-mlp` | Rodar uma etapa do pipeline sem refazer tudo | [§7.1](#71-atalhos-com-makefile) |
| **Treino + export sem notebooks** | `make notebooks && PYTHONPATH=src poetry run python scripts/export_model.py` | Popular `mlruns/` e exportar champion sem subir API | [§5](#5-model-registry-e-seleção-do-champion) |
| **Deploy AWS** | `./infra/deploy.sh` | Subir stack completa em EC2 com CloudFormation | [§17](#17-deploy-na-aws-ec2--cloudformation) |

> **Pré-requisito comum:** os modos que exportam ou servem o modelo
> (`export_model.py`, `docker compose`, `uvicorn`) exigem `mlruns/` populado.
> Rode `make run-all` ou `make notebooks` antes na primeira execução.

## 1. Objetivo

- Estimar a probabilidade de churn por cliente
- Priorizar campanhas de retenção com base em risco e valor
- Medir impacto com métricas técnicas (AUC-ROC, PR-AUC, F1) e de negócio (churn evitado, valor líquido, ROI)

## 2. Pipeline e notebooks

- `01_eda.ipynb`: Exploração e análise dos dados
- `02_baselines.ipynb`: Baselines lineares e de árvore (`Dummy`, `LogReg`, `RandomForest`, `GradientBoosting`), fairness, métricas de negócio, MLflow; exporta splits
- `03_mlp_pytorch.ipynb`: MLP em PyTorch com BatchNorm, Dropout, early stopping, batching, Feature Importance via RF e GB, comparação vs. todos os baselines (lineares + árvores + MLP), análise de custo por threshold e MLflow; resolução dinâmica de paths (compatível com VS Code, Jupyter e `make`)

## 3. Dataset principal

Base: `data/raw/telecom_churn_base_extended.csv`

O dataset é sintético, com problemas controlados de qualidade (duplicidades, missing, inconsistências) para simular um cenário realista. Inclui variáveis de perfil, uso, financeiro, atendimento e satisfação.

Colunas-chave de saída:

- `churn` (target binário)

Importante:

- `churn` é a variável alvo usada nos baselines.

## 4. Estrutura do repositório

```text
mlet-grupo4-tech-challenge/
├── .github/workflows/
│   └── ci_ml_pipeline.yml       # CI/CD: lint, testes, treino, Docker
├── data/
│   ├── raw/                     # CSVs brutos (sintéticos)
│   ├── interim/                 # Artefatos intermediários
│   └── processed/               # Splits processados
├── docs/
│   ├── ml_canvas.md
│   ├── model_card.md
│   ├── eda_metodologia.md
│   └── fase1_doc_tecnica.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_mlp_pytorch.ipynb
├── scripts/
│   ├── analyze_mlruns.py        # Análise de runs MLflow
│   ├── export_model.py          # Seleciona champion e exporta pipeline
│   ├── simulate_drift.py        # Simula data drift na API
│   ├── check_drift.py           # Analisa drift treino vs produção
│   ├── generate_synthetic.py    # Gera dataset sintético
│   └── logging_utils.py
├── src/churn_prediction/
│   ├── config.py                # Constantes e hiperparâmetros
│   ├── data_cleaning.py         # Limpeza de dados
│   ├── preprocessing.py         # Pipeline sklearn (imputer+scaler+OHE)
│   ├── evaluation.py            # Métricas técnicas e de negócio
│   ├── model.py                 # MLP PyTorch
│   ├── monitoring.py            # Drift detection (KS, Chi², PSI)
│   ├── registry.py              # Model Registry: seleção e exportação do champion
│   ├── api/
│   │   ├── main.py              # FastAPI app (/predict, /health)
│   │   └── schemas.py           # Pydantic schemas
│   └── pipelines/
│       └── __init__.py          # Orquestração prepare_data → train
├── tests/
│   ├── test_smoke.py            # Imports dos módulos
│   ├── test_schema.py           # Validação Pydantic + data cleaning
│   ├── test_api.py              # Endpoints com mocks
│   └── test_registry.py         # Seleção e exportação do champion
├── grafana-prometheus/          # Stack de observabilidade (provisionado)
│   ├── prometheus/
│   │   └── prometheus.yml       # Scrape config (alvo: churn-api:8000)
│   └── grafana/
│       ├── provisioning/        # Datasources e providers de dashboards
│       └── dashboards/          # JSON dos dashboards (auto-loaded)
├── infra/
│   ├── deploy.sh                # Deploy automatizado (build + ECR + CloudFormation)
│   ├── template.yaml            # CloudFormation: VPC, EC2, SG, cfn-init
│   ├── Dockerfile.prod          # Imagem otimizada para produção
│   ├── Dockerfile.train         # Imagem para treino remoto na EC2
│   └── .dockerignore.train      # Dockerignore para build de treino
├── Dockerfile
├── docker-compose.yml           # API + MLflow + Prometheus + Grafana
├── pyproject.toml
└── README.md
```

## 5. Model Registry e seleção do champion

O projeto implementa seleção automática do melhor modelo via MLflow Model Registry.

### 5.1 Como funciona

O script `export_model.py` consulta **todos os experimentos** no MLflow (baselines + MLP), rankeia os modelos por **`valor_liquido`** (métrica de negócio) com desempate por **`roc_auc`**, e exporta o vencedor:

```bash
PYTHONPATH=src poetry run python scripts/export_model.py
```

Saída:
```
Champion selecionado: mlp_pytorch_v1 (run_id=7d2323f7) | valor_liquido=1194000.00 | roc_auc=0.8772
============================================================
CHAMPION: mlp_pytorch_v1
  valor_liquido: 1194000.0000
  roc_auc: 0.8772
------------------------------------------------------------
Candidatos avaliados:
  1. mlp_pytorch_v1 | valor_liquido=1194000 | roc_auc=0.8772
  2. log_reg | valor_liquido=1190800 | roc_auc=0.8783
  3. gradient_boosting | valor_liquido=1183300 | roc_auc=0.8815
  4. random_forest | valor_liquido=1103750 | roc_auc=0.8609
============================================================
```

### 5.2 Artefatos gerados

| Arquivo | Descrição |
|---------|-----------|
| `models/churn_pipeline.joblib` | Pipeline serializável (sklearn ou PyTorch wrapper) |
| `models/champion_metadata.json` | Metadados: run_id, métricas, dataset_version, candidatos |

### 5.3 champion_metadata.json

O arquivo responde às perguntas do ciclo de vida do modelo:

| Pergunta | Campo no JSON |
|----------|---------------|
| Qual versão está em produção? | `champion_run_id` + `champion_run_name` |
| Quais parâmetros foram usados? | `metrics` + `dataset_version` |
| Como voltar para a versão anterior? | `all_candidates` (lista rankeada com run_ids) |

### 5.4 Critério de seleção

- **Métrica primária:** `valor_liquido = TP × R$500 − (TP + FP) × R$50`
- **Desempate:** `roc_auc`
- **Justificativa:** O Tech Challenge exige métrica de negócio (custo de churn evitado) e análise de trade-off de custo (FP vs FN). O `valor_liquido` captura ambos.

### 5.5 Suporte a modelos PyTorch

Quando o MLP é o champion, o registry empacota o modelo PyTorch + preprocessor sklearn em um `PyTorchChurnWrapper` que expõe a interface `predict()` / `predict_proba()`. A API não precisa saber o flavor do champion.

## 6. Automação e rastreabilidade

- **Makefile**: targets para rodar EDA, baselines, MLP, análise e MLflow
- **MLflow**: rastreia experimentos, parâmetros, métricas técnicas e de negócio

## 6.1. API FastAPI para Predição e Monitoramento

O projeto inclui uma **API FastAPI em produção** para inferência, monitoramento de drift e feedback de modelo.

### Quick Start

#### Desenvolvimento Local

```bash
# Inicia a API localmente (porta 8000)
PYTHONPATH=src poetry run uvicorn churn_prediction.api.main:app --reload
```

#### Com Docker

```bash
# Inicia API containerizada + MLflow + Prometheus + Grafana
docker compose up --build

# Ou apenas a API
docker compose up churn-api
```

### Endpoints Principais

| Endpoint | Método | Propósito |
|----------|--------|-----------|
| `/health` | GET | Health check e status do modelo |
| `/predict` | POST | Predição de churn para um cliente |
| `/drift/check` | POST | Detecta data drift em batch de dados |
| `/drift/report` | GET | Relatório detalhado de drift por feature |
| `/model/versions` | GET | Lista histórico de versões treinadas |
| `/model/retrain-recommendation` | POST | Recomenda se deve retreinar |
| `/feedback` | POST | Registra feedback sobre uma predição |
| `/feedback/summary` | GET | Sumário de feedback coletado |
| `/metrics` | GET | Métricas Prometheus para observabilidade |
| `/docs` | GET | Documentação interativa (Swagger UI) |

### Exemplo de Uso

#### 1. Predição

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "gender": "male",
    "region": "Sudeste",
    "plan_type": "pos",
    "tenure_months": 24,
    "nps_score": 7
  }'
```

Resposta:
```json
{
  "churn_probability": 0.6512,
  "churn_prediction": 1,
  "risk_level": "alto",
  "model_version": "mlp_pytorch_v1"
}
```

#### 2. Detecção de Drift

```bash
curl -X POST http://localhost:8000/drift/check \
  -H "Content-Type: application/json" \
  -d '[
    {"age": 35, "gender": "male", "tenure_months": 24, "nps_score": 7},
    {"age": 42, "gender": "female", "tenure_months": 36, "nps_score": 8},
    {"age": 28, "gender": "male", "tenure_months": 12, "nps_score": 5}
  ]'
```

Resposta:
```json
{
  "drift_detected": false,
  "drift_ratio": 0.15,
  "recommendation": "Monitor: drift ratio below threshold"
}
```

#### 3. Recomendação de Retreinamento

```bash
curl -X POST "http://localhost:8000/model/retrain-recommendation?drift_ratio=0.3" \
  -H "Content-Type: application/json"
```

Resposta:
```json
{
  "should_retrain": false,
  "reason": "Drift ratio (0.3) below critical threshold (0.5)"
}
```

#### 4. Submissão de Feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_id": "pred_12345",
    "actual_churn": 1,
    "feedback_type": "correct",
    "comment": "Cliente churned, previsão acertou",
    "rating": 5
  }'
```

Resposta:
```json
{
  "feedback_id": "fb_abc123",
  "timestamp": "2026-05-02T10:30:45.123Z",
  "status": "received",
  "message": "Feedback registrado com sucesso"
}
```

### Teste Integrado (End-to-End)

Executar workflow completo: predict → drift → feedback → retrain recommendation

```bash
# Com modelo carregado
PYTHONPATH=src poetry run pytest tests/api/test_integration.py -v
```

### Documentação Detalhada

Para documentação completa sobre schemas, tratamento de erros e exemplos avançados, consulte:

- **[docs/API.md](docs/API.md)** — Guia técnico completo com curl examples e swagger
- **[/docs](http://localhost:8000/docs)** — Swagger UI interativo (rodando a API)

## 7. Ambiente e instalacao

Pre-requisitos:

- Python 3.11+ (recomendado 3.12)
- Poetry
- Docker Desktop (para containerizacao — opcional para desenvolvimento local)

Dependencias-chave para a etapa de fairness:

- `scikit-learn >=1.5,<1.6`
- `fairlearn ^0.11.0`

Instalacao:

```bash
git clone https://github.com/Janoti/mlet-grupo4-tech-challenge.git
cd mlet-grupo4-tech-challenge
poetry install
```

Opcional com Makefile:

```bash
make install
```

## 7.1 Atalhos com Makefile

Comandos para executar notebooks e iniciar MLflow com logs de progresso no terminal:

```bash
make help
make run-all
make notebooks
make notebooks-eda
make notebooks-baselines
make notebooks-mlp
make analyze
make mlflow
make mlflow-up
make mlflow-down
make mlflow-clean
```

Execucao recomendada (fim a fim):

```bash
make install
make run-all
```

O `make run-all` executa estes steps automaticamente:

1. Limpa `mlruns/`.
2. Executa `notebooks/01_eda.ipynb`.
3. Executa `notebooks/02_baselines.ipynb` (exporta splits preprocessados para `data/processed/`).
4. Executa `notebooks/03_mlp_pytorch.ipynb` (MLP com early stopping, métricas e MLflow).
5. Le os runs em `mlruns/` e imprime analise resumida (baselines + MLP).
6. Sobe MLflow em background e mostra o link.

Saidas esperadas no terminal:

- `make notebooks`:
	- `[notebooks-eda] Iniciando execucao de notebooks/01_eda.ipynb...`
	- `[notebooks-eda] Concluido.`
	- `[notebooks-baselines] Iniciando execucao de notebooks/02_baselines.ipynb...`
	- `[notebooks-baselines] Concluido.`
	- `[notebooks-mlp] Iniciando execucao de notebooks/03_mlp_pytorch.ipynb...`
	- `[notebooks-mlp] Concluido.`
	- `[notebooks] Execucao completa finalizada.`
- `make analyze`:
	- `[analysis] Resumo automatico da run`
	- metricas de `dummy_stratified`, `log_reg` e `log_reg_mitigated_equalized_odds`
	- deltas de performance e leitura sugerida para apresentacao
	- resumo da metrica de negocio (`tp`, `fp`, `clientes_abordados`, `valor_liquido`, `valor_por_cliente`)
	- aviso quando algum run esperado ainda nao existe no `mlruns/`
- `make mlflow`:
	- `[mlflow] Subindo MLflow UI em http://127.0.0.1:5000`
- `make mlflow-up`:
	- `[mlflow] PID: ...`
	- `[mlflow] Link: http://127.0.0.1:5000`

Observacao:

- A execucao dos notebooks usa `nbconvert --execute --inplace`, entao as celulas ficam com outputs salvos no proprio arquivo `.ipynb`.
- Para reduzir warnings de IOPub no `nbconvert`, o Makefile usa timeout maior e nivel de log configuravel (`IOPUB_TIMEOUT` e `NB_LOG_LEVEL`).

Padrao de logs dos scripts Python:

- Os scripts usam logging padronizado com formato unico (`timestamp | level | logger | mensagem`).
- Para controlar verbosidade, use a variavel `LOG_LEVEL` (ex.: `LOG_LEVEL=DEBUG make analyze`).
- Scripts com logging padronizado nesta branch: `scripts/generate_dataset.py` e `scripts/analyze_mlruns.py`.

## 8. Geracao dos dados

### Base estendida (recomendada)

```bash
poetry run python scripts/generate_synthetic.py --n-rows 50000 --seed 42 --out-dir data/raw
```

O script da base estendida agora simula problemas de qualidade de dados. Por padrao, ele injeta:

- duplicidade de linhas
- duplicidade de identificadores
- missing adicional em colunas selecionadas
- valores fora de faixa e categorias invalidas

Exemplo com parametrizacao explicita:

```bash
poetry run python scripts/generate_synthetic.py \
	--n-rows 50000 \
	--seed 42 \
	--out-dir data/raw \
	--duplicate-row-rate 0.03 \
	--duplicate-id-rate 0.02 \
	--missing-noise-rate 0.01 \
	--invalid-value-rate 0.01
```

Arquivo gerado:

- `data/raw/telecom_churn_base_extended.csv`

### Base simplificada (legado)

```bash
poetry run python scripts/generate_dataset.py --n-rows 50000 --seed 42 --out-dir data/raw
```

Arquivo gerado:

- `data/raw/telecom_churn_base.csv`

## 9. Fluxo recomendado

1. Gerar/validar dataset em `data/raw`.
2. Executar exploracao e diagnostico em `notebooks/01_eda.ipynb`. O EDA inclui analise de correlacao entre features numericas: foram identificados **14 pares com |r| > 0.7**, sendo os mais criticos:
	- `price_increase_last_3m` ↔ `invoice_shock_flag` (r = 0.99)
	- `avg_bill_last_6m` ↔ `monthly_charges` (r = 0.97)
	- `data_gb_monthly` ↔ `avg_usage_last_3m` (r = 0.97)
	- `late_payments_6m` ↔ `default_flag` (r = 0.92)

	Pares com |r| > 0.9 representam informacao redundante e devem ser consolidados, especialmente para modelos sensiveis a multicolinearidade como a regressao logistica.
3. No proprio EDA da Fase 1, aplicar tratamento minimo para baseline:
	- remocao de duplicados
	- correcao de valores invalidos
	- imputacao simples
	- one-hot encoding das categoricas
4. Separar os dados em treino e teste com split estratificado `80/20`.
5. Treinar baselines com `DummyClassifier` e `LogisticRegression` no notebook `notebooks/02_baselines.ipynb`.
6. Tunar hiperparametros com `GridSearchCV` (5-fold estratificado) variando `C` em [0.001, 0.01, 0.1, 1, 10, 100] e comparar penalidades L1, L2 e ElasticNet. Melhor configuracao encontrada: `C=0.1` com L2 (diferenca entre penalidades < 0.0002 em ROC-AUC).
7. Otimizar threshold de classificacao por metrica de negocio (valor liquido da campanha de retencao), varrendo thresholds de 0.10 a 0.90.
8. Registrar experimentos no MLflow (parametros, metricas e versao do dataset).
9. Avaliar fairness por subgrupos com Fairlearn (`gender`, `age_group`, `region` e `plan_type`).
10. Aplicar mitigacao opcional com `EqualizedOdds` e comparar trade-offs (a versao atual do notebook usa configuracao otimizada para tempo de execucao).
11. Consolidar resultados em `docs/model_card.md`.
12. Revisar premissas de negocio (`V_RETIDO`, `C_ACAO`) com stakeholders para calibrar decisao operacional.

Tratamento adotado na Fase 1:

- Numericas: imputacao por mediana.
- Categoricas: imputacao por moda.
- Categoricas finais: transformacao com one-hot encoding.
- Exclusoes por leakage: `customer_id`, `churn_probability`, `retention_offer_made`, `retention_offer_accepted`, `contract_renewal_date`, `loyalty_end_date`.

Split adotado:

- Treino: `80%`
- Teste: `20%`
- Estratificacao pelo target `churn`

Fairness e rastreabilidade:

- Diagnostico inicial no EDA por subgrupos sensiveis.
- Avaliacao no baseline com Fairlearn (`demographic_parity_difference`, `equalized_odds_difference`).
- Mitigacao de referencia com `ExponentiatedGradient` + `EqualizedOdds`.
- Registro no MLflow com versao do dataset baseada em hash (`dataset_version`).

Metrica de negocio (status atual):

- Ja registrada automaticamente no MLflow para `dummy_stratified`, `log_reg` e `log_reg_mitigated_equalized_odds`.
- Formula operacionalizada no notebook:
	- `valor_liquido = TP * V_RETIDO - (TP + FP) * C_ACAO`
	- `valor_por_cliente = valor_liquido / N`
- Campos consolidados e analisados em `make analyze`:
	- `tp`, `fp`, `clientes_abordados`, `valor_bruto`, `custo_total_acao`, `valor_liquido`, `valor_por_cliente`

## 10. Qualidade de codigo

Lint:

```bash
poetry run ruff check src scripts tests
```

Auto-fix:

```bash
poetry run ruff check src scripts tests --fix
```

Testes:

```bash
poetry run pytest -q                              # resumido
poetry run pytest tests/ -v                       # verbose (todos)
poetry run pytest tests/test_api.py -v            # apenas testes da API
poetry run pytest tests/test_schema.py::TestDataCleaning -v  # classe especifica
```

## 11. MLflow (opcional local)

```bash
poetry run mlflow ui --backend-store-uri ./mlruns
```

Ou via Makefile:

```bash
make mlflow
```

Abrir em http://127.0.0.1:5000

Se a porta 5000 estiver em uso, rode em outra porta:

```bash
MLFLOW_PORT=5001 make mlflow
```

## 12. Documentacao

- Canvas de negocio e modelagem: `docs/ml_canvas.md`
- Explicacao da EDA e das escolhas metodologicas: `docs/eda_metodologia.md`
- Documento tecnico consolidado da Fase 1 (EDA + baseline + automacao): `docs/fase1_doc_tecnica.md`
- Card do modelo: `docs/model_card.md`

## 13. API de inferência (FastAPI)

A API expõe o modelo treinado para predições em tempo real.

### 13.1 Passo a passo para rodar localmente

```bash
# 1. Gerar dados (se ainda não existem)
poetry run python scripts/generate_synthetic.py --n-rows 50000 --seed 42 --out-dir data/raw

# 2. Exportar pipeline treinado (salva em models/churn_pipeline.joblib)
PYTHONPATH=src poetry run python scripts/export_model.py

# 3. Rodar a API (porta 8000)
PYTHONPATH=src poetry run uvicorn churn_prediction.api.main:app --reload --port 8000

# 4. Acessar documentação interativa
#    Swagger UI: http://localhost:8000/docs
#    ReDoc:      http://localhost:8000/redoc
```

### 13.2 Passo a passo via Docker

```bash
# 1. Exportar modelo (necessário antes do build)
PYTHONPATH=src poetry run python scripts/export_model.py

# 2. Build e start dos containers
docker compose up -d --build churn-api          # apenas a API
docker compose up -d --build                    # stack completa (API + MLflow + Prometheus + Grafana)
```

A API fica em http://localhost:8000. Para a stack completa de observabilidade
(Prometheus, Grafana e dashboards provisionados), veja a [seção 14](#14-observabilidade--prometheus--grafana).

### 13.3 Testar a API

```bash
# Health check
curl http://localhost:8000/health

# Predição de churn
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "gender": "female", "plan_type": "pos", "monthly_charges": 120, "nps_score": 3}'
```

Resposta esperada:

```json
{
  "churn_probability": 0.6823,
  "churn_prediction": 1,
  "risk_level": "medio",
  "model_version": "mlp_pytorch_v1"
}
```

### 13.4 Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/` | Informações gerais da API |
| `GET` | `/health` | Status da API e do modelo carregado |
| `POST` | `/predict` | Predição de churn (probabilidade, classe, risco) |
| `GET` | `/metrics` | Métricas Prometheus (formato texto) |
| `GET` | `/docs` | Swagger UI (documentação interativa) |
| `GET` | `/redoc` | ReDoc (documentação formatada) |

Observações:
- Todos os campos de entrada são opcionais (o pipeline imputa valores ausentes)
- Campos inválidos retornam HTTP 422 com detalhes do erro (validação Pydantic)
- A variável `CHURN_MODEL_PATH` permite apontar para outro modelo serializado

## 14. Observabilidade — Prometheus + Grafana

A API expõe métricas Prometheus em `GET /metrics` e o repositório inclui uma stack completa
de observabilidade provisionada em `grafana-prometheus/` — sem cliques manuais para
configurar datasource ou importar dashboards.

### 14.1 Arquitetura

```
┌──────────────┐  scrape  ┌────────────┐  query  ┌─────────┐
│  churn-api   │ ───────▶ │ Prometheus │ ◀────── │ Grafana │
│  (FastAPI)   │  /metrics│  TSDB 15d  │         │  UI     │
│   :8000      │   15s    │   :9090    │         │  :3000  │
└──────────────┘          └────────────┘         └─────────┘
```

Tudo roda na rede Docker `churn-net`, então o Prometheus resolve o serviço `churn-api`
por DNS interno do Compose. Grafana carrega datasource e dashboards via provisioning
automático na primeira partida.

### 14.2 Como subir

```bash
# Sobe API + MLflow + Prometheus + Grafana
docker compose up -d --build

# Validar status
docker compose ps
```

| Serviço | URL | Credencial |
|---------|-----|------------|
| Churn API (Swagger) | http://localhost:8000/docs | — |
| API metrics endpoint | http://localhost:8000/metrics | — |
| MLflow UI | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | `mletg4` / `mletg4` |

Para apenas a API + MLflow (sem observabilidade): `docker compose up -d churn-api mlflow`.

### 14.3 Métricas expostas pela API

| Métrica | Tipo | Labels | Descrição |
|---------|------|--------|-----------|
| `churn_api_http_requests_total` | Counter | `method, endpoint, status_code` | Requisições HTTP processadas |
| `churn_api_http_request_duration_seconds` | Histogram | `method, endpoint` | Latência por request (buckets 5ms–2.5s) |
| `churn_api_predictions_total` | Counter | `risk_level, gender, plan_type, churn_prediction` | Predições segmentadas por perfil e decisão |
| `churn_api_prediction_probability` | Histogram | — | Distribuição da `churn_probability` (19 buckets de 0.05 a 0.95) — base do heatmap de drift |
| `churn_api_model_loaded` | Gauge | — | 1=modelo carregado, 0=não |
| `churn_api_model_info` | Info | `version` | Versão/identificador do modelo ativo |

Os labels `gender` e `plan_type` são normalizados em `prometheus_metrics.py` para um
conjunto fechado de valores (`{male, female, other, unknown}` e `{pre, pos, controle,
empresarial, outro, unknown}`), controlando cardinalidade e evitando "label explosion".

### 14.4 Dashboards

A pasta `Churn` no Grafana traz dois dashboards complementares — o primeiro responde
*"como está o serviço?"* e o segundo responde *"o modelo precisa de retreino?"*.

#### 14.4.1 Churn API — Observabilidade

URL direta: http://localhost:3000/dashboards (folder `Churn`).

| Seção | Painéis | Como ler |
|-------|---------|----------|
| **KPIs** | Modelo carregado · Versão · Customer Profile (total) · Churner Profile (predições=1) · **Taxa de churn predita (1h)** · Throughput | Faixa de cor da taxa: verde até 20% · amarelo 20–35% · laranja 35–50% · vermelho >50% |
| **Customer × Churner** | 4 donuts lado a lado: Gênero (customer/churner) e Plano (customer/churner) | Compare a mesma fatia entre os dois donuts. Ex.: se `male` é 50% do customer mas 70% do churner, há sinal diferencial de risco para esse grupo |
| **Taxa de churn por dimensão (1h)** | Bargauges horizontais por gênero e por plano | Cor segue a mesma escala dos KPIs. Útil para detectar concentração de churn em um grupo específico |
| **Risco e Volume** | Predições por risco (área empilhada) · Predições por gênero · Predições por plano · Matriz Plano×Gênero×Risco | Útil para inspeção temporal: picos súbitos em "alto" indicam mudança de comportamento ou ataque ao endpoint |
| **Operacional** _(colapsada)_ | Latência p50/p95/p99 · Status code · Throughput por endpoint | p99 de `/predict` em produção saudável: <50ms. 5xx >0 dispara investigação |

#### 14.4.2 Churn — Saúde do Modelo (drift & retreino)

| Seção | Painéis | Como ler |
|-------|---------|----------|
| **Status do modelo** | Modelo · Versão · Predições 1h/24h · **Saúde do modelo (score 0–3)** | Score = soma de 3 sinais binários. 0=OK · 1=Atenção · 2=Alerta · 3=Retreino recomendado |
| **Prediction drift** | **Heatmap** da `churn_probability` ao longo do tempo · p50/p90 (1h) · Δ p50/p90 (1h vs 24h) | Heatmap saudável tem padrão vertical *estável*. Se a faixa quente desloca horizontalmente, há drift de saída. \|Δ\| > 10pp é alerta amarelo, >20pp é vermelho |
| **Class balance** | Stack 100% de `risk_level` no tempo · Taxa de churn 1h sobreposta com baseline 24h | A linha 1h afastando-se da linha 24h por mais que 15pp dispara o sinal 1 do score |
| **Input drift** | % `gender=unknown` + % `plan_type=unknown\|outro` · Taxa de 4xx/5xx | Crescimento da taxa de unknown indica novas categorias ou bug upstream. 4xx (especialmente 422) sinaliza payloads fora do schema |
| **Sinais de retreino** | Tabela com 3 gates (Status verde/vermelho) | Use como checklist objetivo antes de disparar pipeline de retreino |

#### 14.4.3 Regra de decisão (KPI "Saúde do modelo")

Score composto = soma de 3 sinais binários:

| # | Sinal | Threshold | Razão |
|---|-------|-----------|-------|
| 1 | Drift na taxa de churn predita | \|Δ\| > 15pp (1h vs 24h) | Output drift — captura mudança de classe agnóstica de feature |
| 2 | Excesso de inputs `unknown` | (gender + plan) > 10% | Input drift — novas categorias ou missing massivo |
| 3 | Concentração em risco `alto` | > 60% das predições | Decision skew — calibração degradou ou inputs adversariais |

Mapeamento: **0=OK** (verde) · **1=Atenção** (amarelo) · **2=Alerta** (laranja) · **3=Retreino recomendado** (vermelho).

### 14.5 Como gerar tráfego para validar

```bash
# Carga randomizada com drift artificial (idade +10, charges +30%, etc)
poetry run python scripts/simulate_drift.py --url http://localhost:8000 --n-requests 300

# Carga determinística — perfil de churn alto (ver seção 13.3 de exemplos)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 22, "gender": "male", "plan_type": "pre", "nps_score": 0,
       "support_calls_30d": 9, "tenure_months": 2, "default_flag": 1,
       "competitor_offer_contact_flag": 1}'
```

Após ~30s de tráfego, o Grafana já mostra dados nos dois dashboards (refresh automático
de 10s no operacional, 30s no de saúde).

### 14.6 Queries PromQL úteis

```promql
# Throughput (req/s, janela 5min)
sum(rate(churn_api_http_requests_total[5m]))

# Latência p95 do /predict
histogram_quantile(0.95, sum by (le) (rate(churn_api_http_request_duration_seconds_bucket{endpoint="/predict"}[5m])))

# Taxa de churn predita (1h móvel)
sum(increase(churn_api_predictions_total{churn_prediction="1"}[1h]))
  / clamp_min(sum(increase(churn_api_predictions_total[1h])), 1)

# Quantil da probabilidade predita
histogram_quantile(0.5, sum by (le) (rate(churn_api_prediction_probability_bucket[1h])))
```

### 14.7 Persistência e troubleshooting

- **Volumes nomeados**: `prometheus-data` (TSDB 15d) e `grafana-data` (config + sessões).
  `docker compose down -v` apaga ambos.
- **Counter zerado após rebuild**: ao mudar labelnames de um Counter, séries antigas viram
  stale e novas só aparecem após o primeiro `inc()`. Gere tráfego (seção 14.5) — em ~15s
  os painéis voltam.
- **Painéis vazios após mudar código de métrica**: requer `docker compose up -d --build
  churn-api`. Hot-reload do uvicorn não recarrega o Prometheus client registry.
- **Empty vector em PromQL**: queries com filtro `{label="x"}` retornam vetor vazio se
  nenhuma série casar — `vazio + escalar = vazio`. Use `or vector(0)` para defaultar
  para zero em queries de KPI/alerta.

### 14.8 Referências adicionais

- [docs/PROMETHEUS_SETUP.md](docs/PROMETHEUS_SETUP.md) — guia conceitual de instrumentação Prometheus
- [examples/prometheus.yml](examples/prometheus.yml) — exemplo de config standalone
- [grafana-prometheus/grafana/dashboards/](grafana-prometheus/grafana/dashboards/) — JSON dos dashboards (versionados em git)

## 15. Monitoramento de drift

### 15.1 Passo a passo para simulação de drift

Pré-requisito: a API deve estar rodando (seção 13).

```bash
# 1. Garantir que a API está no ar
curl http://localhost:8000/health

# 2. Enviar requisições com distribuição alterada (age +10, charges +30%)
poetry run python scripts/simulate_drift.py --url http://localhost:8000 --n-requests 100

# Os resultados são salvos em logs/drift_simulation.jsonl
```

### 15.2 Análise de drift

```bash
# Compara dados de treino com logs de produção
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
Razão de drift: 20.0%
----------------------------------------------------------------------
  ⚠ DRIFT | age                            | kolmogorov_smirnov   | p=0.0001 PSI=0.2541 ⚠ PSI>0.20
  ⚠ DRIFT | monthly_charges                | kolmogorov_smirnov   | p=0.0003 PSI=0.1872
    OK     | nps_score                      | kolmogorov_smirnov   | p=0.4521 PSI=0.0123
======================================================================
```

### 15.3 Testes estatísticos utilizados

| Teste | Tipo de feature | Interpretação |
|-------|----------------|---------------|
| **KS (Kolmogorov-Smirnov)** | Numéricas | p < 0.05 → drift detectado |
| **Chi² (Qui-Quadrado)** | Categóricas | p < 0.05 → drift detectado |
| **PSI** | Numéricas | < 0.10 OK · 0.10-0.20 investigar · > 0.20 retreinar |

## 16. CI/CD (GitHub Actions)

Pipeline automatizado em `.github/workflows/ci_ml_pipeline.yml`:
1. **Quality**: lint (ruff) + pytest em todo push/PR
2. **Train**: gera dados → seleciona champion → exporta modelo (apenas main)
3. **Docker**: valida build da imagem (apenas main)

## 17. Deploy na AWS (EC2 + CloudFormation)

O projeto inclui infraestrutura como código para deploy automatizado na AWS, com stack completa (API + MLflow + Prometheus + Grafana) rodando em uma instância EC2 via docker-compose. O deploy inclui treino remoto com rastreabilidade completa no MLflow (métricas, parâmetros, artifacts e system metrics).

### 17.1 Arquitetura na AWS

```
┌─────────────────────────────────────────────────────────┐
│                     VPC (10.0.0.0/16)                   │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Public Subnet (10.0.1.0/24)          │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │           EC2 (t3.medium)                   │  │  │
│  │  │                                             │  │  │
│  │  │  ┌──────────┐ ┌────────┐ ┌───────────────┐ │  │  │
│  │  │  │churn-api │ │ MLflow │ │  Prometheus   │ │  │  │
│  │  │  │  :8000   │ │ :5000  │ │    :9090      │ │  │  │
│  │  │  └──────────┘ └────────┘ └───────────────┘ │  │  │
│  │  │  ┌───────────────────────────────────────┐ │  │  │
│  │  │  │            Grafana :3000              │ │  │  │
│  │  │  └───────────────────────────────────────┘ │  │  │
│  │  │         docker-compose.prod.yml             │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Elastic  │  │  Security    │  │  VPC Flow Logs    │  │
│  │   IP     │  │  Groups      │  │  → CloudWatch     │  │
│  └──────────┘  └──────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────┘
         │
    ┌────┴─────┐
    │   ECR    │  (imagens Docker: API + treino)
    └──────────┘
```

### 17.2 Pré-requisitos

- AWS CLI v2 configurado (`aws sts get-caller-identity`)
- Docker rodando localmente
- Poetry instalado (para treino local no primeiro deploy)

### 17.3 Deploy automatizado (recomendado)

Um único comando faz tudo — treino, build, push e deploy:

```bash
./infra/deploy.sh
```

**Primeiro deploy (do zero):** o script detecta que a stack não existe e executa:
1. Treino local (notebooks) → exporta modelo
2. Build da imagem → push para ECR
3. Deploy CloudFormation (sobe API + MLflow + Prometheus + Grafana)
4. Treino remoto na EC2 (popula MLflow com métricas, artifacts e system metrics)
5. Rebuild + redeploy com modelo do treino remoto

**Deploys seguintes:** o script detecta a stack existente e executa:
1. Treino remoto na EC2 (registra direto no MLflow de produção)
2. Build da imagem com modelo novo → push para ECR
3. Update da stack CloudFormation

**Flags disponíveis:**

```bash
./infra/deploy.sh                    # deploy completo (treino + build + deploy)
./infra/deploy.sh --skip-train       # build + deploy (sem retreinar)
./infra/deploy.sh --train-only       # só treina e exporta modelo
./infra/deploy.sh --build-only       # só build + push ECR
./infra/deploy.sh --stack-only       # só atualiza CloudFormation
./infra/deploy.sh --destroy          # destrói toda a stack
```

**Variáveis de ambiente configuráveis:**

| Variável | Default | Descrição |
|----------|---------|-----------|
| `STACK_NAME` | `churn-prediction-stack` | Nome da stack CloudFormation |
| `AWS_REGION` | `sa-east-1` | Região AWS |
| `INSTANCE_TYPE` | `t3.medium` | Tipo da instância EC2 |
| `IMAGE_TAG` | auto (`git-hash-timestamp`) | Tag da imagem Docker no ECR |
| `GRAFANA_PASSWORD` | `mletg4` | Senha do admin do Grafana |

### 17.4 Versionamento de imagens

Cada deploy gera uma tag versionada automaticamente (ex: `abc1234-20260502-1830`), baseada no git short hash + timestamp. A tag `latest` é sempre atualizada como alias.

```bash
./infra/deploy.sh                          # → auto: abc1234-20260502-1830
IMAGE_TAG=v2.0.0 ./infra/deploy.sh         # → manual: v2.0.0

# Rollback para versão anterior
IMAGE_TAG=abc1234-20260502-1830 ./infra/deploy.sh --skip-train
```

### 17.5 Treino remoto e MLflow

O treino roda dentro de um container Docker na EC2, conectado diretamente ao MLflow server. Isso garante:

| Dado | Capturado? | Como |
|------|-----------|------|
| Métricas (roc_auc, f1, valor_liquido) | ✅ | `mlflow.log_metrics()` nos notebooks |
| Parâmetros (C, learning_rate, epochs) | ✅ | `mlflow.log_params()` nos notebooks |
| Artifacts (modelo PyTorch, pipeline) | ✅ | `mlflow.pytorch.log_model()` |
| System metrics (CPU, memória, disco) | ✅ | `mlflow.enable_system_metrics_logging()` |
| Tags (champion, deploy_timestamp) | ✅ | `export_model.py` marca o champion |

O champion é selecionado automaticamente entre **todos os modelos** (baselines + MLP) pela métrica `valor_liquido`, com desempate por `roc_auc`. O nome do MLP é versionado com timestamp (ex: `mlp_pytorch_20260502_2013`).

### 17.6 O que o CloudFormation provisiona

| Recurso | Descrição |
|---------|-----------|
| **VPC** | Rede isolada (10.0.0.0/16) com subnet pública |
| **EC2** | Instância com Docker, docker-compose e cfn-init |
| **Elastic IP** | IP fixo para a API |
| **Security Groups** | Portas 8000 (API), 5000 (MLflow), 3000 (Grafana), 9090 (Prometheus) |
| **ECR** | Repositórios de imagens Docker (API + treino) |
| **IAM Role** | Permissões para ECR pull, CloudWatch Logs, SSM |
| **VPC Flow Logs** | Logs de rede → CloudWatch (retenção 14 dias) |

Conformidade com governança:
- Tags obrigatórias (`se_org`, `se_environment`, `se_resource`, `se_pci_machine`, `se_conta_pci`)
- KMS encryption (EBS, ECR, CloudWatch Logs)
- IMDSv2 obrigatório
- VPC Flow Logs habilitados
- SSH desabilitado (acesso via SSM Session Manager)
- iptables hardening (INPUT/OUTPUT policy DROP, anti-exfiltração)
- fail2ban ativo

### 17.7 Validação pós-deploy

Aguarde ~3-5 minutos após o `CREATE_COMPLETE` para os containers inicializarem.

```bash
# Health check
curl http://<ELASTIC_IP>:8000/health

# Predição de teste (cliente com alto risco de churn)
curl -X POST http://<ELASTIC_IP>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 22, "gender": "male", "plan_type": "pre", "nps_score": 0, "support_calls_30d": 9, "tenure_months": 2, "default_flag": 1}'

# Simular drift para testar observabilidade
poetry run python scripts/simulate_drift.py --url http://<ELASTIC_IP>:8000 --n-requests 300

# Analisar drift (compara treino vs produção)
PYTHONPATH=src poetry run python scripts/check_drift.py \
  --reference data/raw/telecom_churn_base_extended.csv \
  --production /tmp/drift_simulation.jsonl
```

| Serviço | URL | Credencial |
|---------|-----|------------|
| API (Swagger) | `http://<ELASTIC_IP>:8000/docs` | — |
| MLflow UI | `http://<ELASTIC_IP>:5000` | — |
| Prometheus | `http://<ELASTIC_IP>:9090` | — |
| Grafana | `http://<ELASTIC_IP>:3000` | `mletg4` / `<GrafanaPassword>` |

### 17.8 Retreino do modelo

Para retreinar e redeployar com o novo modelo:

```bash
# Retreino completo (treina na EC2 + rebuild + redeploy)
./infra/deploy.sh

# Só retreinar (sem redeploy — útil para comparar modelos no MLflow)
./infra/deploy.sh --train-only
```

O MLflow UI mostra o histórico completo de treinos com:
- Comparação de métricas entre versões
- Tag `champion=true` no modelo em produção
- Timestamp de cada deploy
- Artifacts do modelo PyTorch

### 17.9 Destruição da stack

```bash
# Via script (limpa ECR + VPC Endpoints do GuardDuty + deleta stack)
./infra/deploy.sh --destroy

# Ou manualmente
aws cloudformation delete-stack --stack-name churn-prediction-stack --region sa-east-1
aws cloudformation wait stack-delete-complete --stack-name churn-prediction-stack --region sa-east-1
```

## 18. Proximos passos

1. Calibrar os parametros de negocio (`V_RETIDO`, `C_ACAO`) com time de CRM/financas.
2. Definir corte operacional por top-K para retencao.
3. Implementar retreinamento automático via trigger de drift (CT).
4. Adicionar autenticação JWT à API (conforme Cap. 5 do material).
5. ~~Deploy em cloud (Render, AWS, Azure) com autoscaling.~~ ✅ Implementado (seção 17).
6. Integrar SHAP/LIME para explainability do modelo.

## 19. Resultados dos Baselines e Interpretação

### 19.1 Desempenho dos modelos

| Modelo                        | Accuracy | F1    | ROC-AUC | PR-AUC | Valor Líquido    |
|-------------------------------|----------|-------|---------|--------|------------------|
| `dummy_stratified`            | 0.5294   | 0.3937| 0.5046  | 0.3959 | R$ 561.850       |
| `log_reg` (C=1)               | 0.8069   | 0.7425| 0.8783  | 0.8480 | **R$ 1.190.800** |
| `log_reg_best_penalty` (C=0.1)| 0.8069   | 0.7425| 0.8784  | 0.8480 | R$ 1.190.800     |
| `log_reg_mitigated_equalized_odds` | 0.8007 | 0.7352|   --   |   --   | R$ 1.180.950     |

**Interpretação:**
- O DummyClassifier serve como referência mínima, com métricas próximas ao acaso.
- A Regressão Logística supera amplamente o dummy, com ROC-AUC de 0.878, indicando excelente capacidade discriminativa.
- O valor líquido representa o ganho operacional ao aplicar a política de retencao baseada no modelo.
- O modelo mitigado por fairness mantém performance próxima, com pequena perda de F1 e valor líquido, o que é esperado.

### 19.2 Diagnóstico de Overfitting

| Modelo         | delta_roc_auc (treino - teste) | Diagnóstico                        |
|---------------|-------------------------------|------------------------------------|
| `log_reg`     | 0.0013                        | Sem overfitting - generaliza bem   |
| `dummy_stratified` | -0.0085                   | OK (teste ligeiramente melhor)     |

**Interpretação:**
- O delta_roc_auc próximo de zero mostra que o modelo não está memorizando o treino e generaliza bem para novos dados.

### 19.3 Comparação de Penalizações (L1, L2, ElasticNet)

| Penalização | Melhor C | ROC-AUC CV | ROC-AUC Teste |
|-------------|----------|------------|---------------|
| L2 (Ridge)  | 0.1      | 0.8782     | 0.8783        |
| L1 (Lasso)  | 0.1      | 0.8783     | 0.8783        |
| ElasticNet  | 0.1      | 0.8783     | **0.8784**    |

**Interpretação:**
- As três penalizações convergem para C=0.1, com diferença < 0.0002 em ROC-AUC.
- Isso indica que o dataset está bem condicionado e não há ganho relevante em usar penalizações mais complexas.
- Mantém-se L2 como referência pela simplicidade.

### 19.4 Fairness por Grupo Sensível

#### `log_reg` (sem mitigação)

| Atributo sensível | dp_diff | eo_diff |
|-------------------|---------|---------|
| `gender`          | 0.0552  | 0.0741  |
| `age_group`       | 0.1101  | 0.0758  |
| `region`          | 0.1565  | 0.1672  |
| `plan_type`       | 0.2960  | 0.1840  |

- O maior gap está em `plan_type`, indicando risco regulatório de concentrar retencao em clientes pré-pagos.

#### `log_reg_mitigated_equalized_odds` (mitigação por `gender`)

| Métrica           | `log_reg` | Mitigado | Variação |
|-------------------|-----------|----------|----------|
| dp_diff_gender    | 0.0552    | 0.0518   | -6%      |
| eo_diff_gender    | 0.0741    | 0.0847   | +14%     |
| F1                | 0.7425    | 0.7352   | -0.007   |
| Valor Líquido     | R$ 1.190.800 | R$ 1.180.950 | -R$ 9.850 |

- A mitigacao reduz o gap de fairness para gênero, com custo operacional pequeno.
- Os gaps de outros atributos permanecem sem tratamento.

### 19.5 Métrica de Negócio

```
valor_liquido = TP x R$500 - (TP + FP) x R$50
```

- `log_reg`: **R$ 1.190.800** abordando 3.494 clientes (2.731 TP, 763 FP)
- `dummy_stratified`: 2.254 FP para apenas 1.499 acertos -> R$ 561.850
- Mitigacao custa apenas R$ 9.850 a menos — custo aceitável para redução de risco regulatório

**Resumo:**
O pipeline baseline entrega valor de negócio robusto, generaliza bem, e já considera fairness. Os resultados são realistas e prontos para apresentação ou evolução para modelos mais complexos.

## 20. Resultados do MLP e Comparação Completa

### 20.1 Desempenho comparativo (todos os modelos)

| Modelo               | Accuracy | F1     | ROC-AUC | PR-AUC | Valor Líquido    |
|----------------------|----------|--------|---------|--------|------------------|
| `dummy_stratified`   | 0.5294   | 0.3937 | 0.5046  | 0.3959 | R$ 561.850       |
| `random_forest`      | 0.7950   | 0.7154 | 0.8609  | 0.8225 | R$ 1.103.750     |
| `log_reg`            | 0.8069   | 0.7425 | 0.8783  | 0.8480 | R$ 1.190.800     |
| `gradient_boosting`  | 0.8137   | 0.7474 | 0.8815  | 0.8541 | R$ 1.183.300     |
| **`mlp_pytorch`**    | 0.8067   | 0.7427 | 0.8772  | 0.8464 | **R$ 1.192.700** |

**Interpretação:**
- O MLP supera todos os modelos em **valor líquido** (+R$ 9.400 vs. GradientBoosting), com performance técnica comparável.
- ROC-AUC e F1 do MLP ficam apenas 0.004 abaixo do GradientBoosting — diferença não significativa considerando o ganho operacional.
- O MLP confirma a robustez do pipeline: mesmo sem tuning extenso, atinge resultado competitivo.

### 20.2 Feature Importance (RF e Gradient Boosting)

Top features presentes no Top-10 de **ambos** os modelos (7 features consenso):

| Feature                        | Importância RF | Importância GB |
|-------------------------------|----------------|----------------|
| `nps_detractor_flag`          | 0.0976         | 0.1951         |
| `late_payments_6m`            | 0.0595         | 0.2152         |
| `cat__nps_category_detractor` | 0.0801         | 0.0901         |
| `invoice_shock_flag`          | 0.0536         | 0.0872         |
| `plan_price`                  | 0.0356         | 0.1033         |
| `price_increase_last_3m`      | 0.0518         | 0.0258         |
| `nps_promoter_flag`           | 0.0426         | 0.0331         |

**Interpretação:**
- Satisfação (`nps_detractor_flag`) e inadimplência (`late_payments_6m`) são os sinais mais fortes de churn.
- Choques financeiros (`invoice_shock_flag`, `price_increase_last_3m`) e preço do plano também influenciam significativamente.
- Essas features devem ser priorizadas em regras de negócio e campanhas de retenção.

## 21. Contato

Grupo 4 - Tech Challenge FIAP
