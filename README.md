# Rede Neural (MLP) para Previsão de Churn em Telecom (Tech Challenge)

Projeto do **Grupo 4** para construir um pipeline profissional **end-to-end** de Machine Learning para **previsão de churn** (cancelamento de clientes) em uma operadora de telecomunicações.

Nesta **Fase 1**, o foco é: **entendimento do problema, EDA, baselines com Scikit-Learn e tracking no MLflow**, já com estrutura de repositório e boas práticas.

---

## 1) Contexto (Negócio)
Uma operadora está perdendo clientes em ritmo acelerado. O objetivo é criar um modelo preditivo que identifique clientes com maior risco de churn para apoiar ações de retenção.

### Métricas
- **AUC-ROC**: mede a capacidade do modelo de separar churners de não-churners em todos os limiares de decisão possíveis. É robusta ao desbalanceamento de classes (churn ~18% da base) e não depende de um threshold fixo — ideal para comparar modelos candidatos.
- **PR-AUC** (Precision-Recall AUC): complementa a AUC-ROC em cenários desbalanceados. Foca no desempenho sobre a classe positiva (churn), penalizando mais fortemente falsos positivos e falsos negativos dentro do grupo minoritário. Quando o custo de errar sobre quem vai churnar é alto, PR-AUC é mais informativa que AUC-ROC.
- **F1 (classe churn)**: média harmônica entre Precision e Recall para a classe churn = 1. Útil quando precisamos de um único número que balanceie o custo de perder churners reais (baixo Recall) com o custo de acionar desnecessariamente clientes que não iriam cancelar (baixa Precision).
- **Negócio**: churn evitado (receita preservada) vs. custo da campanha de retenção — traduz o desempenho técnico em impacto financeiro real.

---

## 2) Stack / Tecnologias
- **Python 3.12** + **Poetry** (gerência de dependências via `pyproject.toml`)
- **Scikit-Learn** (pipelines, preprocessing, baselines)
- **PyTorch** (MLP — etapas seguintes)
- **MLflow** (tracking de experimentos: parâmetros, métricas, artefatos)
- **FastAPI** (API de inferência — etapas seguintes)
- **Ruff** (lint) e **Pytest** (testes)

---

## 3) Como rodar (Setup)
### Pré-requisitos
- Python **3.12**
- Poetry instalado

### Instalação
```bash
git clone https://github.com/Janoti/mlet-grupo4-tech-challenge.git
cd mlet-grupo4-tech-challenge
poetry install
```

### Gerar o dataset
O script `scripts/generate_dataset.py` gera uma base sintética de 50.000 clientes com label de churn (~18% de taxa), salva em `data/raw/telecom_churn_base.csv`.

```bash
poetry run python scripts/generate_dataset.py
```

Parâmetros opcionais:
```bash
poetry run python scripts/generate_dataset.py --n-rows 100000 --seed 0 --out-dir data/raw
```

O label `churn` é calculado com base em fatores de risco alinhados ao ML Canvas (contrato mensal, tempo de casa, queda de uso, inadimplência, reclamações e chamadas ao suporte).

### Comandos úteis
Rodar lint:
```bash
poetry run ruff check .
```

Rodar testes:
```bash
poetry run pytest -q
```

Subir MLflow local:
```bash
mlflow ui --backend-store-uri ./mlruns
```
Acesse: http://127.0.0.1:5000

> Observação: a pasta `mlruns/` geralmente deve ficar no `.gitignore` (saída local do MLflow).

---

## 4) Estrutura do Repositório (o que cada pasta faz)

```text
mlet-grupo4-tech-challenge/
├── src/
├── data/
├── models/
├── notebooks/
├── tests/
├── docs/
├── pyproject.toml
└── README.md
```

### `src/` — código fonte (produção)
Onde fica o código Python **reutilizável** do projeto (pipelines, treino, utilitários, etc.).  
Exemplo de organização (sugerida):
- `src/churn_prediction/` (pacote principal)
  - `pipelines/` (pipelines de preprocessing e treinamento)
  - `seed.py` (funções para fixar seeds / reprodutibilidade)
  - `logging.py` (logging estruturado)
  - `config.py` (configurações do projeto)


### `data/` — dados (não versionar dados pesados/sensíveis)
Estrutura recomendada:
- `data/raw/` dados brutos (originais)
- `data/interim/` dados intermediários
- `data/processed/` dados finais prontos para treino


### `models/` — artefatos de modelo
- `models/artifacts/` outputs do treino (modelos exportados, encoders, etc.)
- `models/model_registry/` área para organização/versões (se aplicável)

### `notebooks/` — exploração e EDA
Notebooks para:
- EDA
- análises iniciais
- validação de hipóteses

Importante: ao final, o que virar “regra” deve ser migrado para `src/`.

### `tests/` — testes automatizados
Mínimo exigido (>= 3 testes):
- smoke test (import/execução básica)
- schema test (validação de colunas/tipos/target)
- API test (quando FastAPI existir — pode ficar como placeholder inicial)

### `docs/` — documentação
- `ml_canvas.md` (stakeholders, objetivo, métrica de negócio, SLOs)
- `model_card.md` (limitações, vieses, dados, métricas, riscos)

---

## 5) Checklist — Fase 1 (arcabouço + início de baselines)

### Repositório / Engenharia
- [ ] Estrutura criada: `src/`, `data/`, `models/`, `tests/`, `notebooks/`, `docs/`
- [ ] `README.md` com setup + explicação da estrutura + como rodar
- [ ] `pyproject.toml` como single source of truth (deps + ruff + pytest)
- [ ] `.gitignore` adequado (ignorar `mlruns/`, `.venv/`, `data/`, `models/` se necessário)
- [ ] Logging estruturado (evitar `print()` no código de `src/`)
- [ ] Seeds fixados para reprodutibilidade (definir padrão do projeto)

### ML Canvas + Documentação
- [ ] `docs/ml_canvas.md` preenchido (stakeholders, métricas, SLOs, riscos)
- [ ] `docs/model_card.md` iniciado (mesmo que parcial nesta fase)

### EDA + Baselines
- [ ] Notebook de EDA (`notebooks/01_eda.ipynb`) com:
  - [ ] volume, missing, distribuição do target
  - [ ] data readiness (qualidade, vazamento, features suspeitas)
- [ ] Baselines Scikit-Learn:
  - [ ] DummyClassifier
  - [ ] LogisticRegression
- [ ] Validação cruzada estratificada (`StratifiedKFold`)
- [ ] Métrica técnica definida (AUC-ROC / PR-AUC / F1)

### MLflow
- [ ] MLflow rodando local (`mlflow ui`)
- [ ] Experimentos registrados com:
  - [ ] parâmetros
  - [ ] métricas
  - [ ] artefatos (ex.: pipeline salvo, gráficos, tabela)

### Testes (mínimo exigido)
- [ ] `tests/test_smoke.py`
- [ ] `tests/test_schema.py`
- [ ] `tests/test_api.py` (pode ficar como placeholder até a API existir)

---

## 6) Dataset de base no repositório

Este repositório inclui um dataset de base para bootstrap do projeto.

### Onde está o CSV?
Após a geração, o arquivo fica em:

- `data/raw/telecom_churn_base.csv`

Este é o dataset que o grupo deve usar para:
- validar o arcabouço (estrutura do repo),
- rodar EDA inicial,
- treinar baselines,
- registrar experimentos no MLflow.

### Como o dataset é gerado?
O dataset é criado pelo script:

- `scripts/generate_dataset.py`

Ele gera **50.000 linhas** (por padrão no nosso uso atual) com um conjunto de colunas típicas de churn em telecom, incluindo:
- identificador do cliente (`customer_id`)
- variáveis numéricas (ex.: `tenure_months`, `monthly_charges`, `total_charges`, `support_calls_30d`)
- variáveis categóricas (ex.: `contract`, `payment_method`, `internet_service`)
- variável alvo (`churn`, 0/1)

A geração usa um `seed` fixo, então o resultado é **reprodutível** (mesmo arquivo para todos, quando gerado com os mesmos parâmetros).

### Como (re)gerar localmente
A partir da raiz do repositório:

```bash
poetry install
poetry run python scripts/generate_dataset.py

## 7) Próximos passos (imediatos)
1. Preencher `docs/ml_canvas.md` (bem direto, 1 página)
2. Criar notebook de EDA e consolidar “achados” (quality checks)
3. Implementar baselines com CV estratificada + MLflow tracking

