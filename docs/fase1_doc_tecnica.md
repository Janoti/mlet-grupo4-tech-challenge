# Fase 1 - Documentacao Tecnica (EDA + Baseline + Automacao)

## 1. Objetivo desta documentacao

Este documento consolida tecnicamente a Fase 1 do projeto, com foco em:

- como o baseline foi construido e por que esse desenho foi escolhido
- como o EDA foi conduzido e qual seu papel na qualidade do pipeline
- qual a relacao entre EDA e baseline no fluxo de trabalho
- o que era esperado na etapa, o que foi entregue e como foi executado
- como o Makefile funciona como facilitador operacional

## 2. Baseline (separado)

### 2.1 O que foi construido

O baseline foi implementado em `notebooks/02_baselines.ipynb` com os seguintes blocos:

1. Carregamento da base principal (`telecom_churn_base_extended.csv`).
2. Tratamento minimo de dados (duplicidade e padronizacao de campos-chave).
3. Split estratificado treino/teste em 80/20.
4. Pipeline de preprocessamento:
   - numericas com imputacao por mediana
   - categoricas com imputacao por mais frequente
   - one-hot encoding
5. Treino de dois modelos baseline:
   - `DummyClassifier(strategy="stratified")`
   - `LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)`
6. Registro de parametros e metricas no MLflow.
7. Avaliacao de fairness com Fairlearn (`gender` e `age_group`).
8. Mitigacao com `ExponentiatedGradient` + restricao `EqualizedOdds`.

### 2.2 Por que essa arquitetura foi escolhida

- **DummyClassifier**: cria uma referencia minima de desempenho para validar ganho real do modelo.
- **Regressao Logistica**: modelo tabular robusto, rapido, interpretavel e adequado para baseline.
- **Split 80/20 estratificado**: preserva distribuicao do target e melhora confiabilidade da avaliacao.
- **Preprocessamento simples e explicito**: reduz risco de erro por missing/categorias e facilita reproducao.
- **MLflow**: rastreabilidade de execucao (parametros, metricas, versao da base).
- **Fairlearn**: complementa performance global com verificacao de disparidade entre grupos.
- **Mitigacao**: permite quantificar trade-off entre desempenho e equidade em vez de assumir uma decisao ad hoc.

### 2.3 Resultado tecnico esperado do baseline

- `log_reg` deve superar `dummy_stratified` em AUC-ROC, PR-AUC e F1.
- run mitigado deve ser comparado ao baseline logistico para analisar perda/ganho no trade-off performance x fairness.

## 3. EDA (separado)

### 3.1 O que foi feito

O EDA em `notebooks/01_eda.ipynb` foi desenhado para validar readiness da base antes da modelagem:

1. Diagnostico estrutural do dataset.
2. Inspecao de duplicidade, missing, categorias e inconsistencias.
3. Verificacao de colunas com risco de leakage.
4. Leitura de sinais de churn e comportamento de subgrupos.
5. Tratamento minimo da base para habilitar baseline confiavel.

### 3.2 Por que o EDA foi necessario

- A base extended simula cenarios de dados imperfeitos por design.
- Treinar sem auditoria de qualidade e leakage introduz risco de conclusoes falsas.
- O EDA define o contrato minimo de dados para o baseline da Fase 1.

### 3.3 Entrega tecnica do EDA

- Diagnostico de qualidade com justificativa de tratamento.
- Definicao explicita das exclusoes por leakage.
- Base pronta para a etapa de baseline com split e preprocessamento consistentes.

## 4. Relacao entre EDA e Baseline

EDA e baseline nao sao etapas isoladas; sao partes do mesmo pipeline de confiabilidade:

1. **EDA** identifica problemas reais de dados e define regras de saneamento.
2. **Baseline** usa essas regras para treinar e avaliar modelos de forma reproduzivel.
3. **MLflow** registra a execucao da modelagem para auditoria e comparabilidade.
4. **Fairness** valida que desempenho global nao mascara disparidades por subgrupo.

Em resumo: o EDA reduz risco de erro metodologico; o baseline transforma esse diagnostico em evidencia mensuravel.

## 5. Etapa 1 do projeto: o que era esperado vs o que foi feito

### 5.1 O que era esperado na Fase 1

- Entender e preparar dados para modelagem inicial.
- Construir baseline comparavel e rastreavel.
- Medir desempenho com metricas adequadas ao problema.
- Incluir avaliacao de fairness e uma mitigacao de referencia.
- Documentar processo e resultados para decisao tecnica.

### 5.2 O que foi feito

- EDA completo com foco em qualidade e leakage.
- Baseline implementado com Dummy + Regressao Logistica.
- Split estratificado 80/20 e pipeline de preprocessamento consistente.
- Registro de experimentos no MLflow com `dataset_version`.
- Avaliacao de fairness em `gender` e `age_group`.
- Mitigacao com `EqualizedOdds` e comparacao de trade-off.
- Documentacao no README, ML Canvas e Model Card.

### 5.3 Como foi executado

- Codigo principal: `notebooks/01_eda.ipynb` e `notebooks/02_baselines.ipynb`.
- Reprodutibilidade operacional via comandos de Makefile.
- Analise automatica de runs em `scripts/analyze_mlruns.py`.

## 6. Facilitador Make (automacao operacional)

### 6.1 Problema operacional resolvido

Sem automacao, a execucao da Fase 1 depende de passos manuais repetitivos (limpeza de runs, execucao de notebooks, leitura de resultados, subida do MLflow), com maior risco de erro humano.

### 6.2 Solucao implementada

Foi criado um `Makefile` com targets para padronizar a operacao:

- `make install`
- `make mlflow-clean`
- `make notebooks`
- `make analyze`
- `make mlflow-up`
- `make mlflow-down`
- `make run-all` (execucao unica fim a fim)

### 6.3 O que o `make run-all` faz

1. Limpa `mlruns/`.
2. Executa `01_eda.ipynb`.
3. Executa `02_baselines.ipynb`.
4. Analisa automaticamente os runs e imprime resumo tecnico.
5. Sobe MLflow em background e mostra o link.

### 6.4 Beneficios do facilitador

- Reprodutibilidade operacional.
- Menos variacao entre execucoes de equipe.
- Logs padronizados para troubleshooting.
- Menor tempo para demonstracao e validacao.

## 7. Limites da Fase 1 e proxima etapa

### 7.1 Limites atuais

- Base sintetica (nao-producao).
- Mitigacao de fairness em configuracao inicial (trade-off deve ser reavaliado periodicamente).
- Pipeline ainda centrado em notebook (apesar de automatizado por Make).

### 7.2 Evolucoes implementadas apos a Fase 1

As seguintes evolucoes foram construidas sobre a base da Fase 1:

1. **Pipeline reutilizavel** (`src/churn_prediction/`): logica dos notebooks refatorada em modulos Python testaveis — `config.py`, `data_cleaning.py`, `preprocessing.py`, `evaluation.py`, `model.py`, `pipelines/`.
2. **API FastAPI** (`src/churn_prediction/api/`): endpoints `/predict` e `/health` com schemas Pydantic, logging estruturado e faixas de risco (alto/medio/baixo).
3. **Containerizacao** (`Dockerfile`, `docker-compose.yml`): API e MLflow como servicos Docker com health check.
4. **Testes automatizados** (`tests/`): 30 testes cobrindo imports, validacao de schemas, data cleaning e endpoints da API.
5. **CI/CD** (`.github/workflows/ci_ml_pipeline.yml`): pipeline GitHub Actions com lint, testes, treinamento e build Docker.
6. **Monitoramento de drift** (`src/churn_prediction/monitoring.py`): deteccao via KS test, Chi-Quadrado e PSI, com scripts de simulacao (`scripts/simulate_drift.py`) e analise (`scripts/check_drift.py`).

Para instrucoes de uso detalhadas, consulte o `README.md` (secoes 12 a 14).
