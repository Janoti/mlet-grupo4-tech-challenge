
# Predicao de Churn em Telecom - Tech Challenge (Grupo 4)

Projeto de Machine Learning para prever churn em telecom, com foco em EDA, baselines e organizacao de pipeline para evolucao do modelo.

## 1. Objetivo

- Estimar a probabilidade de churn por cliente.
- Priorizar campanhas de retencao com base em risco.
- Medir impacto com metricas tecnicas (AUC-ROC, PR-AUC, top-K) e de negocio (churn evitado, ROI).

## 2. Dataset principal

A base de referencia do projeto e:

- `data/raw/telecom_churn_base_extended.csv`

Nesta etapa do projeto, a base extended nao e perfeitamente limpa de proposito. O gerador atual injeta problemas controlados de qualidade para simular um cenario mais realista de Fase 1, incluindo:

- linhas duplicadas
- `customer_id` duplicado
- valores ausentes adicionais
- valores invalidos e inconsistencias simples

Ela contem dados sinteticos com variaveis de:

- Perfil e contrato:
	- `plan_type`: tipo de plano do cliente (pre, controle, pos, empresarial).
	- `months_to_contract_end`: meses restantes para termino do contrato atual.
	- `has_loyalty`: indica se o cliente esta em periodo de fidelizacao (0/1).
- Qualidade de servico:
	- `network_outages_30d`: quantidade de quedas de rede nos ultimos 30 dias.
	- `avg_signal_quality`: qualidade media de sinal percebida pelo cliente.
	- `call_drop_rate`: taxa de chamadas interrompidas.
- Uso e engajamento:
	- `minutes_monthly`: minutos de voz consumidos no mes.
	- `data_gb_monthly`: consumo mensal de dados moveis em GB.
	- `usage_delta_pct`: variacao percentual recente no uso (tendencia de queda/subida).
	- `app_login_30d`: numero de logins no app nos ultimos 30 dias.
- Financeiro:
	- `monthly_charges`: valor cobrado mensalmente do cliente.
	- `invoice_shock_flag`: flag de aumento relevante de fatura (0/1).
	- `late_payments_6m`: quantidade de pagamentos em atraso nos ultimos 6 meses.
	- `default_flag`: indica historico de inadimplencia (0/1).
- Atendimento:
	- `support_calls_30d`: numero de contatos com suporte nos ultimos 30 dias.
	- `complaints_30d`: indicador de reclamacao recente (0/1).
	- `resolution_time_avg`: tempo medio de resolucao dos atendimentos.
- Satisfacao:
	- `nps_score`: nota NPS individual do cliente (0 a 10).
	- `nps_category`: classificacao NPS (promoter, passive, detractor).
	- `csat_score`: nota de satisfacao (CSAT) do cliente.

Colunas-chave de saida:

- `churn` (target binario)
- `churn_probability` (probabilidade sintetica de referencia)

Importante:

- `churn` e a variavel alvo usada nos baselines.
- `churn_probability` existe apenas como artefato da geracao sintetica e deve ser excluida das features por risco de leakage.

## 3. Estrutura do repositorio

```text
mlet-grupo4-tech-challenge/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
│   ├── ml_canvas.md
│   └── model_card.md
├── models/
│   ├── artifacts/
│   └── model_registry/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_baselines.ipynb
├── scripts/
│   ├── generate_dataset.py
│   └── generate_synthetic.py
├── src/churn_prediction/
├── tests/
├── pyproject.toml
└── README.md
```

## 4. Ambiente e instalacao

Pre-requisitos:

- Python 3.11+ (recomendado 3.12)
- Poetry

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

## 4.1 Atalhos com Makefile

Comandos para executar notebooks e iniciar MLflow com logs de progresso no terminal:

```bash
make help
make run-all
make notebooks
make notebooks-eda
make notebooks-baselines
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
3. Executa `notebooks/02_baselines.ipynb`.
4. Le os runs em `mlruns/` e imprime analise resumida (ganho do `log_reg` vs `dummy` e trade-off da mitigacao).
5. Sobe MLflow em background e mostra o link.

Saidas esperadas no terminal:

- `make notebooks`:
	- `[notebooks-eda] Iniciando execucao de notebooks/01_eda.ipynb...`
	- `[notebooks-eda] Concluido.`
	- `[notebooks-baselines] Iniciando execucao de notebooks/02_baselines.ipynb...`
	- `[notebooks-baselines] Concluido.`
	- `[notebooks] Execucao completa finalizada.`
- `make analyze`:
	- `[analysis] Resumo automatico da run`
	- metricas de `dummy_stratified`, `log_reg` e `log_reg_mitigated_equalized_odds`
	- deltas de performance e leitura sugerida para apresentacao
- `make mlflow`:
	- `[mlflow] Subindo MLflow UI em http://127.0.0.1:5000`
- `make mlflow-up`:
	- `[mlflow] PID: ...`
	- `[mlflow] Link: http://127.0.0.1:5000`

Observacao:

- A execucao dos notebooks usa `nbconvert --execute --inplace`, entao as celulas ficam com outputs salvos no proprio arquivo `.ipynb`.
- Para reduzir warnings de IOPub no `nbconvert`, o Makefile usa timeout maior e nivel de log configuravel (`IOPUB_TIMEOUT` e `NB_LOG_LEVEL`).

## 5. Geracao dos dados

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

## 6. Fluxo recomendado

1. Gerar/validar dataset em `data/raw`.
2. Executar exploracao e diagnostico em `notebooks/01_eda.ipynb`.
3. No proprio EDA da Fase 1, aplicar tratamento minimo para baseline:
	- remocao de duplicados
	- correcao de valores invalidos
	- imputacao simples
	- one-hot encoding das categoricas
4. Separar os dados em treino e teste com split estratificado `80/20`.
5. Treinar baselines com `DummyClassifier` e `LogisticRegression` no notebook `notebooks/02_baselines.ipynb`.
6. Registrar experimentos no MLflow (parametros, metricas e versao do dataset).
7. Avaliar fairness por subgrupos com Fairlearn (`gender` e `age_group`).
8. Aplicar mitigacao opcional com `EqualizedOdds` e comparar trade-offs (a versao atual do notebook usa configuracao otimizada para tempo de execucao).
9. Consolidar resultados em `docs/model_card.md`.

Tratamento adotado na Fase 1:

- Numericas: imputacao por mediana.
- Categoricas: imputacao por moda.
- Categoricas finais: transformacao com one-hot encoding.
- Exclusoes por leakage: `customer_id`, `churn_probability`, `retention_offer_made`, `retention_offer_accepted`.

Split adotado:

- Treino: `80%`
- Teste: `20%`
- Estratificacao pelo target `churn`

Fairness e rastreabilidade:

- Diagnostico inicial no EDA por subgrupos sensiveis.
- Avaliacao no baseline com Fairlearn (`demographic_parity_difference`, `equalized_odds_difference`).
- Mitigacao de referencia com `ExponentiatedGradient` + `EqualizedOdds`.
- Registro no MLflow com versao do dataset baseada em hash (`dataset_version`).

## 7. Qualidade de codigo

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
poetry run pytest -q
```

## 8. MLflow (opcional local)

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

## 9. Documentacao

- Canvas de negocio e modelagem: `docs/ml_canvas.md`
- Explicacao da EDA e das escolhas metodologicas: `docs/eda_metodologia.md`
- Card do modelo: `docs/model_card.md`

## 10. Proximos passos

1. Levar o mesmo tratamento da Fase 1 para pipeline reutilizavel em `src/churn_prediction/pipelines/`.
2. Fechar baseline tabular com comparacao consistente entre treino/teste.
3. Definir corte operacional por top-K para retencao.
4. Criar camada de inferencia (FastAPI) e testes de contrato.

