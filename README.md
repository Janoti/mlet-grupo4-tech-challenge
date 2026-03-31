# Predição de Churn em Telecom - Tech Challenge (Grupo 4)

Projeto de Machine Learning para prever churn em telecom, com pipeline robusto, automação, rastreabilidade e foco em métricas de negócio.

## Status atual

- EDA executável e documentada
- Baselines (`DummyClassifier`, `LogisticRegression`) e fairness com Fairlearn
- Pipeline MLP em PyTorch com rastreabilidade no MLflow
- Métricas de negócio integradas (clientes abordados, valor líquido, ROI)
- Automação fim a fim via Makefile (`run-all`, `analyze`, `mlflow-up/down`)
- Documentação técnica e de negócio atualizada

## 1. Objetivo

- Estimar a probabilidade de churn por cliente
- Priorizar campanhas de retenção com base em risco e valor
- Medir impacto com métricas técnicas (AUC-ROC, PR-AUC, F1) e de negócio (churn evitado, valor líquido, ROI)

## 2. Pipeline e notebooks

- `01_eda.ipynb`: Exploração e análise dos dados
- `02_baselines.ipynb`: Baselines, fairness, métricas de negócio, MLflow; exporta splits preprocessados para o MLP
- `03_mlp_pytorch.ipynb`: MLP em PyTorch com BatchNorm, Dropout, early stopping, batching, análise de custo por threshold e MLflow

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
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
│   ├── ml_canvas.md
│   └── 02_baselines.ipynb
├── scripts/
│   ├── analyze_mlruns.py
│   ├── generate_dataset.py
│   ├── generate_synthetic.py
│   └── logging_utils.py
├── src/churn_prediction/
├── tests/
├── pyproject.toml
└── README.md
```

## 5. Automação e rastreabilidade

- **Makefile**: targets para rodar EDA, baselines, MLP, análise e MLflow
- **MLflow**: rastreia experimentos, parâmetros, métricas técnicas e de negócio

## 6. Ambiente e instalacao

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

## 6.1 Atalhos com Makefile

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

## 7. Geracao dos dados

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

## 8. Fluxo recomendado

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

## 9. Qualidade de codigo

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

## 10. MLflow (opcional local)

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

## 11. Documentacao

- Canvas de negocio e modelagem: `docs/ml_canvas.md`
- Explicacao da EDA e das escolhas metodologicas: `docs/eda_metodologia.md`
- Documento tecnico consolidado da Fase 1 (EDA + baseline + automacao): `docs/fase1_doc_tecnica.md`
- Card do modelo: `docs/model_card.md`

## 12. Proximos passos

1. Levar o mesmo tratamento da Fase 1 para pipeline reutilizavel em `src/churn_prediction/pipelines/`.
2. Fechar baseline tabular com comparacao consistente entre treino/teste.
3. Calibrar os parametros de negocio (`V_RETIDO`, `C_ACAO`) com time de CRM/financas.
4. Definir corte operacional por top-K para retencao.
5. Criar camada de inferencia (FastAPI) e testes de contrato.

## 13. Resultados dos Baselines e Interpretação

### 13.1 Desempenho dos modelos

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

### 13.2 Diagnóstico de Overfitting

| Modelo         | delta_roc_auc (treino - teste) | Diagnóstico                        |
|---------------|-------------------------------|------------------------------------|
| `log_reg`     | 0.0013                        | Sem overfitting - generaliza bem   |
| `dummy_stratified` | -0.0085                   | OK (teste ligeiramente melhor)     |

**Interpretação:**
- O delta_roc_auc próximo de zero mostra que o modelo não está memorizando o treino e generaliza bem para novos dados.

### 13.3 Comparação de Penalizações (L1, L2, ElasticNet)

| Penalização | Melhor C | ROC-AUC CV | ROC-AUC Teste |
|-------------|----------|------------|---------------|
| L2 (Ridge)  | 0.1      | 0.8782     | 0.8783        |
| L1 (Lasso)  | 0.1      | 0.8783     | 0.8783        |
| ElasticNet  | 0.1      | 0.8783     | **0.8784**    |

**Interpretação:**
- As três penalizações convergem para C=0.1, com diferença < 0.0002 em ROC-AUC.
- Isso indica que o dataset está bem condicionado e não há ganho relevante em usar penalizações mais complexas.
- Mantém-se L2 como referência pela simplicidade.

### 13.4 Fairness por Grupo Sensível

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

### 13.5 Métrica de Negócio

```
valor_liquido = TP x R$500 - (TP + FP) x R$50
```

- `log_reg`: **R$ 1.190.800** abordando 3.494 clientes (2.731 TP, 763 FP)
- `dummy_stratified`: 2.254 FP para apenas 1.499 acertos -> R$ 561.850
- Mitigacao custa apenas R$ 9.850 a menos — custo aceitável para redução de risco regulatório

**Resumo:**
O pipeline baseline entrega valor de negócio robusto, generaliza bem, e já considera fairness. Os resultados são realistas e prontos para apresentação ou evolução para modelos mais complexos.

## 14. Contato

Grupo 4 - Tech Challenge FIAP
