
# Model Card - Predição de Churn em Telecom

## 1. Informações gerais

- Projeto: mlet-grupo4-tech-challenge
- Problema: classificação binária de churn (0/1)
- Dataset principal: `data/raw/telecom_churn_base_extended.csv`
- Etapa atual: pipeline robusto com MLP em PyTorch, automação e rastreabilidade em MLflow

## 2. Objetivo do modelo

Estimar o risco de churn por cliente para priorizar campanhas de retenção, maximizando impacto e otimizando custo operacional.

## 3. Dados e preparo

- Fonte: base sintética estendida de telecom
- Tratamento aplicado:
  - Remoção de duplicidades e IDs duplicados
  - Imputação de numéricas por mediana
  - Imputação de categóricas por moda
  - One-hot encoding para categóricas
  - Exclusão de colunas com risco de leakage (`customer_id`, `churn_probability`, `retention_offer_made`, `retention_offer_accepted`)
- Split: treino/teste estratificado 80/20
- Versão do dataset: hash SHA-256 registrado como `dataset_version` no MLflow

## 4. Modelos e pipeline

- Baselines: `DummyClassifier`, `LogisticRegression` (com fairness)
- MLP em PyTorch: pipeline robusto, modular, com automação e rastreabilidade
- Experimentos registrados no MLflow (`churn-baselines`, `churn-mlp-pytorch`)

Fluxo reprodutível:

- `make run-all`: executa EDA, baselines, MLP e análise automática
- `make analyze`: consolida métricas técnicas e de negócio dos experimentos

## 5. Métricas de performance (exemplo)

| Modelo | Accuracy | F1 | ROC-AUC | PR-AUC | Positive Rate |
|---|---:|---:|---:|---:|---:|
| Dummy (stratified) | 0.53 | 0.39 | 0.50 | 0.39 | 0.38 |
| Logistic Regression | 0.80 | 0.74 | 0.88 | 0.85 | 0.36 |
| MLP PyTorch | 0.82 | 0.76 | 0.90 | 0.87 | 0.35 |

## 6. Métricas de negócio

- Clientes abordados, valor bruto, valor líquido, valor por cliente, custo total da ação
- Métricas calculadas e rastreadas no MLflow para todos os experimentos

## 7. Fairness

### 7.1 Atributos sensíveis avaliados

- `gender`
- `age_group` (derivada de `age`)

### 7.2 Métricas de fairness

- `demographic_parity_difference`
- `equalized_odds_difference`
- Gaps por grupo em `selection_rate`, `TPR` e `FPR`

### 7.3 Mitigação implementada

- Equalized Odds (Fairlearn)

## 8. Limitações e próximos passos

- Evoluir arquitetura do MLP (tuning, regularização, explainability)
- Monitoramento contínuo e atualização do pipeline


- Metodo: `ExponentiatedGradient`
- Restricao: `EqualizedOdds`
- Grupo sensivel de referencia na mitigacao: `gender`
- Configuracao atual para desempenho computacional:
	- `mitigation_sample_size = 15000`
	- `mitigation_eps = 0.02`
	- `mitigation_max_iter = 15`

### 6.4 Resultado de fairness (gender)

- Baseline logistico:
	- `dp_diff = 0.0448`
	- `eo_diff = 0.0663`
- Modelo mitigado:
	- `dp_diff_gender = 0.0561`
	- `eo_diff_gender = 0.0739`

Nota: na configuracao rapida atual, a mitigacao nao melhorou os gaps de fairness neste run. Ela foi mantida como referencia tecnica de pipeline e pode ser retreinada com mais iteracoes/amostra para nova comparacao.

## 7. Metricas de negocio (status)

- A metrica de negocio ja e logada automaticamente no MLflow nesta branch.
- Formula operacionalizada no notebook baseline:
	- `valor_liquido = TP * V_RETIDO - (TP + FP) * C_ACAO`
	- `valor_por_cliente = valor_liquido / N`
- Metricas atualmente registradas no MLflow:
	- `tp`, `fp`, `clientes_abordados`, `valor_bruto`, `custo_total_acao`, `valor_liquido`, `valor_por_cliente`

## 8. Limitacoes conhecidas

- Base sintetica (nao representa integralmente o comportamento de producao).
- Fairness foi avaliada em atributos disponiveis (`gender`, `age`) e pode nao cobrir todos os fatores sensiveis relevantes de negocio.
- A mitigacao pode reduzir gap de fairness com perda de performance; a decisao final depende do trade-off acordado com negocio.

## 9. Decisao de uso (a preencher)

- Modelo candidato a piloto: `LogisticRegression` baseline
- Criterio principal de selecao: melhor equilibrio entre ROC-AUC/PR-AUC e simplicidade operacional
- Criterio de fairness minimo aceito: definir com negocio e compliance antes de producao
- Data da decisao: TBD

## 10. Proximos passos

1. Calibrar `V_RETIDO` e `C_ACAO` com negocio para decisao operacional realista.
2. Consolidar threshold operacional de campanha.
3. Definir politica de monitoramento de performance e fairness em producao.
