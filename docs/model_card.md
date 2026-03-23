# Model Card - Predicao de Churn em Telecom

## 1. Informacoes gerais

- Projeto: mlet-grupo4-tech-challenge
- Problema: classificacao binaria de churn (0/1)
- Dataset principal: `data/raw/telecom_churn_base_extended.csv`
- Etapa atual: baseline tabular com rastreabilidade em MLflow e avaliacao de fairness

## 2. Objetivo do modelo

Estimar o risco de churn por cliente para priorizar campanhas de retencao com melhor relacao entre impacto e custo operacional.

## 3. Dados e preparo

- Fonte: base sintetica estendida de telecom
- Tratamento minimo aplicado:
	- remocao de linhas duplicadas e IDs duplicados
	- imputacao de numericas por mediana
	- imputacao de categoricas por moda
	- one-hot encoding para categoricas
	- exclusao de colunas com risco de leakage (`customer_id`, `churn_probability`, `retention_offer_made`, `retention_offer_accepted`)
- Split: treino/teste estratificado 80/20
- Versao do dataset: hash SHA-256 registrado como `dataset_version` no MLflow

## 4. Modelos baseline

- `DummyClassifier(strategy="stratified")` (referencia minima)
- `LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)`

Observacao: os experimentos sao registrados no experimento `churn-baselines` no MLflow.

Fluxo reprodutivel atual:

- `make run-all`: limpa `mlruns/`, executa EDA + baselines, roda analise automatica e sobe MLflow.
- `make analyze`: consolida os ultimos runs esperados (`dummy_stratified`, `log_reg`, `log_reg_mitigated_equalized_odds`).

## 5. Metricas de performance (runs atuais)

| Modelo | Accuracy | F1 | ROC-AUC | PR-AUC | Positive Rate |
|---|---:|---:|---:|---:|---:|
| Dummy (stratified) | 0.5294 | 0.3937 | 0.5046 | 0.3959 | 0.3826 |
| Logistic Regression | 0.8049 | 0.7399 | 0.8786 | 0.8485 | 0.3564 |
| Logistic + mitigacao (EqualizedOdds) | 0.8056 | 0.7389 | N/A | N/A | 0.3510 |

## 6. Fairness

### 6.1 Atributos sensiveis avaliados

- `gender`
- `age_group` (derivada de `age`)

### 6.2 Metricas de fairness

- `demographic_parity_difference`
- `equalized_odds_difference`
- gaps por grupo em `selection_rate`, `TPR` e `FPR`

### 6.3 Mitigacao implementada

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
