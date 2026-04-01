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
  - Exclusão de colunas com risco de leakage (`customer_id`, `churn_probability`, `retention_offer_made`, `retention_offer_accepted`, `contract_renewal_date`, `loyalty_end_date`)
- Split: treino/teste estratificado 80/20; validação interna 10% do treino (para early stopping do MLP)
- Versão do dataset: hash SHA-256 registrado como `dataset_version` no MLflow

## 4. Modelos e pipeline

- Baselines lineares: `DummyClassifier`, `LogisticRegression` (com tuning e fairness)
- Baselines de árvore: `RandomForestClassifier` (200 estimadores), `GradientBoostingClassifier` (100 estimadores)
- MLP em PyTorch: duas camadas ocultas (128→64), BatchNorm, Dropout=0.3, BCEWithLogitsLoss, early stopping, seeds fixadas
- Experimentos registrados no MLflow (`churn-baselines`, `churn-mlp-pytorch`)

Fluxo reprodutível:

- `make run-all`: executa EDA, baselines (com export dos splits), MLP e análise automática
- `make analyze`: consolida métricas técnicas e de negócio dos experimentos

## 5. Métricas de performance

| Modelo | Accuracy | F1 | ROC-AUC | PR-AUC | Valor Líquido |
|---|---:|---:|---:|---:|---:|
| Dummy (stratified) | 0.53 | 0.39 | 0.50 | 0.40 | R$ 561.850 |
| Logistic Regression (C=0.1) | 0.81 | 0.74 | 0.88 | 0.85 | R$ 1.190.800 |
| MLP PyTorch | 0.82 | 0.76 | 0.90 | 0.87 | — |

> Valores do MLP são estimativas até a execução completa do pipeline. Consulte o MLflow para os valores atualizados.

## 6. Métricas de negócio

- Fórmula: `valor_liquido = TP × R$500 − (TP + FP) × R$50`
- Threshold otimizado por varredura de 0.10 a 0.90 (máximo valor líquido)
- Métricas registradas no MLflow: `tp`, `fp`, `clientes_abordados`, `valor_bruto`, `custo_total_acao`, `valor_liquido`, `valor_por_cliente`

## 7. Fairness

### 7.1 Atributos sensíveis avaliados

- `gender`
- `age_group` (derivada de `age`)
- `region`, `plan_type` (avaliação adicional no baseline)

### 7.2 Métricas de fairness

- `demographic_parity_difference`
- `equalized_odds_difference`
- Gaps por grupo em `selection_rate`, `TPR` e `FPR`

### 7.3 Mitigação implementada

- Método: `ExponentiatedGradient`
- Restrição: `EqualizedOdds`
- Grupo sensível de referência: `gender`
- Configuração: `mitigation_sample_size=15000`, `eps=0.02`, `max_iter=15`

### 7.4 Resultado de fairness (gender — baseline logístico)

| Métrica | Sem mitigação | Com mitigação | Variação |
|---------|:---:|:---:|:---:|
| dp_diff_gender | 0.0552 | 0.0518 | −6% |
| eo_diff_gender | 0.0741 | 0.0847 | +14% |
| F1 | 0.7425 | 0.7352 | −0.007 |
| Valor Líquido | R$ 1.190.800 | R$ 1.180.950 | −R$ 9.850 |

Nota: na configuração rápida atual, a mitigação não melhorou o gap `eo_diff`. Pode ser retreinada com mais iterações para nova comparação.

## 8. Limitações conhecidas

- Base sintética (não representa integralmente o comportamento de produção)
- Fairness avaliada apenas nos atributos disponíveis (`gender`, `age`) — pode não cobrir todos os fatores sensíveis relevantes
- A mitigação pode reduzir gap de fairness com perda de performance; a decisão final depende do trade-off acordado com o negócio
- MLP sem avaliação de fairness formal (apenas métricas técnicas e de negócio nesta versão)

## 9. Decisão de uso

- Modelo candidato a piloto: `LogisticRegression` baseline (melhor equilíbrio entre ROC-AUC/PR-AUC e simplicidade operacional)
- Modelo em avaliação: MLP PyTorch (maior capacidade, requer validação de fairness e calibração)
- Critério mínimo de fairness: definir com negócio e compliance antes de produção
- Data da decisão: TBD

## 10. Próximos passos

1. Calibrar `V_RETIDO` e `C_ACAO` com negócio para decisão operacional realista.
2. Consolidar threshold operacional da campanha (varredura 0.10–0.90 já implementada).
3. Avaliar fairness do MLP por subgrupos sensíveis.
4. Definir política de monitoramento de performance e fairness em produção.
5. Evoluir arquitetura (tuning, explainability com SHAP/LIME).
