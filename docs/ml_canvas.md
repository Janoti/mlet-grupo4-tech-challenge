# ML Canvas — Predição de Churn em Telecom

> Padrão: ML Canvas de Louis Dorard (9 blocos)
> Projeto: mlet-grupo4-tech-challenge | Grupo 4 | Atualizado: Abril/2026

---

## Bloco 1 — Value Proposition (Proposta de Valor)

**Problema de negócio:**
Uma operadora de telecomunicações enfrenta alta taxa de cancelamento (churn). A diretoria precisa identificar antecipadamente quais clientes têm maior risco de cancelamento para acionar campanhas de retenção proativas, maximizando receita e eficiência operacional.

**Proposta de valor do modelo:**
Entregar um score de risco de churn por cliente (0–1) que permita ao time de CRM priorizar abordagens, reduzir desperdício de contato em clientes de baixo risco e concentrar ofertas nos clientes com maior probabilidade de cancelar.

**Impacto esperado:**
- Reduzir churn da base ativa via intervenção precoce
- Aumentar ROI das campanhas de retenção (modelo vs. abordagem em massa)
- Fórmula de valor: `valor_liquido = TP × R$500 − (TP + FP) × R$50`
- Referência LogReg: **R$ 1.190.800** por ciclo de campanha vs. R$ 561.850 do Dummy

**Decisão suportada:**
Quem contatar, quando e qual oferta priorizar — decisão final do time de CRM.

---

## Bloco 2 — Data Sources (Fontes de Dados)

**Fonte principal:**
`data/raw/telecom_churn_base_extended.csv` — base sintética de 50.000 clientes de telecom

**Características da base:**
- Gerada por `scripts/generate_synthetic.py` (seed=42, reprodutível)
- Problemas de qualidade injetados intencionalmente: duplicidades de linhas e IDs, missing values, valores fora de faixa, categorias inválidas — simula ambiente real
- Versão rastreada por hash SHA-256 registrado no MLflow como `dataset_version`

**Cobertura temporal:**
Base transversal (snapshot) — sem dimensão temporal explícita

**Fontes externas potenciais (não implementadas):**
- Dados de cobertura de sinal por região
- Ofertas de concorrentes
- Histórico de campanhas anteriores

---

## Bloco 3 — Features (Variáveis de Entrada)

Após limpeza e exclusão de leakage, o pipeline usa os seguintes grupos de variáveis:

| Grupo | Exemplos |
|---|---|
| Perfil e contrato | `age`, `gender`, `region`, `plan_type`, `plan_price`, `months_to_contract_end`, `has_loyalty` |
| Qualidade de serviço | `network_outages_30d`, `avg_signal_quality`, `call_drop_rate`, `service_failures_30d` |
| Uso e engajamento | `minutes_monthly`, `data_gb_monthly`, `usage_delta_pct`, `days_since_last_usage`, `app_login_30d` |
| Financeiro | `monthly_charges`, `avg_bill_last_6m`, `invoice_shock_flag`, `late_payments_6m`, `default_flag` |
| Atendimento | `support_calls_30d`, `support_calls_90d`, `complaints_30d`, `resolution_time_avg` |
| Satisfação | `nps_score`, `nps_category`, `nps_promoter_flag`, `nps_detractor_flag`, `csat_score` |
| Concorrência | `portability_request_flag`, `competitor_offer_contact_flag` |

**Colunas excluídas por leakage:**
`customer_id`, `churn_probability`, `retention_offer_made`, `retention_offer_accepted`, `contract_renewal_date`, `loyalty_end_date`

**Top features identificadas (consenso RF + GB):**

| Feature | Importância RF | Importância GB |
|---|---|---|
| `nps_detractor_flag` | 0.098 | 0.195 |
| `late_payments_6m` | 0.060 | 0.215 |
| `invoice_shock_flag` | 0.054 | 0.087 |
| `plan_price` | 0.036 | 0.103 |

---

## Bloco 4 — Labels (Variável Alvo)

- **Target:** `churn` — variável binária (0 = não cancelou, 1 = cancelou)
- **Balanceamento:** ~26% de churn (classe positiva) — desbalanceamento moderado
- **Estratégia:** split estratificado 80/20; PR-AUC como métrica primária (mais sensível a desbalanceamento)
- **Definição operacional:** churn já ocorrido no período de observação (label histórico)

---

## Bloco 5 — ML Task (Tarefa de ML)

**Tipo:** Classificação binária supervisionada

**Pipeline de pré-processamento** (`src/churn_prediction/`):
1. Remoção de duplicados e IDs duplicados
2. Correção de valores fora de faixa (clip) e padronização de categóricas
3. Imputação: mediana para numéricas, moda para categóricas
4. One-hot encoding para variáveis categóricas
5. StandardScaler para variáveis numéricas

**Modelos treinados:**

| Modelo | Biblioteca | Configuração |
|---|---|---|
| `DummyClassifier` | scikit-learn | strategy=stratified, seed=42 |
| `LogisticRegression` | scikit-learn | C=0.1, L2, GridSearchCV 5-fold |
| `RandomForestClassifier` | scikit-learn | 200 estimadores, max_depth=10, seed=42 |
| `GradientBoostingClassifier` | scikit-learn | 100 estimadores, lr=0.1, max_depth=4, seed=42 |
| `MLP PyTorch` | PyTorch | 128→64, BatchNorm, Dropout=0.3, BCEWithLogitsLoss |

**MLP — detalhes de treinamento:**
- Otimizador: Adam (lr=1e-3, weight_decay=1e-4)
- Mini-batches: BATCH_SIZE=512
- Early stopping: patience=10 sobre val_loss (10% do treino separado)
- `ReduceLROnPlateau`: fator=0.5, patience=5
- Seeds globais fixadas: `random=42`, `numpy=42`, `torch=42`, `cuda=42`
- Artefato salvo no MLflow via `mlflow.pytorch.log_model`

**Rastreabilidade:**
- Experimentos registrados no MLflow: `churn-baselines` e `churn-mlp-pytorch`
- Automação via Makefile: `make run-all` executa todo o pipeline

---

## Bloco 6 — Offline Evaluation (Avaliação Offline)

**Métrica primária:** PR-AUC (mais informativa para classes desbalanceadas)

**Métricas técnicas:**

| Métrica | Justificativa |
|---|---|
| PR-AUC | Prioridade — avalia qualidade na classe positiva (churn) |
| ROC-AUC | Comparação geral entre modelos |
| F1-Score | Equilíbrio precisão/recall |
| Accuracy | Referência — menos informativa com desbalanceamento |
| Confusion Matrix | Diagnóstico de FP e FN |

**Métrica de negócio:**
```
valor_liquido = TP × R$500 − (TP + FP) × R$50
```
- `V_RETIDO = R$500`: valor médio retido por cliente resgatado
- `C_ACAO = R$50`: custo de contato por cliente (desconto + operação)
- Threshold otimizado por varredura 0.10–0.90 (maximização de valor_liquido)

**Resultados offline:**

| Modelo | ROC-AUC | PR-AUC | F1 | Valor Líquido |
|---|---|---|---|---|
| Dummy | 0.505 | 0.396 | 0.394 | R$ 561.850 |
| LogisticRegression | 0.878 | 0.848 | 0.743 | R$ 1.190.800 |
| RandomForest | 0.861 | 0.823 | 0.715 | R$ 1.103.750 |
| GradientBoosting | 0.882 | 0.854 | 0.747 | R$ 1.183.300 |
| **MLP PyTorch** | **0.877** | **0.846** | **0.743** | **R$ 1.192.700** |

**SLOs de qualidade:**
- ROC-AUC ≥ 0.80 em validação
- Recall ≥ 0.70 no top 20% dos clientes com maior score
- Valor líquido > R$ 900.000 por ciclo de campanha

**Diagnóstico de overfitting:**

| Modelo | delta_roc_auc (treino − teste) | Diagnóstico |
|---|---|---|
| LogisticRegression | 0.0013 | Sem overfitting |
| Dummy | −0.0085 | OK |

---

## Bloco 7 — Online Evaluation (Avaliação Online / Monitoramento)

**Monitoramento implementado** (`src/churn_prediction/monitoring.py`):

| Teste | Features | Alerta |
|---|---|---|
| KS (Kolmogorov-Smirnov) | Numéricas | p < 0.05 |
| Chi² (Qui-Quadrado) | Categóricas | p < 0.05 |
| PSI (Population Stability Index) | Numéricas | > 0.20 → retreinar |

**Scripts:**
- `scripts/simulate_drift.py`: simula requisições com distribuição alterada
- `scripts/check_drift.py`: compara dados de treino vs. logs de produção

**Métricas operacionais:**
- ROC-AUC rolling mensal (alerta se < 0.78)
- Taxa de churn real pós-campanha vs. prevista (alerta se divergência > 15%)
- `demographic_parity_difference` e `equalized_odds_difference` mensais

---

## Bloco 8 — Making Predictions (Inferência)

**Modo primário: batch scoring**
- Frequência: mensal (alinhado ao ciclo de campanhas de CRM)
- Input: base de clientes ativa com features atualizadas
- Output: tabela de scores + faixa de risco por cliente
- Entrega: via arquivo ou integração com CRM

**Modo secundário: real-time (implementado)**
- API FastAPI via `POST /predict`
- Input: JSON com features do cliente (todos os campos opcionais)
- Output: `churn_probability`, `churn_prediction`, `risk_level`
- Containerizada com Docker; health check em `GET /health`

**Segmentação de risco:**
- `alto`: score ≥ 0.70 → prioridade máxima de contato
- `medio`: 0.40 ≤ score < 0.70 → abordagem oportunística
- `baixo`: score < 0.40 → sem ação (ou ação passiva)

**Logging de inferência:**
Cada requisição é logada com `timestamp`, `churn_prob`, `risk_level` e `latency_ms` para alimentar o monitoramento de drift.

---

## Bloco 9 — Building / Monitoring (Retreino e Governança)

**Fluxo de retreino:**
1. `check_drift.py` detecta PSI > 0.20 ou KS p < 0.05 em feature crítica
2. Time de Data Science avalia necessidade de retreino
3. `make run-all` executa pipeline completo com dados atualizados
4. `scripts/export_model.py` serializa novo pipeline
5. Novo modelo registrado no MLflow com tag `retrain_trigger`
6. Deploy via Docker Compose com restart da API

**Frequência de retreino:** mensal (ou imediato se drift detectado)

**Fairness — avaliação e mitigação:**

| Atributo sensível | dp_diff | eo_diff |
|---|---|---|
| `gender` | 0.0552 | 0.0741 |
| `age_group` | 0.1101 | 0.0758 |
| `region` | 0.1565 | 0.1672 |
| `plan_type` | 0.2960 | 0.1840 |

Mitigação disponível: `ExponentiatedGradient` + `EqualizedOdds` (Fairlearn)
Custo da mitigação por `gender`: −R$ 9.850 / ciclo (−0.8%)

**CI/CD:** `.github/workflows/ci_ml_pipeline.yml`
- Lint (ruff) + pytest em todo push/PR
- Treino + export automático na `main`
- Validação do build Docker na `main`

**Próximos passos:**
- Calibrar `V_RETIDO` e `C_ACAO` com time de CRM/financeiro
- Retreinamento automático via trigger de drift (CT pipeline)
- SHAP/LIME para explainability por cliente
- Deploy em cloud com autoscaling (AWS/Azure/GCP)
- Autenticação JWT na API
