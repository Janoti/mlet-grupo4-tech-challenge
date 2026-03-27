# ML Canvas - Churn em Telecom (Base Estendida)

## 1. Problema de negócio

A operadora busca reduzir cancelamentos (churn) priorizando clientes de maior risco e valor para campanhas de retenção, maximizando impacto e controlando custos.

## 2. Objetivo de negócio

- Reduzir churn na base ativa
- Preservar receita recorrente, priorizando clientes de maior valor
- Melhorar eficiência das campanhas, evitando contato em massa

## 3. Stakeholders

- Marketing/CRM: define regras e incentivos
- Retenção/Atendimento: executa abordagem dos clientes priorizados
- BI/Analytics: acompanha resultados operacionais e financeiros
- Data Science/MLOps: treina, monitora e atualiza o modelo
- Liderança: avalia impacto em churn, receita e ROI

## 4. Usuário final e decisão suportada

- Usuário principal: time de CRM/retencao
- Entrega: score de churn por cliente (0 a 1) + faixas de risco
- Decisão: quem contatar, quando e qual oferta priorizar

## 5. Dados de entrada (base atual)

Base: `data/raw/telecom_churn_base_extended.csv`

O dataset é sintético, com duplicidades, missing, categorias inválidas e inconsistências para simular ambiente real. O tratamento de dados é parte explícita do pipeline.

Principais blocos de variáveis:

- Perfil e contrato: `age`, `gender`, `region`, `plan_type`, `plan_price`, `is_promotional_plan`, `months_to_contract_end`, `has_loyalty`
- Qualidade de serviço: `network_outages_30d`, `avg_signal_quality`, `call_drop_rate`, `avg_internet_speed`, `service_failures_30d`
- Uso e engajamento: `minutes_monthly`, `data_gb_monthly`, `usage_delta_pct`, `days_since_last_usage`, `app_login_30d`, `self_service_usage_30d`
- Financeiro: `monthly_charges`, `avg_bill_last_6m`, `invoice_shock_flag`, `late_payments_6m`, `default_flag`, `days_past_due`
- Atendimento: `support_calls_30d`, `support_calls_90d`, `complaints_30d`, `resolution_time_avg`, `first_call_resolution_flag`
- Concorrência: `portability_request_flag`, `competitor_offer_contact_flag`, `retention_offer_made`, `retention_offer_accepted`
- Satisfação: `nps_score`, `nps_category`, `nps_promoter_flag`, `nps_detractor_flag`, `csat_score`

Target: `churn` (0/1)

Variável auxiliar: `churn_probability` (probabilidade sintética)

Tratamento aplicado:

- remocao de linhas duplicadas e IDs duplicados
- correcao de valores fora de faixa, convertendo para `NaN` quando necessario
- ajuste de inconsistencias logicas simples
- imputacao de numericas com mediana
- imputacao de categoricas com moda
- one-hot encoding para variaveis categoricas
- exclusao de colunas com risco de leakage

Particionamento para avaliacao inicial:

- treino: 80%
- teste: 20%
- split estratificado pelo target

## 6. Pipeline e decisões técnicas

- EDA detalhada e documentada
- Baselines: DummyClassifier, LogisticRegression, fairness com Fairlearn
- Pipeline MLP em PyTorch (robustez, automação, rastreabilidade)
- Métricas de negócio integradas (clientes abordados, valor líquido, ROI)
- Automação via Makefile e rastreabilidade via MLflow

## 7. Saída do modelo

- Score de churn por cliente.
- Segmentacao recomendada:
  - Alto risco: score >= 0.70
  - Medio risco: 0.40 <= score < 0.70
  - Baixo risco: score < 0.40

Os thresholds devem ser recalibrados com base em custo de campanha e capacidade operacional.

## 8. Métricas de avaliação

Métricas técnicas:

- AUC-ROC (comparacao geral entre modelos)
- PR-AUC (prioritaria para classe positiva desbalanceada)
- Recall no top-K (ex.: top 10% e top 20% mais arriscados)
- Precision no top-K para controlar desperdicio de contato
- Fairness por subgrupo com Fairlearn (`gender`, `age_group`, `region` e `plan_type`)
- Gaps monitorados: `demographic_parity_difference` e `equalized_odds_difference`

Métricas de negócio:

- Reducao da taxa de churn apos campanha
- Receita preservada (clientes retidos x ticket)
- ROI de retencao
- Valor liquido esperado por politica de contato (`TP * V_RETIDO - (TP + FP) * C_ACAO`)

Status de implementacao das metricas de negocio no codigo atual:

- Calculo automatico de valor liquido ja incorporado ao notebook baseline e consolidado no `scripts/analyze_mlruns.py`.
- O fluxo atual registra no MLflow: `tp`, `fp`, `clientes_abordados`, `valor_bruto`, `custo_total_acao`, `valor_liquido`, `valor_por_cliente`.

## 9. SLOs iniciais

- Qualidade:
  - AUC-ROC >= 0.80 em validacao.
  - Recall >= 0.70 no top 20% dos clientes com maior score.
- Operacao:
  - Atualizacao mensal do modelo (ou antes se houver drift).
  - Monitoramento recorrente de drift das principais features.

Requisito minimo de preparo para treino:

- todo baseline deve registrar claramente qual tratamento foi aplicado antes do treino
- treino e teste devem permanecer separados antes da avaliacao final
- colunas identificadas como leakage nao entram na matriz de features

## 10. Riscos e mitigacao

- Desbalanceamento de churn:
  - Mitigar com metrica adequada (PR-AUC/Recall top-K) e validacao estratificada.
- Drift de comportamento:
  - Monitorar distribuicoes e queda de performance ao longo do tempo.
- Vazamento de informacao:
  - Revisar colunas potencialmente proximas do evento de churn antes do treino.
- Viés de acao:
  - Garantir que segmentos nao sejam penalizados por proxies inadequados.
- Viés entre subgrupos sensiveis:
  - Medir diferencas de taxa de selecao/erro por grupo e aplicar mitigacao quando necessario.

## 11. Plano de execucao (curto prazo)

1. Consolidar EDA com foco em qualidade dos dados, target e sinais de risco.
2. Aplicar tratamento minimo reprodutivel para baseline.
3. Separar a base em treino/teste com split estratificado 80/20.
4. Treinar baselines com `DummyClassifier` e `LogisticRegression` em `notebooks/02_baselines.ipynb`.
5. Registrar parametros, metricas e versao do dataset no MLflow.
6. Avaliar fairness por subgrupo e gaps (DP/EO).
7. Testar mitigacao de referencia com `EqualizedOdds` quando houver gap relevante.
8. Definir cutoff operacional para campanha piloto.
9. Documentar resultados no model card e preparar versao inicial de inferencia.
10. Calibrar `V_RETIDO` e `C_ACAO` com negocio para tomada de decisao operacional.

## 12. Status atual da Fase 1

Resultados consolidados do baseline em `notebooks/02_baselines.ipynb`:

- Melhor baseline: `LogisticRegression`.
- Metricas no teste:
  - Accuracy: `0.8069`
  - F1: `0.7425`
  - ROC-AUC: `0.8783`
  - PR-AUC: `0.8480`

Fairness (baseline logistico, por `gender`):

- `demographic_parity_difference`: `0.0552`
- `equalized_odds_difference`: `0.0741`

Mitigacao com `ExponentiatedGradient` + `EqualizedOdds` (configuracao rapida):

- Accuracy: `0.8007`
- F1: `0.7352`
- `dp_diff_gender`: `0.0518`
- `eo_diff_gender`: `0.0847`
- Tempo de execucao aproximado da mitigacao: ~`24s` com amostra estratificada de treino.

Automacao e rastreabilidade ja implementadas no codigo:

- Execucao fim a fim por Make (`make run-all`) com limpeza de runs, execucao dos notebooks, analise automatica e subida do MLflow.
- Analise automatica de runs locais via `scripts/analyze_mlruns.py`.
- Logging padronizado em scripts principais para facilitar troubleshooting e auditoria.

## 13. Próximos passos

- Evoluir arquitetura do modelo (feature engineering, tuning, explainability)
- Monitoramento contínuo e atualização do pipeline
