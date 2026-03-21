# ML Canvas - Churn em Telecom (Base Estendida)

## 1. Problema de negocio

A operadora quer reduzir cancelamentos usando priorizacao de clientes com maior risco de churn.
A decisao pratica e: quais clientes devem entrar primeiro nas campanhas de retencao para maximizar impacto e controlar custo.

## 2. Objetivo de negocio

- Reduzir churn na carteira ativa.
- Preservar receita recorrente com foco em clientes de maior valor.
- Melhorar eficiencia das campanhas, evitando contato em massa sem priorizacao.

## 3. Stakeholders

- Marketing/CRM: define regras de campanha e incentivo.
- Retencao/Atendimento: executa abordagem dos clientes priorizados.
- BI/Analytics: acompanha resultados operacionais e financeiros.
- Data Science/MLOps: treina, monitora e atualiza o modelo.
- Lideranca: avalia impacto em churn, receita e ROI.

## 4. Usuario final e decisao suportada

- Usuario principal: time de CRM/retencao.
- Entrega esperada: score de churn por cliente (0 a 1) + faixas de risco.
- Decisao suportada: quem contatar, quando contatar e qual oferta priorizar.

## 5. Dados de entrada (base atual)

Base de referencia: `data/raw/telecom_churn_base_extended.csv`.

Principais blocos de variaveis:

- Perfil e contrato:
  - `age`, `gender`, `region`, `plan_type`, `plan_price`, `is_promotional_plan`, `months_to_contract_end`, `has_loyalty`.
- Qualidade de servico:
  - `network_outages_30d`, `avg_signal_quality`, `call_drop_rate`, `avg_internet_speed`, `service_failures_30d`.
- Uso e engajamento:
  - `minutes_monthly`, `data_gb_monthly`, `usage_delta_pct`, `days_since_last_usage`, `app_login_30d`, `self_service_usage_30d`.
- Financeiro:
  - `monthly_charges`, `avg_bill_last_6m`, `invoice_shock_flag`, `late_payments_6m`, `default_flag`, `days_past_due`.
- Atendimento e experiencia:
  - `support_calls_30d`, `support_calls_90d`, `complaints_30d`, `resolution_time_avg`, `first_call_resolution_flag`.
- Concorrencia e retencao:
  - `portability_request_flag`, `competitor_offer_contact_flag`, `retention_offer_made`, `retention_offer_accepted`.
- Satisfacao:
  - `nps_score`, `nps_category`, `nps_promoter_flag`, `nps_detractor_flag`, `csat_score`.

Target:

- `churn` (0/1)

Variavel auxiliar para analise:

- `churn_probability` (probabilidade sintetica usada na geracao da base)

## 6. Saida do modelo

- Score de churn por cliente.
- Segmentacao recomendada:
  - Alto risco: score >= 0.70
  - Medio risco: 0.40 <= score < 0.70
  - Baixo risco: score < 0.40

Os thresholds devem ser recalibrados com base em custo de campanha e capacidade operacional.

## 7. Metricas de sucesso

Metricas de negocio:

- Reducao da taxa de churn apos campanha.
- Receita preservada (clientes retidos x ticket).
- ROI de retencao.

Metricas de modelo:

- AUC-ROC (comparacao geral entre modelos).
- PR-AUC (prioritaria para classe positiva desbalanceada).
- Recall no top-K (ex.: top 10% e top 20% mais arriscados).
- Precision no top-K para controlar desperdicio de contato.

## 8. SLOs iniciais

- Qualidade:
  - AUC-ROC >= 0.80 em validacao.
  - Recall >= 0.70 no top 20% dos clientes com maior score.
- Operacao:
  - Atualizacao mensal do modelo (ou antes se houver drift).
  - Monitoramento recorrente de drift das principais features.

## 9. Riscos e mitigacao

- Desbalanceamento de churn:
  - Mitigar com metrica adequada (PR-AUC/Recall top-K) e validacao estratificada.
- Drift de comportamento:
  - Monitorar distribuicoes e queda de performance ao longo do tempo.
- Vazamento de informacao:
  - Revisar colunas potencialmente proximas do evento de churn antes do treino.
- Viés de acao:
  - Garantir que segmentos nao sejam penalizados por proxies inadequados.

## 10. Plano de execucao (curto prazo)

1. Consolidar EDA com foco em qualidade dos dados e sinais de risco.
2. Treinar baselines em `notebooks/02_baselines.ipynb`.
3. Comparar modelos por AUC-ROC, PR-AUC e top-K.
4. Definir cutoff operacional para campanha piloto.
5. Documentar resultados no model card e preparar versao inicial de inferencia.