# Plano de Monitoramento — Churn Prediction

> Documento operacional do pipeline de monitoramento em produção.
> Projeto: mlet-grupo4-tech-challenge | Atualizado: 2026-04-24

---

## 1. Objetivo

Garantir que o modelo servido em produção mantenha performance, fairness e qualidade de dados dentro de limites aceitáveis, detectando degradações com tempo de resposta que minimize perda de valor de negócio.

---

## 2. Métricas monitoradas

### 2.1 Performance técnica

| Métrica | Frequência | Alerta (warning) | Alerta (crítico) |
|---|---|---|---|
| ROC-AUC (validação rolling 30d) | Diária | < 0.82 | < 0.78 |
| F1-Score (validação rolling 30d) | Diária | < 0.70 | < 0.65 |
| PR-AUC (validação rolling 30d) | Diária | < 0.80 | < 0.75 |
| Best threshold drift | Semanal | desvio > 0.05 do baseline | desvio > 0.10 |

### 2.2 Métricas de negócio

| Métrica | Frequência | Alerta (warning) | Alerta (crítico) |
|---|---|---|---|
| Taxa de churn real vs prevista | Mensal | divergência > 10% | divergência > 20% |
| Valor líquido realizado por ciclo | Mensal | < 90% do esperado | < 75% do esperado |
| TP/FP ratio | Mensal | desvio > 15% | desvio > 30% |
| Clientes abordados / total | Mensal | desvio > 20% | desvio > 40% |

### 2.3 Data drift por feature (PSI)

Thresholds **por feature crítica** (top-10 por importância):

| Feature | PSI warning | PSI crítico | Justificativa |
|---|---|---|---|
| `nps_detractor_flag` | 0.15 | 0.25 | Top-1 em importância (GB), sinal mais forte de churn |
| `late_payments_6m` | 0.15 | 0.25 | Top-2 em importância (GB) |
| `nps_category_detractor` | 0.15 | 0.25 | Correlação alta com NPS |
| `invoice_shock_flag` | 0.20 | 0.30 | Feature binária, maior tolerância |
| `plan_price` | 0.15 | 0.25 | Sensível a reajustes |
| `price_increase_last_3m` | 0.20 | 0.30 | Sensível a políticas de preço |
| `nps_promoter_flag` | 0.15 | 0.25 | |
| `nps_score` | 0.15 | 0.25 | Feature contínua — recalibrar se > 0.25 |
| `months_to_contract_end` | 0.20 | 0.30 | Varia sazonalmente |
| `monthly_charges` | 0.15 | 0.25 | |

**Referência PSI:**
- PSI < 0.10 → distribuições estáveis (sem ação)
- 0.10 ≤ PSI < 0.20 → investigar
- PSI ≥ 0.20 → drift significativo, considerar retreino
- PSI ≥ 0.25 em feature crítica → **trigger automático de retreino**

### 2.4 Fairness

| Métrica | Atributo sensível | Frequência | Alerta |
|---|---|---|---|
| `demographic_parity_difference` | `gender` | Mensal | > 0.10 |
| `equalized_odds_difference` | `gender` | Mensal | > 0.10 |
| `demographic_parity_difference` | `age_group` | Mensal | > 0.15 |
| `equalized_odds_difference` | `age_group` | Mensal | > 0.15 |
| `demographic_parity_difference` | `region` | Mensal | > 0.20 |
| `demographic_parity_difference` | `plan_type` | Mensal | > 0.35 (já em 0.296) |

### 2.5 API / Infraestrutura

| Métrica | Frequência | Alerta |
|---|---|---|
| Uptime da API `/health` | 1 min | < 99.5% em 24h |
| Latência P99 do `/predict` | 5 min | > 500ms |
| Latência P50 do `/predict` | 5 min | > 100ms |
| Taxa de erros HTTP 5xx | 5 min | > 1% |
| Taxa de erros HTTP 422 (input inválido) | 1 hora | > 5% |
| `X-Process-Time-Ms` header mediano | Diária | > 50ms |

---

## 3. Ferramentas implementadas

| Ferramenta | Localização | Função |
|---|---|---|
| **PSI / KS / Chi²** | `src/churn_prediction/monitoring.py` | Detecção estatística de drift |
| **`check_drift.py`** | `scripts/check_drift.py` | Relatório CLI de drift entre treino e produção |
| **`simulate_drift.py`** | `scripts/simulate_drift.py` | Geração de tráfego sintético com drift para teste |
| **`InferenceLogger`** | `src/churn_prediction/monitoring.py` | Logger estruturado JSON para auditoria |
| **LatencyMiddleware** | `src/churn_prediction/api/main.py` | Header `X-Process-Time-Ms` em todas as respostas HTTP |
| **MLflow Tracking** | `mlruns/` | Histórico de runs e comparação de métricas |
| **Model Registry** | `src/churn_prediction/registry.py` | Versionamento e promoção de champion |

---

## 4. Playbook de resposta

### 4.1 Queda de performance (ROC-AUC < 0.78)

1. **Verificar drift** com `scripts/check_drift.py` — identificar features com PSI > 0.20
2. **Se drift confirmado** em feature crítica:
   - Executar `make run-all` com dados dos últimos 90 dias
   - Registrar run no MLflow com tag `retrain_trigger=drift_auc`
   - Executar `scripts/export_model.py` — registry selecionará o novo champion
   - Reiniciar container da API: `docker compose restart churn-api`
3. **Se não houver drift**:
   - Verificar qualidade dos labels (atraso de label, ruído)
   - Investigar viés temporal (sazonalidade)
   - Abrir review com time de dados
4. **Tempo de resposta:** 48h para diagnóstico, 1 semana para ação corretiva

### 4.2 Drift detectado (PSI ≥ 0.25 em feature crítica)

1. `scripts/check_drift.py --reference <baseline> --production <logs>` — relatório detalhado
2. Se PSI ≥ 0.25 em **1 ou mais** features do top-10:
   - Trigger automático de retreino
   - Comparar performance do novo modelo vs champion atual
   - Promover novo modelo apenas se `valor_liquido_novo > valor_liquido_champion`
3. **Tempo de resposta:** 24h para retreino automático

### 4.3 Gap de fairness acima do limiar

1. Identificar grupo afetado nas métricas do MLflow
2. Retreinar com `EqualizedOdds` e `max_iter=50` (vs. atual 15)
3. Comparar `eo_diff` antes/depois
4. Se gap persistir após 2 ciclos → **escalar para compliance**
5. Se custo da mitigação > 5% do valor líquido → abrir review de negócio
6. **Tempo de resposta:** 2 semanas para retreino + validação

### 4.4 Erro de schema em produção (HTTP 422 > 5%)

1. Verificar logs estruturados do `InferenceLogger` — identificar campo problemático
2. Se campo novo/alterado no CRM → alinhar pipeline de features com time de dados
3. **Nunca** servir predição com dados incompletos — API retorna 422 corretamente
4. **Tempo de resposta:** 4h para diagnóstico, 24h para fix

### 4.5 Latência P99 > 500ms

1. Verificar `X-Process-Time-Ms` nos logs
2. Confirmar se é issue do modelo ou de rede/infra
3. Se modelo: considerar fallback para modelo mais leve no registry
4. Se infra: escalar container / investigar I/O
5. **Tempo de resposta:** 1h para diagnóstico, 4h para mitigação

### 4.6 `model_loaded: false` no `/health`

1. Confirmar existência de `models/churn_pipeline.joblib` no container
2. Se não existe: rodar `scripts/export_model.py` + reiniciar container
3. Se existe mas falha ao carregar: investigar logs do `lifespan` handler
4. **Ação imediata:** servir fallback (modelo anterior) ou retornar 503 + alertar
5. **Tempo de resposta:** 15min (SLO crítico)

---

## 5. Ciclo de retreino

### Retreino programado
- **Frequência base:** mensal
- **Janela de dados:** últimos 90 dias
- **Aprovação:** automática se `valor_liquido_novo > valor_liquido_champion`

### Retreino por trigger
- Drift crítico (PSI ≥ 0.25 em feature top-10)
- ROC-AUC < 0.78 por 7 dias consecutivos
- Solicitação manual do time de dados/CRM

### Validação antes de promoção
- Checklist no Model Registry:
  - [ ] Todos os testes pytest passando
  - [ ] Fairness: `dp_diff` e `eo_diff` dentro dos limiares
  - [ ] `valor_liquido` >= baseline do champion atual
  - [ ] Calibração (Brier, ECE) dentro dos limiares
  - [ ] Smoke test da API com o novo modelo

---

## 6. Observabilidade

### Logs estruturados (JSON)
- Handler `/predict`: `{event, churn_prob, risk}`
- Middleware HTTP: `{event, method, path, status, latency_ms}`
- Inferência: `src/churn_prediction/monitoring.py::InferenceLogger`

### Artefatos de observabilidade
- `logs/inference.jsonl` — cada predição com features + output
- `logs/drift_simulation.jsonl` — gerado por `simulate_drift.py`
- `mlruns/` — histórico completo de experimentos

### Dashboards recomendados (não implementados)
- Grafana com latência + taxa de erros da API
- Dashboard MLflow com comparação de runs
- Relatório mensal de PSI por feature

---

## 7. Escalação

| Severidade | Canal | SLA de resposta |
|---|---|---|
| Crítico (API down, model_loaded false) | PagerDuty + #incidents | 15 min |
| Alto (drift crítico, perf < threshold) | Slack #data-ops | 4 horas |
| Médio (fairness, drift warning) | Email + Jira | 48 horas |
| Baixo (informativo, PSI warning) | Dashboard | 1 semana |

---

## 8. Referências

- ADR 001 — LogReg como modelo servido
- ADR 002 — Threshold por valor líquido
- [Model Card](model_card.md) — seção "Cenários de falha" e "Plano de monitoramento"
- [FIAP z-fab reference](https://github.com/z-fab/FIAP/tree/master/documentacao-ml) — capítulo 6 (checklist)
