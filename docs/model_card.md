# Model Card - Predição de Churn em Telecom

## 1. Informações gerais

- Projeto: mlet-grupo4-tech-challenge
- Problema: classificação binária de churn (0/1)
- Dataset principal: `data/raw/telecom_churn_base_extended.csv`
- Etapa atual: pipeline completo com MLP em PyTorch, API FastAPI, Docker, CI/CD, monitoramento de drift e rastreabilidade em MLflow

## 2. Objetivo do modelo

Estimar o risco de churn por cliente para priorizar campanhas de retenção, maximizando impacto e otimizando custo operacional.

**Usos pretendidos:**
- Priorização de clientes para campanhas de retenção proativas
- Segmentação de risco (alto / médio / baixo) para definição de oferta e abordagem
- Suporte à decisão do time de CRM/retenção — não substitui julgamento humano

**Usos fora do escopo (não recomendados):**
- Decisões automáticas sem revisão humana (ex.: cancelamento de benefícios, bloqueio de conta)
- Aplicação em bases com distribuição significativamente diferente da base de treino (outro país, outro segmento de mercado)
- Uso como critério único para elegibilidade a serviços, crédito ou ofertas discriminatórias
- Inferência em tempo real com SLA < 200ms sem otimização específica de serving

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
- MLP em PyTorch: arquitetura 101→128→64→1, BatchNorm, Dropout=0.3, BCEWithLogitsLoss, Adam (lr=0.001), batch_size=512, early stopping (patience=10), 21 épocas treinadas, best_threshold=0.1, seeds fixadas
- Experimentos registrados no MLflow (`churn-baselines`, `churn-mlp-pytorch`)

Fluxo reprodutível:

- `make run-all`: executa EDA, baselines (com export dos splits), MLP e análise automática
- `make analyze`: consolida métricas técnicas e de negócio dos experimentos

## 5. Métricas de performance

Valores registrados no MLflow após execução completa do pipeline (`make run-all`, 2026-04-24). Champion selecionado automaticamente por `export_model.py` com base em `valor_liquido` máximo.

| Modelo | Accuracy | F1 | ROC-AUC | PR-AUC | Valor Líquido | Clientes abordados |
|---|---:|---:|---:|---:|---:|---:|
| dummy_stratified | 0.5294 | 0.3937 | 0.5046 | 0.3959 | R$ 561.850 | 3.753 |
| random_forest | 0.7951 | 0.7154 | 0.8609 | 0.8225 | R$ 1.103.750 | 3.205 |
| log_reg_mitigated_equalized_odds | 0.8001 | 0.7343 | — | — | R$ 1.179.050 | 3.519 |
| mlp_pytorch_v1 | 0.8057 | 0.7414 | 0.8772 | 0.8463 | R$ 1.190.600 | 3.508 |
| **log_reg** ⭐ champion | **0.8069** | **0.7425** | **0.8783** | **0.8480** | **R$ 1.190.800** | **3.494** |
| gradient_boosting | 0.8137 | 0.7474 | 0.8815 | 0.8541 | R$ 1.183.300 | 3.374 |

### 5.1 Análise arquitetural: log_reg vs MLP PyTorch

A diferença de valor líquido entre `log_reg` (R$ 1.190.800) e `mlp_pytorch_v1` (R$ 1.190.600) é de apenas **R$ 200 — 0,02%** do valor total. Essa margem é estatisticamente irrelevante e evidencia uma lição arquitetural central em ML para dados tabulares:

> Redes neurais não trazem necessariamente ganho sobre modelos lineares bem calibrados. Em dados estruturados com features bem construídas, a regressão logística com regularização adequada já captura os padrões relevantes. O MLP, apesar de maior capacidade expressiva (101→128→64→1, 21 épocas), não encontra representações adicionais que justifiquem seu custo computacional e de manutenção.

Note ainda que `gradient_boosting` obtém o melhor ROC-AUC (0.8815) e F1 (0.7474), mas entrega **R$ 7.500 menos** de valor líquido que o `log_reg` — consequência de um perfil de threshold que reduz o número de clientes abordados (3.374 vs 3.494). Isso demonstra que a métrica de otimização escolhida (valor líquido de campanha) diverge das métricas técnicas clássicas e deve guiar a seleção de modelo em contextos de negócio.

**O `log_reg` é o champion operacional do registry**: máximo valor líquido, maior interpretabilidade, menor latência de inferência e sem dependência de GPU.

**O `mlp_pytorch_v1` documenta o requisito obrigatório de rede neural** do Tech Challenge e valida experimentalmente que arquiteturas mais complexas nem sempre superam modelos lineares em datasets tabulares bem preparados.

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
| F1 | 0.7425 | 0.7343 | −0.008 |
| Valor Líquido | R$ 1.190.800 | R$ 1.179.050 | −R$ 11.750 |

Nota: na configuração rápida atual, a mitigação não melhorou o gap `eo_diff`. Pode ser retreinada com mais iterações para nova comparação.

## 8. Cenários de falha conhecidos

| Cenário | Causa provável | Impacto | Mitigação |
|---|---|---|---|
| Queda brusca de ROC-AUC em produção | Data drift — mudança no perfil de uso ou planos | Campanhas ineficientes, custo desperdiçado | Monitorar distribuição de features mensalmente; retreinar se AUC < 0.78 |
| Alto volume de falsos positivos | Threshold muito baixo ou mudança sazonal (ex.: promoções) | Custo de campanha elevado sem retorno | Recalibrar threshold via varredura; revisar `valor_liquido` |
| Alto volume de falsos negativos | Threshold muito alto; churn concentrado em grupos sub-representados | Perda de receita por clientes não abordados | Revisar recall segmentado por grupo; retreinar com dados recentes |
| Viés por subgrupo demográfico | Desbalanceamento histórico nos dados de treinamento | Clientes de certos grupos sistematicamente ignorados ou super-abordados | Monitorar `dp_diff` e `eo_diff` mensalmente; aplicar mitigação se gap > 0.10 |
| Falha no pipeline de dados | Feature ausente ou schema alterado na entrada | Erro em inferência ou predição degenerada | Validação de schema com pandera antes da predição; smoke test na API |
| Leakage em produção | Variável correlacionada ao churn inserida por engano | Métricas infladas sem generalização real | Auditoria de features a cada atualização de pipeline |

## 9. Arquitetura de deploy

**Modo escolhido: Batch (offline scoring)**

Justificativa:
- O time de CRM executa campanhas em ciclos mensais — não há necessidade de score em tempo real
- O volume de clientes (~50k–200k) é adequado para batch overnight
- Simplifica auditoria e rastreabilidade (cada score tem timestamp e versão de modelo)

**Fluxo proposto:**
```
Dados de produção → Pipeline sklearn (pré-processamento) → MLP PyTorch → Score por cliente
→ Tabela de priorização (CRM) → Campanha de retenção → Feedback de resultado
```

**Alternativa real-time (implementada):**
- FastAPI + pipeline sklearn serializado (`churn_pipeline.joblib`) via `/predict`
- Schemas Pydantic validam entrada automaticamente (HTTP 422 para dados inválidos)
- Retorna probabilidade, predição binária e faixa de risco (alto/medio/baixo)
- Containerizado com Docker (Dockerfile + docker-compose)
- Health check em `/health` para monitoramento de disponibilidade
- Logging estruturado de cada inferência (timestamp, probabilidade, latência)

## 10. Plano de monitoramento

### Métricas a monitorar

| Métrica | Frequência | Alerta |
|---|---|---|
| ROC-AUC no conjunto de validação rolling | Mensal | < 0.78 |
| Taxa de churn real pós-campanha vs. prevista | Mensal | Divergência > 15% |
| `demographic_parity_difference` por `gender` | Mensal | > 0.10 |
| `equalized_odds_difference` por `gender` | Mensal | > 0.10 |
| Distribuição de features principais (PSI) | Mensal | PSI > 0.20 em qualquer feature top-10 |
| Taxa de erros da API `/predict` | Diária (se real-time) | > 1% de erros HTTP 5xx |

### Playbook de resposta

1. **Queda de AUC (< 0.78):**
   - Verificar drift de features com PSI
   - Se PSI > 0.20 em features críticas → retreinar com janela de dados mais recente
   - Registrar novo run no MLflow com tag `retrain_trigger=drift`

2. **Gap de fairness acima do limiar:**
   - Retreinar com `EqualizedOdds` e mais iterações (`max_iter=50`)
   - Comparar `eo_diff` antes/depois no MLflow
   - Escalar para compliance se gap persistir após 2 ciclos

3. **Erro em produção (schema inválido):**
   - Pipeline de validação rejeita entrada e retorna erro estruturado
   - Alertar time de dados via log estruturado (`logger.error`)
   - Não servir predição com dados incompletos

4. **Performance abaixo do SLO por 2 ciclos consecutivos:**
   - Abrir processo de revisão de modelo (nova rodada experimental)
   - Considerar re-engenharia de features ou arquitetura alternativa

## 11. Limitações conhecidas

- Base sintética (não representa integralmente o comportamento de produção)
- Fairness avaliada apenas nos atributos disponíveis (`gender`, `age`) — pode não cobrir todos os fatores sensíveis relevantes
- A mitigação pode reduzir gap de fairness com perda de performance; a decisão final depende do trade-off acordado com o negócio
- MLP sem avaliação de fairness formal (apenas métricas técnicas e de negócio nesta versão)

## 12. Decisão de uso

- **Champion operacional:** `log_reg` — selecionado automaticamente pelo registry (`export_model.py`) com base em `valor_liquido` máximo (R$ 1.190.800). Melhor equilíbrio entre resultado de negócio, interpretabilidade e custo operacional.
- **Modelo de estudo (Tech Challenge):** `mlp_pytorch_v1` — cumpre o requisito obrigatório de rede neural; diferença de R$ 200 (0,02%) em relação ao champion confirma que a complexidade adicional não se justifica operacionalmente para este dataset tabular.
- **Critério mínimo de fairness:** `dp_diff_gender` < 0.10 e `eo_diff_gender` < 0.10 (monitoramento mensal conforme seção 10)
- **Data da decisão:** 2026-04-24

## 13. Ferramentas de monitoramento implementadas

- **`scripts/simulate_drift.py`**: Gera requisições com distribuição alterada para testar detecção de drift
- **`scripts/check_drift.py`**: Analisa drift entre dados de treino e logs de produção
- **`src/churn_prediction/monitoring.py`**: Módulo com KS test, Chi², PSI e InferenceLogger
- **CI/CD** (`.github/workflows/ci_ml_pipeline.yml`): Pipeline automatizado com lint, testes, treino e build Docker

## 14. Próximos passos

1. Calibrar `V_RETIDO` e `C_ACAO` com negócio para decisão operacional realista.
2. Consolidar threshold operacional da campanha (varredura 0.10–0.90 já implementada).
3. Avaliar fairness do MLP por subgrupos sensíveis.
4. Implementar retreinamento automático via trigger de drift (Continuous Training).
5. Adicionar autenticação JWT à API.
6. Deploy em cloud com autoscaling.
7. Evoluir arquitetura (tuning, explainability com SHAP/LIME).
