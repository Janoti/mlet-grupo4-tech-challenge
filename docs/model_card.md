# Model Card — Predição de Churn em Telecom

> Padrão: Google Model Cards (Mitchell et al., 2019)
> Projeto: mlet-grupo4-tech-challenge | Grupo 4 | Atualizado: Abril/2026

---

## 1. Model Details (Detalhes do Modelo)

| Campo | Valor |
|---|---|
| **Nome** | Churn Prediction — Telecom |
| **Versão** | 1.0.0 |
| **Tipo** | Classificação binária supervisionada |
| **Arquitetura primária** | LogisticRegression (sklearn) — modelo de produção |
| **Arquitetura em avaliação** | MLP PyTorch (128→64, BatchNorm, Dropout=0.3) |
| **Biblioteca** | scikit-learn 1.5, PyTorch 2.3 |
| **Rastreamento** | MLflow — experimentos `churn-baselines` e `churn-mlp-pytorch` |
| **Serialização** | `models/churn_pipeline.joblib` (sklearn pipeline) |
| **Dataset** | `data/raw/telecom_churn_base_extended.csv` (50.000 clientes, sintético) |
| **Data de treinamento** | Abril/2026 |
| **Equipe** | Grupo 4 — Tech Challenge FIAP Machine Learning Engineering |

---

## 2. Intended Use (Uso Pretendido)

### Uso pretendido

- Priorização de clientes para campanhas de retenção proativas
- Segmentação de risco (alto / médio / baixo) para definição de oferta e abordagem
- Suporte à decisão do time de CRM — não substitui julgamento humano

### Usuários pretendidos

- **Time de CRM/Retenção:** consome scores para priorizar contatos
- **Data Science/MLOps:** mantém, retreina e monitora o modelo
- **BI/Analytics:** acompanha KPIs pós-campanha

### Usos fora do escopo (não recomendados)

- Decisões automáticas sem revisão humana (ex.: cancelamento de benefícios, bloqueio de conta)
- Aplicação em bases com distribuição significativamente diferente da base de treino (outro país, segmento B2B)
- Uso como critério único para elegibilidade a crédito, serviços ou ofertas que possam ser discriminatórias
- Inferência em tempo real com SLA < 200ms sem otimização específica de serving

---

## 3. Training Data (Dados de Treinamento)

- **Fonte:** `data/raw/telecom_churn_base_extended.csv` (base sintética)
- **Volume:** 50.000 clientes, ~26% de churn (classe positiva)
- **Geração:** `scripts/generate_synthetic.py --n-rows 50000 --seed 42`
- **Split:** treino 80% / teste 20%, estratificado pelo target `churn`
- **Validação interna (MLP):** 10% do treino separado para early stopping

**Tratamento aplicado:**
1. Remoção de duplicatas e IDs duplicados
2. Clip de valores fora de faixa → `NaN`
3. Padronização de categóricas (lowercase, trim)
4. Imputação: mediana (numéricas), moda (categóricas)
5. One-hot encoding para variáveis categóricas
6. StandardScaler para variáveis numéricas

**Colunas excluídas por risco de leakage:**
`customer_id`, `churn_probability`, `retention_offer_made`, `retention_offer_accepted`, `contract_renewal_date`, `loyalty_end_date`

**Rastreabilidade:** hash SHA-256 do CSV registrado como `dataset_version` no MLflow

---

## 4. Evaluation Data (Dados de Avaliação)

- **Conjunto de teste:** 20% da base original, estratificado pelo target
- **Separação:** realizada antes de qualquer fitting (sem data leakage)
- **Distribuição:** mantém proporção de ~26% de churn (estratificação)
- **Fairness:** avaliação por subgrupos: `gender`, `age_group`, `region`, `plan_type`

---

## 5. Metrics (Métricas)

### Desempenho comparativo

| Modelo | Accuracy | F1 | ROC-AUC | PR-AUC | Valor Líquido |
|---|---:|---:|---:|---:|---:|
| Dummy (stratified) | 0.529 | 0.394 | 0.505 | 0.396 | R$ 561.850 |
| LogisticRegression (C=0.1) | 0.807 | 0.743 | 0.878 | 0.848 | R$ 1.190.800 |
| RandomForest (200 est.) | 0.795 | 0.715 | 0.861 | 0.823 | R$ 1.103.750 |
| GradientBoosting (100 est.) | 0.814 | 0.747 | 0.882 | 0.854 | R$ 1.183.300 |
| **MLP PyTorch** | **0.807** | **0.743** | **0.877** | **0.846** | **R$ 1.192.700** |
| LogReg + EqualizedOdds | 0.801 | 0.735 | — | — | R$ 1.180.950 |

### Métrica de negócio

```
valor_liquido = TP × R$500 − (TP + FP) × R$50
```

- `V_RETIDO = R$500`: receita preservada por cliente retido
- `C_ACAO = R$50`: custo de contato (desconto + operação)
- Threshold otimizado por varredura 0.10–0.90

**LogisticRegression no threshold ótimo:**
- 2.731 TP, 763 FP, 3.494 clientes abordados
- Valor líquido: **R$ 1.190.800** por ciclo de campanha

### Diagnóstico de overfitting

| Modelo | delta ROC-AUC (treino − teste) | Diagnóstico |
|---|---|---|
| LogisticRegression | 0.0013 | Sem overfitting — generaliza bem |
| Dummy | −0.0085 | OK |

---

## 6. Factors (Fatores Relevantes)

### Atributos sensíveis avaliados

- `gender` (male / female)
- `age_group` (derivada de `age`: <25, 25–34, 35–44, 45–54, 55+)
- `region` (Sudeste, Sul, Norte, Nordeste, Centro-Oeste)
- `plan_type` (pre, pos, controle)

### Por que esses fatores importam

Modelos de churn que não consideram equidade podem concentrar campanhas em segmentos específicos, gerando tratamento diferenciado com base em atributos demográficos — o que pode violar regulamentações ou princípios de equidade corporativa.

---

## 7. Quantitative Analyses (Análises Quantitativas de Fairness)

### LogisticRegression — sem mitigação

| Atributo sensível | dp_diff | eo_diff | Interpretação |
|---|---|---|---|
| `gender` | 0.0552 | 0.0741 | Gap moderado — aceitável |
| `age_group` | 0.1101 | 0.0758 | Gap relevante — monitorar |
| `region` | 0.1565 | 0.1672 | Gap alto — investigar |
| `plan_type` | 0.2960 | 0.1840 | Gap crítico — risco regulatório |

> `dp_diff` = demographic parity difference | `eo_diff` = equalized odds difference
> Limiar de alerta: dp_diff ou eo_diff > 0.10

### LogisticRegression — com mitigação (EqualizedOdds, por `gender`)

| Métrica | Sem mitigação | Com mitigação | Variação |
|---|:---:|:---:|:---:|
| dp_diff_gender | 0.0552 | 0.0518 | −6% |
| eo_diff_gender | 0.0741 | 0.0847 | +14% |
| F1 | 0.7425 | 0.7352 | −0.007 |
| Valor Líquido | R$ 1.190.800 | R$ 1.180.950 | −R$ 9.850 |

**Método:** `ExponentiatedGradient` + `EqualizedOdds` (Fairlearn)
**Configuração:** `eps=0.02`, `max_iter=15`, `mitigation_sample_size=15000`

> Nota: na configuração rápida atual, a mitigação não melhorou `eo_diff`. Pode ser retreinada com `max_iter=50` para comparação mais completa.

---

## 8. Ethical Considerations (Considerações Éticas)

- **Base sintética:** não representa integralmente comportamento real de produção; métricas podem variar em dados reais
- **Fairness parcial:** avaliada apenas nos atributos disponíveis (`gender`, `age`, `region`, `plan_type`) — podem existir outros fatores sensíveis não cobertos
- **MLP sem fairness formal:** o MLP foi avaliado apenas em métricas técnicas e de negócio; avaliação de fairness do MLP é um próximo passo
- **Decisão assistida:** o modelo deve ser usado como suporte à decisão humana, não como substituto
- **LGPD:** dados de clientes reais devem ser tratados conforme LGPD; base atual é sintética

---

## 9. Caveats and Recommendations (Limitações e Recomendações)

### Limitações conhecidas

1. **Dataset sintético:** pode não capturar todas as nuances do comportamento real de churn
2. **Threshold fixo:** o threshold ótimo foi calculado sobre o conjunto de teste — pode precisar de recalibração com dados reais
3. **Fairness por `plan_type`:** gap crítico (dp_diff=0.296) sem mitigação implementada
4. **MLP sem serialização sklearn:** o MLP PyTorch não está no pipeline sklearn exportado para a API; a API usa o modelo logístico
5. **Custo de campanha fixo:** `V_RETIDO` e `C_ACAO` são estimativas — devem ser calibrados com o time de CRM/financeiro

### Recomendações

- **Para produção imediata:** usar `LogisticRegression` (melhor equilíbrio performance/interpretabilidade/operação)
- **Para evolução:** avaliar GradientBoosting (melhor ROC-AUC/F1) ou MLP (melhor valor_liquido) com fairness completa
- **Antes do go-live:** validar threshold com dados de produção reais; calibrar `V_RETIDO` e `C_ACAO`
- **Retreino:** monitorar PSI mensal; retreinar se PSI > 0.20 em features críticas

---

## 10. Cenários de Falha Conhecidos

| Cenário | Causa provável | Impacto | Mitigação |
|---|---|---|---|
| Queda de ROC-AUC em produção | Data drift — mudança no perfil de uso | Campanhas ineficientes | Monitorar PSI mensal; retreinar se AUC < 0.78 |
| Alto volume de falsos positivos | Threshold baixo ou mudança sazonal | Custo de campanha elevado | Recalibrar threshold; revisar `valor_liquido` |
| Alto volume de falsos negativos | Threshold alto; subgrupos sub-representados | Perda de receita | Revisar recall segmentado; retreinar |
| Viés por subgrupo demográfico | Desbalanceamento histórico no treino | Tratamento desigual | Monitorar `dp_diff`/`eo_diff`; aplicar mitigação |
| Falha no pipeline de dados | Feature ausente ou schema alterado | Erro de inferência | Validação de schema com pandera; smoke test |
| Leakage em produção | Feature correlacionada inserida por engano | Métricas infladas | Auditoria de features a cada atualização |
| `model_loaded: false` na API | `churn_pipeline.joblib` não gerado | API retorna 503 | Rodar `scripts/export_model.py` antes de subir Docker |

---

## 11. Arquitetura de Deploy

**Modo primário: batch scoring (offline)**

Justificativa:
- O time de CRM executa campanhas em ciclos mensais — score em tempo real não é necessário
- Simplifica auditoria e rastreabilidade (cada score tem timestamp e versão de modelo)
- Volume adequado para processamento overnight

**Fluxo batch:**
```
dados de produção
  → pipeline sklearn (limpeza + pré-processamento)
  → modelo LogisticRegression
  → score por cliente
  → tabela de priorização (CRM)
  → campanha de retenção
  → coleta de resultado → feedback loop
```

**Modo secundário: real-time (implementado)**

- FastAPI + `models/churn_pipeline.joblib` via `POST /predict`
- Schemas Pydantic validam entrada (HTTP 422 para dados inválidos)
- Retorna: `churn_probability`, `churn_prediction`, `risk_level`
- Containerizado com Docker (`Dockerfile` + `docker-compose.yml`)
- Health check em `GET /health` para monitoramento de disponibilidade
- Logging estruturado de cada inferência (timestamp, probabilidade, latência)

**Como subir:**
```bash
# 1. Gerar modelo
PYTHONPATH=src poetry run python scripts/export_model.py

# 2. Subir container
docker compose up --build churn-api

# 3. Testar
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age": 45, "monthly_charges": 120, "nps_score": 3}'
```

---

## 12. Plano de Monitoramento

### Métricas a monitorar

| Métrica | Frequência | Alerta |
|---|---|---|
| ROC-AUC rolling (validação) | Mensal | < 0.78 |
| Taxa de churn real pós-campanha vs. prevista | Mensal | Divergência > 15% |
| PSI das features principais | Mensal | > 0.20 |
| `demographic_parity_difference` por `gender` | Mensal | > 0.10 |
| `equalized_odds_difference` por `gender` | Mensal | > 0.10 |
| Taxa de erros HTTP 5xx da API | Diária | > 1% |
| Latência P99 do `/predict` | Diária | > 500ms |

### Playbook de resposta

**1. Queda de AUC (< 0.78):**
- Verificar PSI das top features com `scripts/check_drift.py`
- Se PSI > 0.20 em feature crítica → `make run-all` com dados atualizados
- Registrar novo run no MLflow com tag `retrain_trigger=drift`
- Reexportar modelo e reiniciar container

**2. Gap de fairness acima do limiar:**
- Retreinar com `EqualizedOdds` e `max_iter=50`
- Comparar `eo_diff` antes/depois no MLflow
- Escalar para compliance se gap persistir após 2 ciclos

**3. Erro de schema em produção:**
- Pipeline de validação rejeita entrada e retorna HTTP 422
- Alertar time de dados via log estruturado (`logger.error`)
- Não servir predição com dados incompletos
- Verificar se pipeline de features foi atualizado sem alinhamento com a API

**4. `model_loaded: false` na API:**
- Confirmar existência de `models/churn_pipeline.joblib`
- Se não existe: `PYTHONPATH=src poetry run python scripts/export_model.py`
- Reiniciar container: `docker compose restart churn-api`

**5. Performance abaixo do SLO por 2 ciclos consecutivos:**
- Abrir processo de revisão de modelo
- Considerar re-engenharia de features ou nova arquitetura
- Avaliar inclusão de dados externos (cobertura de rede, concorrência)

---

## 13. Ferramentas implementadas

| Ferramenta | Localização | Função |
|---|---|---|
| FastAPI + Pydantic | `src/churn_prediction/api/` | API de inferência com validação de schema |
| MLflow Tracking | `mlruns/` | Rastreamento de experimentos, parâmetros e artefatos |
| Fairlearn | `notebooks/02_baselines.ipynb` | Avaliação e mitigação de fairness |
| Drift detection | `src/churn_prediction/monitoring.py` | KS test, Chi², PSI |
| `simulate_drift.py` | `scripts/` | Geração de tráfego com drift para teste |
| `check_drift.py` | `scripts/` | Análise de drift treino vs. produção |
| `export_model.py` | `scripts/` | Serialização do pipeline para a API |
| CI/CD | `.github/workflows/ci_ml_pipeline.yml` | Lint, testes, treino e build Docker automatizados |
| Docker | `Dockerfile` + `docker-compose.yml` | Containerização da API e MLflow |
| pytest (30 testes) | `tests/` | Smoke, schema, API — todos passando |

---

## 14. Decisão de uso e próximos passos

**Modelo recomendado para produção:** `LogisticRegression` (C=0.1, L2)
- Melhor equilíbrio entre ROC-AUC, interpretabilidade e simplicidade operacional
- Pipeline sklearn serializado e pronto para uso na API

**Modelo em avaliação para evolução:** MLP PyTorch
- Maior capacidade de capturar não-linearidades
- Requer avaliação de fairness formal e calibração de threshold em dados reais

**Próximos passos:**
1. Calibrar `V_RETIDO` e `C_ACAO` com time de CRM/financeiro
2. Validar threshold ótimo com dados de produção reais
3. Implementar fairness formal para o MLP PyTorch
4. Retreinamento automático via trigger de drift (CT pipeline)
5. Deploy em cloud com autoscaling (AWS/Azure/GCP)
6. Adicionar autenticação JWT à API
7. Integrar SHAP/LIME para explainability por cliente
