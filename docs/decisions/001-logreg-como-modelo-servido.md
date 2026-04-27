# ADR 001 — LogisticRegression como modelo servido (não MLP)

**Status:** ✅ Accepted
**Data:** 2026-04-24
**Autores:** Grupo 4

## Contexto

O Tech Challenge exige a entrega de uma rede neural (MLP em PyTorch) como parte obrigatória do projeto. No entanto, a decisão de qual modelo **servir em produção** via API é separada da obrigação acadêmica.

Após rodar o pipeline completo (`make run-all`, 2026-04-24), obtivemos estas métricas:

| Modelo | ROC-AUC | F1 | Valor Líquido |
|---|---|---|---|
| `log_reg` | 0.8783 | 0.7425 | **R$ 1.190.800** |
| `mlp_pytorch_v1` | 0.8772 | 0.7414 | R$ 1.190.600 |
| `gradient_boosting` | 0.8815 | 0.7474 | R$ 1.183.300 |

A diferença entre LogReg e MLP em valor líquido é de **R$ 200 — 0,02%** do total. GradientBoosting tem melhor ROC-AUC/F1, mas entrega R$ 7.500 a menos em valor líquido pelo perfil de threshold.

## Alternativas consideradas

1. **Servir o MLP PyTorch** — maior capacidade expressiva, mas requer PyTorch runtime, maior latência e dependência de GPU opcional
2. **Servir o GradientBoosting** — melhor ROC-AUC, mas valor líquido inferior
3. **Ensemble (stacking)** — ganho marginal a custo de complexidade operacional
4. **LogisticRegression** — modelo servido atual

## Decisão

Servir **`LogisticRegression` (C=0.1, L2)** via API. O Model Registry (`src/churn_prediction/registry.py`) seleciona automaticamente o champion por `valor_liquido`, e atualmente o LogReg é o vencedor.

## Consequências

### Positivas
- **Latência P99 < 10ms** sem necessidade de otimização de serving
- **Sem dependência de GPU** em produção
- **Interpretabilidade nativa** via coeficientes — facilita auditoria e compliance
- **Pipeline sklearn serializável** com `joblib` (~13 KB) — deploy trivial
- **Menor custo de manutenção** — não precisa de versionamento de PyTorch

### Negativas
- Não aproveita capacidade não-linear do MLP
- Se novos dados/features introduzirem não-linearidades fortes, LogReg pode degradar primeiro

### Neutras
- O MLP permanece treinado e registrado no MLflow como **modelo de estudo** (requisito do Tech Challenge) e como **challenger** no Model Registry
- Se o MLP ultrapassar LogReg em `valor_liquido` em futuros retreinos, o registry promoverá automaticamente

## Lição arquitetural

Em dados tabulares bem preparados, modelos lineares regularizados frequentemente igualam ou superam redes neurais em tarefas de classificação binária. O ganho de capacidade expressiva do MLP só se justifica quando há padrões não-lineares fortes que o pré-processamento não captura.
