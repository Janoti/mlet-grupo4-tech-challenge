# ADR 004 — EqualizedOdds como restrição de fairness

**Status:** ✅ Accepted
**Data:** 2026-04-24
**Autores:** Grupo 4

## Contexto

O projeto avalia fairness por atributos sensíveis (`gender`, `age_group`, `region`, `plan_type`) e oferece mitigação via `fairlearn`. Precisamos escolher qual definição de fairness adotar como restrição.

Gaps observados sem mitigação (LogReg):

| Atributo | dp_diff | eo_diff |
|---|---|---|
| `gender` | 0.0552 | 0.0741 |
| `age_group` | 0.1101 | 0.0758 |
| `region` | 0.1565 | 0.1672 |
| `plan_type` | 0.2960 | 0.1840 |

## Alternativas consideradas

| Restrição | Definição | Prós | Contras |
|---|---|---|---|
| **Demographic Parity (DP)** | P(Ŷ=1\|A=a) igual entre grupos | Simples, visível | Ignora labels reais — pode prejudicar grupos com mais churn real |
| **Equalized Odds (EO)** | TPR e FPR iguais entre grupos | Considera labels reais | Mais restritivo; pode custar mais performance |
| **Equal Opportunity** | Apenas TPR igual | Balanceia oportunidade | Permite FPR desigual |
| **Calibration** | P(Y=1\|Ŝ=s,A=a) igual | Alinhado a probabilidades | Difícil conciliar com EO |

## Decisão

Adotar **`EqualizedOdds`** via `fairlearn.reductions.ExponentiatedGradient` como restrição de referência. Configuração:

```python
ExponentiatedGradient(
    estimator=LogisticRegression(C=0.1),
    constraints=EqualizedOdds(),
    eps=0.02,
    max_iter=15,
)
```

Grupo sensível de referência: `gender` (maior visibilidade regulatória e menor cardinalidade).

## Consequências

### Positivas
- **Captura ambos os tipos de erro** (FP e FN) desigualmente distribuídos entre grupos
- **Coerente com métrica de negócio** — TPR (recall) afeta receita, FPR afeta custo
- **Regulatoriamente defensável** — EO é mais conservador que DP e aceito em literatura regulatória

### Negativas
- **Custo de −R$ 11.750 em valor líquido** (−1,0%) para reduzir `dp_diff` de 0.0552 para 0.0518
- **Não reduz `eo_diff`** com `max_iter=15` — pode precisar de `max_iter=50+` para convergir
- **Mais restritivo** que DP → reduz flexibilidade do modelo

### Neutras
- Métricas DP e EO de todos os grupos continuam reportadas no MLflow para transparência
- A decisão de mitigar em produção depende do trade-off aceito com negócio/compliance

## Escalação

- Se `eo_diff_gender > 0.10` persistir após 2 ciclos de retreino → escalar para compliance
- Se custo da mitigação > 5% do valor líquido → revisar necessidade e avaliar DP como alternativa
- `plan_type` (gap 0.296) é o maior risco regulatório — próximo candidato a mitigação

## Referências

- Hardt, Price, Srebro (2016) — "Equality of Opportunity in Supervised Learning"
- Fairlearn docs — https://fairlearn.org/
