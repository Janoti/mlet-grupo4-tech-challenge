# ADR 002 — Threshold otimizado por valor líquido de negócio

**Status:** ✅ Accepted
**Data:** 2026-04-24
**Autores:** Grupo 4

## Contexto

Modelos de classificação binária retornam probabilidades (0–1) que precisam ser convertidas em decisões (contatar/não contatar) via um threshold. A escolha padrão de 0.5 raramente é ótima em contextos de negócio com custos assimétricos.

No nosso caso:
- **Falso Positivo (FP):** contatar cliente que não ia cancelar → custo de campanha R$ 50
- **Falso Negativo (FN):** não contatar cliente que cancelará → perda de R$ 500 de receita
- **Verdadeiro Positivo (TP):** contatar cliente que ia cancelar → ganho de R$ 500 evitando churn

Otimizar só por F1 ou ROC-AUC ignora essa assimetria de custos.

## Alternativas consideradas

1. **Threshold fixo em 0.5** — simples, mas ignora custos de negócio
2. **Maximizar F1** — equilibra precisão/recall, mas não considera custo absoluto
3. **Maximizar Youden's J** (TPR − FPR) — estatisticamente balanceado, não financeiro
4. **Varredura de threshold por valor líquido** — escolhida

## Decisão

Calcular threshold ótimo via varredura de 0.10 a 0.90 (passo 0.01), escolhendo o ponto que maximiza:

```
valor_liquido = TP × R$500 − (TP + FP) × R$50
```

Essa varredura é aplicada por modelo no notebook `02_baselines.ipynb` e `03_mlp_pytorch.ipynb`, e o threshold ótimo é registrado no MLflow como parâmetro `best_threshold`.

## Consequências

### Positivas
- **Alinhamento direto com KPI de negócio** — o modelo é avaliado no que importa para o CRM
- **Controle operacional explícito** — thresholds distintos por modelo refletem seus perfis de erro
- **Transparência:** fórmula é auditável e recalibrável sem retreino

### Negativas
- **Dependência de `V_RETIDO` e `C_ACAO`** — valores atuais (R$ 500 e R$ 50) são estimativas que devem ser calibradas com o time de CRM/financeiro antes do go-live
- **Instabilidade em pequenas amostras** — varredura em conjunto de teste pequeno pode sobreajustar o threshold

### Neutras
- Alternativa F1 ainda é reportada no MLflow para comparação técnica
- A escolha de threshold é reavaliada a cada retreino; o registry promove o champion por valor líquido

## Mitigação de riscos

- Recalibrar `V_RETIDO` e `C_ACAO` trimestralmente com dados reais da campanha
- Validar threshold com A/B test em produção antes de uso em massa
- Monitorar razão TP/FP mensalmente — desvio de >15% aciona revisão do threshold
