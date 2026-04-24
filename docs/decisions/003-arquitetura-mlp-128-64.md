# ADR 003 — Arquitetura MLP 128→64 com BatchNorm e Dropout=0.3

**Status:** ✅ Accepted
**Data:** 2026-04-24
**Autores:** Grupo 4

## Contexto

O Tech Challenge requer uma rede neural (MLP) em PyTorch. Precisamos definir:
- Número de camadas ocultas e neurônios
- Função de ativação
- Regularização
- Normalização
- Loss function

A base tem ~101 features após one-hot encoding e 49.049 amostras de treino, com churn desbalanceado (~26%).

## Alternativas consideradas

| Arquitetura | Parâmetros | Prós | Contras |
|---|---|---|---|
| 64→32 (rasa) | ~8k | Rápida, baixo overfitting | Capacidade limitada |
| 128→64 | ~22k | Equilíbrio capacidade/overfitting | — |
| 256→128→64 (profunda) | ~55k | Alta capacidade | Alto risco de overfitting, mais lenta |
| 512→256→128 (muito profunda) | ~200k | — | Injustificável para 101 features |

## Decisão

**Arquitetura:** `input → 128 → 64 → 1`

**Componentes:**
- **Ativação:** `ReLU` nas camadas ocultas
- **Normalização:** `BatchNorm1d` após cada camada linear (estabiliza treinamento com batch_size=512)
- **Regularização:** `Dropout(0.3)` após ativação (previne co-adaptação)
- **Loss:** `BCEWithLogitsLoss` (numericamente estável, evita Sigmoid manual)
- **Otimizador:** `Adam` (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** `ReduceLROnPlateau` (factor=0.5, patience=5)
- **Early stopping:** patience=10 sobre val_loss (10% do treino separado)

## Consequências

### Positivas
- **Capacidade adequada** ao dataset: ~22k parâmetros para 101 features × 49k amostras (razão parâmetros/amostra ≈ 0,45)
- **Regularização em camadas** (BatchNorm + Dropout) reduz overfitting em dados tabulares
- **Treinamento estável** — convergência em 21 épocas com early stopping
- **BCEWithLogitsLoss** evita instabilidade numérica do Sigmoid + BCELoss separados

### Negativas
- Modelo final não superou LogReg em valor líquido (ver ADR 001)
- Dropout=0.3 pode ser agressivo; Dropout=0.2 não foi testado

### Neutras
- Seeds fixadas (`random=42`, `numpy=42`, `torch=42`, `torch.cuda=42`) garantem reprodutibilidade
- Artefato salvo via `mlflow.pytorch.log_model` — permite reload do modelo

## Próximos passos

- Testar tuning bayesiano (Optuna) sobre [hidden_size, dropout, lr, weight_decay]
- Avaliar arquiteturas alternativas: ResNet-like com skip connections, Transformer tabular (TabNet)
- Se o MLP superar LogReg em futuros retreinos, o registry promove automaticamente
