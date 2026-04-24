# Architecture Decision Records (ADRs)

Este diretório documenta decisões arquiteturais e técnicas do projeto, seguindo o padrão ADR (Architecture Decision Record) de [Michael Nygard](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).

Cada ADR descreve **uma** decisão relevante, com seu contexto, alternativas consideradas, decisão tomada e consequências esperadas.

## Formato

- **Título:** enunciado curto da decisão
- **Status:** proposed / accepted / deprecated / superseded
- **Contexto:** situação e forças que motivam a decisão
- **Decisão:** o que foi decidido
- **Consequências:** resultados positivos, negativos e neutros

## Índice

| # | Título | Status |
|---|---|---|
| [001](001-logreg-como-modelo-servido.md) | LogisticRegression como modelo servido (não MLP) | ✅ Accepted |
| [002](002-threshold-por-valor-liquido.md) | Threshold otimizado por valor líquido de negócio | ✅ Accepted |
| [003](003-arquitetura-mlp-128-64.md) | Arquitetura MLP 128→64 com BatchNorm e Dropout=0.3 | ✅ Accepted |
| [004](004-fairness-equalized-odds.md) | EqualizedOdds como restrição de fairness | ✅ Accepted |
| [005](005-batch-scoring-como-modo-primario.md) | Batch scoring como modo primário de deploy | ✅ Accepted |
