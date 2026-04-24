# ADR 005 — Batch scoring como modo primário de deploy

**Status:** ✅ Accepted
**Data:** 2026-04-24
**Autores:** Grupo 4

## Contexto

O modelo de churn precisa servir predições para o time de CRM. Há duas arquiteturas típicas:

1. **Batch scoring (offline):** scores gerados periodicamente e consumidos via tabela/arquivo
2. **Real-time (online):** API REST recebe requisições e retorna score na hora

Precisamos decidir qual é o **modo primário** para priorizar infraestrutura e SLA.

### Requisitos do cliente (CRM)
- Campanhas executadas em **ciclos mensais** (não diárias)
- Volume de clientes ativos: ~50–200k
- Necessidade: tabela de priorização com faixas de risco
- Não há caso de uso imediato para score em tempo real (ex.: chamada de atendimento com decisão ao vivo)

## Alternativas consideradas

| Arquitetura | Prós | Contras |
|---|---|---|
| **Batch-only** | Infra simples, audit trail completo, custo baixo | Não atende caso real-time futuro |
| **Real-time-only** | Flexibilidade máxima | Complexidade de infra, SLO de latência, custo contínuo |
| **Batch primário + Real-time opcional** | Atende caso atual e suporta evolução | Exige manter dois fluxos |

## Decisão

**Modo primário: Batch scoring mensal.**

**Modo secundário: Real-time via FastAPI** (já implementado) para:
- Testes de integração e validação
- Eventuais chamadas de atendimento inbound
- Simulação de drift (`scripts/simulate_drift.py`)

### Arquitetura batch (proposta)
```
Base de clientes ativa (CSV ou tabela)
  → scripts/batch_score.py (a criar)
  → aplica src/churn_prediction/pipelines/
  → tabela de priorização com colunas:
     [customer_id, churn_probability, risk_level, model_version, scored_at]
  → entrega via S3/SFTP para CRM
```

### Arquitetura real-time (existente)
```
CRM frontend / atendimento
  → POST /predict (FastAPI + Docker)
  → score individual + latência < 200ms (P99)
```

## Consequências

### Positivas
- **Infra mínima para batch:** um script + scheduler (cron/Airflow)
- **Audit trail completo:** cada score tem timestamp e versão de modelo → fácil rastreabilidade
- **Custo operacional baixo:** ~1 execução/mês vs. container 24/7
- **Melhor observabilidade:** tabela histórica permite análise retrospectiva de performance
- **Permite feature lookup complexo** (agregações, janelas) sem preocupação com latência

### Negativas
- **Latência de dados:** clientes que mudam perfil entre ciclos só são rescored no próximo mês
- **Dois caminhos de código para manter** (batch + real-time) — mitigado usando o mesmo pipeline em `src/`

### Neutras
- A API FastAPI continua disponível e usada para tests, drift simulation e integrações futuras
- Se caso de uso real-time surgir (ex.: atendimento ativo), a infra já existe
- Latency middleware (`X-Process-Time-Ms`) em ambos os fluxos

## Implementação pendente

- [ ] `scripts/batch_score.py` — gera tabela de priorização para o CRM
- [ ] Agendamento mensal (cron/Airflow/CI)
- [ ] Pipeline de entrega (S3/SFTP/API externa do CRM)
- [ ] Dashboard de acompanhamento (TP, FP, valor_liquido realizado por ciclo)
