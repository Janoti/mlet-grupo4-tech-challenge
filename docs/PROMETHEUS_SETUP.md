# Prometheus & Grafana — Setup de Monitoramento

Guia para configurar Prometheus e Grafana para monitorar a API de Churn Prediction.

## Métricas Expostas

A API expõe métricas Prometheus no endpoint `GET /metrics`:

| Métrica | Tipo | Descrição |
|---------|------|-----------|
| `churn_api_http_requests_total` | Counter | Total de requisições HTTP (labels: method, endpoint, status_code) |
| `churn_api_http_request_duration_seconds` | Histogram | Latência das requisições em segundos (labels: method, endpoint) |
| `churn_api_predictions_total` | Counter | Total de predições realizadas (label: risk_level) |
| `churn_api_model_loaded` | Gauge | Modelo carregado: 1=sim, 0=não |
| `churn_api_model_info` | Info | Versão do modelo ativo |

## 1. Rodando a API Localmente

```bash
# Instalar dependências
poetry install

# Executar pipeline completo (limpa mlruns, roda notebooks, analisa)
make run-all

# Exportar modelo treinado
PYTHONPATH=src poetry run python scripts/export_model.py

# Iniciar API (porta configurável via SERVER_PORT)
PYTHONPATH=src poetry run uvicorn churn_prediction.api.main:app \
  --host 0.0.0.0 --port ${SERVER_PORT:-8000} --reload
```

Ou via Docker:

```bash
docker compose up --build churn-api
```

### Verificar métricas

```bash
curl -s http://localhost:8000/metrics | head -30
```

## 2. Formato da Saída `/metrics`

O endpoint retorna métricas no formato texto do Prometheus:

```
# HELP churn_api_http_requests_total Total de requisições HTTP recebidas
# TYPE churn_api_http_requests_total counter
churn_api_http_requests_total{endpoint="/health",method="GET",status_code="200"} 5.0
churn_api_http_requests_total{endpoint="/predict",method="POST",status_code="200"} 12.0

# HELP churn_api_http_request_duration_seconds Latência das requisições HTTP em segundos
# TYPE churn_api_http_request_duration_seconds histogram
churn_api_http_request_duration_seconds_bucket{endpoint="/predict",method="POST",le="0.05"} 8.0
churn_api_http_request_duration_seconds_bucket{endpoint="/predict",method="POST",le="0.1"} 11.0
churn_api_http_request_duration_seconds_bucket{endpoint="/predict",method="POST",le="+Inf"} 12.0
churn_api_http_request_duration_seconds_sum{endpoint="/predict",method="POST"} 0.874
churn_api_http_request_duration_seconds_count{endpoint="/predict",method="POST"} 12.0

# HELP churn_api_predictions_total Total de predições realizadas
# TYPE churn_api_predictions_total counter
churn_api_predictions_total{risk_level="alto"} 4.0
churn_api_predictions_total{risk_level="medio"} 5.0
churn_api_predictions_total{risk_level="baixo"} 3.0

# HELP churn_api_model_loaded Indica se o modelo está carregado (1=sim, 0=não)
# TYPE churn_api_model_loaded gauge
churn_api_model_loaded 1.0
```

## 3. Configuração do Prometheus

### prometheus.yml (scrape config)

Um exemplo completo está em [`examples/prometheus.yml`](../examples/prometheus.yml):

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'churn-api'
    metrics_path: '/metrics'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
        labels:
          service: 'churn-prediction'
```

### Rodando Prometheus via Docker

```bash
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/examples/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus:latest
```

> **Nota Docker:** Se a API roda no host, use `host.docker.internal:8000` como target em vez de `localhost:8000`.

Acesse a UI em http://localhost:9090 e verifique em **Status → Targets** que o job `churn-api` está com status **UP**.

## 4. Queries PromQL para Grafana

### Adicionar Data Source

1. Grafana → **Configuration → Data Sources → Add → Prometheus**
2. URL: `http://localhost:9090`
3. **Save & Test**

### Painéis Recomendados

#### Throughput (requisições por segundo)

```promql
rate(churn_api_http_requests_total[1m])
```

Para agrupar por endpoint:

```promql
sum(rate(churn_api_http_requests_total[1m])) by (endpoint)
```

#### Latência P95

```promql
histogram_quantile(0.95, rate(churn_api_http_request_duration_seconds_bucket[5m]))
```

#### Latência Média

```promql
rate(churn_api_http_request_duration_seconds_sum[5m])
  / rate(churn_api_http_request_duration_seconds_count[5m])
```

#### Distribuição de Predições por Risco

```promql
sum(rate(churn_api_predictions_total[5m])) by (risk_level)
```

#### Proporção de Alto Risco

```promql
sum(rate(churn_api_predictions_total{risk_level="alto"}[5m]))
  / sum(rate(churn_api_predictions_total[5m]))
```

#### Taxa de Erros (5xx)

```promql
rate(churn_api_http_requests_total{status_code=~"5.."}[1m])
```

#### Status do Modelo

```promql
churn_api_model_loaded
```

## 5. Variáveis de Ambiente

| Variável | Default | Descrição |
|----------|---------|-----------|
| `SERVER_PORT` | `8000` | Porta do servidor |
| `CHURN_MODEL_PATH` | `models/churn_pipeline.joblib` | Caminho do modelo serializado |
| `LOG_LEVEL` | `INFO` | Nível de logging |

## 6. Troubleshooting

| Problema | Solução |
|----------|---------|
| `/metrics` retorna 404 | Verificar se API está rodando (`curl localhost:8000/health`) |
| Prometheus target DOWN | Verificar IP/porta; em Docker use `host.docker.internal` |
| Métricas não aparecem no Grafana | Esperar 1-2 scrape_intervals; verificar data source no Grafana |
| Contador não incrementa | Gerar tráfego: `curl localhost:8000/health` e verificar novamente |
