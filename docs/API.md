# Churn Prediction API — Documentação Completa

## Visão Geral

API FastAPI para predição de churn em clientes telecom. Fornece endpoints para:
- **Inferência**: Predições de churn com probabilidade e nível de risco
- **Monitoramento**: Detecção de data drift e alertas
- **Gerenciamento**: Versionamento de modelos e recomendações de retreinamento
- **Feedback**: Coleta de feedback para melhoria contínua

**Base URL:** `http://localhost:8000`  
**Swagger Docs:** `http://localhost:8000/docs`  
**Version:** 1.0.0

---

## Início Rápido

### 1. Iniciar a API

```bash
# Via Docker
docker compose up churn-api

# Ou local
PYTHONPATH=src poetry run uvicorn churn_prediction.api.main:app --reload
```

Acesse `http://localhost:8000/docs` para explorar endpoints interativamente.

### 2. Fazer Predição

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "gender": "male",
    "region": "Sudeste",
    "plan_type": "pos",
    "tenure_months": 24,
    "nps_score": 7,
    "support_calls_30d": 2
  }'
```

**Resposta:**
```json
{
  "churn_probability": 0.3215,
  "churn_prediction": 0,
  "risk_level": "medio",
  "model_version": "mlp_pytorch_12345678"
}
```

---

## Endpoints

### Monitoramento de Saúde

#### `GET /health`
Verifica status da API e se modelo está carregado.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "mlp_pytorch_20250430"
}
```

---

### Inferência

#### `POST /predict`
Realiza predição de churn para um cliente.

**Request Body:**
```json
{
  "age": 35,
  "gender": "male",
  "region": "Sudeste",
  "plan_type": "pos",
  "plan_price": 89.90,
  "tenure_months": 24,
  "monthly_charges": 95.50,
  "nps_score": 7,
  "support_calls_30d": 2
}
```

**Response:**
```json
{
  "churn_probability": 0.3215,
  "churn_prediction": 0,
  "risk_level": "medio",
  "model_version": "mlp_pytorch_20250430"
}
```

**Campos de Entrada Opcionais:**
- `age`: Idade (0-120)
- `gender`: 'male', 'female', 'other'
- `region`: 'Sudeste', 'Sul', 'Nordeste', etc
- `plan_type`: 'pre', 'pos', 'controle', 'empresarial'
- `tenure_months`: Meses como cliente (>= 0)
- Mais de 60 features de uso, qualidade, comportamento e satisfação

**Risk Levels:**
- `baixo`: churn_probability < 0.40 → ação: monitorar
- `medio`: 0.40 ≤ churn_probability < 0.70 → ação: contactar com oferta
- `alto`: churn_probability ≥ 0.70 → ação: escalação imediata

---

### Monitoramento de Drift

#### `POST /drift/check`
Detecta data drift em batch de dados de produção.

**Request Body:**
```json
[
  {
    "age": 35,
    "gender": "male",
    "region": "Sudeste",
    "plan_type": "pos",
    "tenure_months": 24
  },
  {
    "age": 42,
    "gender": "female",
    "region": "Sul",
    "plan_type": "pre",
    "tenure_months": 12
  }
]
```

**Response:**
```json
{
  "timestamp": "2025-05-02T10:30:00Z",
  "total_features_checked": 58,
  "drift_alerts": 8,
  "drift_ratio": 0.138,
  "features_with_drift": ["tenure_months", "avg_signal_quality", "nps_score"],
  "recommendation": "investigate"
}
```

**Recomendações:**
- `monitor`: drift_ratio < 10% → continue monitorando normalmente
- `investigate`: 10% ≤ drift_ratio < 30% → analise features com drift
- `retrain`: drift_ratio ≥ 30% → retreine o modelo urgentemente

#### `GET /drift/report`
Gera relatório detalhado com testes estatísticos por feature.

**Query Params:**
- `sample_size` (int): Número de amostras (default: 100)

**Response:**
```json
{
  "timestamp": "2025-05-02T10:30:00Z",
  "total_features": 58,
  "drift_alerts": 8,
  "drift_ratio": 0.138,
  "features": {
    "tenure_months": {
      "feature_name": "tenure_months",
      "feature_type": "numeric",
      "test_name": "kolmogorov_smirnov",
      "statistic": 0.185,
      "p_value": 0.0023,
      "drift_detected": true,
      "psi": 0.082
    },
    "nps_score": {
      "feature_name": "nps_score",
      "feature_type": "numeric",
      "test_name": "kolmogorov_smirnov",
      "statistic": 0.092,
      "p_value": 0.234,
      "drift_detected": false,
      "psi": 0.031
    }
  }
}
```

---

### Gerenciamento de Modelo

#### `GET /model/versions`
Lista histórico de versões treinadas.

**Response:**
```json
{
  "total_versions": 12,
  "champion_version": "abc12345",
  "versions": [
    {
      "version_id": "abc12345",
      "model_name": "mlp_pytorch_abc12345",
      "metrics": {
        "roc_auc": 0.847,
        "f1_score": 0.712,
        "valor_liquido": 145000
      },
      "params": {
        "hidden_dim": 64,
        "dropout": 0.3,
        "learning_rate": 0.001
      },
      "registered_at": null,
      "is_champion": true
    },
    {
      "version_id": "def67890",
      "model_name": "mlp_pytorch_def67890",
      "metrics": {
        "roc_auc": 0.821,
        "f1_score": 0.698,
        "valor_liquido": 132000
      },
      "params": {
        "hidden_dim": 32,
        "dropout": 0.2,
        "learning_rate": 0.0005
      },
      "registered_at": null,
      "is_champion": false
    }
  ]
}
```

#### `POST /model/retrain-recommendation`
Recomenda se deve retreinar baseado em drift e idade do modelo.

**Query Params:**
- `drift_ratio` (float): Resultado de `/drift/check` (0-1). Default: 0
- `days_since_retrain` (int): Dias desde último treino. Default: null

**Example:**
```bash
curl -X POST "http://localhost:8000/model/retrain-recommendation?drift_ratio=0.35&days_since_retrain=45"
```

**Response:**
```json
{
  "should_retrain": true,
  "reason": "Drift significativo detectado (35.0% das features)",
  "metrics_degradation": null,
  "last_retrain_days_ago": 45,
  "estimated_retrain_cost": "high"
}
```

---

### Feedback

#### `POST /feedback`
Registra feedback do usuário sobre uma predição.

**Request Body:**
```json
{
  "prediction_id": "abc123def456",
  "actual_churn": 1,
  "feedback_type": "incorrect",
  "comment": "Cliente churned mesmo com baixa probabilidade predicted",
  "rating": 2
}
```

**Response:**
```json
{
  "feedback_id": "fb78901a",
  "timestamp": "2025-05-02T10:35:00Z",
  "status": "received",
  "message": "Feedback registrado com sucesso"
}
```

**feedback_type:**
- `correct`: Predição acertou
- `incorrect`: Predição errou
- `uncertain`: Resultado ainda incerto

#### `GET /feedback/summary`
Retorna sumário agregado de feedback.

**Response:**
```json
{
  "total_feedback": 234,
  "accuracy": 0.782,
  "avg_rating": 3.4,
  "feedback_by_type": {
    "correct": 182,
    "incorrect": 42,
    "uncertain": 10
  }
}
```

---

### Monitoramento Prometheus

#### `GET /metrics`
Expõe métricas em formato Prometheus para scraping.

**Response (text/plain):**
```
# HELP predictions_total Total number of predictions
# TYPE predictions_total counter
predictions_total{churn_prediction="0",gender="male",plan_type="pos",risk_level="medio"} 23.0

# HELP prediction_probability Histogram of churn probabilities
# TYPE prediction_probability histogram
prediction_probability_bucket{le="0.1"} 45.0
prediction_probability_bucket{le="0.5"} 189.0
prediction_probability_bucket{le="+Inf"} 234.0
```

---

## Guias de Integração

### Frontend (React / Vue / Angular)

```javascript
// 1. Fazer predição
async function predictChurn(customerData) {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(customerData)
  });
  return response.json();
}

// 2. Enviar feedback
async function submitFeedback(predictionId, actualChurn, rating) {
  const response = await fetch('http://localhost:8000/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prediction_id: predictionId,
      actual_churn: actualChurn,
      feedback_type: actualChurn === prediction ? 'correct' : 'incorrect',
      rating: rating
    })
  });
  return response.json();
}

// 3. Verificar drift
async function checkDrift(recentData) {
  const response = await fetch('http://localhost:8000/drift/check', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(recentData)
  });
  return response.json();
}
```

### Python / Requests

```python
import requests

BASE_URL = "http://localhost:8000"

# Predição
response = requests.post(
    f"{BASE_URL}/predict",
    json={
        "age": 35,
        "gender": "male",
        "tenure_months": 24,
        "nps_score": 7
    }
)
prediction = response.json()
print(f"Churn Probability: {prediction['churn_probability']}")

# Feedback
requests.post(
    f"{BASE_URL}/feedback",
    json={
        "prediction_id": prediction.get("model_version"),
        "actual_churn": 1,
        "feedback_type": "incorrect",
        "rating": 2
    }
)
```

---

## Resolução de Problemas

### Erro: "Modelo não carregado"

**Causa:** Arquivo `models/churn_pipeline.joblib` não existe ou CHURN_MODEL_PATH está incorreto.

**Solução:**
```bash
# Treinar modelo
make notebooks-mlp

# Ou exportar modelo existente
PYTHONPATH=src poetry run python scripts/export_model.py
```

### Erro: "Referência ou produção vazios" em /drift/check

**Causa:** Dados de referência não carregados em `data/processed/train_final.csv`.

**Solução:**
```bash
# Executar pipeline completo
make run-all
```

### CORS bloqueando requisições do frontend

**Verificar:**
1. Browser console para CORS errors
2. Confirmar CORS middleware está habilitado: `GET /health` deve incluir headers `Access-Control-Allow-*`

**Teste:**
```bash
curl -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type" \
  -X OPTIONS http://localhost:8000/predict
```

---

## Roadmap de Recursos Futuros

- [ ] Explicabilidade: endpoint `/explain/{prediction_id}` com SHAP values
- [ ] Batch processing: `/predict/batch` para processar 1000+ clientes
- [ ] Model serving: A/B testing com múltiplos modelos
- [ ] Auto-retraining: Trigger de retreinamento automático baseado em drift
- [ ] Data versioning: Rastreamento de datasets com DVC
- [ ] Advanced monitoring: Detectar output drift e fairness degradation

---

## Contato & Suporte

- **Issues:** [GitHub Issues](https://github.com/grupo4-tech-challenge/issues)
- **Docs:** Veja `notebooks/` para exemplos e análises
- **MLflow:** http://localhost:5000 (histórico de treinamentos)
