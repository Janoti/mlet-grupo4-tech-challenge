FROM python:3.12-slim

WORKDIR /app

# Dependências de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia apenas arquivos de dependência primeiro (melhor cache)
COPY pyproject.toml poetry.lock ./

# Instala poetry e dependências (sem dev)
RUN pip install --no-cache-dir poetry==1.8.* \
    && poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi \
    && pip uninstall -y poetry

# Copia código fonte
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Variáveis de ambiente
ENV CHURN_MODEL_PATH=/app/models/churn_pipeline.joblib
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8000/health'); assert r.status_code==200"

# Uvicorn com 2 workers (ajustar conforme recursos)
CMD ["python", "-m", "uvicorn", "churn_prediction.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
