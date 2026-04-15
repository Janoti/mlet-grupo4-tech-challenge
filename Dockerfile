FROM python:3.12-slim

WORKDIR /app

# Copia apenas arquivos de dependência primeiro (melhor cache)
COPY pyproject.toml poetry.lock ./

# Estratégia: exporta requirements.txt via Poetry e instala com pip
# Isso evita o timeout do Poetry no download de pacotes grandes (nvidia-cublas ~400MB)
RUN pip install --no-cache-dir poetry==1.8.* \
    && poetry export --only main --without-hashes -f requirements.txt -o requirements.txt \
    && pip uninstall -y poetry

# Instala dependências com pip (timeout alto + retries para pacotes grandes)
RUN pip install --no-cache-dir --default-timeout=300 --retries=5 -r requirements.txt

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
