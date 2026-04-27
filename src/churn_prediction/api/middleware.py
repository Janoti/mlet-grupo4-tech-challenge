"""Middleware unificado de observabilidade HTTP.

Responsabilidades (medidas em uma única passagem com `time.perf_counter`):
- Injeta header `X-Process-Time-Ms` no response (contrato consumido por clientes
  que precisam medir latência sem instrumentação própria).
- Emite log estruturado por request (campo `event=http_request`).
- Registra Counter `churn_api_http_requests_total` e Histogram
  `churn_api_http_request_duration_seconds` no Prometheus.
"""

from __future__ import annotations

import logging
import time

from fastapi import Request
from prometheus_client import Counter, Histogram

logger = logging.getLogger("churn_api")

http_requests_total = Counter(
    "churn_api_http_requests_total",
    "Total de requisições HTTP recebidas",
    labelnames=["method", "endpoint", "status_code"],
)

http_request_duration_seconds = Histogram(
    "churn_api_http_request_duration_seconds",
    "Latência das requisições HTTP em segundos",
    labelnames=["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)


async def observability_middleware(request: Request, call_next):
    """Mede latência uma vez e propaga para header, log estruturado e Prometheus."""
    start = time.perf_counter()
    endpoint = request.url.path
    method = request.method
    status_code: int | None = None
    response = None

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        duration_s = time.perf_counter() - start
        latency_ms = duration_s * 1000

        # Header HTTP só é injetado se a request não falhou (response existe).
        if response is not None:
            response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"

        # Métricas Prometheus — sempre registradas, inclusive em exceções.
        http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(
            duration_s
        )
        http_requests_total.labels(
            method=method, endpoint=endpoint, status_code=str(status_code)
        ).inc()

        # Log estruturado (formato compatível com o anterior LatencyMiddleware).
        logger.info(
            '{"event":"http_request","method":"%s","path":"%s","status":%d,"latency_ms":%.2f}',
            method,
            endpoint,
            status_code,
            latency_ms,
        )
