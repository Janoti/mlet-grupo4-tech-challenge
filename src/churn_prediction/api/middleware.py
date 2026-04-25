"""Middleware Prometheus para instrumentação automática de requests HTTP."""

from __future__ import annotations

import time

from fastapi import Request
from prometheus_client import Counter, Histogram

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


async def prometheus_middleware(request: Request, call_next):
    """Registra contagem e latência de cada request HTTP."""
    start_time = time.perf_counter()
    endpoint = request.url.path
    method = request.method

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        status_code = 500
        raise
    finally:
        duration = time.perf_counter() - start_time
        http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
        http_requests_total.labels(
            method=method, endpoint=endpoint, status_code=str(status_code)
        ).inc()

    return response
