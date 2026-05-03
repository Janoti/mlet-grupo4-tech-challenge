"""Schemas Pydantic para validação de entrada/saída da API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Dados de entrada para predição de churn de um cliente.

    Campos obrigatórios correspondem às features do dataset de telecom.
    Campos opcionais aceitam None (tratados por imputação no pipeline).
    """

    age: float | None = Field(None, ge=0, le=120, description="Idade do cliente")
    gender: str | None = Field(None, description="Gênero (male, female, other)")
    region: str | None = Field(None, description="Região geográfica")
    plan_type: str | None = Field(None, description="Tipo de plano (pre, pos, controle, empresarial)")
    plan_price: float | None = Field(None, ge=0)
    is_promotional_plan: int | None = Field(None, ge=0, le=1)
    discount_amount: float | None = Field(None, ge=0)
    price_increase_last_3m: float | None = None
    months_to_contract_end: float | None = None
    has_loyalty: int | None = Field(None, ge=0, le=1)
    months_to_loyalty_end: float | None = None
    internet_service: str | None = None
    has_tv: int | None = Field(None, ge=0, le=1)
    has_fixed: int | None = Field(None, ge=0, le=1)
    tenure_months: float | None = Field(None, ge=0)
    network_outages_30d: float | None = Field(None, ge=0)
    avg_signal_quality: float | None = Field(None, ge=0, le=5)
    call_drop_rate: float | None = Field(None, ge=0, le=1)
    avg_internet_speed: float | None = Field(None, ge=0)
    service_failures_30d: float | None = Field(None, ge=0)
    installation_delay_days: float | None = None
    repair_visits_90d: float | None = Field(None, ge=0)
    minutes_monthly: float | None = Field(None, ge=0)
    data_gb_monthly: float | None = Field(None, ge=0)
    sms_monthly: float | None = Field(None, ge=0)
    usage_delta_pct: float | None = None
    days_since_last_usage: float | None = Field(None, ge=0)
    active_days_30d: float | None = Field(None, ge=0, le=30)
    night_usage_pct: float | None = Field(None, ge=0, le=1)
    roaming_usage: float | None = Field(None, ge=0)
    topup_frequency: float | None = Field(None, ge=0)
    avg_usage_last_3m: float | None = Field(None, ge=0)
    usage_trend_3m: float | None = None
    payment_method: str | None = None
    billing_type: str | None = None
    late_payments_6m: float | None = Field(None, ge=0)
    invoice_shock_flag: int | None = Field(None, ge=0, le=1)
    avg_bill_last_6m: float | None = Field(None, ge=0)
    bill_variation_pct: float | None = None
    collection_attempts_30d: float | None = Field(None, ge=0)
    monthly_charges: float | None = Field(None, ge=0)
    default_flag: int | None = Field(None, ge=0, le=1)
    days_past_due: float | None = Field(None, ge=0)
    support_calls_30d: float | None = Field(None, ge=0)
    support_calls_90d: float | None = Field(None, ge=0)
    complaints_30d: float | None = Field(None, ge=0)
    tickets_open_30d: float | None = Field(None, ge=0)
    last_complaint_reason: str | None = None
    resolution_time_avg: float | None = Field(None, ge=0)
    first_call_resolution_flag: int | None = Field(None, ge=0, le=1)
    transferred_calls_count: float | None = Field(None, ge=0)
    last_contact_channel: str | None = None
    app_login_30d: float | None = Field(None, ge=0)
    self_service_usage_30d: float | None = Field(None, ge=0)
    marketing_open_rate: float | None = Field(None, ge=0, le=1)
    campaign_response_flag: int | None = Field(None, ge=0, le=1)
    last_offer_accepted: str | None = None
    days_since_last_interaction: float | None = Field(None, ge=0)
    portability_request_flag: int | None = Field(None, ge=0, le=1)
    sim_swap_count: float | None = Field(None, ge=0)
    competitor_offer_contact_flag: int | None = Field(None, ge=0, le=1)
    nps_score: float | None = Field(None, ge=0, le=10)
    nps_category: str | None = None
    nps_promoter_flag: int | None = Field(None, ge=0, le=1)
    nps_detractor_flag: int | None = Field(None, ge=0, le=1)
    csat_score: float | None = Field(None, ge=1, le=5)

    model_config = {"json_schema_extra": {"examples": [
        {
            "age": 35,
            "gender": "male",
            "region": "Sudeste",
            "plan_type": "pos",
            "plan_price": 89.90,
            "tenure_months": 24,
            "monthly_charges": 95.50,
            "nps_score": 7,
            "support_calls_30d": 2,
        }
    ]}}


class PredictionResponse(BaseModel):
    """Resposta da predição de churn."""

    churn_probability: float = Field(
        ..., ge=0, le=1, description="Probabilidade de churn (0 a 1)"
    )
    churn_prediction: int = Field(
        ..., ge=0, le=1, description="Predição binária (0=não churn, 1=churn)"
    )
    risk_level: str = Field(
        ..., description="Faixa de risco (alto, medio, baixo)"
    )
    model_version: str = Field(
        ..., description="Versão/hash do modelo utilizado"
    )


class HealthResponse(BaseModel):
    """Resposta do health check."""

    status: str
    model_loaded: bool
    model_version: str | None = None


class DriftFeatureResult(BaseModel):
    """Resultado de detecção de drift para uma feature."""

    feature_name: str
    feature_type: str = Field(..., description="'numeric' ou 'categorical'")
    test_name: str = Field(..., description="Tipo de teste estatístico aplicado")
    statistic: float = Field(..., description="Valor da estatística do teste")
    p_value: float = Field(..., description="P-value do teste (rejeita H0 se < alpha)")
    drift_detected: bool = Field(..., description="True se drift foi detectado (p_value < alpha)")
    psi: float | None = Field(None, description="Population Stability Index (para numéricas)")


class DriftCheckResponse(BaseModel):
    """Resposta de um check rápido de drift."""

    timestamp: str
    total_features_checked: int
    drift_alerts: int
    drift_ratio: float = Field(..., description="Proporção de features com drift (0-1)")
    features_with_drift: list[str] = Field(
        default_factory=list,
        description="Nomes das features que apresentam drift"
    )
    recommendation: str = Field(
        ...,
        description="Recomendação de ação: 'monitor', 'investigate', ou 'retrain'"
    )


class DriftReportResponse(BaseModel):
    """Relatório completo de drift com detalhes por feature."""

    timestamp: str
    total_features: int
    drift_alerts: int
    drift_ratio: float
    features: dict[str, DriftFeatureResult] = Field(
        ...,
        description="Mapa de feature_name -> resultado do teste"
    )


class ModelVersionInfo(BaseModel):
    """Informações sobre uma versão de modelo registrada no MLflow."""

    version_id: str = Field(..., description="Run ID ou versão do modelo")
    model_name: str
    metrics: dict = Field(default_factory=dict, description="Métricas do treinamento (ROC-AUC, F1, etc)")
    params: dict = Field(default_factory=dict, description="Hiperparâmetros utilizados")
    registered_at: str | None = None
    is_champion: bool = Field(default=False, description="Se é o modelo em produção")


class ModelVersionsResponse(BaseModel):
    """Lista de versões de modelo disponíveis."""

    total_versions: int
    champion_version: str | None
    versions: list[ModelVersionInfo]


class RetargetRecommendation(BaseModel):
    """Recomendação de retreinamento baseada em critérios de negócio."""

    should_retrain: bool
    reason: str = Field(..., description="Explicação da recomendação")
    metrics_degradation: dict | None = Field(
        None,
        description="Métrica e degradação relativa (ex: {'auc': -0.05})"
    )
    last_retrain_days_ago: int | None = None
    estimated_retrain_cost: str | None = Field(None, description="Ex: 'low', 'medium', 'high'")


class FeedbackRequest(BaseModel):
    """Feedback do usuário sobre uma predição."""

    prediction_id: str = Field(..., description="ID ou hash da predição realizada")
    actual_churn: int | None = Field(
        None,
        description="Resultado real (0=não churned, 1=churned)"
    )
    feedback_type: str = Field(
        ...,
        description="Tipo de feedback: 'correct', 'incorrect', 'uncertain'"
    )
    comment: str | None = Field(None, description="Comentário adicional do usuário")
    rating: int | None = Field(None, ge=1, le=5, description="Avaliação da predição (1-5)")


class FeedbackResponse(BaseModel):
    """Confirmação de recebimento de feedback."""

    feedback_id: str
    timestamp: str
    status: str = Field(..., description="'received' ou 'error'")
    message: str = ""
