"""Constantes e configurações compartilhadas do pipeline de churn."""

# ---------------------------------------------------------------------------
# Colunas com risco de leakage (removidas antes do treinamento)
# ---------------------------------------------------------------------------
LEAKAGE_COLS: list[str] = [
    "customer_id",
    "churn_probability",
    "retention_offer_made",
    "retention_offer_accepted",
    "contract_renewal_date",
    "loyalty_end_date",
]

# ---------------------------------------------------------------------------
# Atributos sensíveis para avaliação de fairness
# ---------------------------------------------------------------------------
SENSITIVE_FEATURES: list[str] = ["gender", "age_group", "region", "plan_type"]

# ---------------------------------------------------------------------------
# Métricas de negócio
# ---------------------------------------------------------------------------
V_RETIDO: float = 500.0   # Valor retido por churn evitado (R$)
C_ACAO: float = 50.0      # Custo da ação de retenção por cliente (R$)

# ---------------------------------------------------------------------------
# Split e reprodutibilidade
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Regras de clipping para variáveis numéricas
# ---------------------------------------------------------------------------
BUSINESS_CLIP_RULES: dict[str, tuple[float | None, float | None]] = {
    "age": (18, 100),
    "monthly_charges": (0, None),
    "avg_signal_quality": (0, 5),
    "call_drop_rate": (0, 1),
    "nps_score": (0, 10),
    "csat_score": (1, 5),
}

# ---------------------------------------------------------------------------
# Valores de pagamento válidos
# ---------------------------------------------------------------------------
VALID_PAYMENT_METHODS: set[str] = {"boleto", "debito", "card", "pix"}

# ---------------------------------------------------------------------------
# Bins para age_group
# ---------------------------------------------------------------------------
AGE_BINS: list[int] = [0, 24, 34, 44, 54, 120]
AGE_LABELS: list[str] = ["<25", "25-34", "35-44", "45-54", "55+"]

# ---------------------------------------------------------------------------
# Hiperparâmetros padrão dos modelos
# ---------------------------------------------------------------------------
LOG_REG_KWARGS: dict = {
    "max_iter": 2000,
    "solver": "liblinear",
    "random_state": RANDOM_STATE,
}

RF_KWARGS: dict = {
    "n_estimators": 200,
    "max_depth": 10,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

GB_KWARGS: dict = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
}

# ---------------------------------------------------------------------------
# Hiperparâmetros do MLP PyTorch
# ---------------------------------------------------------------------------
MLP_HIDDEN1: int = 128
MLP_HIDDEN2: int = 64
MLP_DROPOUT: float = 0.3
MLP_BATCH_SIZE: int = 512
MLP_LR: float = 1e-3
MLP_WEIGHT_DECAY: float = 1e-4
MLP_EPOCHS: int = 100
MLP_PATIENCE: int = 10
