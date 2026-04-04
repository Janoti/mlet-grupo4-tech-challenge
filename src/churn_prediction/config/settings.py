from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataSettings:
    """Configuração relacionada aos dados de entrada e aos metadados principais do conjunto de dados."""

    raw_filename: str = "telecom_churn_base_extended.csv"
    target_column: str = "churn"
    id_column: str = "customer_id"


@dataclass(frozen=True)
class SplitSettings:
    """Configuração relacionada à divisão do conjunto de dados."""

    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


@dataclass(frozen=True)
class TrackingSettings:
    """Configuração relacionada ao rastreamento de experimentos."""

    experiment_name: str = "churn-prediction"


@dataclass(frozen=True)
class AppSettings:
    """Configurações de nível superior do aplicativo."""

    data: DataSettings = field(default_factory=DataSettings)
    split: SplitSettings = field(default_factory=SplitSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)


settings = AppSettings()