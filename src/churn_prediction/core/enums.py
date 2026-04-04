from enum import Enum


class DatasetSplit(str, Enum):
    """Partições lógicas do conjunto de dados."""

    TRAIN = "train"
    TEST = "test"
    FULL = "full"


class ModelFamily(str, Enum):
    """Apoio a famílias modelo de alto nível."""

    BASELINE = "baseline"
    NEURAL = "neural"