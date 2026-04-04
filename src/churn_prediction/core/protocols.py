from pathlib import Path
from typing import Protocol

import pandas as pd


class DataFrameTransformer(Protocol):
    """Protocolo para transformadores baseados em dataframes."""

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "DataFrameTransformer":
        ...

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


class Trainable(Protocol):
    """Protocolo para serviços de modelos treináveis."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        ...


class ArtifactWriter(Protocol):
    """Protocolo para classes que persistem artefatos."""

    def save(self, output_dir: Path) -> None:
        ...