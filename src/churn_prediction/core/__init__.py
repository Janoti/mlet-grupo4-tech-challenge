"""Abstrações e utilitários compartilhados utilizados em todo o projeto."""

from .enums import DatasetSplit, ModelFamily
from .paths import ProjectPaths, paths

__all__ = [
    "DatasetSplit",
    "ModelFamily",
    "ProjectPaths",
    "paths",
]