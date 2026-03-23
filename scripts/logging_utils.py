from __future__ import annotations

import logging
import os
from typing import Any


def setup_logging(level: str | None = None) -> None:
    """Configure project-wide logging for scripts with a simple consistent format."""
    resolved_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(getattr(logging, resolved_level, logging.INFO))
        return

    logging.basicConfig(
        level=getattr(logging, resolved_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_kv(logger: logging.Logger, event: str, **kwargs: Any) -> None:
    if not kwargs:
        logger.info(event)
        return
    fields = " ".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    logger.info("%s | %s", event, fields)
