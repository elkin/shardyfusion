"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
from typing import Any

LOGGER = logging.getLogger("slatedb_spark_sharded")


def log_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Emit a single structured log line."""

    if not LOGGER.isEnabledFor(level):
        return
    payload = json.dumps(fields, sort_keys=True, default=str)
    LOGGER.log(level, "%s %s", event, payload)
