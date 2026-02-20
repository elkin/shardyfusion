"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
import traceback
from enum import Enum
from typing import Any

LOGGER = logging.getLogger("slatedb_spark_sharded")


class FailureSeverity(str, Enum):
    """Severity classification for operational failures.

    Maps to logging levels and indicates expected operator response:
      TRANSIENT  - temporary; retry may succeed (WARNING)
      ERROR      - operation failed; not retryable (ERROR)
      CRITICAL   - state inconsistency or data integrity risk (CRITICAL)
    """

    TRANSIENT = "transient"
    ERROR = "error"
    CRITICAL = "critical"


_SEVERITY_TO_LEVEL = {
    FailureSeverity.TRANSIENT: logging.WARNING,
    FailureSeverity.ERROR: logging.ERROR,
    FailureSeverity.CRITICAL: logging.CRITICAL,
}


def log_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Emit a single structured log line."""

    if not LOGGER.isEnabledFor(level):
        return
    payload = json.dumps(fields, sort_keys=True, default=str)
    LOGGER.log(level, "%s %s", event, payload)


def log_failure(
    event: str,
    *,
    severity: FailureSeverity,
    error: BaseException | None = None,
    include_traceback: bool = False,
    **fields: Any,
) -> None:
    """Emit a structured log line for a failure, using severity-based level.

    Parameters
    ----------
    event:
        Short snake_case event name (e.g. ``s3_put_transient_failure``).
    severity:
        Determines the log level and adds a ``severity`` field to the payload.
    error:
        Optional exception whose ``repr`` is added as the ``error`` field.
    include_traceback:
        When *True*, the formatted traceback is included as ``traceback``.
    **fields:
        Arbitrary key/value context appended to the JSON payload.
    """

    level = _SEVERITY_TO_LEVEL[severity]
    if not LOGGER.isEnabledFor(level):
        return

    merged: dict[str, Any] = {"severity": severity.value, **fields}
    if error is not None:
        merged["error"] = repr(error)
        if include_traceback:
            merged["traceback"] = traceback.format_exception(error)
    payload = json.dumps(merged, sort_keys=True, default=str)
    LOGGER.log(level, "%s %s", event, payload)
