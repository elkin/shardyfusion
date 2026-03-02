"""Structured logging helpers."""

import logging
import traceback
from enum import Enum
from typing import Any

LOGGER = logging.getLogger("shardyfusion")

_SEVERITY_TO_LOG_LEVEL: dict[str, int] = {
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "FATAL": logging.CRITICAL,
}


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


def get_logger(name: str) -> logging.Logger:
    """Return a logger in the ``shardyfusion`` hierarchy.

    Strips the package prefix when *name* already starts with it so that
    callers can pass ``__name__`` directly::

        _logger = get_logger(__name__)
    """
    prefix = "shardyfusion."
    suffix = name[len(prefix) :] if name.startswith(prefix) else name
    return logging.getLogger(f"shardyfusion.{suffix}")


def log_event(
    event: str,
    *,
    level: int = logging.INFO,
    logger: logging.Logger | None = None,
    **fields: Any,
) -> None:
    """Emit a single structured log line.

    Structured fields are attached via ``extra={"slatedb": fields}`` so
    that handlers/formatters can access them without parsing the message
    string.  The default stdlib formatter simply shows the *event* name.
    """
    _log = logger or LOGGER
    if not _log.isEnabledFor(level):
        return
    _log.log(level, event, extra={"slatedb": fields})


def log_failure(
    event: str,
    *,
    severity: FailureSeverity,
    error: BaseException | None = None,
    include_traceback: bool = False,
    logger: logging.Logger | None = None,
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
    logger:
        Explicit logger instance; falls back to the package root logger.
    **fields:
        Arbitrary key/value context attached to the ``extra["slatedb"]`` dict.
    """
    _log = logger or LOGGER
    level = _SEVERITY_TO_LEVEL[severity]
    if not _log.isEnabledFor(level):
        return

    merged: dict[str, Any] = {"severity": severity.value, **fields}
    if error is not None:
        merged["error"] = repr(error)
        if include_traceback:
            merged["traceback"] = traceback.format_exception(error)
    _log.log(level, event, extra={"slatedb": merged})
