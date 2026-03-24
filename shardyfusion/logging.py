"""Structured logging helpers."""

import contextvars
import json
import logging
import traceback
from enum import Enum
from typing import Any

LOGGER = logging.getLogger("shardyfusion")

_log_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "shardyfusion_log_context",
)


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

    Structured fields are attached via ``extra={"shardyfusion": fields}`` so
    that handlers/formatters can access them without parsing the message
    string.  The default stdlib formatter simply shows the *event* name.
    """
    _log = logger or LOGGER
    if not _log.isEnabledFor(level):
        return
    merged = {**_log_context.get({}), **fields}
    _log.log(level, event, extra={"shardyfusion": merged})


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
        Arbitrary key/value context attached to the ``extra["shardyfusion"]`` dict.
    """
    _log = logger or LOGGER
    level = _SEVERITY_TO_LEVEL[severity]
    if not _log.isEnabledFor(level):
        return

    merged: dict[str, Any] = {
        **_log_context.get({}),
        "severity": severity.value,
        **fields,
    }
    if error is not None:
        merged["error"] = repr(error)
        if include_traceback:
            merged["traceback"] = traceback.format_exception(error)
    _log.log(level, event, extra={"shardyfusion": merged})


class LogContext:
    """Context manager that binds fields to all log calls within its scope.

    Works correctly with both threading and asyncio — ``asyncio.create_task()``
    inherits the parent's context automatically via contextvars.

    Example::

        with LogContext(run_id="abc", writer_type="spark"):
            log_event("write_started", logger=_logger)
            # -> extra includes run_id="abc", writer_type="spark"

    Nesting is supported; inner contexts override outer ones for overlapping keys.
    """

    def __init__(self, **fields: Any) -> None:
        self._fields = fields
        self._token: contextvars.Token[dict[str, Any]] | None = None

    def __enter__(self) -> "LogContext":
        current = _log_context.get({})
        self._token = _log_context.set({**current, **self._fields})
        return self

    def __exit__(self, *args: object) -> None:
        if self._token is not None:
            _log_context.reset(self._token)


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON.

    Flattens ``extra["shardyfusion"]`` fields into the top-level JSON object alongside
    standard fields (``timestamp``, ``level``, ``logger``, ``event``).
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        sf_fields = getattr(record, "shardyfusion", None)
        if isinstance(sf_fields, dict):
            entry.update(sf_fields)
        return json.dumps(entry, default=str)


def configure_logging(
    level: int = logging.INFO,
    json_format: bool = False,
    stream: Any = None,
) -> None:
    """Configure the ``shardyfusion.*`` logger hierarchy.

    This is a convenience for users — libraries must not call this automatically.

    Args:
        level: Log level for the shardyfusion logger hierarchy.
        json_format: If True, use JsonFormatter for JSON output.
        stream: Output stream (default: sys.stderr).
    """
    import sys

    logger = logging.getLogger("shardyfusion")
    logger.setLevel(level)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))

    # Remove existing handlers to avoid duplicate output
    logger.handlers.clear()
    logger.addHandler(handler)
