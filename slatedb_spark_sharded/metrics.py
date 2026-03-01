"""Configurable metrics collection for observability."""

from enum import Enum, unique
from typing import Any, Protocol


@unique
class MetricEvent(str, Enum):
    """Metric events emitted during write and read operations.

    Events are grouped by subsystem.  Implementations should silently
    ignore any events they do not recognise so that new events can be
    added without breaking existing collectors.
    """

    # Writer lifecycle
    WRITE_STARTED = "write.started"
    SHARDING_COMPLETED = "write.sharding_completed"
    SHARD_WRITE_STARTED = "write.shard.started"
    SHARD_WRITE_COMPLETED = "write.shard.completed"
    SHARD_WRITES_COMPLETED = "write.shards_completed"
    MANIFEST_PUBLISHED = "write.manifest_published"
    CURRENT_PUBLISHED = "write.current_published"
    WRITE_COMPLETED = "write.completed"
    BATCH_WRITTEN = "write.batch_written"

    # Reader lifecycle
    READER_INITIALIZED = "reader.initialized"
    READER_GET = "reader.get"
    READER_MULTI_GET = "reader.multi_get"
    READER_REFRESHED = "reader.refreshed"
    READER_CLOSED = "reader.closed"

    # Infrastructure
    S3_RETRY = "s3.retry"
    S3_RETRY_EXHAUSTED = "s3.retry_exhausted"
    RATE_LIMITER_THROTTLED = "rate_limiter.throttled"


class MetricsCollector(Protocol):
    """Protocol for receiving metric events.

    Implementations must be thread-safe.  The ``emit()`` method is called
    synchronously; buffer internally if blocking is a concern.
    """

    def emit(self, event: MetricEvent, payload: dict[str, Any]) -> None: ...


def emit(
    collector: MetricsCollector | None,
    event: MetricEvent,
    payload: dict[str, Any],
) -> None:
    """Emit a metric if a collector is configured.  No-op when *None*."""
    if collector is not None:
        collector.emit(event, payload)
