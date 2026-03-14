"""Metric event definitions."""

from enum import Enum, unique


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
    RATE_LIMITER_DENIED = "rate_limiter.denied"

    # Circuit breaker
    CIRCUIT_BREAKER_OPENED = "circuit_breaker.opened"
    CIRCUIT_BREAKER_CLOSED = "circuit_breaker.closed"
    CIRCUIT_BREAKER_REJECTED = "circuit_breaker.rejected"
