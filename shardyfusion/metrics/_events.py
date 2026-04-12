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

    # Writer retry
    SHARD_WRITE_RETRIED = "write.shard.retried"
    SHARD_WRITE_RETRY_EXHAUSTED = "write.shard.retry_exhausted"

    # Infrastructure
    S3_RETRY = "s3.retry"
    S3_RETRY_EXHAUSTED = "s3.retry_exhausted"
    RATE_LIMITER_THROTTLED = "rate_limiter.throttled"
    RATE_LIMITER_DENIED = "rate_limiter.denied"

    # Vector writer lifecycle
    VECTOR_WRITE_STARTED = "vector.write.started"
    VECTOR_SHARD_WRITE_COMPLETED = "vector.write.shard.completed"
    VECTOR_WRITE_COMPLETED = "vector.write.completed"

    # Vector reader lifecycle
    VECTOR_READER_INITIALIZED = "vector.reader.initialized"
    VECTOR_SEARCH = "vector.reader.search"
    VECTOR_SHARD_SEARCH = "vector.reader.shard_search"
    VECTOR_READER_REFRESHED = "vector.reader.refreshed"
    VECTOR_READER_CLOSED = "vector.reader.closed"
