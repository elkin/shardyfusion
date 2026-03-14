"""TypedDict definitions for MetricEvent payloads.

These are type-checking aids only — the ``MetricsCollector.emit()`` Protocol
signature stays ``dict[str, Any]``.  Annotating internal ``emit()`` call sites
lets pyright catch payload mismatches at type-check time.
"""

from typing import TypedDict


class WriteStartedPayload(TypedDict):
    elapsed_ms: int


class WriteCompletedPayload(TypedDict):
    elapsed_ms: int
    rows_written: int


class ShardingCompletedPayload(TypedDict):
    elapsed_ms: int
    duration_ms: int


class ShardWriteStartedPayload(TypedDict):
    elapsed_ms: int


class ShardWriteCompletedPayload(TypedDict):
    elapsed_ms: int
    duration_ms: int
    row_count: int


class ShardWritesCompletedPayload(TypedDict):
    elapsed_ms: int
    duration_ms: int
    rows_written: int


class BatchWrittenPayload(TypedDict):
    elapsed_ms: int
    batch_size: int


class ReaderGetPayload(TypedDict):
    duration_ms: int
    found: bool


class ReaderMultiGetPayload(TypedDict):
    duration_ms: int
    num_keys: int


class ReaderRefreshedPayload(TypedDict):
    changed: bool


class ReaderClosedPayload(TypedDict):
    num_handles: int


class S3RetryPayload(TypedDict):
    attempt: int
    max_retries: int
    delay_s: float


class S3RetryExhaustedPayload(TypedDict):
    attempts: int


class RateLimiterThrottledPayload(TypedDict):
    wait_seconds: float
    limiter_type: str


class RateLimiterDeniedPayload(TypedDict):
    deficit_seconds: float
    limiter_type: str


class CircuitBreakerOpenedPayload(TypedDict, total=False):
    reason: str
    failure_count: int


class CircuitBreakerClosedPayload(TypedDict):
    pass


class CircuitBreakerRejectedPayload(TypedDict):
    pass
