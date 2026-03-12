"""Token-bucket rate limiter shared by all writers.

``RateLimiter`` is the Protocol that writers depend on; ``TokenBucket`` is the
default (single-threaded) implementation.
"""

import logging
import time
from dataclasses import dataclass
from typing import Protocol

from .logging import get_logger, log_event
from .metrics import MetricEvent, MetricsCollector

_logger = get_logger(__name__)


@dataclass(slots=True)
class AcquireResult:
    """Result of a non-blocking ``try_acquire()`` call.

    *acquired* is ``True`` when tokens were successfully deducted.
    *deficit* is the number of seconds until enough tokens would be
    available; ``0.0`` when *acquired* is ``True``.
    """

    acquired: bool
    deficit: float

    def __bool__(self) -> bool:
        return self.acquired


class RateLimiter(Protocol):
    """Minimal interface consumed by all writer paths."""

    def acquire(self, tokens: int = 1) -> None: ...
    def try_acquire(self, tokens: int = 1) -> AcquireResult: ...


class TokenBucket:
    """Token bucket for rate limiting write_batch calls."""

    def __init__(
        self,
        rate: float,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        self._rate = rate  # tokens (batches) per second
        self._tokens = rate  # start full
        self._last = time.monotonic()
        self._metrics = metrics_collector

    def _replenish(self) -> None:
        """Add tokens accrued since the last call."""
        now = time.monotonic()
        self._tokens = min(self._rate, self._tokens + (now - self._last) * self._rate)
        self._last = now

    def acquire(self, tokens: int = 1) -> None:
        """Block until *tokens* are available."""

        while True:
            self._replenish()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return
            wait = (tokens - self._tokens) / self._rate
            log_event(
                "rate_limiter_throttled",
                level=logging.DEBUG,
                logger=_logger,
                wait_seconds=wait,
                tokens_requested=tokens,
            )
            if self._metrics is not None:
                self._metrics.emit(
                    MetricEvent.RATE_LIMITER_THROTTLED,
                    {
                        "wait_seconds": wait,
                    },
                )
            time.sleep(wait)

    def try_acquire(self, tokens: int = 1) -> AcquireResult:
        """Non-blocking acquire: return immediately whether tokens were available."""

        self._replenish()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return AcquireResult(acquired=True, deficit=0.0)

        deficit = (tokens - self._tokens) / self._rate
        log_event(
            "rate_limiter_denied",
            level=logging.DEBUG,
            logger=_logger,
            deficit_seconds=deficit,
            tokens_requested=tokens,
        )
        if self._metrics is not None:
            self._metrics.emit(
                MetricEvent.RATE_LIMITER_DENIED,
                {
                    "deficit_seconds": deficit,
                },
            )
        return AcquireResult(acquired=False, deficit=deficit)


# Future: ThreadSafeTokenBucket(TokenBucket) can override acquire() with a lock when needed.
