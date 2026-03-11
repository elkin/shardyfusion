"""Token-bucket rate limiter shared by all writers.

``RateLimiter`` is the Protocol that writers depend on; ``TokenBucket`` is the
default (single-threaded) implementation.
"""

import logging
import time
from typing import Protocol

from .logging import get_logger, log_event
from .metrics import MetricEvent, MetricsCollector

_logger = get_logger(__name__)


class RateLimiter(Protocol):
    """Minimal interface consumed by all writer paths."""

    def acquire(self, tokens: int = 1) -> None: ...


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

    def acquire(self, tokens: int = 1) -> None:
        """Block until *tokens* are available."""

        while True:
            now = time.monotonic()
            self._tokens = min(
                self._rate, self._tokens + (now - self._last) * self._rate
            )
            self._last = now
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


# Future: ThreadSafeTokenBucket(TokenBucket) can override acquire() with a lock when needed.
