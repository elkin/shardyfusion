"""Token-bucket rate limiter shared by all writers."""

import logging
import threading
import time

from .logging import get_logger, log_event
from .metrics import MetricEvent, MetricsCollector
from .metrics import emit as emit_metric

_logger = get_logger(__name__)


class TokenBucket:
    """Thread-safe token bucket for rate limiting write_batch calls."""

    def __init__(
        self,
        rate: float,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        self._rate = rate  # tokens (batches) per second
        self._tokens = rate  # start full
        self._last = time.monotonic()
        self._lock = threading.Lock()
        self._metrics = metrics_collector

    def acquire(self, tokens: int = 1) -> None:
        """Block until *tokens* are available."""

        while True:
            with self._lock:
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
            emit_metric(
                self._metrics,
                MetricEvent.RATE_LIMITER_THROTTLED,
                {
                    "wait_seconds": wait,
                },
            )
            time.sleep(wait)
