"""Token-bucket rate limiter shared by all writers."""

import threading
import time


class TokenBucket:
    """Thread-safe token bucket for rate limiting write_batch calls."""

    def __init__(self, rate: float) -> None:
        self._rate = rate  # tokens (batches) per second
        self._tokens = rate  # start full
        self._last = time.monotonic()
        self._lock = threading.Lock()

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
            time.sleep(wait)
