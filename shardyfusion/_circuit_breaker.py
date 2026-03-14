"""Circuit breaker for S3 operations.

Tracks failure patterns across operations and fails fast when the backend
is degraded, instead of burning time on retries that will almost certainly fail.
"""

import time
from dataclasses import dataclass
from enum import Enum, unique

from .logging import get_logger, log_event
from .metrics import MetricEvent, MetricsCollector

_logger = get_logger(__name__)


@unique
class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(slots=True, frozen=True)
class CircuitBreakerConfig:
    """Configuration for the circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout_s: float = 30.0
    half_open_max_calls: int = 1


class CircuitBreaker:
    """Tracks S3 failure patterns and fails fast when backend is degraded.

    States: CLOSED (normal) -> OPEN (fail fast) -> HALF_OPEN (probe).
    """

    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        self._config = config or CircuitBreakerConfig()
        self._metrics = metrics_collector
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state, transitioning OPEN->HALF_OPEN if recovery timeout has elapsed."""
        if self._state == CircuitState.OPEN:
            if (
                time.monotonic() - self._last_failure_time
                >= self._config.recovery_timeout_s
            ):
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    def allow_request(self) -> bool:
        """Return True if a request should be allowed through."""
        current = self.state
        if current == CircuitState.CLOSED:
            return True
        if current == CircuitState.HALF_OPEN:
            if self._half_open_calls < self._config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
        # OPEN
        return False

    def record_success(self) -> None:
        """Record a successful operation. Transitions HALF_OPEN->CLOSED."""
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
            log_event("circuit_breaker_closed", logger=_logger)
            if self._metrics is not None:
                self._metrics.emit(MetricEvent.CIRCUIT_BREAKER_CLOSED, {})
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed operation. May transition CLOSED->OPEN or HALF_OPEN->OPEN."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            log_event(
                "circuit_breaker_opened",
                logger=_logger,
                reason="half_open_probe_failed",
            )
            if self._metrics is not None:
                self._metrics.emit(
                    MetricEvent.CIRCUIT_BREAKER_OPENED,
                    {"reason": "half_open_probe_failed"},
                )
        elif (
            self._state == CircuitState.CLOSED
            and self._failure_count >= self._config.failure_threshold
        ):
            self._state = CircuitState.OPEN
            log_event(
                "circuit_breaker_opened",
                logger=_logger,
                reason="failure_threshold_reached",
                failure_count=self._failure_count,
            )
            if self._metrics is not None:
                self._metrics.emit(
                    MetricEvent.CIRCUIT_BREAKER_OPENED,
                    {
                        "reason": "failure_threshold_reached",
                        "failure_count": self._failure_count,
                    },
                )
