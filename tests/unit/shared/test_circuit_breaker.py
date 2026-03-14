"""Unit tests for the circuit breaker."""

from __future__ import annotations

from unittest.mock import patch

from shardyfusion._circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)
from shardyfusion.metrics import MetricEvent


def test_starts_closed() -> None:
    """New circuit breaker starts in CLOSED state."""
    breaker = CircuitBreaker()
    assert breaker.state == CircuitState.CLOSED


def test_allows_requests_when_closed() -> None:
    """CLOSED circuit allows all requests."""
    breaker = CircuitBreaker()
    assert breaker.allow_request() is True


def test_closed_to_open_after_threshold() -> None:
    """CLOSED → OPEN after N consecutive failures."""
    config = CircuitBreakerConfig(failure_threshold=3)
    breaker = CircuitBreaker(config=config)

    for _ in range(3):
        breaker.record_failure()

    assert breaker.state == CircuitState.OPEN


def test_open_rejects_requests() -> None:
    """OPEN circuit rejects requests immediately."""
    config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout_s=60.0)
    breaker = CircuitBreaker(config=config)

    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state == CircuitState.OPEN
    assert breaker.allow_request() is False


def test_open_to_half_open_after_timeout() -> None:
    """OPEN → HALF_OPEN after recovery timeout elapsed."""
    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout_s=10.0)
    with patch(
        "shardyfusion._circuit_breaker.time.monotonic", side_effect=fake_monotonic
    ):
        breaker = CircuitBreaker(config=config)

        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Advance past recovery timeout
        clock[0] += 11.0
        assert breaker.state == CircuitState.HALF_OPEN


def test_half_open_allows_probe() -> None:
    """HALF_OPEN allows a single probe request."""
    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    config = CircuitBreakerConfig(
        failure_threshold=2, recovery_timeout_s=5.0, half_open_max_calls=1
    )
    with patch(
        "shardyfusion._circuit_breaker.time.monotonic", side_effect=fake_monotonic
    ):
        breaker = CircuitBreaker(config=config)
        breaker.record_failure()
        breaker.record_failure()

        clock[0] += 6.0  # trigger HALF_OPEN
        assert breaker.allow_request() is True
        # Second request should be rejected
        assert breaker.allow_request() is False


def test_half_open_to_closed_on_success() -> None:
    """HALF_OPEN → CLOSED on successful probe."""
    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout_s=5.0)
    with patch(
        "shardyfusion._circuit_breaker.time.monotonic", side_effect=fake_monotonic
    ):
        breaker = CircuitBreaker(config=config)
        breaker.record_failure()
        breaker.record_failure()

        clock[0] += 6.0
        assert breaker.state == CircuitState.HALF_OPEN

        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED


def test_half_open_to_open_on_failure() -> None:
    """HALF_OPEN → OPEN on failed probe."""
    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout_s=5.0)
    with patch(
        "shardyfusion._circuit_breaker.time.monotonic", side_effect=fake_monotonic
    ):
        breaker = CircuitBreaker(config=config)
        breaker.record_failure()
        breaker.record_failure()

        clock[0] += 6.0
        assert breaker.state == CircuitState.HALF_OPEN

        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN


def test_success_resets_failure_count() -> None:
    """Success in CLOSED state resets the consecutive failure count."""
    config = CircuitBreakerConfig(failure_threshold=3)
    breaker = CircuitBreaker(config=config)

    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()  # reset
    breaker.record_failure()
    breaker.record_failure()

    # Should still be CLOSED (only 2 consecutive, not 3)
    assert breaker.state == CircuitState.CLOSED


def test_metrics_emitted_on_state_transitions() -> None:
    """Metrics are emitted when circuit breaker changes state."""
    events: list[tuple[MetricEvent, dict]] = []

    class Recorder:
        def emit(self, event: MetricEvent, payload: dict) -> None:
            events.append((event, payload))

    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout_s=5.0)
    with patch(
        "shardyfusion._circuit_breaker.time.monotonic", side_effect=fake_monotonic
    ):
        breaker = CircuitBreaker(config=config, metrics_collector=Recorder())

        # Trigger OPEN
        breaker.record_failure()
        breaker.record_failure()

        opened = [e for e in events if e[0] == MetricEvent.CIRCUIT_BREAKER_OPENED]
        assert len(opened) == 1

        # Trigger HALF_OPEN → CLOSED
        clock[0] += 6.0
        _ = breaker.state  # triggers transition
        breaker.record_success()

        closed = [e for e in events if e[0] == MetricEvent.CIRCUIT_BREAKER_CLOSED]
        assert len(closed) == 1
