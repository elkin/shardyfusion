"""Dedicated unit tests for TokenBucket rate limiter."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

from shardyfusion._rate_limiter import TokenBucket


def test_acquire_immediate_when_full() -> None:
    """Full bucket acquires without delay."""
    bucket = TokenBucket(rate=100.0)

    start = time.monotonic()
    bucket.acquire(1)
    elapsed = time.monotonic() - start

    assert elapsed < 0.5


def test_acquire_blocks_when_empty() -> None:
    """Empty bucket blocks — verified via mocked time, no real sleep."""
    sleeps: list[float] = []
    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    def fake_sleep(secs: float) -> None:
        sleeps.append(secs)
        clock[0] += secs

    with (
        patch("shardyfusion._rate_limiter.time.monotonic", side_effect=fake_monotonic),
        patch("shardyfusion._rate_limiter.time.sleep", side_effect=fake_sleep),
    ):
        bucket = TokenBucket(rate=10.0)  # 10 tokens/sec, starts full

        # Drain all tokens
        for _ in range(10):
            bucket.acquire(1)

        # Next acquire should trigger a sleep
        bucket.acquire(1)

    assert len(sleeps) >= 1
    assert sleeps[0] > 0


def test_tokens_replenish_over_time() -> None:
    """After time passes, tokens become available again — mocked time."""
    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    def fake_sleep(secs: float) -> None:
        clock[0] += secs

    with (
        patch("shardyfusion._rate_limiter.time.monotonic", side_effect=fake_monotonic),
        patch("shardyfusion._rate_limiter.time.sleep", side_effect=fake_sleep),
    ):
        bucket = TokenBucket(rate=10.0)

        # Drain all tokens
        for _ in range(10):
            bucket.acquire(1)

        # Simulate 0.2s passing → ~2 tokens replenished
        clock[0] += 0.2

        # Should acquire immediately (no sleep needed)
        bucket.acquire(1)


def test_multi_token_acquire() -> None:
    """acquire(n) with n > 1 works correctly — mocked time."""
    sleeps: list[float] = []
    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    def fake_sleep(secs: float) -> None:
        sleeps.append(secs)
        clock[0] += secs

    with (
        patch("shardyfusion._rate_limiter.time.monotonic", side_effect=fake_monotonic),
        patch("shardyfusion._rate_limiter.time.sleep", side_effect=fake_sleep),
    ):
        bucket = TokenBucket(rate=10.0)

        # Acquire all tokens at once — should be immediate (bucket starts full)
        bucket.acquire(10)
        assert len(sleeps) == 0

        # Next single acquire should block
        bucket.acquire(1)
        assert len(sleeps) >= 1


def test_sequential_acquires_drain_bucket() -> None:
    """Repeated acquires deplete the bucket, then block — mocked time."""
    sleeps: list[float] = []
    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    def fake_sleep(secs: float) -> None:
        sleeps.append(secs)
        clock[0] += secs

    with (
        patch("shardyfusion._rate_limiter.time.monotonic", side_effect=fake_monotonic),
        patch("shardyfusion._rate_limiter.time.sleep", side_effect=fake_sleep),
    ):
        bucket = TokenBucket(rate=5.0)

        # First 5 acquires should be fast (bucket starts full)
        for _ in range(5):
            bucket.acquire(1)
        assert len(sleeps) == 0

        # 6th acquire should block
        bucket.acquire(1)
        assert len(sleeps) >= 1


def test_concurrent_acquires_are_thread_safe() -> None:
    """10 threads doing concurrent acquires complete without deadlock."""
    rate = 100.0
    bucket = TokenBucket(rate=rate)
    errors: list[Exception] = []
    barrier = threading.Barrier(10)

    def worker() -> None:
        try:
            barrier.wait(timeout=5.0)
            for _ in range(5):
                bucket.acquire(1)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors
    assert all(not t.is_alive() for t in threads)


def test_high_rate_no_meaningful_delay() -> None:
    """High-rate bucket (100k/s) introduces no overhead."""
    bucket = TokenBucket(rate=100_000.0)

    start = time.monotonic()
    for _ in range(100):
        bucket.acquire(1)
    elapsed = time.monotonic() - start

    assert elapsed < 0.5


def test_metrics_emitted_on_throttle() -> None:
    """MetricsCollector receives RATE_LIMITER_THROTTLED events."""
    from shardyfusion.metrics import MetricEvent

    events: list[tuple[MetricEvent, dict]] = []

    class Recorder:
        def emit(self, event: MetricEvent, payload: dict) -> None:
            events.append((event, payload))

    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    def fake_sleep(secs: float) -> None:
        clock[0] += secs

    with (
        patch("shardyfusion._rate_limiter.time.monotonic", side_effect=fake_monotonic),
        patch("shardyfusion._rate_limiter.time.sleep", side_effect=fake_sleep),
    ):
        bucket = TokenBucket(rate=5.0, metrics_collector=Recorder())

        # Drain
        for _ in range(5):
            bucket.acquire(1)

        # Trigger throttle
        bucket.acquire(1)

    throttle_events = [e for e in events if e[0] == MetricEvent.RATE_LIMITER_THROTTLED]
    assert len(throttle_events) >= 1
    assert throttle_events[0][1]["wait_seconds"] > 0
