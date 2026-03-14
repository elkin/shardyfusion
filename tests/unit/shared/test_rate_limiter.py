"""Dedicated unit tests for TokenBucket rate limiter."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from shardyfusion._rate_limiter import AcquireResult, TokenBucket


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


def test_token_bucket_satisfies_rate_limiter_protocol() -> None:
    """TokenBucket structurally satisfies the RateLimiter protocol."""
    from shardyfusion._rate_limiter import RateLimiter

    bucket: RateLimiter = TokenBucket(rate=10.0)
    bucket.acquire(1)
    bucket.acquire()

    result = bucket.try_acquire(1)
    assert isinstance(result, AcquireResult)
    result2 = bucket.try_acquire()
    assert isinstance(result2, AcquireResult)


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


# ---------------------------------------------------------------------------
# try_acquire tests
# ---------------------------------------------------------------------------


def test_try_acquire_succeeds_when_full() -> None:
    """Full bucket try_acquire returns success with zero deficit."""
    clock = [0.0]

    with patch("shardyfusion._rate_limiter.time.monotonic", return_value=clock[0]):
        bucket = TokenBucket(rate=10.0)

    with patch("shardyfusion._rate_limiter.time.monotonic", return_value=clock[0]):
        result = bucket.try_acquire(1)

    assert result.acquired is True
    assert result.deficit == 0.0
    assert bool(result) is True


def test_try_acquire_fails_when_empty() -> None:
    """Drained bucket try_acquire returns failure with positive deficit."""
    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    with patch("shardyfusion._rate_limiter.time.monotonic", side_effect=fake_monotonic):
        bucket = TokenBucket(rate=5.0)

        # Drain all tokens
        for _ in range(5):
            bucket.try_acquire(1)

        # Now should fail
        result = bucket.try_acquire(1)

    assert result.acquired is False
    assert result.deficit > 0
    assert bool(result) is False


def test_try_acquire_does_not_block() -> None:
    """try_acquire never sleeps, even when tokens are unavailable."""
    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    with (
        patch("shardyfusion._rate_limiter.time.monotonic", side_effect=fake_monotonic),
        patch("shardyfusion._rate_limiter.time.sleep") as mock_sleep,
    ):
        bucket = TokenBucket(rate=5.0)

        # Drain
        for _ in range(5):
            bucket.try_acquire(1)

        # Denied — should NOT sleep
        bucket.try_acquire(1)

    mock_sleep.assert_not_called()


def test_try_acquire_deducts_tokens() -> None:
    """Successful try_acquire calls deduct tokens from the bucket."""
    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    with patch("shardyfusion._rate_limiter.time.monotonic", side_effect=fake_monotonic):
        bucket = TokenBucket(rate=2.0)

        # Two should succeed
        r1 = bucket.try_acquire(1)
        r2 = bucket.try_acquire(1)
        assert r1.acquired is True
        assert r2.acquired is True

        # Third should fail
        r3 = bucket.try_acquire(1)
        assert r3.acquired is False


def test_try_acquire_deficit_accuracy() -> None:
    """Deficit should approximate the time needed for 1 token at the given rate."""
    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    with patch("shardyfusion._rate_limiter.time.monotonic", side_effect=fake_monotonic):
        bucket = TokenBucket(rate=10.0)

        # Drain all 10 tokens
        bucket.try_acquire(10)

        # 1 token at rate=10 → deficit ≈ 0.1s
        result = bucket.try_acquire(1)
        assert result.acquired is False
        assert abs(result.deficit - 0.1) < 0.01


def test_try_acquire_replenishes_after_time() -> None:
    """After time passes, try_acquire succeeds again."""
    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    with patch("shardyfusion._rate_limiter.time.monotonic", side_effect=fake_monotonic):
        bucket = TokenBucket(rate=10.0)

        # Drain
        bucket.try_acquire(10)
        assert bucket.try_acquire(1).acquired is False

        # Advance clock by 0.5s → 5 tokens replenished
        clock[0] += 0.5
        result = bucket.try_acquire(1)
        assert result.acquired is True


def test_try_acquire_metrics_on_denial() -> None:
    """RATE_LIMITER_DENIED is emitted when try_acquire fails."""
    from shardyfusion.metrics import MetricEvent

    events: list[tuple[MetricEvent, dict]] = []

    class Recorder:
        def emit(self, event: MetricEvent, payload: dict) -> None:
            events.append((event, payload))

    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    with patch("shardyfusion._rate_limiter.time.monotonic", side_effect=fake_monotonic):
        bucket = TokenBucket(rate=5.0, metrics_collector=Recorder())

        # Drain
        for _ in range(5):
            bucket.try_acquire(1)

        # Trigger denial
        bucket.try_acquire(1)

    denied_events = [e for e in events if e[0] == MetricEvent.RATE_LIMITER_DENIED]
    assert len(denied_events) == 1
    assert denied_events[0][1]["deficit_seconds"] > 0


def test_try_acquire_no_metrics_on_success() -> None:
    """No RATE_LIMITER_DENIED metric is emitted on successful try_acquire."""
    from shardyfusion.metrics import MetricEvent

    events: list[tuple[MetricEvent, dict]] = []

    class Recorder:
        def emit(self, event: MetricEvent, payload: dict) -> None:
            events.append((event, payload))

    clock = [0.0]

    with patch("shardyfusion._rate_limiter.time.monotonic", return_value=clock[0]):
        bucket = TokenBucket(rate=10.0, metrics_collector=Recorder())

    with patch("shardyfusion._rate_limiter.time.monotonic", return_value=clock[0]):
        bucket.try_acquire(1)

    denied_events = [e for e in events if e[0] == MetricEvent.RATE_LIMITER_DENIED]
    assert len(denied_events) == 0


# ---------------------------------------------------------------------------
# acquire_async tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acquire_async_immediate_when_full() -> None:
    """Full bucket acquire_async returns without sleeping."""

    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    sleeps: list[float] = []

    async def fake_async_sleep(secs: float) -> None:
        sleeps.append(secs)
        clock[0] += secs

    with (
        patch("shardyfusion._rate_limiter.time.monotonic", side_effect=fake_monotonic),
        patch("asyncio.sleep", side_effect=fake_async_sleep),
    ):
        bucket = TokenBucket(rate=10.0)
        await bucket.acquire_async(1)

    assert len(sleeps) == 0


@pytest.mark.asyncio
async def test_acquire_async_sleeps_on_deficit() -> None:
    """Drained bucket acquire_async sleeps via asyncio.sleep."""

    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    sleeps: list[float] = []

    async def fake_async_sleep(secs: float) -> None:
        sleeps.append(secs)
        clock[0] += secs

    with (
        patch("shardyfusion._rate_limiter.time.monotonic", side_effect=fake_monotonic),
        patch("asyncio.sleep", side_effect=fake_async_sleep),
    ):
        bucket = TokenBucket(rate=5.0)

        # Drain
        for _ in range(5):
            bucket.try_acquire(1)

        # Should trigger async sleep
        await bucket.acquire_async(1)

    assert len(sleeps) >= 1
    assert sleeps[0] > 0


@pytest.mark.asyncio
async def test_acquire_async_never_calls_time_sleep() -> None:
    """acquire_async must never use time.sleep — only asyncio.sleep."""

    clock = [0.0]

    def fake_monotonic() -> float:
        return clock[0]

    async def fake_async_sleep(secs: float) -> None:
        clock[0] += secs

    with (
        patch("shardyfusion._rate_limiter.time.monotonic", side_effect=fake_monotonic),
        patch("shardyfusion._rate_limiter.time.sleep") as mock_sync_sleep,
        patch("asyncio.sleep", side_effect=fake_async_sleep),
    ):
        bucket = TokenBucket(rate=5.0)

        # Drain
        for _ in range(5):
            bucket.try_acquire(1)

        await bucket.acquire_async(1)

    mock_sync_sleep.assert_not_called()
