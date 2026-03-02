"""Dedicated unit tests for TokenBucket rate limiter."""

from __future__ import annotations

import threading
import time

from shardyfusion._rate_limiter import TokenBucket


def test_acquire_immediate_when_full() -> None:
    """Full bucket acquires without delay."""
    bucket = TokenBucket(rate=100.0)

    start = time.monotonic()
    bucket.acquire(1)
    elapsed = time.monotonic() - start

    assert elapsed < 0.1


def test_acquire_blocks_when_empty() -> None:
    """Empty bucket blocks for ~1/rate seconds per token."""
    rate = 10.0  # 10 tokens/sec → ~0.1s per token
    bucket = TokenBucket(rate=rate)

    # Drain all tokens
    for _ in range(int(rate)):
        bucket.acquire(1)

    # Next acquire should block
    start = time.monotonic()
    bucket.acquire(1)
    elapsed = time.monotonic() - start

    assert 0.05 <= elapsed <= 0.5


def test_tokens_replenish_over_time() -> None:
    """After sleeping, tokens become available again."""
    rate = 10.0
    bucket = TokenBucket(rate=rate)

    # Drain all tokens
    for _ in range(int(rate)):
        bucket.acquire(1)

    # Wait for replenishment (~0.2s → ~2 tokens)
    time.sleep(0.2)

    # Should acquire immediately (tokens replenished)
    start = time.monotonic()
    bucket.acquire(1)
    elapsed = time.monotonic() - start

    assert elapsed < 0.05


def test_multi_token_acquire() -> None:
    """acquire(n) with n > 1 works correctly."""
    rate = 10.0
    bucket = TokenBucket(rate=rate)

    # Acquire all tokens at once
    start = time.monotonic()
    bucket.acquire(int(rate))
    elapsed = time.monotonic() - start

    # Should be immediate (bucket starts full)
    assert elapsed < 0.1

    # Next single acquire should block
    start = time.monotonic()
    bucket.acquire(1)
    elapsed = time.monotonic() - start

    assert 0.05 <= elapsed <= 0.5


def test_sequential_acquires_drain_bucket() -> None:
    """Repeated acquires deplete the bucket, then block."""
    rate = 5.0  # 5 tokens/sec
    bucket = TokenBucket(rate=rate)

    # First 5 acquires should be fast (bucket starts full)
    start = time.monotonic()
    for _ in range(int(rate)):
        bucket.acquire(1)
    fast_elapsed = time.monotonic() - start

    assert fast_elapsed < 0.1

    # 6th acquire should block
    start = time.monotonic()
    bucket.acquire(1)
    slow_elapsed = time.monotonic() - start

    assert slow_elapsed >= 0.05


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

    assert elapsed < 0.1
