"""Unit tests for shardyfusion._async_bridge.

These tests exercise the typed sync -> async bridge surface
(:func:`call_sync`, :func:`gather_sync`,
:class:`BridgeTimeoutError`) without requiring slatedb. The bridge
loop is the real shardyfusion-owned daemon-thread loop because we want
to verify cross-thread submission, timeout-driven cancellation, and
fan-out behavior end-to-end. The loop is process-global and lazy so
running this module alongside slatedb tests is safe.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from shardyfusion._async_bridge import call_sync, gather_sync
from shardyfusion.errors import BridgeTimeoutError, ShardyfusionError

# ---------------------------------------------------------------------------
# call_sync
# ---------------------------------------------------------------------------


def test_call_sync_returns_coroutine_result() -> None:
    async def _add(x: int, y: int) -> int:
        return x + y

    assert call_sync(_add(2, 3)) == 5


def test_call_sync_propagates_coroutine_exception() -> None:
    class BoomError(RuntimeError):
        pass

    async def _boom() -> None:
        raise BoomError("nope")

    with pytest.raises(BoomError, match="nope"):
        call_sync(_boom())


def test_call_sync_timeout_raises_bridge_timeout_error() -> None:
    """Timeout maps to BridgeTimeoutError (retryable, ShardyfusionError)."""

    async def _slow() -> None:
        await asyncio.sleep(5.0)

    with pytest.raises(BridgeTimeoutError) as excinfo:
        call_sync(_slow(), timeout=0.05)

    assert excinfo.value.retryable is True
    assert isinstance(excinfo.value, ShardyfusionError)
    assert "timeout=0.05" in str(excinfo.value)


def test_call_sync_timeout_cancels_coroutine_on_loop() -> None:
    """A timed-out coroutine must observe CancelledError on the loop.

    Without ``loop.call_soon_threadsafe(future.cancel)`` the bridge
    would happily keep driving the work after the caller gave up,
    leading to half-done writes and resource leaks.
    """
    cancelled = threading.Event()
    started = threading.Event()

    async def _watch() -> None:
        started.set()
        try:
            await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with pytest.raises(BridgeTimeoutError):
        call_sync(_watch(), timeout=0.05)

    # Give the loop a brief moment to deliver the cancellation; this is
    # not a wall-clock-sensitive assertion since we set an unbounded
    # wait below.
    assert started.wait(timeout=1.0), "coroutine never started"
    assert cancelled.wait(timeout=1.0), (
        "timed-out coroutine was not cancelled on the bridge loop"
    )


def test_call_sync_no_timeout_blocks_until_completion() -> None:
    """timeout=None preserves the unbounded ``run_coro`` semantics."""

    async def _delayed() -> str:
        await asyncio.sleep(0.05)
        return "done"

    assert call_sync(_delayed(), timeout=None) == "done"


def test_call_sync_runs_concurrently_across_threads() -> None:
    """Distinct threads can submit to the bridge in parallel.

    Sanity-check that the bridge does not serialize unrelated callers
    \u2014 two coroutines that each sleep 0.1s should complete in well
    under 0.2s when launched from two threads.
    """
    results: list[float] = []
    lock = threading.Lock()

    async def _sleep_then_return(value: float) -> float:
        await asyncio.sleep(0.1)
        return value

    def _worker(value: float) -> None:
        out = call_sync(_sleep_then_return(value))
        with lock:
            results.append(out)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(4)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start

    assert sorted(results) == [0.0, 1.0, 2.0, 3.0]
    # Generous bound \u2014 serial would be 0.4s; concurrent should be ~0.1s
    # plus bridge overhead. Use 0.3s to avoid CI flake.
    assert elapsed < 0.3, (
        f"bridge appears to serialize callers (elapsed={elapsed:.3f}s)"
    )


# ---------------------------------------------------------------------------
# gather_sync
# ---------------------------------------------------------------------------


def test_gather_sync_returns_results_in_submission_order() -> None:
    async def _identity(x: int) -> int:
        await asyncio.sleep(0)
        return x

    out = gather_sync([_identity(i) for i in range(5)])
    assert out == [0, 1, 2, 3, 4]


def test_gather_sync_empty_returns_empty_list() -> None:
    assert gather_sync([]) == []


def test_gather_sync_runs_in_parallel_on_one_bridge_hop() -> None:
    """Four 0.1s sleeps gathered should complete in ~0.1s, not 0.4s."""

    async def _sleep() -> None:
        await asyncio.sleep(0.1)

    start = time.perf_counter()
    gather_sync([_sleep() for _ in range(4)])
    elapsed = time.perf_counter() - start

    assert elapsed < 0.25, (
        f"gather_sync did not parallelize coroutines (elapsed={elapsed:.3f}s)"
    )


def test_gather_sync_propagates_first_exception_by_default() -> None:
    class GatherBoomError(RuntimeError):
        pass

    async def _ok() -> int:
        return 1

    async def _boom() -> int:
        raise GatherBoomError("boom")

    with pytest.raises(GatherBoomError, match="boom"):
        gather_sync([_ok(), _boom(), _ok()])


def test_gather_sync_return_exceptions_collects_errors() -> None:
    async def _ok() -> int:
        return 1

    async def _boom() -> int:
        raise ValueError("nope")

    out = gather_sync([_ok(), _boom(), _ok()], return_exceptions=True)
    assert out[0] == 1
    assert isinstance(out[1], ValueError)
    assert out[2] == 1


def test_gather_sync_timeout_cancels_all_pending() -> None:
    """Timeout must cancel every still-running coroutine, not just one.

    Otherwise a partial gather leaves orphaned work pinning slatedb
    handles on the bridge loop after the caller returns.
    """
    cancellations: list[int] = []
    lock = threading.Lock()

    async def _cancellable(idx: int) -> None:
        try:
            await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            with lock:
                cancellations.append(idx)
            raise

    with pytest.raises(BridgeTimeoutError):
        gather_sync(
            [_cancellable(i) for i in range(3)],
            timeout=0.05,
        )

    # asyncio.gather cancels the wrapper task on timeout, which
    # propagates to children. Allow a brief drain.
    deadline = time.perf_counter() + 1.0
    while time.perf_counter() < deadline and len(cancellations) < 3:
        time.sleep(0.01)
    assert sorted(cancellations) == [0, 1, 2], (
        f"not all gathered coroutines were cancelled: {cancellations}"
    )


def test_gather_sync_cancels_pending_on_first_error_and_closes_successes() -> None:
    """On first failure with return_exceptions=False: cancel pending siblings
    AND best-effort close any successful results so DbReader-like handles
    are not orphaned on the bridge loop."""
    cancellations: list[int] = []
    closed: list[int] = []
    lock = threading.Lock()

    class _FakeReader:
        def __init__(self, idx: int) -> None:
            self.idx = idx

        async def shutdown(self) -> None:
            with lock:
                closed.append(self.idx)

    async def _quick_ok(idx: int) -> _FakeReader:
        return _FakeReader(idx)

    async def _slow_cancel(idx: int) -> _FakeReader:
        try:
            await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            with lock:
                cancellations.append(idx)
            raise
        return _FakeReader(idx)

    async def _boom() -> _FakeReader:
        await asyncio.sleep(0.05)
        raise RuntimeError("open failed")

    with pytest.raises(RuntimeError, match="open failed"):
        gather_sync([_quick_ok(1), _boom(), _slow_cancel(2), _slow_cancel(3)])

    deadline = time.perf_counter() + 1.0
    while time.perf_counter() < deadline and (
        len(cancellations) < 2 or len(closed) < 1
    ):
        time.sleep(0.01)
    assert sorted(cancellations) == [2, 3], (
        f"sibling opens were not cancelled: cancellations={cancellations}"
    )
    assert closed == [1], (
        f"completed reader was not closed on partial-failure: closed={closed}"
    )


def test_gather_sync_return_exceptions_does_not_close_successes() -> None:
    """In return_exceptions=True mode the caller owns all results, so
    successful handles must NOT be auto-closed by the bridge."""
    closed: list[int] = []

    class _FakeReader:
        def __init__(self, idx: int) -> None:
            self.idx = idx

        async def shutdown(self) -> None:
            closed.append(self.idx)

    async def _ok(idx: int) -> _FakeReader:
        return _FakeReader(idx)

    async def _boom() -> _FakeReader:
        raise RuntimeError("x")

    out = gather_sync([_ok(1), _boom(), _ok(2)], return_exceptions=True)
    assert isinstance(out[0], _FakeReader) and out[0].idx == 1
    assert isinstance(out[1], RuntimeError)
    assert isinstance(out[2], _FakeReader) and out[2].idx == 2
    assert closed == [], f"return_exceptions=True must not close handles: {closed}"
