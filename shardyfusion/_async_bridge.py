"""Typed sync\u2192async bridge for the slatedb 0.12 uniffi async API.

This module is the high-level front door for sync code that needs to
invoke async slatedb (or any other async coroutine) from a thread that
has no running event loop. It builds on the lower-level loop owner in
:mod:`shardyfusion._slatedb_runtime` and adds three things the bare
``run_coro`` helper does not provide:

* **Optional per-call timeouts** that actually cancel the in-flight
  coroutine on the bridge loop instead of merely abandoning it. A
  timeout surfaces as :class:`shardyfusion.errors.BridgeTimeoutError`
  (retryable) so callers can map it onto their retry policy.

* **Ctrl-C / cooperative cancellation safety**. If the calling thread
  is interrupted while blocked on ``future.result(...)`` we cancel the
  coroutine on the loop before re-raising. Without this the loop would
  keep driving the work for a caller that has already given up, which
  for slatedb means a half-published manifest or a still-running flush.

* **Fan-out** via :func:`gather_sync`. Submitting N coroutines through
  one ``run_coroutine_threadsafe`` hop costs one bridge round trip
  instead of N (the fixed cost is ~16\u201318 \u00b5s per hop on this
  workstation, see ``tests/integration/perf/test_slatedb_bridge_overhead.py``)
  *and* lets the slatedb Tokio runtime drive them concurrently.

Design notes
------------
* :func:`call_sync` is the shape every sync adapter method should reach
  for. The bare :func:`shardyfusion._slatedb_runtime.run_coro` is kept
  for back-compat and for low-level call sites that intentionally want
  no timeout.
* Cancellation goes through ``loop.call_soon_threadsafe(future.cancel)``
  because :class:`concurrent.futures.Future` returned by
  :func:`asyncio.run_coroutine_threadsafe` is the asyncio Future
  proxy: cancelling it is what propagates ``CancelledError`` into the
  coroutine on the bridge loop.
* The module deliberately exposes no class \u2014 this is a function
  surface so that adapter code can stay procedural and the helpers can
  be tested in isolation without a SlateDB dependency.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine, Iterable
from typing import Any, TypeVar

from shardyfusion._slatedb_runtime import get_loop, run_coro
from shardyfusion.errors import BridgeTimeoutError
from shardyfusion.logging import FailureSeverity, get_logger, log_failure

_T = TypeVar("_T")

_logger = get_logger(__name__)

__all__ = [
    "BridgeTimeoutError",
    "call_sync",
    "gather_sync",
    "run_coro",
]


def _is_running_on_bridge() -> bool:
    """Return True if the current thread is executing the bridge loop."""
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        return False
    # ``get_loop`` is cheap once the loop exists; we don't want to start
    # the loop just to perform a re-entrancy check.
    from shardyfusion import _slatedb_runtime  # noqa: PLC0415

    return running is _slatedb_runtime._loop  # type: ignore[attr-defined]


def call_sync(
    coro: Coroutine[Any, Any, _T],
    *,
    timeout: float | None = None,
) -> _T:
    """Run ``coro`` on the bridge loop and block, with timeout + cancel safety.

    Parameters
    ----------
    coro:
        The coroutine to schedule. Must be a fresh coroutine \u2014
        coroutines may only be awaited once. Construct it at the call
        site, e.g. ``call_sync(db.write(wb), timeout=30.0)``.
    timeout:
        Optional wall-clock limit in seconds. ``None`` (default) blocks
        forever, matching the behavior of
        :func:`shardyfusion._slatedb_runtime.run_coro`.

    Raises
    ------
    BridgeTimeoutError
        If ``timeout`` elapses before ``coro`` completes. The coroutine
        is cancelled on the bridge loop before this exception is
        raised, so the loop is not left holding stale work.
    RuntimeError
        If called from inside the bridge loop itself (would deadlock).
        Async callers should ``await coro`` directly. Also raised by
        :func:`asyncio.run_coroutine_threadsafe` if ``coro`` has
        already been awaited — construct a fresh coroutine per call.
    KeyboardInterrupt
        Re-raised after cancelling the in-flight coroutine on the loop.
    Exception
        Any exception raised by ``coro`` is propagated to the caller.
    """
    if _is_running_on_bridge():
        raise RuntimeError(
            "call_sync invoked from inside the shardyfusion slatedb loop; "
            "await the coroutine directly instead"
        )
    loop = get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError as exc:
        loop.call_soon_threadsafe(future.cancel)
        raise BridgeTimeoutError(
            f"slatedb bridge call exceeded timeout={timeout}s"
        ) from exc
    except KeyboardInterrupt:
        loop.call_soon_threadsafe(future.cancel)
        raise


def gather_sync(
    coros: Iterable[Coroutine[Any, Any, _T]],
    *,
    timeout: float | None = None,
    return_exceptions: bool = False,
) -> list[_T]:
    """Submit N coroutines to the bridge loop in one hop and block on all.

    Functionally equivalent to ``[call_sync(c) for c in coros]`` but
    pays a single bridge round trip instead of N, and lets slatedb's
    Tokio runtime drive the coroutines concurrently. For workloads that
    open several shard readers in parallel or fan out object-store
    puts, the wall-clock difference is the gather time \u2014 not N\u00d7
    the slowest call.

    Parameters
    ----------
    coros:
        Iterable of fresh coroutines. Iteration is performed once at
        call time so generators are supported.
    timeout:
        Optional wall-clock limit for the *whole* gather. On timeout,
        every still-running coroutine is cancelled on the bridge loop
        before :class:`BridgeTimeoutError` is raised.
    return_exceptions:
        Forwarded to :func:`asyncio.gather`. When ``True``, exceptions
        are returned as elements of the result list instead of being
        re-raised.

    Returns
    -------
    list
        Results in submission order, matching :func:`asyncio.gather`.

    Raises
    ------
    BridgeTimeoutError, RuntimeError, KeyboardInterrupt
        Same semantics as :func:`call_sync`.
    """
    coro_list = list(coros)
    if not coro_list:
        return []
    return call_sync(
        _gather(coro_list, return_exceptions=return_exceptions),
        timeout=timeout,
    )


async def _gather(
    coros: list[Coroutine[Any, Any, _T]],
    *,
    return_exceptions: bool,
) -> list[Any]:
    # Wrapped in a coroutine so the bridge loop sees a single awaitable.
    # Return type widened to ``list[Any]`` because asyncio.gather with
    # ``return_exceptions=True`` interleaves exceptions into the result
    # list; the public ``gather_sync`` signature documents this.
    if return_exceptions:
        # asyncio.gather already preserves all results (including
        # exceptions) in this mode; nothing to clean up.
        return await asyncio.gather(*coros, return_exceptions=True)

    # ``return_exceptions=False`` mode: the public contract is "raise
    # the first exception". Plain ``asyncio.gather`` propagates it but
    # leaves sibling coroutines running on the loop, which for
    # ``SlateDbReaderFactory.open_many`` orphans background reader
    # builds and silently leaks any DbReader that finishes after the
    # failure. Wrap each coroutine in a Task, cancel pending Tasks on
    # the first failure, and best-effort close any already-completed
    # readers so callers do not get half-built handles.
    tasks: list[asyncio.Task[Any]] = [asyncio.ensure_future(c) for c in coros]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    except BaseException:
        # ``asyncio.wait`` itself was cancelled (e.g. KeyboardInterrupt
        # routed through ``call_sync``). Cancel everything we spawned.
        for task in tasks:
            if not task.done():
                task.cancel()
        # Drain so cancellations are observed by the loop.
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    first_exc: BaseException | None = None
    for task in done:
        exc = task.exception()
        if exc is not None:
            first_exc = exc
            break

    if first_exc is None:
        # All done tasks succeeded; if any are still pending (FIRST_EXCEPTION
        # only stops on first exception, but if none raised, ``done`` already
        # contains all tasks). Just unpack in original order.
        return [t.result() for t in tasks]

    # A task failed: cancel pending siblings, drain them, and close any
    # already-completed successful results.
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    for task in done:
        if task.exception() is not None:
            continue
        result = task.result()
        await _best_effort_close(result)

    raise first_exc


async def _best_effort_close(obj: Any) -> None:
    """Close a partially-constructed result so we don't leak resources.

    Used by :func:`_gather` when one coroutine in a fan-out fails: any
    siblings that already produced a result (e.g. an open SlateDB
    ``DbReader``) need their ``shutdown``/``close`` method invoked so
    Tokio/object-store resources are released. Errors from the close
    itself do not prevent the original failure from propagating, but
    they are logged so genuine cleanup bugs (corrupt state, hung Tokio
    task) are not lost.
    """
    for attr in ("shutdown", "aclose", "close"):
        fn = getattr(obj, attr, None)
        if fn is None:
            continue
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            log_failure(
                "bridge_best_effort_close_failed",
                severity=FailureSeverity.TRANSIENT,
                logger=_logger,
                error=exc,
                attr=attr,
            )
        return
