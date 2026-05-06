"""Process-global asyncio runtime for bridging sync code to slatedb.uniffi.

The slatedb 0.12 uniffi bindings expose async-only methods on ``Db``,
``DbReader``, ``WriteBatch`` (write ops), and ``Admin``. shardyfusion's
sync writer/reader paths (Spark UDFs, Dask workers, the synchronous
``ShardedReader``, etc.) need to call these from threads that have no
running event loop.

This module owns a single daemon-thread event loop for the lifetime of
the process. Every sync caller submits coroutines to that loop via
:func:`run_coro` and blocks on the resulting future. The bridge thread
runs ``loop.run_forever()``, so coroutines submitted via
``asyncio.run_coroutine_threadsafe`` see the bridge loop as their
running loop and the slatedb.uniffi bindings discover it through
``asyncio.get_running_loop()`` without any global registration.

Async callers (``AsyncShardedReader`` etc.) should *not* go through this
module — they already have their own running loop and can ``await``
slatedb coroutines directly. Importantly, this module does **not** call
``uniffi_set_event_loop``: doing so would pin the bridge loop in
uniffi's process-global slot and cause async callers' awaited futures
to be created on the wrong loop.

Design notes
------------
* The loop is started lazily on first use to avoid import-time side
  effects (matters for `just docs`, lint, and test collection on
  machines without slatedb installed).
* The loop runs in a daemon thread so it never blocks interpreter exit.
* An ``atexit`` hook stops the loop cleanly when possible; failures are
  swallowed because exit-time teardown order is unpredictable.
* ``run_coro`` is safe to call from any thread *except* a thread that is
  itself running the bridge loop. We assert against that to surface
  programming errors early.
"""

from __future__ import annotations

import asyncio
import atexit
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

_T = TypeVar("_T")

_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    finally:
        try:
            loop.close()
        except Exception:
            pass


def get_loop() -> asyncio.AbstractEventLoop:
    """Return the shardyfusion-owned bridge loop, starting it if needed.

    We deliberately do *not* call ``slatedb.uniffi`` ``uniffi_set_event_loop``
    here. That function pins a process-global event-loop slot that the
    generated bindings consult *before* falling back to
    ``asyncio.get_running_loop()`` — pinning the bridge loop would make
    every uniffi async call (including ones the application awaits from
    its own loop via :class:`AsyncSlateDbReaderFactory`) create futures
    on the bridge loop, raising cross-loop errors or hanging. The bridge
    thread runs ``loop.run_forever()``, so any coroutine submitted via
    :func:`asyncio.run_coroutine_threadsafe` already has the bridge loop
    as its running loop and the bindings find it without registration.
    """
    global _loop, _thread
    with _lock:
        if _loop is not None and _loop.is_running():
            return _loop
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=_run_loop,
            args=(loop,),
            name="shardyfusion-slatedb-loop",
            daemon=True,
        )
        thread.start()
        _loop = loop
        _thread = thread
    return loop


def run_coro(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run ``coro`` on the bridge loop and block until it completes.

    Must not be called from inside the bridge loop itself; doing so
    would deadlock. Async callers with their own loop should ``await``
    the slatedb coroutine directly instead of using this helper.
    """
    loop = get_loop()
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        raise RuntimeError(
            "run_coro called from inside the shardyfusion slatedb loop; "
            "await the coroutine directly instead"
        )
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def _shutdown() -> None:
    global _loop, _thread
    with _lock:
        loop = _loop
        thread = _thread
        _loop = None
        _thread = None
    if loop is None:
        return
    try:
        loop.call_soon_threadsafe(loop.stop)
    except Exception:
        return
    if thread is not None:
        thread.join(timeout=1.0)


atexit.register(_shutdown)
