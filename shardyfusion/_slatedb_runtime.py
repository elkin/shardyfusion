"""Process-global asyncio runtime for bridging sync code to slatedb.uniffi.

The slatedb 0.12 uniffi bindings expose async-only methods on ``Db``,
``DbReader``, ``WriteBatch`` (write ops), and ``Admin``. shardyfusion's
sync writer/reader paths (Spark UDFs, Dask workers, the synchronous
``ShardedReader``, etc.) need to call these from threads that have no
running event loop.

This module owns a single daemon-thread event loop for the lifetime of
the process. Every sync caller submits coroutines to that loop via
:func:`run_coro` and blocks on the resulting future. The loop is also
registered with uniffi via ``uniffi_set_event_loop`` so that any
internal callbacks the bindings schedule end up on the same loop.

Async callers (``AsyncShardedReader`` etc.) should *not* go through this
module — they already have their own running loop and can ``await``
slatedb coroutines directly.

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
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from asyncio import BaseEventLoop

_T = TypeVar("_T")

_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_uniffi_registered = False


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    finally:
        try:
            loop.close()
        except Exception:
            pass


def _register_with_uniffi(loop: asyncio.AbstractEventLoop) -> None:
    """Tell slatedb.uniffi which loop to use for its internal callbacks.

    ``uniffi_set_event_loop`` is defined inside the generated uniffi
    bindings module ``slatedb.uniffi._slatedb_uniffi.slatedb`` (it is
    not re-exported from the ``slatedb.uniffi`` package facade). We
    pin ``slatedb>=0.12,<0.13`` in ``pyproject.toml`` so the import
    path is stable; if a future patch release changes the layout we
    want a loud ``ImportError`` here at first writer/reader call
    rather than a confusing "no running event loop" deeper in the
    bindings.

    Registration is technically only required when uniffi fires Python
    callbacks from a thread that does not own a running loop (e.g. a
    Rust-spawned thread). The shardyfusion bridge always submits
    coroutines via :func:`asyncio.run_coroutine_threadsafe`, which
    runs them on the bridge thread where the loop *is* running, so the
    common path works without registration. We register defensively
    anyway so any future uniffi callback that does need to look up the
    global loop finds ours.
    """
    global _uniffi_registered
    if _uniffi_registered:
        return
    try:
        from slatedb.uniffi._slatedb_uniffi.slatedb import (  # type: ignore[attr-defined]
            uniffi_set_event_loop,
        )
    except ImportError:
        # slatedb is an optional extra. The bridge itself works without
        # it (it just runs arbitrary coroutines on a background loop);
        # uniffi registration is only meaningful when slatedb is
        # installed. Mark registered so we do not retry on every call.
        _uniffi_registered = True
        return

    # uniffi declares ``eventloop: BaseEventLoop``; the concrete asyncio
    # loops we hand it (Selector/Proactor) inherit BaseEventLoop, but the
    # static type from ``new_event_loop()`` is ``AbstractEventLoop``.
    uniffi_set_event_loop(cast("BaseEventLoop", loop))
    _uniffi_registered = True


def get_loop() -> asyncio.AbstractEventLoop:
    """Return the shardyfusion-owned bridge loop, starting it if needed."""
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
    _register_with_uniffi(loop)
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
