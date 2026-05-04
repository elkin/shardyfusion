"""Unit tests for _SlateDbReaderHandle.close lifecycle.

The sync handle wraps a uniffi ``DbReader``. Closing the handle must
invoke the underlying ``DbReader.shutdown()`` so Tokio/object-store
resources are released eagerly; otherwise long-running services that
refresh snapshots leak handles until Python finalization.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from shardyfusion.reader._types import _SlateDbReaderHandle


class _FakeDbReader:
    def __init__(self) -> None:
        self.shutdowns: list[float] = []
        self._lock = threading.Lock()

    async def shutdown(self) -> None:
        # Mark and yield once so we exercise the bridge round-trip.
        await asyncio.sleep(0)
        with self._lock:
            self.shutdowns.append(1.0)


def test_close_invokes_uniffi_shutdown() -> None:
    inner = _FakeDbReader()
    handle = _SlateDbReaderHandle(inner, iterator_chunk_size=64)
    handle.close()
    assert inner.shutdowns == [1.0], (
        f"DbReader.shutdown() was not invoked on close: {inner.shutdowns}"
    )


def test_double_close_invokes_shutdown_only_once() -> None:
    inner = _FakeDbReader()
    handle = _SlateDbReaderHandle(inner, iterator_chunk_size=64)
    handle.close()
    handle.close()
    assert len(inner.shutdowns) == 1, f"close() was not idempotent: {inner.shutdowns}"


def test_close_swallows_shutdown_errors() -> None:
    """A shutdown hiccup must not break the caller's cleanup path."""

    class _BoomReader:
        async def shutdown(self) -> None:
            raise RuntimeError("simulated shutdown failure")

    handle = _SlateDbReaderHandle(_BoomReader(), iterator_chunk_size=64)
    # Must not raise.
    handle.close()


def test_close_tolerates_reader_without_shutdown() -> None:
    """Older fakes / test doubles may not implement shutdown; close() must work."""

    class _NoShutdownReader:
        pass

    handle = _SlateDbReaderHandle(_NoShutdownReader(), iterator_chunk_size=64)
    handle.close()  # no AttributeError


@pytest.mark.asyncio
async def test_async_handle_close_invokes_shutdown() -> None:
    """The async-side wrapper has the same shutdown obligation."""
    from shardyfusion.reader.async_reader import _SlateDbAsyncShardReader

    inner = _FakeDbReader()
    handle = _SlateDbAsyncShardReader(inner)
    await handle.close()
    assert inner.shutdowns == [1.0]
    # Idempotent
    await handle.close()
    assert inner.shutdowns == [1.0]
