"""Unit tests for _SlateDbReaderHandle.close + scan_iter lifecycle.

The sync handle wraps a uniffi ``DbReader``. Closing the handle must
invoke the underlying ``DbReader.shutdown()`` so Tokio/object-store
resources are released eagerly; otherwise long-running services that
refresh snapshots leak handles until Python finalization.

``scan_iter`` opens a per-call uniffi ``DbIterator`` on the bridge loop;
its cleanup must fire on consumer ``break`` / generator GC so iterator
prefetch tasks release without waiting for the parent ``DbReader``
shutdown cascade.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from shardyfusion.errors import ConfigValidationError
from shardyfusion.reader._types import SlateDbReaderFactory, _SlateDbReaderHandle


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


# ---------------------------------------------------------------------------
# scan_iter cleanup (W1): consumer ``break`` must release iterator resources
# ---------------------------------------------------------------------------


class _FakeKv:
    def __init__(self, key: bytes, value: bytes) -> None:
        self.key = key
        self.value = value


class _FakeKeyRange:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeIterator:
    """uniffi-shaped async iterator fake.

    Yields a fixed sequence of ``(key, value)`` pairs via ``next()``,
    then ``None`` to signal exhaustion. Records ``aclose()`` calls so
    tests can assert lifecycle. Intentionally omits ``next_batch`` so
    ``scan_iter`` exercises the per-row branch (W2 capability probe).
    """

    def __init__(self, items: list[tuple[bytes, bytes]]) -> None:
        self._items = list(items)
        self._idx = 0
        self.aclose_count = 0

    async def next(self) -> Any:
        if self._idx >= len(self._items):
            return None
        kv = _FakeKv(*self._items[self._idx])
        self._idx += 1
        return kv

    async def aclose(self) -> None:
        self.aclose_count += 1


class _FakeDbReaderForScan:
    """uniffi-shaped reader fake exposing only ``scan``."""

    def __init__(self, items: list[tuple[bytes, bytes]]) -> None:
        self._items = items
        self.opened: list[_FakeIterator] = []

    async def scan(self, key_range: _FakeKeyRange) -> _FakeIterator:
        del key_range
        it = _FakeIterator(self._items)
        self.opened.append(it)
        return it


@pytest.fixture
def fake_key_range_class(monkeypatch: pytest.MonkeyPatch) -> type:
    """Patch ``get_key_range_class`` to return our positional-only fake."""
    from shardyfusion.reader import _types as types_mod

    monkeypatch.setattr(types_mod, "get_key_range_class", lambda: _FakeKeyRange)
    return _FakeKeyRange


def test_scan_iter_aclose_called_on_early_break(fake_key_range_class: type) -> None:
    """W1: ``try/finally`` cleanup must invoke ``aclose`` when the consumer
    closes the generator early — Tokio prefetch tasks should release
    without waiting for the parent ``DbReader.shutdown`` cascade.
    """
    items = [(f"k{i}".encode(), f"v{i}".encode()) for i in range(8)]
    reader = _FakeDbReaderForScan(items)
    handle = _SlateDbReaderHandle(reader, iterator_chunk_size=4)

    gen = handle.scan_iter(b"a", b"z")
    first = next(gen)
    gen.close()  # raises GeneratorExit inside the generator → fires finally

    assert first == (b"k0", b"v0")
    assert len(reader.opened) == 1
    assert reader.opened[0].aclose_count == 1


def test_scan_iter_aclose_called_on_natural_exhaustion(
    fake_key_range_class: type,
) -> None:
    """Cleanup also fires when the consumer drains the iterator fully."""
    items = [(f"k{i}".encode(), f"v{i}".encode()) for i in range(3)]
    reader = _FakeDbReaderForScan(items)
    handle = _SlateDbReaderHandle(reader, iterator_chunk_size=4)

    seen = list(handle.scan_iter(b"a", b"z"))

    assert seen == items
    assert reader.opened[0].aclose_count == 1


def test_scan_iter_aclose_called_when_consumer_raises(
    fake_key_range_class: type,
) -> None:
    """Cleanup also fires when the consumer raises mid-iteration."""
    items = [(f"k{i}".encode(), f"v{i}".encode()) for i in range(8)]
    reader = _FakeDbReaderForScan(items)
    handle = _SlateDbReaderHandle(reader, iterator_chunk_size=4)

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        for _i, _ in enumerate(handle.scan_iter(b"a", b"z")):
            raise _Boom

    assert reader.opened[0].aclose_count == 1


def test_scan_iter_tolerates_iterator_without_aclose(
    fake_key_range_class: type,
) -> None:
    """uniffi 0.12.1 has no ``aclose``; cleanup is gated by ``hasattr``."""

    class _NoAcloseIterator:
        def __init__(self) -> None:
            self._done = False

        async def next(self) -> Any:
            if self._done:
                return None
            self._done = True
            return _FakeKv(b"k", b"v")

    class _NoAcloseReader:
        async def scan(self, key_range: _FakeKeyRange) -> _NoAcloseIterator:
            del key_range
            return _NoAcloseIterator()

    handle = _SlateDbReaderHandle(_NoAcloseReader(), iterator_chunk_size=4)
    seen = list(handle.scan_iter(b"a", b"z"))
    assert seen == [(b"k", b"v")]


# ---------------------------------------------------------------------------
# Factory validation (W6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, -1024])
def test_factory_rejects_non_positive_iterator_chunk_size(bad: int) -> None:
    """W6: ``iterator_chunk_size <= 0`` would silently yield empty scans
    (``range(0)`` is a no-op), so reject at construction time.
    """
    with pytest.raises(ConfigValidationError, match="iterator_chunk_size"):
        SlateDbReaderFactory(iterator_chunk_size=bad)
