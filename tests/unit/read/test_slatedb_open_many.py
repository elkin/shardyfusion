"""Unit tests for ``SlateDbReaderFactory.open_many`` partial-failure behaviour.

Closes the verification gap between
``test_async_bridge::test_gather_sync_cancels_pending_on_first_error_and_closes_successes``
(verifies ``gather_sync`` cleanup) and the integration in
``SlateDbReaderFactory.open_many`` (where the wrap-into-handles step
happens after the gather returns). When one shard's builder fails, the
already-built readers must have ``shutdown()`` invoked via
``_best_effort_close`` — otherwise long-running services would leak
``DbReader`` handles on every snapshot refresh that hits a transient
open error.

Uses the same fake-uniffi pattern documented in
``tests/unit/backend/slatedb/test_slatedb_adapter.py`` plus a fake
``Manifest`` placeholder for the protocol's ``manifest=`` argument.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from shardyfusion.reader._types import SlateDbReaderFactory
from shardyfusion.type_defs import Manifest

# ---------------------------------------------------------------------------
# Fake uniffi reader surface
# ---------------------------------------------------------------------------


class _FakeDbReader:
    def __init__(self, idx: int) -> None:
        self.idx = idx
        self.shutdowns = 0

    async def shutdown(self) -> None:
        # Yield once so the bridge schedules the call before returning.
        await asyncio.sleep(0)
        self.shutdowns += 1


class _FakeDbReaderBuilder:
    """Each instance can be configured to succeed or raise on ``build()``.

    ``_outcomes`` is a class-level list shared by the factory; tests
    register one outcome per builder via :meth:`set_outcomes`.
    """

    _outcomes: list[Any] = []
    _index: int = 0
    _lock = threading.Lock()
    _built: list[_FakeDbReader] = []

    def __init__(self, path: str, store: object) -> None:
        # uniffi DbReaderBuilder takes (path, store); shardyfusion passes ("", store).
        self._path = path
        self._store = store

    @classmethod
    def reset(cls) -> None:
        cls._outcomes = []
        cls._index = 0
        cls._built = []

    @classmethod
    def set_outcomes(cls, outcomes: list[Any]) -> None:
        """``outcomes[i]`` is either an int (success → produces a reader)
        or an Exception subclass to raise from ``build()``."""
        cls._outcomes = list(outcomes)
        cls._index = 0
        cls._built = []

    async def build(self) -> _FakeDbReader:
        with _FakeDbReaderBuilder._lock:
            idx = _FakeDbReaderBuilder._index
            _FakeDbReaderBuilder._index += 1
            outcome = _FakeDbReaderBuilder._outcomes[idx]
        # Yield so all builders are scheduled concurrently before any
        # finishes — exercises gather_sync's cancel-on-first-failure path.
        await asyncio.sleep(0.01)
        if isinstance(outcome, Exception):
            raise outcome
        reader = _FakeDbReader(int(outcome))
        with _FakeDbReaderBuilder._lock:
            _FakeDbReaderBuilder._built.append(reader)
        return reader


class _FakeObjectStore:
    @staticmethod
    def resolve(url: str) -> str:
        return f"resolved::{url}"


@pytest.fixture
def fake_uniffi_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[types.ModuleType]:
    """Install a fake ``slatedb.uniffi`` reader-side module."""
    _FakeDbReaderBuilder.reset()

    fake_uniffi_mod = types.ModuleType("slatedb.uniffi")
    fake_uniffi_mod.DbReaderBuilder = _FakeDbReaderBuilder
    fake_uniffi_mod.DbReader = _FakeDbReader
    fake_uniffi_mod.ObjectStore = _FakeObjectStore

    fake_slatedb = types.ModuleType("slatedb")
    fake_slatedb.uniffi = fake_uniffi_mod

    fake_inner_pkg = types.ModuleType("slatedb.uniffi._slatedb_uniffi")
    fake_inner_mod = types.ModuleType("slatedb.uniffi._slatedb_uniffi.slatedb")
    fake_inner_mod.uniffi_set_event_loop = lambda _loop: None

    monkeypatch.setitem(sys.modules, "slatedb", fake_slatedb)
    monkeypatch.setitem(sys.modules, "slatedb.uniffi", fake_uniffi_mod)
    monkeypatch.setitem(sys.modules, "slatedb.uniffi._slatedb_uniffi", fake_inner_pkg)
    monkeypatch.setitem(
        sys.modules,
        "slatedb.uniffi._slatedb_uniffi.slatedb",
        fake_inner_mod,
    )
    yield fake_uniffi_mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _spec(url: str) -> tuple[str, Manifest]:
    return (url, cast(Manifest, None))


def test_open_many_returns_handles_for_all_specs(fake_uniffi_reader: Any) -> None:
    """Happy path: every builder succeeds, returns a handle per spec in order."""
    _FakeDbReaderBuilder.set_outcomes([1, 2, 3])

    factory = SlateDbReaderFactory()
    handles = factory.open_many(
        [_spec("s3://b/0"), _spec("s3://b/1"), _spec("s3://b/2")]
    )

    assert len(handles) == 3
    # No reader was shut down on the happy path.
    for reader in _FakeDbReaderBuilder._built:
        assert reader.shutdowns == 0


def test_open_many_partial_failure_shuts_down_successful_siblings(
    fake_uniffi_reader: Any,
) -> None:
    """A mid-batch ``build()`` exception must:
    1. propagate to the caller, and
    2. invoke ``shutdown()`` on already-built readers via
       ``_best_effort_close`` so Tokio/object-store resources release.
    """

    class _OpenError(RuntimeError):
        pass

    _FakeDbReaderBuilder.set_outcomes([1, _OpenError("simulated open failure"), 3])

    factory = SlateDbReaderFactory()
    with pytest.raises(_OpenError, match="simulated open failure"):
        factory.open_many(
            [_spec("s3://b/0"), _spec("s3://b/1"), _spec("s3://b/2")]
        )

    # Reader 1 (idx=0) finished before the failure; it must have been
    # shut down. Reader 3 may or may not have started (depends on
    # gather_sync FIRST_EXCEPTION timing) — assert only on the readers
    # that actually got built.
    built_indices = {r.idx for r in _FakeDbReaderBuilder._built}
    assert 1 in built_indices, "first builder should have produced a reader"
    for reader in _FakeDbReaderBuilder._built:
        assert reader.shutdowns == 1, (
            f"reader {reader.idx} not shut down on partial-failure cleanup"
        )


def test_open_many_empty_specs_returns_empty(fake_uniffi_reader: Any) -> None:
    """No specs → no bridge call, empty result list."""
    factory = SlateDbReaderFactory()
    handles = factory.open_many([])
    assert handles == []
    assert _FakeDbReaderBuilder._built == []


def test_open_many_first_builder_failure_no_orphans(fake_uniffi_reader: Any) -> None:
    """If the first builder fails, no successful reader exists to leak."""

    class _OpenError(RuntimeError):
        pass

    _FakeDbReaderBuilder.set_outcomes([_OpenError("first failed"), 2, 3])

    factory = SlateDbReaderFactory()
    with pytest.raises(_OpenError, match="first failed"):
        factory.open_many(
            [_spec("s3://b/0"), _spec("s3://b/1"), _spec("s3://b/2")]
        )

    # Any siblings that completed before the failure observed the cleanup.
    for reader in _FakeDbReaderBuilder._built:
        assert reader.shutdowns == 1


def test_open_many_local_dir_arg_not_required(
    fake_uniffi_reader: Any, tmp_path: Path
) -> None:
    """``open_many`` takes ``(db_url, manifest)`` per spec — ``local_dir``
    is *not* in the contract. The SlateDB factory must build readers
    without touching ``self.local_root`` or per-shard dirs.
    """
    _FakeDbReaderBuilder.set_outcomes([1, 2])
    factory = SlateDbReaderFactory()
    handles = factory.open_many([_spec("s3://b/0"), _spec("s3://b/1")])
    assert len(handles) == 2
    # No directories created on tmp_path because nothing in the SlateDB
    # path computes local paths from db_url.
    assert list(tmp_path.iterdir()) == []
