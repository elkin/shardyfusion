"""Tests for adapter lifecycle failure modes in write_shard_core().

Covers: flush() failure, checkpoint() returning None/invalid,
context manager __exit__ raising, factory failure.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

import pytest

from shardyfusion._shard_writer import ShardWriteParams, write_shard_core
from shardyfusion.errors import ShardWriteError, ShardyfusionError
from shardyfusion.serde import make_key_encoder
from shardyfusion.sharding_types import KeyEncoding


def _rows(n: int = 3) -> list[tuple[int, bytes, None]]:
    return [(i, f"val-{i}".encode(), None) for i in range(n)]


class _FlushFailAdapter:
    """Adapter whose flush() raises."""

    def __init__(self, *, db_url: str, local_dir: Path) -> None:
        self.written: list[list[tuple[bytes, bytes]]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        self.written.append(list(pairs))

    def flush(self) -> None:
        raise OSError("disk full during flush")

    def seal(self) -> None:
        return None

    def db_bytes(self) -> int:
        return 0


class _BadCheckpointAdapter:
    """Adapter whose ``seal()`` returns ``None`` (the only valid return).

    Pre-slatedb-0.12 the adapter Protocol exposed ``checkpoint() -> str | None``
    and the writer surfaced that string as ``ShardAttemptResult.checkpoint_id``.
    The Protocol now uses ``seal() -> None`` and the writer stamps a
    shardyfusion-generated UUID hex unconditionally; this adapter exists to
    cover the "adapter contributes no metadata" path.
    """

    def __init__(self, *, db_url: str, local_dir: Path) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        pass

    def flush(self) -> None:
        pass

    def seal(self) -> None:
        return None

    def db_bytes(self) -> int:
        return 0


class _ExitFailAdapter:
    """Adapter whose __exit__ raises."""

    def __init__(self, *, db_url: str, local_dir: Path) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> bool:
        raise RuntimeError("cleanup explosion")

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        pass

    def flush(self) -> None:
        pass

    def seal(self) -> None:
        return None

    def db_bytes(self) -> int:
        return 0


class _FactoryFailFactory:
    """Factory that raises on __call__."""

    def __call__(self, *, db_url: str, local_dir: Path) -> Any:
        raise ConnectionError("cannot connect to S3")


def _params(factory: Any, tmp_path: Path | None = None) -> ShardWriteParams:
    return ShardWriteParams(
        db_id=0,
        attempt=0,
        run_id="test-run",
        db_url="s3://bucket/prefix/db=00000/attempt=00",
        local_dir=tmp_path or Path("/tmp/test"),
        factory=factory,
        key_encoder=make_key_encoder(KeyEncoding.U64BE),
        batch_size=100,
        ops_limiter=None,
        bytes_limiter=None,
        metrics_collector=None,
        started=time.perf_counter(),
    )


class TestFlushFailure:
    def test_flush_failure_raises_shard_write_error(self, tmp_path: Path) -> None:
        """When flush() raises, write_shard_core wraps it in ShardWriteError."""
        params = _params(_FlushFailAdapter, tmp_path)
        with pytest.raises(ShardWriteError, match="disk full"):
            write_shard_core(params, iter(_rows()))


class TestBadCheckpoint:
    def test_seal_returning_none_still_stamps_uuid_checkpoint(
        self, tmp_path: Path
    ) -> None:
        """``seal()`` returns ``None``; writer still stamps a UUID checkpoint_id.

        Post-slatedb-0.12 the adapter Protocol no longer carries checkpoint
        metadata. The shard writer mints ``uuid.uuid4().hex`` for every
        successful shard via ``shardyfusion._checkpoint_id.generate_checkpoint_id``.
        """
        params = _params(_BadCheckpointAdapter, tmp_path)
        result = write_shard_core(params, iter(_rows()))
        assert result.checkpoint_id is not None
        assert isinstance(result.checkpoint_id, str)
        assert len(result.checkpoint_id) == 32
        # Hex-only, lowercase (uuid4().hex contract).
        int(result.checkpoint_id, 16)
        assert result.row_count == 3


class TestExitFailure:
    def test_exit_failure_propagates(self, tmp_path: Path) -> None:
        """When __exit__ raises, the error propagates."""
        params = _params(_ExitFailAdapter, tmp_path)
        with pytest.raises((RuntimeError, ShardyfusionError)):
            write_shard_core(params, iter(_rows()))


class TestFactoryFailure:
    def test_factory_raises_on_call(self, tmp_path: Path) -> None:
        """Factory that can't create adapter wraps error in ShardWriteError."""
        params = _params(_FactoryFailFactory(), tmp_path)
        with pytest.raises((ConnectionError, ShardWriteError)):
            write_shard_core(params, iter(_rows()))


class TestEmptyRowsCheckpoint:
    def test_empty_rows_produce_zero_row_count(self, tmp_path: Path) -> None:
        """Zero rows → shard result with row_count=0."""
        params = _params(_BadCheckpointAdapter, tmp_path)
        result = write_shard_core(params, iter([]))
        assert result.row_count == 0
