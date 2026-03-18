"""Tests for Python writer parallel mode error paths."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Self

import pytest

from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
from shardyfusion.errors import ShardyfusionError
from shardyfusion.manifest_store import InMemoryManifestStore
from shardyfusion.writer.python.writer import _write_parallel


class _CrashingAdapter:
    """Adapter that raises after a few writes to simulate worker crash."""

    def __init__(self, crash_after: int = 2) -> None:
        self._count = 0
        self._crash_after = crash_after

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        pass

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        self._count += 1
        if self._count >= self._crash_after:
            raise RuntimeError("simulated adapter crash")

    def flush(self) -> None:
        pass

    def checkpoint(self) -> str | None:
        return None

    def close(self) -> None:
        pass


class _GoodAdapter:
    """Adapter that works normally."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        pass

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        pass

    def flush(self) -> None:
        pass

    def checkpoint(self) -> str | None:
        return "cp"

    def close(self) -> None:
        pass


def _crashing_factory(*, db_url: str, local_dir: Path) -> _CrashingAdapter:
    return _CrashingAdapter(crash_after=1)


def _good_factory(*, db_url: str, local_dir: Path) -> _GoodAdapter:
    return _GoodAdapter()


def _make_config(num_dbs: int = 2) -> WriteConfig:
    return WriteConfig(
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/test",
        adapter_factory=_good_factory,
        manifest=ManifestOptions(store=InMemoryManifestStore()),
        output=OutputOptions(run_id="test-run"),
    )


def test_worker_crash_raises_error() -> None:
    """Worker process crash is detected and raises ShardyfusionError."""
    config = WriteConfig(
        num_dbs=2,
        s3_prefix="s3://bucket/test",
        adapter_factory=_crashing_factory,
        manifest=ManifestOptions(store=InMemoryManifestStore()),
        output=OutputOptions(run_id="crash-test"),
    )

    records = [{"id": i, "val": b"x"} for i in range(100)]

    with pytest.raises(ShardyfusionError):
        _write_parallel(
            records=records,
            config=config,
            num_dbs=config.num_dbs,
            run_id="crash-test",
            factory=_crashing_factory,
            key_fn=lambda r: r["id"],
            value_fn=lambda r: r["val"],
            max_queue_size=10,
            max_writes_per_second=None,
            max_write_bytes_per_second=None,
        )


def test_parallel_succeeds_with_good_factory() -> None:
    """Parallel mode completes successfully with well-behaved adapters."""
    config = _make_config(num_dbs=2)
    records = [{"id": i, "val": b"x"} for i in range(20)]

    results = _write_parallel(
        records=records,
        config=config,
        num_dbs=config.num_dbs,
        run_id="good-test",
        factory=_good_factory,
        key_fn=lambda r: r["id"],
        value_fn=lambda r: r["val"],
        max_queue_size=10,
        max_writes_per_second=None,
        max_write_bytes_per_second=None,
    )

    assert len(results) == 2
    assert sum(r.row_count for r in results) == 20
