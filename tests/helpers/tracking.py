"""Shared test doubles for writer tests.

TrackingAdapter / TrackingFactory record all write_batch calls so tests can
inspect what the writer produced without touching S3 or SlateDB.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Self

import pytest

from shardyfusion._rate_limiter import AcquireResult


class TrackingAdapter:
    """In-memory adapter that records all write_batch calls."""

    def __init__(self) -> None:
        self.write_calls: list[list[tuple[bytes, bytes]]] = []
        self.flushed = False
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        self.write_calls.append(list(pairs))

    def flush(self) -> None:
        self.flushed = True

    def seal(self) -> None:
        return None

    def db_bytes(self) -> int:
        return 0

    def close(self) -> None:
        self.closed = True


class TrackingFactory:
    """Factory that creates tracking adapters and keeps references.

    Stores adapters in a list — ``adapters[0]`` is the first adapter created,
    ``adapters[1]`` the second, and so on.
    """

    def __init__(self) -> None:
        self.adapters: list[TrackingAdapter] = []

    def __call__(self, *, db_url: str, local_dir: Path) -> TrackingAdapter:
        adapter = TrackingAdapter()
        self.adapters.append(adapter)
        return adapter


class RecordingTokenBucket:
    """Fake TokenBucket that records __init__, acquire, and try_acquire calls without delay.

    Class-level ``instances`` list collects all created instances.  Reset it
    via ``RecordingTokenBucket.instances = []`` before each test (typically in
    a fixture that also monkeypatches the real ``TokenBucket``).
    """

    instances: list[RecordingTokenBucket] = []

    def __init__(self, rate: float, **kwargs: object) -> None:
        self.rate = rate
        self.limiter_type: str = str(kwargs.get("limiter_type", "ops"))
        self.acquire_calls: list[int] = []
        self.try_acquire_calls: list[int] = []
        RecordingTokenBucket.instances.append(self)

    def acquire(self, tokens: int = 1) -> None:
        self.acquire_calls.append(tokens)

    def try_acquire(self, tokens: int = 1) -> AcquireResult:
        self.try_acquire_calls.append(tokens)
        return AcquireResult(acquired=True, deficit=0.0)

    async def acquire_async(self, tokens: int = 1) -> None:
        self.acquire_calls.append(tokens)


def patch_token_bucket_fixture(module_path: str):
    """Return a pytest fixture that patches ``TokenBucket`` in *module_path*.

    Usage in a test module::

        _patch_token_bucket = patch_token_bucket_fixture(
            "shardyfusion.writer.spark.single_db_writer"
        )
    """

    @pytest.fixture()
    def _patch_token_bucket(
        monkeypatch: pytest.MonkeyPatch,
    ) -> list[RecordingTokenBucket]:
        RecordingTokenBucket.instances = []
        monkeypatch.setattr(
            f"{module_path}.TokenBucket",
            RecordingTokenBucket,
        )
        return RecordingTokenBucket.instances

    return _patch_token_bucket
