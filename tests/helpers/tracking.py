"""Shared test doubles for writer tests.

TrackingAdapter / TrackingFactory record all write_batch calls so tests can
inspect what the writer produced without touching S3 or SlateDB.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Self


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

    def checkpoint(self) -> str | None:
        return "fake-checkpoint"

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
    """Fake TokenBucket that records __init__ and acquire calls without delay.

    Class-level ``instances`` list collects all created instances.  Reset it
    via ``RecordingTokenBucket.instances = []`` before each test (typically in
    a fixture that also monkeypatches the real ``TokenBucket``).
    """

    instances: list[RecordingTokenBucket] = []

    def __init__(self, rate: float, **kwargs: object) -> None:
        self.rate = rate
        self.acquire_calls: list[int] = []
        RecordingTokenBucket.instances.append(self)

    def acquire(self, tokens: int = 1) -> None:
        self.acquire_calls.append(tokens)
