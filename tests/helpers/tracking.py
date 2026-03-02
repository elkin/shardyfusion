"""Shared test doubles for writer tests.

TrackingAdapter / TrackingFactory record all write_batch calls so tests can
inspect what the writer produced without touching S3 or SlateDB.

InMemoryPublisher stores manifest artifacts in a plain dict so tests can
verify manifest content without S3.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self

from shardyfusion.manifest import ManifestArtifact
from shardyfusion.publish import ManifestPublisher


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

    def __call__(self, *, db_url: str, local_dir: str) -> TrackingAdapter:
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


class InMemoryPublisher(ManifestPublisher):
    """Manifest publisher that stores artifacts in a plain dict."""

    def __init__(self) -> None:
        self.objects: dict[str, ManifestArtifact] = {}

    def publish_manifest(
        self, *, name: str, artifact: ManifestArtifact, run_id: str
    ) -> str:
        ref = f"mem://manifests/run_id={run_id}/{name}"
        self.objects[ref] = artifact
        return ref

    def publish_current(self, *, name: str, artifact: ManifestArtifact) -> str | None:
        ref = f"mem://{name}"
        self.objects[ref] = artifact
        return ref
