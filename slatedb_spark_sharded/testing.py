"""Testing helpers that must be importable from Spark worker processes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class FakeDb:
    writes: int = 0


class FakeSlateDbAdapter:
    """Minimal adapter implementation for tests without real SlateDB."""

    def open(
        self,
        *,
        local_dir: str,
        db_url: str,
        env_file: str | None,
        settings: dict[str, Any] | None,
    ) -> FakeDb:
        _ = (local_dir, db_url, env_file, settings)
        return FakeDb()

    def write_pairs(self, db: FakeDb, pairs) -> None:
        db.writes += len(list(pairs))

    def flush_wal_if_supported(self, db: FakeDb) -> None:
        _ = db

    def create_checkpoint_if_supported(self, db: FakeDb) -> str | None:
        _ = db
        return "fake-checkpoint"

    def close_if_supported(self, db: FakeDb) -> None:
        _ = db


def fake_adapter_factory() -> FakeSlateDbAdapter:
    """Return a worker-serializable fake adapter instance."""

    return FakeSlateDbAdapter()
