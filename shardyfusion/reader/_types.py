"""Data types and factories for sharded readers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from shardyfusion._slatedb_symbols import get_slatedb_reader_class
from shardyfusion.credentials import (
    CredentialProvider,
    resolve_env_file,
)
from shardyfusion.errors import DbAdapterError
from shardyfusion.logging import (
    get_logger,
)
from shardyfusion.routing import SnapshotRouter
from shardyfusion.sharding_types import (
    KeyEncoding,
    ShardHashAlgorithm,
    ShardingStrategy,
)
from shardyfusion.type_defs import (
    Manifest,
    ShardReader,
)

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Null shard reader for empty shards (db_url=None)
# ---------------------------------------------------------------------------


class _NullShardReader:
    """No-op reader for empty shards — always returns ``None``."""

    def get(self, key: bytes) -> bytes | None:
        return None

    def close(self) -> None:
        pass


@dataclass(slots=True, frozen=True)
class SnapshotInfo:
    """Read-only snapshot of manifest metadata."""

    run_id: str
    num_dbs: int
    sharding: ShardingStrategy
    created_at: datetime
    manifest_ref: str
    row_count: int = 0
    key_encoding: KeyEncoding = KeyEncoding.U64BE
    hash_algorithm: ShardHashAlgorithm = ShardHashAlgorithm.XXH3_64


@dataclass(slots=True, frozen=True)
class ShardDetail:
    """Per-shard metadata exposed by ``shard_details()``."""

    db_id: int
    row_count: int
    min_key: int | float | str | bytes | None
    max_key: int | float | str | bytes | None
    db_url: str | None


@dataclass(slots=True, frozen=True)
class ReaderHealth:
    """Diagnostic snapshot of reader state."""

    status: Literal["healthy", "degraded", "unhealthy"]
    manifest_ref: str
    manifest_age: timedelta
    num_shards: int
    is_closed: bool


@dataclass(slots=True)
class SlateDbReaderFactory:
    """Picklable reader factory that captures SlateDB-specific config."""

    env_file: str | None = None
    credential_provider: CredentialProvider | None = None

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        manifest: Manifest,
    ) -> ShardReader:
        try:
            reader_cls = get_slatedb_reader_class()
        except DbAdapterError as exc:  # pragma: no cover - runtime dependent
            raise DbAdapterError(
                "slatedb package is required for reading shards"
            ) from exc

        with resolve_env_file(self.env_file, self.credential_provider) as env_path:
            return reader_cls(
                str(local_dir),
                url=db_url,
                env_file=env_path,
                checkpoint_id=checkpoint_id,
            )


def _shard_details_from_router(router: SnapshotRouter) -> list[ShardDetail]:
    """Extract per-shard metadata from a router's shard list."""
    return [
        ShardDetail(
            db_id=s.db_id,
            row_count=s.row_count,
            min_key=s.min_key,
            max_key=s.max_key,
            db_url=s.db_url,
        )
        for s in router.shards
    ]
