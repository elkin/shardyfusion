"""Thin compatibility adapter around SlateDB Python bindings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, cast

from .errors import SlateDbApiError


class SlateDbAdapter(Protocol):
    """Adapter protocol used by partition writers."""

    def open(
        self,
        *,
        local_dir: str,
        db_url: str,
        env_file: str | None,
        settings: dict[str, Any] | None,
    ) -> Any:
        """Open/create a DB handle."""

    def write_pairs(self, db: Any, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        """Write one batch of key/value pairs."""

    def flush_wal_if_supported(self, db: Any) -> None:
        """Flush WAL if supported by binding."""

    def create_checkpoint_if_supported(self, db: Any) -> str | None:
        """Create a durable checkpoint if supported."""

    def close_if_supported(self, db: Any) -> None:
        """Close DB handle if supported."""


@dataclass(slots=True)
class DefaultSlateDbAdapter:
    """Adapter using the official SlateDB Python binding constructor contract."""

    def open(
        self,
        *,
        local_dir: str,
        db_url: str,
        env_file: str | None,
        settings: dict[str, Any] | None,
    ) -> Any:
        try:
            from slatedb import SlateDB
        except ImportError as exc:
            raise SlateDbApiError("slatedb package is required at runtime") from exc

        settings_payload: str | None = None
        if settings is not None:
            settings_payload = json.dumps(
                settings, sort_keys=True, separators=(",", ":")
            )

        db_ctor = cast(Any, SlateDB)
        try:
            return db_ctor(
                local_dir,
                url=db_url,
                env_file=env_file,
                settings=settings_payload,
            )
        except TypeError as exc:
            raise SlateDbApiError(
                "Unable to construct SlateDB using the official Python binding "
                "signature `SlateDB(path, url=..., env_file=..., settings=...)`."
            ) from exc

    def write_pairs(self, db: Any, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        batch = list(pairs)
        if not batch:
            return

        # Most efficient option: write(WriteBatch-like) if exposed.
        if hasattr(db, "new_write_batch") and hasattr(db, "write"):
            wb = db.new_write_batch()
            for key, value in batch:
                wb.put(key, value)
            db.write(wb)
            return

        if hasattr(db, "write_batch"):
            db.write_batch(batch)
            return

        if hasattr(db, "put"):
            for key, value in batch:
                db.put(key, value)
            return

        raise SlateDbApiError("SlateDB binding exposes no supported write API")

    def flush_wal_if_supported(self, db: Any) -> None:
        if hasattr(db, "flush_with_options"):
            try:
                db.flush_with_options("wal")
                return
            except TypeError:
                pass

        if hasattr(db, "flush"):
            db.flush()

    def create_checkpoint_if_supported(self, db: Any) -> str | None:
        if not hasattr(db, "create_checkpoint"):
            return None
        try:
            checkpoint = db.create_checkpoint(scope="durable")
        except TypeError:
            checkpoint = db.create_checkpoint()

        if checkpoint is None:
            return None
        if isinstance(checkpoint, str):
            return checkpoint
        if isinstance(checkpoint, dict):
            value = checkpoint.get("id")
            return str(value) if value is not None else None
        if hasattr(checkpoint, "id"):
            value = checkpoint.id
            return str(value) if value is not None else None
        return str(checkpoint)

    def close_if_supported(self, db: Any) -> None:
        if hasattr(db, "close"):
            db.close()


def default_adapter_factory() -> SlateDbAdapter:
    """Factory for default adapter instances."""

    return DefaultSlateDbAdapter()
