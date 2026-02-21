"""Thin adapter around official SlateDB Python bindings."""

from __future__ import annotations

import json
from typing import Iterable, Protocol

from .errors import SlateDbApiError
from .logging import FailureSeverity, log_failure
from .type_defs import JsonObject


class SlateDbAdapter(Protocol):
    """Adapter protocol used by partition writers."""

    def __enter__(self) -> "SlateDbAdapter":
        """Enter adapter context."""
        ...

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit adapter context and release resources."""
        ...

    @property
    def db(self) -> object:
        """Underlying db handle (for advanced/testing access)."""
        ...

    def write_pairs(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        """Write one batch of key/value pairs."""
        ...

    def flush_wal_if_supported(self) -> None:
        """Flush WAL if supported by binding."""
        ...

    def create_checkpoint_if_supported(self) -> str | None:
        """Create a durable checkpoint if supported."""
        ...

    def close(self) -> None:
        """Close DB handle if supported."""
        ...


class SlateDbAdapterFactory(Protocol):
    """Factory protocol for opening one shard writer adapter."""

    def __call__(
        self,
        *,
        local_dir: str,
        db_url: str,
        env_file: str | None,
        settings: JsonObject | None,
    ) -> SlateDbAdapter:
        """Construct an opened adapter instance."""
        ...


class DefaultSlateDbAdapter:
    """Adapter using the official SlateDB Python binding constructor contract."""

    def __init__(
        self,
        *,
        local_dir: str,
        db_url: str,
        env_file: str | None,
        settings: JsonObject | None,
    ) -> None:
        try:
            from slatedb import SlateDB
        except ImportError as exc:
            raise SlateDbApiError("slatedb package is required at runtime") from exc

        settings_payload: str | None = None
        if settings is not None:
            settings_payload = json.dumps(
                settings, sort_keys=True, separators=(",", ":")
            )

        try:
            self._db = SlateDB(
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

    @property
    def db(self) -> object:
        return self._db

    def __enter__(self) -> "DefaultSlateDbAdapter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write_pairs(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        batch = list(pairs)
        if not batch:
            return

        try:
            from slatedb import WriteBatch
        except ImportError as exc:
            raise SlateDbApiError("slatedb package is required at runtime") from exc

        wb = WriteBatch()
        for key, value in batch:
            wb.put(key, value)
        self._db.write(wb)

    def flush_wal_if_supported(self) -> None:
        self._db.flush_with_options("wal")

    def create_checkpoint_if_supported(self) -> str | None:
        checkpoint = self._db.create_checkpoint(scope="durable")
        return checkpoint["id"]

    def close(self) -> None:
        try:
            self._db.close()
        except Exception as exc:
            log_failure(
                "slatedb_adapter_close_failed",
                severity=FailureSeverity.ERROR,
                error=exc,
            )
            raise


def default_adapter_factory(
    *,
    local_dir: str,
    db_url: str,
    env_file: str | None,
    settings: JsonObject | None,
) -> SlateDbAdapter:
    """Factory for default adapter instances."""

    return DefaultSlateDbAdapter(
        local_dir=local_dir,
        db_url=db_url,
        env_file=env_file,
        settings=settings,
    )
