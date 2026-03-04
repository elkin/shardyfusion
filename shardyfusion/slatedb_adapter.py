"""Thin adapter around official SlateDB Python bindings."""

import json
import logging
import types
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

from .errors import SlateDbApiError
from .logging import FailureSeverity, get_logger, log_event, log_failure
from .type_defs import JsonObject

_logger = get_logger(__name__)


class DbAdapter(Protocol):
    """Adapter protocol used by partition writers."""

    def __enter__(self) -> Self:
        """Enter adapter context."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        """Exit adapter context and release resources."""
        ...

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        """Write one batch of key/value pairs."""
        ...

    def flush(self) -> None:
        """Flush WAL if supported by binding."""
        ...

    def checkpoint(self) -> str | None:
        """Create a durable checkpoint if supported."""
        ...

    def close(self) -> None:
        """Close DB handle if supported."""
        ...


class DbAdapterFactory(Protocol):
    """Factory protocol for opening one shard writer adapter."""

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
    ) -> DbAdapter:
        """Construct an opened adapter instance."""
        ...


@dataclass(slots=True)
class SlateDbFactory:
    """Picklable factory that captures SlateDB-specific config at construction time."""

    env_file: str | None = None
    settings: JsonObject | None = None

    def __call__(self, *, db_url: str, local_dir: Path) -> "DefaultSlateDbAdapter":
        return DefaultSlateDbAdapter(
            local_dir=local_dir,
            db_url=db_url,
            env_file=self.env_file,
            settings=self.settings,
        )


class DefaultSlateDbAdapter:
    """Adapter using the official SlateDB Python binding constructor contract."""

    def __init__(
        self,
        *,
        local_dir: Path,
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

        self._db_url = db_url

        try:
            self._db = SlateDB(
                str(local_dir),
                url=db_url,
                env_file=env_file,
                settings=settings_payload,
            )
        except TypeError as exc:
            raise SlateDbApiError(
                "Unable to construct SlateDB using the official Python binding "
                "signature `SlateDB(path, url=..., env_file=..., settings=...)`."
            ) from exc

        log_event(
            "shard_adapter_opened",
            level=logging.DEBUG,
            logger=_logger,
            db_url=db_url,
        )

    @property
    def db(self) -> object:
        return self._db

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        self.close()

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
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

    def flush(self) -> None:
        self._db.flush_with_options("wal")
        log_event(
            "shard_adapter_flushed",
            level=logging.DEBUG,
            logger=_logger,
            db_url=getattr(self, "_db_url", "<unknown>"),
        )

    def checkpoint(self) -> str | None:
        checkpoint = self._db.create_checkpoint(scope="durable")
        checkpoint_id = checkpoint["id"]
        log_event(
            "shard_adapter_checkpointed",
            level=logging.DEBUG,
            logger=_logger,
            db_url=getattr(self, "_db_url", "<unknown>"),
            checkpoint_id=checkpoint_id,
        )
        return checkpoint_id

    def close(self) -> None:
        try:
            self._db.close()
        except Exception as exc:
            log_failure(
                "slatedb_adapter_close_failed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=exc,
            )
            raise
        log_event(
            "shard_adapter_closed",
            level=logging.DEBUG,
            logger=_logger,
            db_url=getattr(self, "_db_url", "<unknown>"),
        )


def default_adapter_factory(
    *,
    db_url: str,
    local_dir: Path,
) -> DbAdapter:
    """Factory for default adapter instances (uses SlateDbFactory with no config)."""

    return SlateDbFactory()(db_url=db_url, local_dir=local_dir)
