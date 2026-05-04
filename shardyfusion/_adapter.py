"""Backend-neutral writer adapter protocols.

Every backend (SlateDB, SQLite, SQLite-vec, LanceDB, …) implements
:class:`DbAdapter` and exposes an associated :class:`DbAdapterFactory`.
The framework writers (Spark / Dask / Ray / Python) only depend on
these protocols, never on a concrete backend module.

Keeping the protocols in their own module avoids dragging optional
backend imports (SlateDB, APSW, LanceDB) onto every writer code path
and makes the contract easy to audit in one place.
"""

from __future__ import annotations

import types
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, Self

__all__ = ["DbAdapter", "DbAdapterFactory"]


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

    def seal(self) -> None:
        """Finalize the shard so it is safe to read.

        Called after the final :meth:`flush` and before :meth:`close`.
        Backends that need a finalization step (e.g. SQLite ``COMMIT`` +
        ``PRAGMA optimize`` + file-size capture) do their work here.
        SlateDB and LanceDB-style adapters that have nothing to do
        beyond ``flush`` may implement this as a no-op.
        """
        ...

    def db_bytes(self) -> int:
        """Return the materialized shard size in bytes (0 if unknown)."""
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
