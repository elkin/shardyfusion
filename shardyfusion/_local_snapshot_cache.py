"""Concurrency-safe download-and-cache helper for shard reader backends.

Multiple shardyfusion processes (CLI, pytest workers, ``tox -p N``) often
share ``local_root`` (default ``/tmp/shardyfusion``). The download-and-cache
readers (SQLite, SQLite-vec) materialize a shard's database file under
``local_root / shard=NNNNN / shard.db`` and then mmap it for reads. Without
serialization, concurrent processes can race on the cache directory and
produce two failure modes:

* **Torn reads.** Process A reads cached bytes (which look current to A's
  identity check) while process B has just truncated and is mid-rewriting
  the same path. SQLite's mmap returns garbage / older content, surfacing
  as silently wrong query results.
* **Crashes.** If A mmaps the file while B truncates it,
  subsequent page accesses in A raise ``SIGBUS``.

This helper centralises the safe pattern:

1. Acquire an exclusive file lock on a sibling ``.cache.lock`` so only one
   process at a time mutates the cache directory.
2. Re-check the cached identity *inside* the lock so duplicate downloads
   are skipped when a peer just finished one.
3. Write the database to ``shard.db.tmp.<pid>``, fsync, and ``os.replace``
   it over ``shard.db`` (POSIX-atomic rename).
4. Write the identity file via the same temp+replace dance.
5. Release the lock.

Readers may safely keep mmaps open across renames: ``os.replace`` unlinks
the old inode but readers retain access until they close the connection.
The identity file rename guarantees that the next reader sees a consistent
{db file, identity} pair.
"""

from __future__ import annotations

import errno
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import fcntl  # type: ignore[import]
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]


_LOCK_FILENAME = ".cache.lock"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically via temp+fsync+replace.

    ``os.write`` is allowed to return after writing fewer bytes than
    requested (POSIX ``write(2)`` only guarantees a partial write). For
    multi-GB shard databases a short write would silently produce a
    truncated cache file that the identity check then trusts as valid.
    Loop until every byte is written.
    """
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(f"os.write returned {written} writing to {tmp}")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _read_identity(identity_path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(identity_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _identity_matches(
    identity_path: Path, db_path: Path, expected: dict[str, Any]
) -> bool:
    if not db_path.exists() or not identity_path.exists():
        return False
    cached = _read_identity(identity_path)
    return cached == expected


def ensure_cached_snapshot(
    *,
    local_dir: Path,
    db_filename: str,
    identity_filename: str,
    expected_identity: dict[str, Any],
    downloader: Callable[[], bytes],
) -> Path:
    """Ensure ``local_dir / db_filename`` matches ``expected_identity``.

    Returns the absolute path to the cached database file.

    Concurrency: safe across threads and processes via ``fcntl.flock``.
    On platforms without ``fcntl`` (Windows), falls back to the
    legacy non-atomic behaviour (acceptable since shardyfusion's
    targeted deployments are POSIX).
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    db_path = local_dir / db_filename
    identity_path = local_dir / identity_filename

    # Fast path: identity matches without taking the lock.
    if _identity_matches(identity_path, db_path, expected_identity):
        return db_path

    if fcntl is None:  # pragma: no cover - Windows fallback
        data = downloader()
        _atomic_write_bytes(db_path, data)
        _atomic_write_text(identity_path, json.dumps(expected_identity, sort_keys=True))
        return db_path

    lock_path = local_dir / _LOCK_FILENAME
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        except OSError as exc:  # pragma: no cover - very unusual
            if exc.errno not in (errno.EBADF, errno.EINVAL):
                raise

        # Re-check inside the lock: a peer may have just finished.
        if _identity_matches(identity_path, db_path, expected_identity):
            return db_path

        data = downloader()
        _atomic_write_bytes(db_path, data)
        _atomic_write_text(identity_path, json.dumps(expected_identity, sort_keys=True))
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:  # pragma: no cover
            pass
        os.close(lock_fd)

    return db_path


__all__ = ["ensure_cached_snapshot"]
