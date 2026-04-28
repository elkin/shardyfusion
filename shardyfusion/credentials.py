"""Credential provider abstractions for S3 authentication.

Separates identity (who you are) from transport (how you connect).
All dataclasses are frozen for immutability and picklability across
Spark/Dask/Ray/multiprocessing boundaries.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

_logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class S3Credentials:
    """Resolved S3 identity — access key, secret, and optional session token."""

    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None


@runtime_checkable
class CredentialProvider(Protocol):
    """Pull-based credential resolver.

    Implementations return an ``S3Credentials`` snapshot.  The caller
    decides *when* to resolve (e.g. once at construction, or before
    each request for future rotation support).

    .. note::

       Custom implementations must be **picklable** so they can be
       serialized by Spark/Dask/Ray workers and by the Python writer's
       multiprocessing ``spawn`` mode. Use ``@dataclass(slots=True, frozen=True)``
       or implement ``__getstate__`` / ``__setstate__`` explicitly.
       The built-in :class:`StaticCredentialProvider` and
       :class:`EnvCredentialProvider` are already picklable.
    """

    def resolve(self) -> S3Credentials: ...


@dataclass(slots=True, frozen=True)
class StaticCredentialProvider:
    """Credential provider with inline identity fields.

    Flat fields (not composition) for ergonomic construction::

        provider = StaticCredentialProvider(
            access_key_id="AKIA...",
            secret_access_key="wJal...",
        )
    """

    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None

    def resolve(self) -> S3Credentials:
        return S3Credentials(
            access_key_id=self.access_key_id,
            secret_access_key=self.secret_access_key,
            session_token=self.session_token,
        )


@dataclass(slots=True, frozen=True)
class EnvCredentialProvider:
    """Credential provider that reads ``S3_*`` / ``AWS_*`` environment variables.

    Env var precedence per field (first non-empty wins):
      - ``S3_ACCESS_KEY_ID``  / ``AWS_ACCESS_KEY_ID``
      - ``S3_SECRET_ACCESS_KEY`` / ``AWS_SECRET_ACCESS_KEY``
      - ``S3_SESSION_TOKEN`` / ``AWS_SESSION_TOKEN``
    """

    def resolve(self) -> S3Credentials:
        return S3Credentials(
            access_key_id=os.getenv("S3_ACCESS_KEY_ID")
            or os.getenv("AWS_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY")
            or os.getenv("AWS_SECRET_ACCESS_KEY"),
            session_token=os.getenv("S3_SESSION_TOKEN")
            or os.getenv("AWS_SESSION_TOKEN"),
        )


def materialize_env_file(creds: S3Credentials) -> str:
    """Write credentials to a temporary ``.env`` file for SlateDB's object-store.

    Returns the file path.  The file is created with ``0o600`` permissions
    so only the current user can read it.  Caller is responsible for
    calling :func:`remove_env_file` after use.
    """
    lines: list[str] = []
    if creds.access_key_id:
        lines.append(f"AWS_ACCESS_KEY_ID={creds.access_key_id}")
    if creds.secret_access_key:
        lines.append(f"AWS_SECRET_ACCESS_KEY={creds.secret_access_key}")
    if creds.session_token:
        lines.append(f"AWS_SESSION_TOKEN={creds.session_token}")

    fd, path = tempfile.mkstemp(prefix="shardyfusion_creds_", suffix=".env")
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, "\n".join(lines).encode("utf-8"))
    finally:
        os.close(fd)
    return path


def remove_env_file(path: str) -> None:
    """Safely delete a credentials env file created by :func:`materialize_env_file`."""
    try:
        os.unlink(path)
    except OSError:
        _logger.warning("Failed to remove credentials env file: %s", path)


class _EnvFileContext:
    """Context manager that resolves an env file path for SlateDB.

    If *env_file* is already set, yields it unchanged (explicit takes precedence).
    Otherwise, if *credential_provider* is set, materializes a temp env file,
    yields its path, and cleans it up on exit.  Yields ``None`` when neither
    is provided.
    """

    __slots__ = ("_env_file", "_credential_provider", "_materialized")

    def __init__(
        self,
        env_file: str | None,
        credential_provider: CredentialProvider | None,
    ) -> None:
        self._env_file = env_file
        self._credential_provider = credential_provider
        self._materialized: str | None = None

    def __enter__(self) -> str | None:
        if self._env_file is not None:
            return self._env_file
        if self._credential_provider is not None:
            self._materialized = materialize_env_file(
                self._credential_provider.resolve()
            )
            return self._materialized
        return None

    def __exit__(self, *args: object) -> None:
        if self._materialized is not None:
            remove_env_file(self._materialized)
            self._materialized = None


def resolve_env_file(
    env_file: str | None,
    credential_provider: CredentialProvider | None,
) -> _EnvFileContext:
    """Create a context manager that resolves an env file for SlateDB."""
    return _EnvFileContext(env_file, credential_provider)
