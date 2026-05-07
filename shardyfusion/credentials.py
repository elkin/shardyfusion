"""Credential provider abstractions for S3 authentication.

Separates identity (who you are) from transport (how you connect).
All dataclasses are frozen for immutability and picklability across
Spark/Dask/Ray/multiprocessing boundaries.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from shardyfusion.type_defs import S3ConnectionOptions

_logger = logging.getLogger(__name__)

# Serialises ``_AppliedEnvContext`` bodies across threads in the same
# process so concurrent context entries cannot race on the
# ``os.environ`` snapshot/restore pattern (see ``_AppliedEnvContext``).
_env_mutation_lock = threading.Lock()


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


def materialize_env_file(
    creds: S3Credentials,
    *,
    connection_options: S3ConnectionOptions | None = None,
) -> str:
    """Write credentials to a temporary ``.env`` file for SlateDB's object-store.

    Returns the file path.  The file is created with ``0o600`` permissions
    so only the current user can read it.  Caller is responsible for
    calling :func:`remove_env_file` after use.

    When *connection_options* is supplied, transport overrides are translated
    into the standard ``AWS_*`` env vars honored by the Apache ``object_store``
    crate (the Rust dependency wrapped by ``slatedb.uniffi``):

    * ``endpoint_url``       → ``AWS_ENDPOINT_URL`` (and ``AWS_ALLOW_HTTP=true``
      when the endpoint is plain ``http://``, which the crate otherwise rejects).
    * ``region_name``        → ``AWS_REGION``.
    * ``addressing_style``   → ``AWS_VIRTUAL_HOSTED_STYLE_REQUEST`` (``"path"``
      → ``false``, ``"virtual"`` → ``true``).  Required for path-style stores
      such as Garage / MinIO / Ceph that lack wildcard DNS.

    Identity (creds) and transport (options) are kept on separate parameters
    so callers can pass connection options without supplying credentials and
    vice-versa; both are merged into the same dotenv.
    """
    lines: list[str] = []
    if creds.access_key_id:
        lines.append(f"AWS_ACCESS_KEY_ID={creds.access_key_id}")
    if creds.secret_access_key:
        lines.append(f"AWS_SECRET_ACCESS_KEY={creds.secret_access_key}")
    if creds.session_token:
        lines.append(f"AWS_SESSION_TOKEN={creds.session_token}")

    if connection_options is not None:
        endpoint = connection_options.get("endpoint_url")
        if endpoint:
            lines.append(f"AWS_ENDPOINT_URL={endpoint}")
            if endpoint.startswith("http://"):
                lines.append("AWS_ALLOW_HTTP=true")
        region = connection_options.get("region_name")
        if region:
            lines.append(f"AWS_REGION={region}")
        addressing = connection_options.get("addressing_style")
        if addressing == "path":
            lines.append("AWS_VIRTUAL_HOSTED_STYLE_REQUEST=false")
        elif addressing == "virtual":
            lines.append("AWS_VIRTUAL_HOSTED_STYLE_REQUEST=true")

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

    If *env_file* is already set, yields it unchanged (explicit takes
    precedence; *connection_options* is ignored in that case — the
    caller's file is the source of truth).  Otherwise, if
    *credential_provider* or *connection_options* is set, materializes
    a temp env file, yields its path, and cleans it up on exit.  Yields
    ``None`` when nothing is provided.
    """

    __slots__ = (
        "_env_file",
        "_credential_provider",
        "_connection_options",
        "_materialized",
    )

    def __init__(
        self,
        env_file: str | None,
        credential_provider: CredentialProvider | None,
        connection_options: S3ConnectionOptions | None = None,
    ) -> None:
        self._env_file = env_file
        self._credential_provider = credential_provider
        self._connection_options = connection_options
        self._materialized: str | None = None

    def __enter__(self) -> str | None:
        if self._env_file is not None:
            return self._env_file
        if (
            self._credential_provider is not None
            or self._connection_options is not None
        ):
            creds = (
                self._credential_provider.resolve()
                if self._credential_provider is not None
                else S3Credentials()
            )
            self._materialized = materialize_env_file(
                creds,
                connection_options=self._connection_options,
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
    connection_options: S3ConnectionOptions | None = None,
) -> _EnvFileContext:
    """Create a context manager that resolves an env file for SlateDB."""
    return _EnvFileContext(env_file, credential_provider, connection_options)


def _parse_dotenv(text: str) -> dict[str, str]:
    """Minimal dotenv parser.

    Accepts lines of the form ``KEY=VALUE``; ignores blank lines and
    comments starting with ``#``. Surrounding single or double quotes
    are stripped from values. Inline ``export`` prefixes are tolerated.

    This intentionally does not implement the full dotenv grammar
    (variable interpolation, multiline values, escape sequences). Our
    only producer is :func:`materialize_env_file`, which writes a
    flat ``KEY=VALUE`` file. User-supplied env files are expected to
    follow the same shape.
    """
    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        out[key] = value
    return out


class _AppliedEnvContext:
    """Apply env_file contents to ``os.environ`` for the context body.

    ``os.environ`` is process-global, so two threads entering this
    context concurrently with overlapping keys would race on the
    snapshot/restore pattern: a later-entering thread can capture the
    earlier thread's mutation as its "prior" and restore the wrong
    value on exit. To keep the invariant *env on exit equals env on
    entry*, the entire context body is serialised across threads via
    :data:`_env_mutation_lock`.

    The lock is held only while ``env_file`` is non-empty and readable,
    and only as long as the caller's body runs (typically a handful of
    ``os.environ`` writes plus one ``ObjectStore.resolve`` — well under
    a millisecond per call). Reads on hot paths (``get``/``multi_get``)
    do not touch this lock.
    """

    __slots__ = ("_env_file", "_previous", "_locked")

    def __init__(self, env_file: str | None) -> None:
        self._env_file = env_file
        self._previous: dict[str, str | None] = {}
        self._locked = False

    def __enter__(self) -> None:
        if self._env_file is None:
            return
        try:
            with open(self._env_file, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            _logger.warning("Failed to read env file %s: %s", self._env_file, exc)
            return
        pairs = _parse_dotenv(text)
        if not pairs:
            return
        _env_mutation_lock.acquire()
        self._locked = True
        for key, value in pairs.items():
            self._previous[key] = os.environ.get(key)
            os.environ[key] = value

    def __exit__(self, *args: object) -> None:
        if not self._locked:
            return
        try:
            for key, prior in self._previous.items():
                if prior is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prior
            self._previous.clear()
        finally:
            _env_mutation_lock.release()
            self._locked = False


def apply_env_file(env_file: str | None) -> _AppliedEnvContext:
    """Context manager that loads ``env_file`` into ``os.environ``.

    Restores the prior environment on exit. ``None`` is a no-op.
    Used by the slatedb.uniffi adapter so per-shard ``ObjectStore.resolve(url)``
    calls can pick up credentials from a caller-supplied dotenv file
    without leaking them across worker processes.
    """
    return _AppliedEnvContext(env_file)
