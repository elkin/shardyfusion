"""Testing helpers that must be importable from Spark worker processes."""

import base64
import json
import re
import types
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, TypedDict
from urllib.parse import quote

from ._slatedb_symbols import get_slatedb_writer_symbols
from .errors import DbAdapterError
from .metrics import MetricEvent
from .storage import parse_s3_url


@dataclass(slots=True)
class FakeDb:
    writes: int = 0


class FakeSlateDbAdapter:
    """Minimal adapter implementation for tests without real SlateDB."""

    def __init__(
        self,
        *,
        db_url: str,
        local_dir: Path,
    ) -> None:
        _ = (local_dir, db_url)
        self._db = FakeDb()

    @property
    def db(self) -> FakeDb:
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
        self._db.writes += len(list(pairs))

    def flush(self) -> None:
        return None

    def checkpoint(self) -> str | None:
        return "fake-checkpoint"

    def db_bytes(self) -> int:
        return 0

    def close(self) -> None:
        return None


def fake_adapter_factory(
    *,
    db_url: str,
    local_dir: Path,
) -> FakeSlateDbAdapter:
    """Return a worker-serializable fake adapter instance."""

    return FakeSlateDbAdapter(
        db_url=db_url,
        local_dir=local_dir,
    )


@dataclass(slots=True)
class FileBackedDb:
    """File-backed fake DB handle keyed by db_url."""

    file_path: str


class FileBackedSlateDbAdapter:
    """Fake adapter that persists written kv pairs to local files."""

    def __init__(
        self,
        root_dir: str,
        *,
        db_url: str,
        local_dir: Path,
    ) -> None:
        _ = local_dir
        self._root_dir = root_dir
        Path(self._root_dir).mkdir(parents=True, exist_ok=True)
        self._db = FileBackedDb(file_path=_file_path_for_db_url(self._root_dir, db_url))

    @property
    def db(self) -> FileBackedDb:
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
        Path(self._db.file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._db.file_path, "ab") as handle:
            for key, value in pairs:
                payload = {
                    "key": base64.b64encode(bytes(key)).decode("ascii"),
                    "value": base64.b64encode(bytes(value)).decode("ascii"),
                }
                handle.write(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
                handle.write(b"\n")

    def flush(self) -> None:
        return None

    def checkpoint(self) -> str | None:
        return "file-backed-checkpoint"

    def db_bytes(self) -> int:
        return 0

    def close(self) -> None:
        return None


@dataclass(slots=True)
class FileBackedSlateDbAdapterFactory:
    """Serializable adapter factory for Spark workers."""

    root_dir: str

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
    ) -> FileBackedSlateDbAdapter:
        return FileBackedSlateDbAdapter(
            self.root_dir,
            db_url=db_url,
            local_dir=local_dir,
        )


def file_backed_adapter_factory(root_dir: str) -> FileBackedSlateDbAdapterFactory:
    """Return a serializable file-backed adapter factory."""

    return FileBackedSlateDbAdapterFactory(root_dir=root_dir)


def file_backed_load_db(root_dir: str, db_url: str) -> dict[bytes, bytes]:
    """Load latest key/value view for one db_url from file-backed fake adapter output."""

    file_path = _file_path_for_db_url(root_dir, db_url)
    if not Path(file_path).exists():
        return {}

    result: dict[bytes, bytes] = {}
    with open(file_path, "rb") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line.decode("utf-8"))
            key = base64.b64decode(payload["key"])
            value = base64.b64decode(payload["value"])
            result[key] = value
    return result


def _file_path_for_db_url(root_dir: str, db_url: str) -> str:
    safe_name = quote(db_url, safe="")
    return str(Path(root_dir) / f"{safe_name}.jsonl")


def map_s3_db_url_to_file_url(db_url: str, object_store_root: str) -> str:
    """Map a shard db_url like s3://bucket/key to file://... for local tests."""

    bucket, key = parse_s3_url(db_url)
    object_path = Path(object_store_root) / bucket / key
    object_path.mkdir(parents=True, exist_ok=True)
    return f"file://{object_path}"


def local_dir_for_file_shard(object_store_root: str, db_url: str) -> Path:
    """Derive a deterministic SlateDB local-cache dir from a shard db_url.

    SlateDB uses the local-path argument as a namespace prefix inside the
    object store.  Writer and reader **must** use the same local path for a
    given shard, otherwise the reader cannot find the writer's manifest.

    This helper derives the path from ``(object_store_root, db_url)`` so that
    any code holding the same two values produces an identical directory.
    """
    bucket, key = parse_s3_url(db_url)
    local_path = Path(object_store_root) / ".slatedb-local" / bucket / key
    local_path.mkdir(parents=True, exist_ok=True)
    return local_path


class _SlateDbOpenKwargs(TypedDict, total=False):
    url: str
    env_file: str
    settings: str


class RealSlateDbFileAdapter:
    """Real SlateDB adapter that writes to a local file:// object-store path."""

    def __init__(
        self,
        object_store_root: str,
        *,
        db_url: str,
        local_dir: Path,
    ) -> None:
        self._object_store_root = object_store_root
        _ = local_dir  # ignored; SlateDB requires a deterministic path
        try:
            slatedb_cls, _ = get_slatedb_writer_symbols()
        except DbAdapterError as exc:
            raise RuntimeError("slatedb.SlateDB is unavailable at runtime") from exc

        mapped_url = map_s3_db_url_to_file_url(db_url, self._object_store_root)
        shard_local = local_dir_for_file_shard(self._object_store_root, db_url)
        kwargs: _SlateDbOpenKwargs = {"url": mapped_url}
        self._db = slatedb_cls(str(shard_local), **kwargs)

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
        try:
            _, write_batch_cls = get_slatedb_writer_symbols()
        except DbAdapterError as exc:
            raise RuntimeError("slatedb.WriteBatch is unavailable at runtime") from exc
        wb = write_batch_cls()
        for key, value in pairs:
            wb.put(bytes(key), bytes(value))
        self._db.write(wb)

    def flush(self) -> None:
        self._db.flush_with_options("wal")

    def checkpoint(self) -> str | None:
        checkpoint = self._db.create_checkpoint(scope="durable")
        return checkpoint["id"]

    def db_bytes(self) -> int:
        return 0

    def close(self) -> None:
        self._db.close()


@dataclass(slots=True)
class RealSlateDbFileAdapterFactory:
    """Serializable real SlateDB adapter factory for Spark worker processes."""

    object_store_root: str

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
    ) -> RealSlateDbFileAdapter:
        return RealSlateDbFileAdapter(
            self.object_store_root,
            db_url=db_url,
            local_dir=local_dir,
        )


def real_file_adapter_factory(object_store_root: str) -> RealSlateDbFileAdapterFactory:
    """Return a serializable real SlateDB file-backed adapter factory."""

    return RealSlateDbFileAdapterFactory(object_store_root=object_store_root)


_DB_ID_RE = re.compile(r"(?:^|/)db=(\d+)(?:/|$)")
_ATTEMPT_RE = re.compile(r"(?:^|/)attempt=(\d+)(?:/|$)")


def _parse_db_url_identity(db_url: str) -> tuple[int | None, int | None]:
    bucket, key = parse_s3_url(db_url)
    del bucket
    db_match = _DB_ID_RE.search(key)
    attempt_match = _ATTEMPT_RE.search(key)
    db_id = int(db_match.group(1)) if db_match is not None else None
    attempt = int(attempt_match.group(1)) if attempt_match is not None else None
    return db_id, attempt


class _FailOnceAdapter:
    """Proxy adapter that injects one retryable write failure."""

    def __init__(
        self,
        inner: Any,
        *,
        marker_path: Path | None,
        error_message: str,
    ) -> None:
        self._inner = inner
        self._marker_path = marker_path
        self._error_message = error_message

    @property
    def db(self) -> Any:
        return self._inner.db

    def __enter__(self) -> Self:
        entered = self._inner.__enter__()
        if entered is not None:
            self._inner = entered
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        self._inner.__exit__(exc_type, exc, tb)

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        if self._marker_path is not None:
            self._marker_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._marker_path.touch(exist_ok=False)
            except FileExistsError:
                pass
            else:
                raise RuntimeError(self._error_message)
        self._inner.write_batch(pairs)

    def flush(self) -> None:
        self._inner.flush()

    def checkpoint(self) -> str | None:
        return self._inner.checkpoint()

    def db_bytes(self) -> int:
        return self._inner.db_bytes()

    def close(self) -> None:
        self._inner.close()


@dataclass(slots=True)
class FailOnceAdapterFactory:
    """Wrap another adapter factory and fail once for selected db ids/attempts."""

    inner_factory: Any
    marker_root: str
    fail_db_ids: tuple[int, ...] = (0,)
    fail_attempts: tuple[int, ...] = (0,)
    error_message: str = "simulated transient write failure"

    def __call__(self, *, db_url: str, local_dir: Path) -> _FailOnceAdapter:
        db_id, attempt = _parse_db_url_identity(db_url)
        marker_path: Path | None = None
        if db_id in self.fail_db_ids and attempt in self.fail_attempts:
            marker_name = f"db={db_id:05d}_attempt={attempt:02d}.once"
            marker_path = Path(self.marker_root) / marker_name

        return _FailOnceAdapter(
            self.inner_factory(db_url=db_url, local_dir=local_dir),
            marker_path=marker_path,
            error_message=self.error_message,
        )


@dataclass(slots=True)
class ListMetricsCollector:
    """Test helper: collects all metric events for assertions."""

    events: list[tuple[MetricEvent, dict[str, Any]]] = field(default_factory=list)

    def emit(self, event: MetricEvent, payload: dict[str, Any]) -> None:
        self.events.append((event, dict(payload)))
