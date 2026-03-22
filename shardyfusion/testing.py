"""Testing helpers that must be importable from Spark worker processes."""

import base64
import json
import types
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, TypedDict
from urllib.parse import quote

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

        from slatedb import SlateDB

        mapped_url = map_s3_db_url_to_file_url(db_url, self._object_store_root)
        shard_local = local_dir_for_file_shard(self._object_store_root, db_url)
        kwargs: _SlateDbOpenKwargs = {"url": mapped_url}
        self._db = SlateDB(str(shard_local), **kwargs)

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
        from slatedb import WriteBatch

        wb = WriteBatch()
        for key, value in pairs:
            wb.put(bytes(key), bytes(value))
        self._db.write(wb)

    def flush(self) -> None:
        self._db.flush_with_options("wal")

    def checkpoint(self) -> str | None:
        checkpoint = self._db.create_checkpoint(scope="durable")
        return checkpoint["id"]

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


@dataclass(slots=True)
class ListMetricsCollector:
    """Test helper: collects all metric events for assertions."""

    events: list[tuple[MetricEvent, dict[str, Any]]] = field(default_factory=list)

    def emit(self, event: MetricEvent, payload: dict[str, Any]) -> None:
        self.events.append((event, dict(payload)))
