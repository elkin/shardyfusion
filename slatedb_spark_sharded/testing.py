"""Testing helpers that must be importable from Spark worker processes."""

from __future__ import annotations

import base64
import json
import os
import types
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypedDict
from urllib.parse import quote

from .storage import parse_s3_url
from .type_defs import JsonObject


@dataclass(slots=True)
class FakeDb:
    writes: int = 0


class FakeSlateDbAdapter:
    """Minimal adapter implementation for tests without real SlateDB."""

    def __init__(
        self,
        *,
        local_dir: str,
        db_url: str,
        env_file: str | None,
        settings: JsonObject | None,
    ) -> None:
        _ = (local_dir, db_url, env_file, settings)
        self._db = FakeDb()

    @property
    def db(self) -> FakeDb:
        return self._db

    def __enter__(self) -> "FakeSlateDbAdapter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        self.close()

    def write_pairs(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        self._db.writes += len(list(pairs))

    def flush_wal_if_supported(self) -> None:
        return None

    def create_checkpoint_if_supported(self) -> str | None:
        return "fake-checkpoint"

    def close(self) -> None:
        return None


def fake_adapter_factory(
    *,
    local_dir: str,
    db_url: str,
    env_file: str | None,
    settings: JsonObject | None,
) -> FakeSlateDbAdapter:
    """Return a worker-serializable fake adapter instance."""

    return FakeSlateDbAdapter(
        local_dir=local_dir,
        db_url=db_url,
        env_file=env_file,
        settings=settings,
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
        local_dir: str,
        db_url: str,
        env_file: str | None,
        settings: JsonObject | None,
    ) -> None:
        _ = (local_dir, env_file, settings)
        self._root_dir = root_dir
        os.makedirs(self._root_dir, exist_ok=True)
        self._db = FileBackedDb(file_path=_file_path_for_db_url(self._root_dir, db_url))

    @property
    def db(self) -> FileBackedDb:
        return self._db

    def __enter__(self) -> "FileBackedSlateDbAdapter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        self.close()

    def write_pairs(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        os.makedirs(os.path.dirname(self._db.file_path), exist_ok=True)
        with open(self._db.file_path, "ab") as handle:
            for key, value in pairs:
                payload = {
                    "key": base64.b64encode(bytes(key)).decode("ascii"),
                    "value": base64.b64encode(bytes(value)).decode("ascii"),
                }
                handle.write(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
                handle.write(b"\n")

    def flush_wal_if_supported(self) -> None:
        return None

    def create_checkpoint_if_supported(self) -> str | None:
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
        local_dir: str,
        db_url: str,
        env_file: str | None,
        settings: JsonObject | None,
    ) -> FileBackedSlateDbAdapter:
        return FileBackedSlateDbAdapter(
            self.root_dir,
            local_dir=local_dir,
            db_url=db_url,
            env_file=env_file,
            settings=settings,
        )


def file_backed_adapter_factory(root_dir: str) -> FileBackedSlateDbAdapterFactory:
    """Return a serializable file-backed adapter factory."""

    return FileBackedSlateDbAdapterFactory(root_dir=root_dir)


def file_backed_load_db(root_dir: str, db_url: str) -> dict[bytes, bytes]:
    """Load latest key/value view for one db_url from file-backed fake adapter output."""

    file_path = _file_path_for_db_url(root_dir, db_url)
    if not os.path.exists(file_path):
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
    return os.path.join(root_dir, f"{safe_name}.jsonl")


def map_s3_db_url_to_file_url(db_url: str, object_store_root: str) -> str:
    """Map a shard db_url like s3://bucket/key to file://... for local tests."""

    bucket, key = parse_s3_url(db_url)
    object_path = os.path.join(object_store_root, bucket, key)
    os.makedirs(object_path, exist_ok=True)
    return f"file://{object_path}"


def writer_local_dir_for_db_url(db_url: str, local_root: str) -> str:
    """Reconstruct writer local_dir from canonical db_url path segments."""

    _, key = parse_s3_url(db_url)
    segments = [segment for segment in key.split("/") if segment]
    if len(segments) < 3:
        raise ValueError(
            f"Unexpected db_url for writer local dir reconstruction: {db_url}"
        )

    run_segment = segments[-3]
    db_segment = segments[-2]
    attempt_segment = segments[-1]
    return os.path.join(local_root, run_segment, db_segment, attempt_segment)


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
        local_dir: str,
        db_url: str,
        env_file: str | None,
        settings: JsonObject | None,
    ) -> None:
        self._object_store_root = object_store_root

        from slatedb import SlateDB

        mapped_url = map_s3_db_url_to_file_url(db_url, self._object_store_root)
        kwargs: _SlateDbOpenKwargs = {"url": mapped_url}
        if env_file is not None:
            kwargs["env_file"] = env_file
        if settings is not None:
            kwargs["settings"] = json.dumps(
                settings, sort_keys=True, separators=(",", ":")
            )
        self._db = SlateDB(local_dir, **kwargs)

    @property
    def db(self) -> object:
        return self._db

    def __enter__(self) -> "RealSlateDbFileAdapter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        self.close()

    def write_pairs(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        from slatedb import WriteBatch

        wb = WriteBatch()
        for key, value in pairs:
            wb.put(bytes(key), bytes(value))
        self._db.write(wb)

    def flush_wal_if_supported(self) -> None:
        self._db.flush_with_options("wal")

    def create_checkpoint_if_supported(self) -> str | None:
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
        local_dir: str,
        db_url: str,
        env_file: str | None,
        settings: JsonObject | None,
    ) -> RealSlateDbFileAdapter:
        return RealSlateDbFileAdapter(
            self.object_store_root,
            local_dir=local_dir,
            db_url=db_url,
            env_file=env_file,
            settings=settings,
        )


def real_file_adapter_factory(object_store_root: str) -> RealSlateDbFileAdapterFactory:
    """Return a serializable real SlateDB file-backed adapter factory."""

    return RealSlateDbFileAdapterFactory(object_store_root=object_store_root)
