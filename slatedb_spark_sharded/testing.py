"""Testing helpers that must be importable from Spark worker processes."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
from typing import Any
from urllib.parse import quote

from .storage import parse_s3_url


@dataclass(slots=True)
class FakeDb:
    writes: int = 0


class FakeSlateDbAdapter:
    """Minimal adapter implementation for tests without real SlateDB."""

    def open(
        self,
        *,
        local_dir: str,
        db_url: str,
        env_file: str | None,
        settings: dict[str, Any] | None,
    ) -> FakeDb:
        _ = (local_dir, db_url, env_file, settings)
        return FakeDb()

    def write_pairs(self, db: FakeDb, pairs) -> None:
        db.writes += len(list(pairs))

    def flush_wal_if_supported(self, db: FakeDb) -> None:
        _ = db

    def create_checkpoint_if_supported(self, db: FakeDb) -> str | None:
        _ = db
        return "fake-checkpoint"

    def close_if_supported(self, db: FakeDb) -> None:
        _ = db


def fake_adapter_factory() -> FakeSlateDbAdapter:
    """Return a worker-serializable fake adapter instance."""

    return FakeSlateDbAdapter()


@dataclass(slots=True)
class FileBackedDb:
    """File-backed fake DB handle keyed by db_url."""

    file_path: str


class FileBackedSlateDbAdapter:
    """Fake adapter that persists written kv pairs to local files."""

    def __init__(self, root_dir: str) -> None:
        self._root_dir = root_dir

    def open(
        self,
        *,
        local_dir: str,
        db_url: str,
        env_file: str | None,
        settings: dict[str, Any] | None,
    ) -> FileBackedDb:
        _ = (local_dir, env_file, settings)
        os.makedirs(self._root_dir, exist_ok=True)
        return FileBackedDb(file_path=_file_path_for_db_url(self._root_dir, db_url))

    def write_pairs(self, db: FileBackedDb, pairs) -> None:
        os.makedirs(os.path.dirname(db.file_path), exist_ok=True)
        with open(db.file_path, "ab") as handle:
            for key, value in pairs:
                payload = {
                    "key": base64.b64encode(bytes(key)).decode("ascii"),
                    "value": base64.b64encode(bytes(value)).decode("ascii"),
                }
                handle.write(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
                handle.write(b"\n")

    def flush_wal_if_supported(self, db: FileBackedDb) -> None:
        _ = db

    def create_checkpoint_if_supported(self, db: FileBackedDb) -> str | None:
        _ = db
        return "file-backed-checkpoint"

    def close_if_supported(self, db: FileBackedDb) -> None:
        _ = db


@dataclass(slots=True)
class FileBackedSlateDbAdapterFactory:
    """Serializable adapter factory for Spark workers."""

    root_dir: str

    def __call__(self) -> FileBackedSlateDbAdapter:
        return FileBackedSlateDbAdapter(self.root_dir)


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
        raise ValueError(f"Unexpected db_url for writer local dir reconstruction: {db_url}")

    run_segment = segments[-3]
    db_segment = segments[-2]
    attempt_segment = segments[-1]
    return os.path.join(local_root, run_segment, db_segment, attempt_segment)


class RealSlateDbFileAdapter:
    """Real SlateDB adapter that writes to a local file:// object-store path."""

    def __init__(self, object_store_root: str) -> None:
        self._object_store_root = object_store_root

    def open(
        self,
        *,
        local_dir: str,
        db_url: str,
        env_file: str | None,
        settings: dict[str, Any] | None,
    ):
        from slatedb import SlateDB

        mapped_url = map_s3_db_url_to_file_url(db_url, self._object_store_root)
        kwargs: dict[str, Any] = {"url": mapped_url}
        if env_file is not None:
            kwargs["env_file"] = env_file
        if isinstance(settings, str):
            kwargs["settings"] = settings
        return SlateDB(local_dir, **kwargs)

    def write_pairs(self, db: Any, pairs) -> None:
        for key, value in pairs:
            db.put(bytes(key), bytes(value))

    def flush_wal_if_supported(self, db: Any) -> None:
        db.flush_with_options("wal")

    def create_checkpoint_if_supported(self, db: Any) -> str | None:
        checkpoint = db.create_checkpoint(scope="durable")
        if isinstance(checkpoint, dict):
            checkpoint_id = checkpoint.get("id")
            return str(checkpoint_id) if checkpoint_id is not None else None
        return str(checkpoint)

    def close_if_supported(self, db: Any) -> None:
        db.close()


@dataclass(slots=True)
class RealSlateDbFileAdapterFactory:
    """Serializable real SlateDB adapter factory for Spark worker processes."""

    object_store_root: str

    def __call__(self) -> RealSlateDbFileAdapter:
        return RealSlateDbFileAdapter(self.object_store_root)


def real_file_adapter_factory(object_store_root: str) -> RealSlateDbFileAdapterFactory:
    """Return a serializable real SlateDB file-backed adapter factory."""

    return RealSlateDbFileAdapterFactory(object_store_root=object_store_root)
