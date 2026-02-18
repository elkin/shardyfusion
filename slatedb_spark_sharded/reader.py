"""Service-side sharded SlateDB reader helpers."""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from queue import Queue
from typing import Any, Sequence

from .errors import SlateDbApiError
from .manifest import ParsedManifest
from .manifest_readers import DefaultS3ManifestReader, ManifestReader
from .publish import DefaultS3Publisher, ManifestPublisher
from .routing import SnapshotRouter


@dataclass(slots=True)
class _ReaderState:
    manifest_ref: str
    router: SnapshotRouter
    handles: dict[int, "_ShardHandle"]
    refcount: int = 0
    retired: bool = False


@dataclass(slots=True)
class _ShardHandle:
    mode: str
    reader: Any | None = None
    lock: threading.Lock | None = None
    pool: "_ReaderPool | None" = None


class _ReaderPool:
    def __init__(self, readers: list[Any]) -> None:
        if not readers:
            raise ValueError("Reader pool requires at least one reader")
        self._readers = readers
        self._indexes: Queue[int] = Queue()
        for idx in range(len(readers)):
            self._indexes.put(idx)

    @contextmanager
    def checkout(self):
        idx = self._indexes.get()
        try:
            yield self._readers[idx]
        finally:
            self._indexes.put(idx)

    def close(self) -> None:
        for reader in self._readers:
            reader.close()


class SlateShardedReader:
    """Load latest snapshot manifest and perform routed key lookups."""

    def __init__(
        self,
        *,
        s3_prefix: str,
        local_root: str,
        publisher: ManifestPublisher | None = None,
        manifest_reader: ManifestReader | None = None,
        current_name: str = "_CURRENT",
        slate_env_file: str | None = None,
        slate_settings: dict | None = None,
        thread_safety: str = "lock",
        max_workers: int | None = None,
    ) -> None:
        if thread_safety not in {"lock", "pool"}:
            raise ValueError("thread_safety must be 'lock' or 'pool'")

        if publisher is not None and not isinstance(publisher, DefaultS3Publisher):
            if manifest_reader is None:
                raise ValueError(
                    "Custom publisher detected. Provide a custom ManifestReader that can "
                    "load CURRENT and decode manifest refs."
                )

        self.s3_prefix = s3_prefix
        self.local_root = local_root
        self.current_name = current_name
        self.slate_env_file = slate_env_file
        self.slate_settings = slate_settings
        self.thread_safety = thread_safety
        self.max_workers = max_workers

        if manifest_reader is not None:
            self._manifest_reader = manifest_reader
        else:
            self._manifest_reader = DefaultS3ManifestReader(
                s3_prefix,
                current_name=current_name,
            )

        self._state_lock = threading.Lock()
        self._closed = False
        self._state = self._load_initial_state()

    def get(self, key: int | str | bytes) -> bytes | None:
        """Get one key from the currently loaded snapshot."""

        state = self._acquire_state()
        try:
            db_id = state.router.route_one(key)
            key_bytes = state.router.encode_lookup_key(key)
            handle = state.handles[db_id]
            return _read_one(handle, key_bytes)
        finally:
            self._release_state(state)

    def multi_get(
        self, keys: Sequence[int | str | bytes]
    ) -> dict[int | str | bytes, bytes | None]:
        """Get multiple keys with per-shard grouping and optional shard parallelism."""

        key_list = list(keys)
        state = self._acquire_state()
        try:
            grouped = state.router.group_keys(key_list)
            results: dict[int | str | bytes, bytes | None] = {}

            if self.max_workers and self.max_workers > 1 and len(grouped) > 1:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [
                        executor.submit(
                            _read_group,
                            state.router,
                            state.handles[db_id],
                            shard_keys,
                        )
                        for db_id, shard_keys in grouped.items()
                    ]
                    for future in futures:
                        results.update(future.result())
            else:
                for db_id, shard_keys in grouped.items():
                    results.update(
                        _read_group(state.router, state.handles[db_id], shard_keys)
                    )

            return {key: results.get(key) for key in key_list}
        finally:
            self._release_state(state)

    def refresh(self) -> bool:
        """Reload CURRENT and manifest, atomically swapping readers when ref changes."""

        current = self._manifest_reader.load_current()
        if current is None:
            raise ValueError("CURRENT pointer not found during refresh")

        current_ref = current.manifest_ref
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Reader is closed")
            if current_ref == self._state.manifest_ref:
                return False

        manifest = self._manifest_reader.load_manifest(
            current_ref,
            current.manifest_content_type,
        )
        new_state = self._build_state(current_ref, manifest)

        old_state: _ReaderState
        should_close_old = False
        with self._state_lock:
            if self._closed:
                _close_state(new_state)
                raise RuntimeError("Reader is closed")
            old_state = self._state
            self._state = new_state
            old_state.retired = True
            should_close_old = old_state.refcount == 0

        if should_close_old:
            _close_state(old_state)

        return True

    def close(self) -> None:
        """Close all active readers and prevent further operations."""

        state_to_close: _ReaderState | None = None
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._state.retired = True
            if self._state.refcount == 0:
                state_to_close = self._state

        if state_to_close is not None:
            _close_state(state_to_close)

    def _load_initial_state(self) -> _ReaderState:
        current = self._manifest_reader.load_current()
        if current is None:
            raise ValueError("CURRENT pointer not found")

        manifest = self._manifest_reader.load_manifest(
            current.manifest_ref,
            current.manifest_content_type,
        )
        return self._build_state(current.manifest_ref, manifest)

    def _build_state(self, manifest_ref: str, manifest: ParsedManifest) -> _ReaderState:
        router = SnapshotRouter(manifest.required_build, manifest.shards)
        handles: dict[int, _ShardHandle] = {}

        try:
            for shard in manifest.shards:
                local_path = os.path.join(self.local_root, f"shard={shard.db_id:05d}")
                os.makedirs(local_path, exist_ok=True)

                if self.thread_safety == "lock":
                    reader = _open_slatedb_reader(
                        local_path=local_path,
                        db_url=shard.db_url,
                        checkpoint_id=shard.checkpoint_id,
                        env_file=self.slate_env_file,
                        settings=self.slate_settings,
                    )
                    handles[shard.db_id] = _ShardHandle(
                        mode="lock",
                        reader=reader,
                        lock=threading.Lock(),
                    )
                else:
                    pool_size = self.max_workers or 4
                    readers = [
                        _open_slatedb_reader(
                            local_path=local_path,
                            db_url=shard.db_url,
                            checkpoint_id=shard.checkpoint_id,
                            env_file=self.slate_env_file,
                            settings=self.slate_settings,
                        )
                        for _ in range(pool_size)
                    ]
                    handles[shard.db_id] = _ShardHandle(
                        mode="pool",
                        pool=_ReaderPool(readers),
                    )
        except Exception:
            for handle in handles.values():
                _close_handle(handle)
            raise

        return _ReaderState(manifest_ref=manifest_ref, router=router, handles=handles)

    def _acquire_state(self) -> _ReaderState:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Reader is closed")
            state = self._state
            state.refcount += 1
            return state

    def _release_state(self, state: _ReaderState) -> None:
        should_close = False
        with self._state_lock:
            state.refcount -= 1
            should_close = state.retired and state.refcount == 0

        if should_close:
            _close_state(state)


def _read_group(
    router: SnapshotRouter,
    handle: _ShardHandle,
    keys: list[int | str | bytes],
) -> dict[int | str | bytes, bytes | None]:
    results: dict[int | str | bytes, bytes | None] = {}
    for key in keys:
        key_bytes = router.encode_lookup_key(key)
        results[key] = _read_one(handle, key_bytes)
    return results


def _read_one(handle: _ShardHandle, key: bytes) -> bytes | None:
    if handle.mode == "lock":
        assert handle.lock is not None
        assert handle.reader is not None
        with handle.lock:
            return handle.reader.get(key)

    assert handle.pool is not None
    with handle.pool.checkout() as reader:
        return reader.get(key)


def _open_slatedb_reader(
    *,
    local_path: str,
    db_url: str,
    checkpoint_id: str | None,
    env_file: str | None,
    settings: dict[str, Any] | None,
) -> Any:
    try:
        from slatedb import SlateDBReader
    except ImportError as exc:  # pragma: no cover - runtime dependent
        raise SlateDbApiError(
            "slatedb package is required for SlateShardedReader"
        ) from exc

    reader_kwargs: dict[str, Any] = {"url": db_url}
    if checkpoint_id is not None:
        reader_kwargs["checkpoint_id"] = checkpoint_id
    if env_file is not None:
        reader_kwargs["env_file"] = env_file
    if settings is not None:
        reader_kwargs["settings"] = json.dumps(
            settings, sort_keys=True, separators=(",", ":")
        )

    try:
        reader = SlateDBReader(local_path, **reader_kwargs)
    except TypeError as exc:
        raise SlateDbApiError(
            "Unable to construct SlateDBReader using the official Python binding "
            "signature `SlateDBReader(path, url=..., checkpoint_id=..., "
            "env_file=..., settings=...)`."
        ) from exc

    return reader


def _close_state(state: _ReaderState) -> None:
    for handle in state.handles.values():
        _close_handle(handle)


def _close_handle(handle: _ShardHandle) -> None:
    if handle.mode == "lock" and handle.reader is not None:
        handle.reader.close()
    if handle.mode == "pool" and handle.pool is not None:
        handle.pool.close()
