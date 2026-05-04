"""Unit tests for the local snapshot cache helper."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

import pytest

from shardyfusion._local_snapshot_cache import ensure_cached_snapshot


def test_first_call_downloads_and_writes_identity(tmp_path: Path) -> None:
    calls: list[int] = []

    def downloader() -> bytes:
        calls.append(1)
        return b"\x01" * 1024

    db_path = ensure_cached_snapshot(
        local_dir=tmp_path / "shard",
        db_filename="db",
        identity_filename="ident.json",
        expected_identity={"db_url": "s3://b/x", "checkpoint_id": "c1"},
        downloader=downloader,
    )

    assert calls == [1]
    assert db_path.read_bytes() == b"\x01" * 1024
    identity = json.loads((tmp_path / "shard" / "ident.json").read_text())
    assert identity == {"db_url": "s3://b/x", "checkpoint_id": "c1"}


def test_second_call_with_matching_identity_skips_download(tmp_path: Path) -> None:
    calls: list[int] = []

    def downloader() -> bytes:
        calls.append(1)
        return b"\x02" * 16

    expected = {"db_url": "s3://b/x", "checkpoint_id": "c1"}
    ensure_cached_snapshot(
        local_dir=tmp_path / "shard",
        db_filename="db",
        identity_filename="ident.json",
        expected_identity=expected,
        downloader=downloader,
    )
    ensure_cached_snapshot(
        local_dir=tmp_path / "shard",
        db_filename="db",
        identity_filename="ident.json",
        expected_identity=expected,
        downloader=downloader,
    )
    assert calls == [1]


def test_changed_identity_triggers_redownload(tmp_path: Path) -> None:
    calls: list[bytes] = []

    def make_downloader(payload: bytes):
        def d() -> bytes:
            calls.append(payload)
            return payload

        return d

    ensure_cached_snapshot(
        local_dir=tmp_path / "shard",
        db_filename="db",
        identity_filename="ident.json",
        expected_identity={"db_url": "s3://b/x", "checkpoint_id": "c1"},
        downloader=make_downloader(b"v1"),
    )
    db_path = ensure_cached_snapshot(
        local_dir=tmp_path / "shard",
        db_filename="db",
        identity_filename="ident.json",
        expected_identity={"db_url": "s3://b/x", "checkpoint_id": "c2"},
        downloader=make_downloader(b"v2"),
    )

    assert calls == [b"v1", b"v2"]
    assert db_path.read_bytes() == b"v2"


def test_threaded_concurrent_callers_download_once(tmp_path: Path) -> None:
    download_count = 0
    lock = threading.Lock()

    def downloader() -> bytes:
        nonlocal download_count
        with lock:
            download_count += 1
        time.sleep(0.05)
        return b"\xab" * 4096

    expected = {"db_url": "s3://b/x", "checkpoint_id": "c"}
    paths: list[Path] = []
    threads = [
        threading.Thread(
            target=lambda: paths.append(
                ensure_cached_snapshot(
                    local_dir=tmp_path / "shard",
                    db_filename="db",
                    identity_filename="ident.json",
                    expected_identity=expected,
                    downloader=downloader,
                )
            )
        )
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert download_count == 1
    assert {p.read_bytes() for p in paths} == {b"\xab" * 4096}


def _proc_worker(args: tuple[str, str, bytes, int]) -> tuple[int, bytes]:
    cache_dir, db_url, payload, sleep_ms = args
    time.sleep(sleep_ms / 1000)

    def downloader() -> bytes:
        return payload

    db_path = ensure_cached_snapshot(
        local_dir=Path(cache_dir),
        db_filename="db",
        identity_filename="ident.json",
        expected_identity={"db_url": db_url, "checkpoint_id": "c"},
        downloader=downloader,
    )
    return os.getpid(), db_path.read_bytes()


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only fcntl path")
def test_multiprocess_concurrent_callers_safe(tmp_path: Path) -> None:
    """Eight processes hammer the same cache dir; all must read the same bytes."""
    cache_dir = str(tmp_path / "shard")
    payload = bytes(range(256)) * 32  # 8 KB
    args = [(cache_dir, "s3://b/x", payload, 0) for _ in range(8)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        results = pool.map(_proc_worker, args)

    assert len({pid for pid, _ in results}) == 8  # truly different processes
    for _, payload_seen in results:
        assert payload_seen == payload
