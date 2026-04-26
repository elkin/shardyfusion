"""Tests for ShardyRepl interactive REPL."""

from __future__ import annotations

import json
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from io import StringIO
from typing import Any

from shardyfusion.cli.config import OutputConfig
from shardyfusion.cli.interactive import ShardyRepl
from shardyfusion.reader import ReaderHealth, ShardDetail, SnapshotInfo
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy

_FAKE_CREATED_AT = datetime.fromisoformat("2026-01-01T00:00:00+00:00")


class _FakeReader:
    """Minimal mock reader for REPL testing."""

    def __init__(
        self,
        store: dict[Any, bytes] | None = None,
        key_encoding: str = "u64be",
    ) -> None:
        self._store = store or {}
        self._key_encoding = key_encoding
        self._manifest_store: Any = None

    @property
    def key_encoding(self) -> str:
        return self._key_encoding

    def get(self, key: Any, **kwargs: Any) -> bytes | None:
        return self._store.get(key)

    def multi_get(self, keys: list[Any], **kwargs: Any) -> dict[Any, bytes | None]:
        return {k: self._store.get(k) for k in keys}

    def refresh(self) -> bool:
        return False

    def snapshot_info(self) -> SnapshotInfo:
        return SnapshotInfo(
            run_id="test-run",
            num_dbs=2,
            sharding=ShardingStrategy.HASH,
            created_at=_FAKE_CREATED_AT,
            manifest_ref="s3://bucket/manifests/test",
            key_encoding=KeyEncoding.from_value(self._key_encoding),
            row_count=len(self._store),
        )

    def shard_details(self) -> list[ShardDetail]:
        return [
            ShardDetail(
                db_id=0,
                row_count=10,
                min_key=0,
                max_key=49,
                db_url="s3://b/shard=00000",
            ),
            ShardDetail(
                db_id=1,
                row_count=20,
                min_key=50,
                max_key=99,
                db_url="s3://b/shard=00001",
            ),
        ]

    def route_key(self, key: Any, **kwargs: Any) -> int:
        return 0 if (isinstance(key, int) and key < 50) else 1

    def health(self, *, staleness_threshold: timedelta | None = None) -> ReaderHealth:
        return ReaderHealth(
            status="healthy",
            manifest_ref="s3://bucket/manifests/test",
            manifest_age=timedelta(seconds=5.0),
            num_shards=2,
            is_closed=False,
        )

    def close(self) -> None:
        pass


def _make_repl(
    store: dict[Any, bytes] | None = None,
) -> ShardyRepl:
    reader = _FakeReader(store=store)
    return ShardyRepl(reader, OutputConfig())


def test_do_get_found() -> None:
    repl = _make_repl({42: b"hello"})
    buf = StringIO()
    with redirect_stdout(buf):
        repl.onecmd("get 42")
    output = buf.getvalue()
    assert "hello" in output or "aGVsbG8=" in output  # base64 or raw


def test_do_get_not_found() -> None:
    repl = _make_repl({})
    buf = StringIO()
    with redirect_stdout(buf):
        repl.onecmd("get 999")
    output = buf.getvalue()
    assert "null" in output or "None" in output.lower()


def test_do_get_no_args() -> None:
    repl = _make_repl()
    buf = StringIO()
    with redirect_stdout(buf):
        repl.onecmd("get")
    # Should print error to stderr but not crash


def test_do_multiget() -> None:
    repl = _make_repl({1: b"a", 2: b"b"})
    buf = StringIO()
    with redirect_stdout(buf):
        repl.onecmd("multiget 1 2 3")
    output = buf.getvalue()
    parsed = json.loads(output)
    results = parsed["results"]
    assert results[0]["found"] is True
    assert results[1]["found"] is True
    assert results[2]["found"] is False


def test_do_multiget_no_args() -> None:
    repl = _make_repl()
    buf = StringIO()
    with redirect_stdout(buf):
        repl.onecmd("multiget")
    # Should print error but not crash


def test_do_refresh() -> None:
    repl = _make_repl()
    buf = StringIO()
    with redirect_stdout(buf):
        repl.onecmd("refresh")
    output = buf.getvalue()
    assert output.strip() != ""


def test_do_info() -> None:
    repl = _make_repl()
    buf = StringIO()
    with redirect_stdout(buf):
        repl.onecmd("info")
    output = buf.getvalue()
    assert "test-run" in output or output.strip() != ""


def test_do_quit_returns_true() -> None:
    repl = _make_repl()
    assert repl.onecmd("quit") is True


def test_do_exit_returns_true() -> None:
    repl = _make_repl()
    assert repl.onecmd("exit") is True


def test_do_eof_returns_true() -> None:
    repl = _make_repl()
    buf = StringIO()
    with redirect_stdout(buf):
        assert repl.onecmd("EOF") is True


def test_do_shards() -> None:
    repl = _make_repl()
    buf = StringIO()
    with redirect_stdout(buf):
        repl.onecmd("shards")
    parsed = json.loads(buf.getvalue())
    assert parsed["op"] == "shards"
    assert len(parsed["shards"]) == 2


def test_do_route() -> None:
    repl = _make_repl()
    buf = StringIO()
    with redirect_stdout(buf):
        repl.onecmd("route 10")
    parsed = json.loads(buf.getvalue())
    assert parsed["op"] == "route"
    assert parsed["db_id"] == 0


def test_do_route_no_args() -> None:
    repl = _make_repl()
    buf = StringIO()
    with redirect_stdout(buf):
        repl.onecmd("route")
    # Should print error but not crash


def test_print_banner() -> None:
    repl = _make_repl()
    buf = StringIO()
    with redirect_stdout(buf):
        repl.print_banner()
    # Just verify it doesn't crash


def test_do_health() -> None:
    repl = _make_repl()
    buf = StringIO()
    with redirect_stdout(buf):
        repl.onecmd("health")
    parsed = json.loads(buf.getvalue())
    assert parsed["op"] == "health"
    assert parsed["status"] == "healthy"
    assert parsed["num_shards"] == 2


def test_do_health_with_threshold() -> None:
    repl = _make_repl()
    buf = StringIO()
    with redirect_stdout(buf):
        repl.onecmd("health 300")
    parsed = json.loads(buf.getvalue())
    assert parsed["op"] == "health"
