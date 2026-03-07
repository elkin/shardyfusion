"""Tests for SlateReaderRepl interactive REPL."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from typing import Any

from shardyfusion.cli.config import OutputConfig
from shardyfusion.cli.interactive import SlateReaderRepl
from shardyfusion.reader.reader import SnapshotInfo


class _FakeReader:
    """Minimal mock reader for REPL testing."""

    def __init__(
        self,
        store: dict[Any, bytes] | None = None,
        key_encoding: str = "u64be",
    ) -> None:
        self._store = store or {}
        self._key_encoding = key_encoding

    @property
    def key_encoding(self) -> str:
        return self._key_encoding

    def get(self, key: Any) -> bytes | None:
        return self._store.get(key)

    def multi_get(self, keys: list[Any]) -> dict[Any, bytes | None]:
        return {k: self._store.get(k) for k in keys}

    def refresh(self) -> bool:
        return False

    def snapshot_info(self) -> SnapshotInfo:
        return SnapshotInfo(
            run_id="test-run",
            num_dbs=2,
            sharding="hash",
            created_at="2026-01-01T00:00:00+00:00",
            manifest_ref="s3://bucket/manifests/test",
        )

    def close(self) -> None:
        pass


def _make_repl(
    store: dict[Any, bytes] | None = None,
) -> SlateReaderRepl:
    reader = _FakeReader(store=store)
    return SlateReaderRepl(reader, OutputConfig())


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
    import json

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


def test_print_banner() -> None:
    repl = _make_repl()
    buf = StringIO()
    with redirect_stdout(buf):
        repl.print_banner()
    # Just verify it doesn't crash
