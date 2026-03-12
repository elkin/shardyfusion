"""Tests for batch script loading and execution."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from io import StringIO
from typing import Any

import pytest

from shardyfusion.cli.batch import load_script, run_script
from shardyfusion.cli.config import OutputConfig
from shardyfusion.reader.reader import ShardDetail, SnapshotInfo
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy

_FAKE_CREATED_AT = datetime.fromisoformat("2026-01-01T00:00:00+00:00")


class _FakeReader:
    def __init__(self, store: dict[Any, bytes] | None = None) -> None:
        self._store = store or {}

    @property
    def key_encoding(self) -> str:
        return "u64be"

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
            sharding=ShardingStrategy.HASH,
            created_at=_FAKE_CREATED_AT,
            manifest_ref="s3://bucket/manifests/test",
            key_encoding=KeyEncoding.U64BE,
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

    def route_key(self, key: Any) -> int:
        return 0 if (isinstance(key, int) and key < 50) else 1


def _write_script(content: str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False)
    f.write(content)
    f.close()
    return f.name


def test_load_script_valid() -> None:
    path = _write_script("commands:\n  - op: get\n    key: 42\n  - op: info\n")
    data = load_script(path)
    assert "commands" in data
    assert len(data["commands"]) == 2


def test_load_script_not_a_mapping() -> None:
    path = _write_script("- item1\n- item2\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_script(path)


def test_load_script_missing_commands() -> None:
    path = _write_script("some_key: value\n")
    with pytest.raises(ValueError, match="commands"):
        load_script(path)


def test_run_script_get() -> None:
    path = _write_script("commands:\n  - op: get\n    key: 42\n")
    reader = _FakeReader({42: b"hello"})
    out = StringIO()
    errors = run_script(reader, path, OutputConfig(), output_file=out)
    assert errors == 0
    assert out.getvalue().strip() != ""


def test_run_script_multiget() -> None:
    path = _write_script("commands:\n  - op: multiget\n    keys: [1, 2]\n")
    reader = _FakeReader({1: b"a", 2: b"b"})
    out = StringIO()
    errors = run_script(reader, path, OutputConfig(), output_file=out)
    assert errors == 0
    parsed = json.loads(out.getvalue().strip())
    assert all(r["found"] is True for r in parsed["results"])


def test_run_script_info() -> None:
    path = _write_script("commands:\n  - op: info\n")
    reader = _FakeReader()
    out = StringIO()
    errors = run_script(reader, path, OutputConfig(), output_file=out)
    assert errors == 0


def test_run_script_refresh() -> None:
    path = _write_script("commands:\n  - op: refresh\n")
    reader = _FakeReader()
    out = StringIO()
    errors = run_script(reader, path, OutputConfig(), output_file=out)
    assert errors == 0


def test_run_script_unknown_op() -> None:
    path = _write_script("commands:\n  - op: bogus\n")
    reader = _FakeReader()
    out = StringIO()
    errors = run_script(reader, path, OutputConfig(), output_file=out)
    assert errors == 1


def test_run_script_non_mapping_command() -> None:
    path = _write_script("commands:\n  - just_a_string\n")
    reader = _FakeReader()
    out = StringIO()
    errors = run_script(reader, path, OutputConfig(), output_file=out)
    assert errors == 1


def test_run_script_on_error_continue() -> None:
    path = _write_script("on_error: continue\ncommands:\n  - op: bogus\n  - op: info\n")
    reader = _FakeReader()
    out = StringIO()
    errors = run_script(reader, path, OutputConfig(), output_file=out)
    assert errors == 1  # one error, but execution continued


def test_run_script_on_error_stop() -> None:
    path = _write_script("on_error: stop\ncommands:\n  - op: bogus\n  - op: info\n")
    reader = _FakeReader()
    out = StringIO()
    errors = run_script(reader, path, OutputConfig(), output_file=out)
    assert errors == 1  # stopped after first error


def test_run_script_missing_key_in_get() -> None:
    path = _write_script("commands:\n  - op: get\n")
    reader = _FakeReader()
    out = StringIO()
    errors = run_script(reader, path, OutputConfig(), output_file=out)
    assert errors == 1


def test_run_script_shards() -> None:
    path = _write_script("commands:\n  - op: shards\n")
    reader = _FakeReader()
    out = StringIO()
    errors = run_script(reader, path, OutputConfig(), output_file=out)
    assert errors == 0
    parsed = json.loads(out.getvalue().strip())
    assert parsed["op"] == "shards"
    assert len(parsed["shards"]) == 2


def test_run_script_route() -> None:
    path = _write_script("commands:\n  - op: route\n    key: 10\n")
    reader = _FakeReader()
    out = StringIO()
    errors = run_script(reader, path, OutputConfig(), output_file=out)
    assert errors == 0
    parsed = json.loads(out.getvalue().strip())
    assert parsed["op"] == "route"
    assert parsed["db_id"] == 0
