"""Tests for CLI batch script error handling edge cases.

Covers: malformed YAML content, missing op fields, reader exception
during execution, on_error behavior edge cases.
"""

from __future__ import annotations

import tempfile
from io import StringIO
from typing import Any

import pytest

from shardyfusion.cli.batch import load_script, run_script
from shardyfusion.cli.config import OutputConfig
from shardyfusion.reader import ShardDetail, SnapshotInfo
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy


class _FakeReader:
    def __init__(
        self,
        *,
        get_raises: Exception | None = None,
    ) -> None:
        self._get_raises = get_raises

    @property
    def key_encoding(self) -> str:
        return "u64be"

    def get(self, key: Any, **kw: Any) -> bytes | None:
        if self._get_raises:
            raise self._get_raises
        return b"val"

    def multi_get(self, keys: list[Any], **kw: Any) -> dict[Any, bytes | None]:
        if self._get_raises:
            raise self._get_raises
        return {k: b"val" for k in keys}

    def refresh(self) -> bool:
        return False

    def snapshot_info(self) -> SnapshotInfo:
        from datetime import datetime

        return SnapshotInfo(
            run_id="r",
            num_dbs=1,
            sharding=ShardingStrategy.HASH,
            created_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
            manifest_ref="s3://b/m",
            key_encoding=KeyEncoding.U64BE,
            row_count=0,
        )

    def shard_details(self) -> list[ShardDetail]:
        return []

    def route_key(self, key: Any, **kw: Any) -> int:
        return 0


def _write_script(content: str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False)
    f.write(content)
    f.close()
    return f.name


class TestLoadScriptErrors:
    def test_empty_commands_list(self) -> None:
        path = _write_script("commands: []\n")
        data = load_script(path)
        assert data["commands"] == []

    def test_commands_is_string_raises(self) -> None:
        path = _write_script("commands: not_a_list\n")
        with pytest.raises(ValueError, match="list"):
            load_script(path)

    def test_commands_is_int_raises(self) -> None:
        path = _write_script("commands: 42\n")
        with pytest.raises(ValueError, match="list"):
            load_script(path)


class TestRunScriptErrors:
    def test_missing_op_field(self) -> None:
        """Command dict without 'op' key → error."""
        path = _write_script("commands:\n  - key: 42\n")
        reader = _FakeReader()
        out = StringIO()
        errors = run_script(reader, path, OutputConfig(), output_file=out)
        assert errors == 1

    def test_get_missing_key_field(self) -> None:
        path = _write_script("commands:\n  - op: get\n")
        reader = _FakeReader()
        out = StringIO()
        errors = run_script(reader, path, OutputConfig(), output_file=out)
        assert errors == 1

    def test_multiget_empty_keys(self) -> None:
        path = _write_script("commands:\n  - op: multiget\n    keys: []\n")
        reader = _FakeReader()
        out = StringIO()
        errors = run_script(reader, path, OutputConfig(), output_file=out)
        assert errors == 1

    def test_multiget_keys_not_list(self) -> None:
        path = _write_script("commands:\n  - op: multiget\n    keys: single\n")
        reader = _FakeReader()
        out = StringIO()
        errors = run_script(reader, path, OutputConfig(), output_file=out)
        assert errors == 1

    def test_route_missing_key(self) -> None:
        path = _write_script("commands:\n  - op: route\n")
        reader = _FakeReader()
        out = StringIO()
        errors = run_script(reader, path, OutputConfig(), output_file=out)
        assert errors == 1

    def test_reader_exception_during_get(self) -> None:
        """Reader raising during get() → caught and reported as error."""
        path = _write_script("commands:\n  - op: get\n    key: 42\n")
        reader = _FakeReader(get_raises=RuntimeError("db crash"))
        out = StringIO()
        errors = run_script(reader, path, OutputConfig(), output_file=out)
        assert errors == 1

    def test_on_error_continue_processes_all(self) -> None:
        """With on_error=continue, subsequent commands still run after error."""
        script = (
            "on_error: continue\n"
            "commands:\n"
            "  - op: get\n"  # missing key → error
            "  - op: get\n"  # missing key → error
            "  - op: info\n"  # succeeds
        )
        path = _write_script(script)
        reader = _FakeReader()
        out = StringIO()
        errors = run_script(reader, path, OutputConfig(), output_file=out)
        assert errors == 2  # two errors, but info still ran

    def test_on_error_stop_halts_at_first(self) -> None:
        """With on_error=stop (default), processing stops at first error."""
        script = "on_error: stop\ncommands:\n  - not_a_mapping\n  - op: info\n"
        path = _write_script(script)
        reader = _FakeReader()
        out = StringIO()
        errors = run_script(reader, path, OutputConfig(), output_file=out)
        assert errors == 1
        # Only one line of output (the error), not two
        lines = [line for line in out.getvalue().strip().split("\n") if line]
        assert len(lines) == 1
