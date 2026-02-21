"""Unit tests for the slate-reader Click CLI (app.py).

All tests mock the reader construction to avoid real S3 / SlateDB dependencies.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import click.testing

from slatedb_spark_sharded.cli.app import cli

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeReader:
    """Minimal mock of SlateShardedReader for CLI testing."""

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

    def snapshot_info(self) -> dict[str, Any]:
        return {
            "run_id": "test-run",
            "num_dbs": 2,
            "sharding": "hash",
            "created_at": "2026-01-01T00:00:00+00:00",
            "manifest_ref": "manifest-001.json",
        }

    def close(self) -> None:
        pass

    def __enter__(self) -> "_FakeReader":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _invoke(
    args: list[str],
    reader: _FakeReader | None = None,
    env: dict[str, str] | None = None,
) -> click.testing.Result:
    """Invoke the CLI with a mocked reader and CURRENT_URL env var."""
    if reader is None:
        reader = _FakeReader()

    effective_env = {"SLATE_READER_CURRENT": "s3://bucket/prefix/_CURRENT"}
    if env:
        effective_env.update(env)

    with patch("slatedb_spark_sharded.cli.app._build_reader", return_value=reader):
        runner = click.testing.CliRunner()
        return runner.invoke(cli, args, env=effective_env)


# ---------------------------------------------------------------------------
# --current-url is now an option, not a positional argument
# ---------------------------------------------------------------------------


class TestCurrentUrlOption:
    def test_get_subcommand_not_consumed_as_url(self) -> None:
        """Regression: 'get' should be parsed as a subcommand, not as CURRENT_URL."""
        reader = _FakeReader(store={42: b"hello"})
        result = _invoke(["get", "42"], reader=reader)
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "get"
        assert parsed["found"] is True

    def test_current_url_option_works(self) -> None:
        reader = _FakeReader(store={42: b"v"})
        result = _invoke(
            ["--current-url", "s3://b/p/_CURRENT", "get", "42"],
            reader=reader,
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Key coercion
# ---------------------------------------------------------------------------


class TestKeyCoercion:
    def test_get_coerces_int_for_u64be(self) -> None:
        reader = _FakeReader(store={42: b"value"}, key_encoding="u64be")
        result = _invoke(["get", "42"], reader=reader)
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["found"] is True

    def test_get_non_numeric_u64be_fails(self) -> None:
        reader = _FakeReader(key_encoding="u64be")
        result = _invoke(["get", "abc"], reader=reader)
        assert result.exit_code == 1
        err = json.loads(result.stderr)
        assert "error" in err

    def test_get_string_key_utf8(self) -> None:
        reader = _FakeReader(store={"hello": b"world"}, key_encoding="utf8")
        result = _invoke(["get", "hello"], reader=reader)
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["found"] is True

    def test_multiget_coerces_keys(self) -> None:
        reader = _FakeReader(store={1: b"a", 2: b"b"}, key_encoding="u64be")
        result = _invoke(["multiget", "1", "2"], reader=reader)
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "multiget"
        assert len(parsed["results"]) == 2


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


class TestSubcommands:
    def test_info(self) -> None:
        result = _invoke(["info"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "info"
        assert parsed["run_id"] == "test-run"

    def test_refresh(self) -> None:
        result = _invoke(["refresh"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "refresh"
        assert parsed["changed"] is False

    def test_get_not_found(self) -> None:
        reader = _FakeReader(store={}, key_encoding="utf8")
        result = _invoke(["get", "missing"], reader=reader)
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["found"] is False

    def test_output_format_override(self) -> None:
        reader = _FakeReader(store={42: b"v"})
        result = _invoke(["--output-format", "text", "get", "42"], reader=reader)
        assert result.exit_code == 0
        assert "42=" in result.output
