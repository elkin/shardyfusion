"""Unit tests for the shardy Click CLI (app.py).

All tests mock the reader construction to avoid real S3 / SlateDB dependencies.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import click.testing

from shardyfusion.cli.app import _build_manifest_store, cli
from shardyfusion.cli.config import ManifestStoreConfig
from shardyfusion.reader import ReaderHealth, ShardDetail, SnapshotInfo
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy

_FAKE_CREATED_AT = datetime.fromisoformat("2026-01-01T00:00:00+00:00")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeReader:
    """Minimal mock of ConcurrentShardedReader for CLI testing."""

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
            manifest_ref="manifest-001.json",
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
        status = "healthy"
        if staleness_threshold is not None and staleness_threshold.total_seconds() < 1:
            status = "degraded"
        return ReaderHealth(
            status=status,
            manifest_ref="manifest-001.json",
            manifest_age=timedelta(seconds=42.0),
            num_shards=2,
            is_closed=False,
        )

    def close(self) -> None:
        pass

    def __enter__(self) -> _FakeReader:
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

    effective_env = {"SHARDY_CURRENT": "s3://bucket/prefix/_CURRENT"}
    if env:
        effective_env.update(env)

    with patch("shardyfusion.cli.app._build_reader", return_value=reader):
        runner = click.testing.CliRunner()
        return runner.invoke(cli, args, env=effective_env)


# ---------------------------------------------------------------------------
# --current-url is now an option, not a positional argument
# ---------------------------------------------------------------------------


class TestVersionFlag:
    def test_version_output(self) -> None:
        runner = click.testing.CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "shardy" in result.output


class TestSubcommandHelp:
    def test_subcommand_help_does_not_require_current_url(self) -> None:
        runner = click.testing.CliRunner()

        for command in ("get", "search", "cleanup", "history"):
            result = runner.invoke(
                cli,
                [command, "--help"],
                env={"SHARDY_CURRENT": ""},
            )

            assert result.exit_code == 0, command
            assert f"Usage: cli {command}" in result.output


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
        assert all(r["found"] is True for r in parsed["results"])


# ---------------------------------------------------------------------------
# Stdin multiget
# ---------------------------------------------------------------------------


class TestMultigetStdin:
    def test_reads_keys_from_stdin(self) -> None:
        reader = _FakeReader(store={1: b"a", 2: b"b"}, key_encoding="u64be")
        effective_env = {"SHARDY_CURRENT": "s3://bucket/prefix/_CURRENT"}
        with patch("shardyfusion.cli.app._build_reader", return_value=reader):
            runner = click.testing.CliRunner()
            result = runner.invoke(
                cli, ["multiget", "-"], input="1\n2\n", env=effective_env
            )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "multiget"
        assert len(parsed["results"]) == 2
        assert all(r["found"] is True for r in parsed["results"])

    def test_empty_stdin_errors(self) -> None:
        reader = _FakeReader()
        effective_env = {"SHARDY_CURRENT": "s3://bucket/prefix/_CURRENT"}
        with patch("shardyfusion.cli.app._build_reader", return_value=reader):
            runner = click.testing.CliRunner()
            result = runner.invoke(cli, ["multiget", "-"], input="", env=effective_env)
        assert result.exit_code != 0


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
        assert parsed["key_encoding"] == "u64be"
        assert parsed["row_count"] == 0

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

    def test_shards(self) -> None:
        result = _invoke(["shards"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "shards"
        assert len(parsed["shards"]) == 2
        assert parsed["shards"][0]["db_id"] == 0
        assert parsed["shards"][1]["row_count"] == 20

    def test_route(self) -> None:
        result = _invoke(["route", "10"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "route"
        assert parsed["key"] == "10"
        assert parsed["db_id"] == 0

    def test_route_high_key(self) -> None:
        result = _invoke(["route", "80"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["db_id"] == 1


# ---------------------------------------------------------------------------
# _build_manifest_store dispatch
# ---------------------------------------------------------------------------


class TestBuildManifestStore:
    def test_s3_backend(self) -> None:
        store_cfg = ManifestStoreConfig(backend="s3")
        params: dict[str, Any] = {
            "s3_prefix": "s3://bucket/prefix",
            "current_pointer_key": "_CURRENT",
            "credential_provider": None,
            "s3_connection_options": {},
        }
        with patch(
            "shardyfusion.manifest_store.S3ManifestStore", autospec=True
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            store = _build_manifest_store(store_cfg, params)
            mock_cls.assert_called_once_with(
                "s3://bucket/prefix",
                current_pointer_key="_CURRENT",
                credential_provider=None,
                s3_connection_options={},
            )
            assert store is mock_cls.return_value

    def test_postgres_backend(self) -> None:
        store_cfg = ManifestStoreConfig(
            backend="postgres", dsn="host=localhost dbname=test"
        )
        params: dict[str, Any] = {
            "s3_prefix": "s3://bucket/prefix",
            "current_pointer_key": "_CURRENT",
            "credential_provider": None,
            "s3_connection_options": {},
        }
        with patch(
            "shardyfusion.db_manifest_store.PostgresManifestStore", autospec=True
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            store = _build_manifest_store(store_cfg, params)
            mock_cls.assert_called_once()
            assert store is mock_cls.return_value


# ---------------------------------------------------------------------------
# DB backend: s3_prefix required, --current-url warns
# ---------------------------------------------------------------------------


class TestDbBackendCliFlow:
    def _write_config(self, tmp_path: Any, toml_content: str) -> str:
        config_file = tmp_path / "reader.toml"
        config_file.write_text(toml_content)
        return str(config_file)

    def test_db_backend_requires_s3_prefix(self, tmp_path: Any) -> None:
        cfg_path = self._write_config(
            tmp_path,
            """\
[manifest_store]
backend = "postgres"
dsn = "host=localhost dbname=test"
""",
        )
        reader = _FakeReader()
        runner = click.testing.CliRunner()
        with patch("shardyfusion.cli.app._build_reader", return_value=reader):
            result = runner.invoke(
                cli,
                ["--config", cfg_path, "info"],
                env={"SHARDY_CURRENT": ""},
            )
        assert result.exit_code != 0
        assert "s3_prefix is required" in (result.output + (result.stderr or ""))

    def test_db_backend_current_url_warns(self, tmp_path: Any) -> None:
        cfg_path = self._write_config(
            tmp_path,
            """\
[reader]
s3_prefix = "s3://bucket/prefix"

[manifest_store]
backend = "postgres"
dsn = "host=localhost dbname=test"
""",
        )
        reader = _FakeReader()
        runner = click.testing.CliRunner()
        with patch("shardyfusion.cli.app._build_reader", return_value=reader):
            result = runner.invoke(
                cli,
                ["--config", cfg_path, "--current-url", "s3://b/p/_CURRENT", "info"],
            )
        assert result.exit_code == 0
        assert "--current-url is ignored" in result.output

    def test_db_backend_works_without_current_url(self, tmp_path: Any) -> None:
        cfg_path = self._write_config(
            tmp_path,
            """\
[reader]
s3_prefix = "s3://bucket/prefix"

[manifest_store]
backend = "postgres"
dsn = "host=localhost dbname=test"
""",
        )
        reader = _FakeReader()
        runner = click.testing.CliRunner()
        with patch("shardyfusion.cli.app._build_reader", return_value=reader):
            result = runner.invoke(
                cli,
                ["--config", cfg_path, "info"],
            )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "info"


# ---------------------------------------------------------------------------
# pool_checkout_timeout pass-through
# ---------------------------------------------------------------------------


class TestPoolCheckoutTimeout:
    def test_timeout_passed_to_reader(self, tmp_path: Any) -> None:
        cfg_path = tmp_path / "reader.toml"
        cfg_path.write_text(
            """\
[reader]
current_url = "s3://bucket/prefix/_CURRENT"
pool_checkout_timeout = 15.0
"""
        )
        reader = _FakeReader()
        captured_kwargs: dict[str, Any] = {}

        def _capture_reader(**kwargs: Any) -> _FakeReader:
            captured_kwargs.update(kwargs)
            return reader

        with (
            patch(
                "shardyfusion.reader.ConcurrentShardedReader",
                side_effect=_capture_reader,
            ),
            patch(
                "shardyfusion.cli.app._build_manifest_store",
                return_value=MagicMock(),
            ),
        ):
            runner = click.testing.CliRunner()
            result = runner.invoke(cli, ["--config", str(cfg_path), "info"])

        assert result.exit_code == 0
        assert captured_kwargs.get("pool_checkout_timeout") == timedelta(seconds=15.0)


# ---------------------------------------------------------------------------
# reader_backend selection
# ---------------------------------------------------------------------------


class TestReaderBackendSelection:
    def test_slatedb_default_no_factory_override(self, tmp_path: Any) -> None:
        cfg_path = tmp_path / "reader.toml"
        cfg_path.write_text(
            """\
[reader]
current_url = "s3://bucket/prefix/_CURRENT"
"""
        )
        captured_kwargs: dict[str, Any] = {}

        def _capture_reader(**kwargs: Any) -> _FakeReader:
            captured_kwargs.update(kwargs)
            return _FakeReader()

        with (
            patch(
                "shardyfusion.reader.ConcurrentShardedReader",
                side_effect=_capture_reader,
            ),
            patch(
                "shardyfusion.cli.app._build_manifest_store",
                return_value=MagicMock(),
            ),
        ):
            runner = click.testing.CliRunner()
            result = runner.invoke(cli, ["--config", str(cfg_path), "info"])

        assert result.exit_code == 0
        assert captured_kwargs.get("reader_factory") is None

    def test_sqlite_injects_sqlite_reader_factory(self, tmp_path: Any) -> None:
        """Default sqlite_mode='auto' wires up AdaptiveSqliteReaderFactory."""
        cfg_path = tmp_path / "reader.toml"
        cfg_path.write_text(
            """\
[reader]
current_url = "s3://bucket/prefix/_CURRENT"
reader_backend = "sqlite"
"""
        )
        captured_kwargs: dict[str, Any] = {}

        def _capture_reader(**kwargs: Any) -> _FakeReader:
            captured_kwargs.update(kwargs)
            return _FakeReader()

        with (
            patch(
                "shardyfusion.reader.ConcurrentShardedReader",
                side_effect=_capture_reader,
            ),
            patch(
                "shardyfusion.cli.app._build_manifest_store",
                return_value=MagicMock(),
            ),
        ):
            runner = click.testing.CliRunner()
            result = runner.invoke(cli, ["--config", str(cfg_path), "info"])

        assert result.exit_code == 0
        from shardyfusion.sqlite_adapter import AdaptiveSqliteReaderFactory

        assert isinstance(
            captured_kwargs.get("reader_factory"), AdaptiveSqliteReaderFactory
        )

    def test_sqlite_mode_download_injects_download_factory(self, tmp_path: Any) -> None:
        cfg_path = tmp_path / "reader.toml"
        cfg_path.write_text(
            """\
[reader]
current_url = "s3://bucket/prefix/_CURRENT"
reader_backend = "sqlite"
sqlite_mode = "download"
"""
        )
        captured_kwargs: dict[str, Any] = {}

        def _capture_reader(**kwargs: Any) -> _FakeReader:
            captured_kwargs.update(kwargs)
            return _FakeReader()

        with (
            patch(
                "shardyfusion.reader.ConcurrentShardedReader",
                side_effect=_capture_reader,
            ),
            patch(
                "shardyfusion.cli.app._build_manifest_store",
                return_value=MagicMock(),
            ),
        ):
            runner = click.testing.CliRunner()
            result = runner.invoke(cli, ["--config", str(cfg_path), "info"])

        assert result.exit_code == 0
        from shardyfusion.sqlite_adapter import SqliteReaderFactory

        assert isinstance(captured_kwargs.get("reader_factory"), SqliteReaderFactory)

    def test_sqlite_mode_range_injects_range_factory(self, tmp_path: Any) -> None:
        cfg_path = tmp_path / "reader.toml"
        cfg_path.write_text(
            """\
[reader]
current_url = "s3://bucket/prefix/_CURRENT"
reader_backend = "sqlite"
sqlite_mode = "range"
"""
        )
        captured_kwargs: dict[str, Any] = {}

        def _capture_reader(**kwargs: Any) -> _FakeReader:
            captured_kwargs.update(kwargs)
            return _FakeReader()

        with (
            patch(
                "shardyfusion.reader.ConcurrentShardedReader",
                side_effect=_capture_reader,
            ),
            patch(
                "shardyfusion.cli.app._build_manifest_store",
                return_value=MagicMock(),
            ),
        ):
            runner = click.testing.CliRunner()
            result = runner.invoke(cli, ["--config", str(cfg_path), "info"])

        assert result.exit_code == 0
        from shardyfusion.sqlite_adapter import SqliteRangeReaderFactory

        assert isinstance(
            captured_kwargs.get("reader_factory"), SqliteRangeReaderFactory
        )

    def test_sqlite_mode_cli_flag_overrides_toml(self, tmp_path: Any) -> None:
        """`--sqlite-mode download` on the CLI overrides reader.toml's value."""
        cfg_path = tmp_path / "reader.toml"
        cfg_path.write_text(
            """\
[reader]
current_url = "s3://bucket/prefix/_CURRENT"
reader_backend = "sqlite"
sqlite_mode = "auto"
"""
        )
        captured_kwargs: dict[str, Any] = {}

        def _capture_reader(**kwargs: Any) -> _FakeReader:
            captured_kwargs.update(kwargs)
            return _FakeReader()

        with (
            patch(
                "shardyfusion.reader.ConcurrentShardedReader",
                side_effect=_capture_reader,
            ),
            patch(
                "shardyfusion.cli.app._build_manifest_store",
                return_value=MagicMock(),
            ),
        ):
            runner = click.testing.CliRunner()
            result = runner.invoke(
                cli,
                [
                    "--config",
                    str(cfg_path),
                    "--sqlite-mode",
                    "download",
                    "info",
                ],
            )

        assert result.exit_code == 0
        from shardyfusion.sqlite_adapter import (
            AdaptiveSqliteReaderFactory,
            SqliteReaderFactory,
        )

        factory = captured_kwargs.get("reader_factory")
        assert isinstance(factory, SqliteReaderFactory)
        assert not isinstance(factory, AdaptiveSqliteReaderFactory)

    def test_sqlite_auto_threshold_overrides_propagate(self, tmp_path: Any) -> None:
        cfg_path = tmp_path / "reader.toml"
        cfg_path.write_text(
            """\
[reader]
current_url = "s3://bucket/prefix/_CURRENT"
reader_backend = "sqlite"
"""
        )
        captured_kwargs: dict[str, Any] = {}

        def _capture_reader(**kwargs: Any) -> _FakeReader:
            captured_kwargs.update(kwargs)
            return _FakeReader()

        with (
            patch(
                "shardyfusion.reader.ConcurrentShardedReader",
                side_effect=_capture_reader,
            ),
            patch(
                "shardyfusion.cli.app._build_manifest_store",
                return_value=MagicMock(),
            ),
        ):
            runner = click.testing.CliRunner()
            result = runner.invoke(
                cli,
                [
                    "--config",
                    str(cfg_path),
                    "--sqlite-auto-per-shard-bytes",
                    "1024",
                    "--sqlite-auto-total-bytes",
                    "8192",
                    "info",
                ],
            )

        assert result.exit_code == 0
        from shardyfusion.sqlite_adapter import (
            AdaptiveSqliteReaderFactory,
            _ThresholdPolicy,
        )

        factory = captured_kwargs.get("reader_factory")
        assert isinstance(factory, AdaptiveSqliteReaderFactory)
        policy = factory._policy
        assert isinstance(policy, _ThresholdPolicy)
        assert policy.per_shard_threshold == 1024
        assert policy.total_budget == 8192


class TestSqliteOverrideValidation:
    """--sqlite-auto-* overrides go through ReaderConfig.model_validate.

    Negative values must produce a clean UsageError (exit 2) — not a stack
    trace from pydantic's ValidationError.
    """

    def _cfg(self, tmp_path: Any) -> Any:
        cfg_path = tmp_path / "reader.toml"
        cfg_path.write_text(
            """\
[reader]
current_url = "s3://bucket/prefix/_CURRENT"
reader_backend = "sqlite"
"""
        )
        return cfg_path

    def test_negative_per_shard_threshold_emits_usage_error(
        self, tmp_path: Any
    ) -> None:
        cfg_path = self._cfg(tmp_path)
        runner = click.testing.CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg_path),
                "--sqlite-auto-per-shard-bytes",
                "-1",
                "info",
            ],
        )

        assert result.exit_code != 0
        # Click formats UsageError with "Error:" prefix; ValidationError would
        # surface as an unhandled exception (different output / non-zero exit
        # without "Error:" prefix).
        assert "Invalid SQLite override" in result.output
        # Make sure we didn't leak a raw traceback
        assert "Traceback" not in result.output

    def test_negative_total_budget_emits_usage_error(self, tmp_path: Any) -> None:
        cfg_path = self._cfg(tmp_path)
        runner = click.testing.CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg_path),
                "--sqlite-auto-total-bytes",
                "-100",
                "info",
            ],
        )

        assert result.exit_code != 0
        assert "Invalid SQLite override" in result.output
        assert "Traceback" not in result.output


# ---------------------------------------------------------------------------
# --ref / --offset must never call set_current()
# ---------------------------------------------------------------------------


class TestManifestTargetingDoesNotMutate:
    """Verify that --ref and --offset never mutate _CURRENT via set_current()."""

    def _make_manifest_store_mock(self) -> MagicMock:
        """Build a MagicMock manifest store with realistic return values."""
        from shardyfusion.manifest import ManifestRef

        store = MagicMock()
        ref = ManifestRef(
            ref="manifests/2026-01-01T00:00:00.000000Z_run_id=abc/manifest",
            run_id="abc",
            published_at=_FAKE_CREATED_AT,
        )
        store.load_current.return_value = ref
        store.list_manifests.return_value = [ref]
        manifest_mock = MagicMock()
        manifest_mock.required_build.run_id = "abc"
        store.load_manifest.return_value = manifest_mock
        return store

    def test_get_with_ref_does_not_call_set_current(self) -> None:
        reader = _FakeReader(store={42: b"hello"})
        manifest_store = self._make_manifest_store_mock()

        with (
            patch(
                "shardyfusion.cli.app._build_manifest_store",
                return_value=manifest_store,
            ),
            patch("shardyfusion.cli.app._build_reader", return_value=reader),
        ):
            runner = click.testing.CliRunner()
            result = runner.invoke(
                cli,
                ["--ref", "some-ref", "get", "42"],
                env={"SHARDY_CURRENT": "s3://bucket/prefix/_CURRENT"},
            )

        assert result.exit_code == 0
        manifest_store.set_current.assert_not_called()

    def test_get_with_offset_does_not_call_set_current(self) -> None:
        reader = _FakeReader(store={42: b"hello"})
        manifest_store = self._make_manifest_store_mock()

        with (
            patch(
                "shardyfusion.cli.app._build_manifest_store",
                return_value=manifest_store,
            ),
            patch("shardyfusion.cli.app._build_reader", return_value=reader),
        ):
            runner = click.testing.CliRunner()
            result = runner.invoke(
                cli,
                ["--offset", "0", "get", "42"],
                env={"SHARDY_CURRENT": "s3://bucket/prefix/_CURRENT"},
            )

        assert result.exit_code == 0
        manifest_store.set_current.assert_not_called()

    def test_cleanup_with_ref_does_not_call_set_current(self) -> None:
        manifest_store = self._make_manifest_store_mock()

        with (
            patch(
                "shardyfusion.cli.app._build_manifest_store",
                return_value=manifest_store,
            ),
            patch(
                "shardyfusion.cli.app._build_reader",
                return_value=_FakeReader(),
            ),
            patch(
                "shardyfusion._writer_core.cleanup_stale_attempts",
                return_value=[],
            ),
            patch(
                "shardyfusion.storage.create_s3_client",
                return_value=MagicMock(),
            ),
        ):
            runner = click.testing.CliRunner()
            result = runner.invoke(
                cli,
                ["--ref", "some-ref", "cleanup", "--dry-run"],
                env={"SHARDY_CURRENT": "s3://bucket/prefix/_CURRENT"},
            )

        assert result.exit_code == 0
        manifest_store.set_current.assert_not_called()

    def test_info_with_offset_does_not_call_set_current(self) -> None:
        reader = _FakeReader()
        manifest_store = self._make_manifest_store_mock()

        with (
            patch(
                "shardyfusion.cli.app._build_manifest_store",
                return_value=manifest_store,
            ),
            patch("shardyfusion.cli.app._build_reader", return_value=reader),
        ):
            runner = click.testing.CliRunner()
            result = runner.invoke(
                cli,
                ["--offset", "0", "info"],
                env={"SHARDY_CURRENT": "s3://bucket/prefix/_CURRENT"},
            )

        assert result.exit_code == 0
        manifest_store.set_current.assert_not_called()


# ---------------------------------------------------------------------------
# health subcommand
# ---------------------------------------------------------------------------


class TestHealthSubcommand:
    def test_health_healthy(self) -> None:
        result = _invoke(["health"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "health"
        assert parsed["status"] == "healthy"
        assert parsed["num_shards"] == 2
        assert parsed["is_closed"] is False

    def test_health_degraded_exit_code(self) -> None:
        """A degraded reader should exit with code 1."""
        reader = _FakeReader()
        result = _invoke(["health", "--staleness-threshold", "0.001"], reader=reader)
        assert result.exit_code == 1
        parsed = json.loads(result.output)
        assert parsed["status"] == "degraded"

    def test_health_text_format(self) -> None:
        result = _invoke(["--output-format", "text", "health"])
        assert result.exit_code == 0
        assert "status=healthy" in result.output


# ---------------------------------------------------------------------------
# get --strict
# ---------------------------------------------------------------------------


class TestGetStrict:
    def test_strict_key_found_exits_0(self) -> None:
        reader = _FakeReader(store={42: b"hello"})
        result = _invoke(["get", "--strict", "42"], reader=reader)
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["found"] is True

    def test_strict_key_not_found_exits_1(self) -> None:
        reader = _FakeReader(store={}, key_encoding="utf8")
        result = _invoke(["get", "--strict", "missing"], reader=reader)
        assert result.exit_code == 1
        # Result is still emitted to stdout
        parsed = json.loads(result.output)
        assert parsed["found"] is False

    def test_without_strict_not_found_exits_0(self) -> None:
        reader = _FakeReader(store={}, key_encoding="utf8")
        result = _invoke(["get", "missing"], reader=reader)
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["found"] is False


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def _invoke_search(self, args: list[str]) -> click.testing.Result:
        runner = click.testing.CliRunner(
            env={"SHARDY_CURRENT": "s3://bucket/prefix/_CURRENT"}
        )
        # Patch _build_reader so existing commands still work, but search builds
        # its own reader from the manifest store.
        with patch(
            "shardyfusion.cli.app._build_reader",
            return_value=_FakeReader(),
        ):
            return runner.invoke(cli, args, obj={})

    def test_search_no_query_or_file_errors(self) -> None:
        result = self._invoke_search(["search"])
        assert result.exit_code != 0
        assert "Provide either" in result.output or "Usage:" in result.output

    def test_search_comma_separated_query(self) -> None:
        fake_manifest = MagicMock()
        fake_manifest.custom = {
            "vector": {
                "dim": 3,
                "metric": "cosine",
                "unified": False,
            }
        }
        fake_ref = MagicMock()
        fake_ref.ref = "manifest.json"
        fake_store = MagicMock()
        fake_store.load_current.return_value = fake_ref
        fake_store.load_manifest.return_value = fake_manifest

        fake_response = MagicMock()
        fake_response.num_shards_queried = 1
        fake_response.latency_ms = 1.5
        fake_response.results = [
            MagicMock(id="a", score=0.1, payload={"x": 1}),
        ]

        with patch(
            "shardyfusion.cli.app._build_manifest_store", return_value=fake_store
        ):
            with patch("shardyfusion.vector.reader.ShardedVectorReader") as MockReader:
                MockReader.return_value.__enter__ = lambda self: self
                MockReader.return_value.__exit__ = lambda *args: None
                MockReader.return_value.search.return_value = fake_response
                result = self._invoke_search(["search", "0.1,0.2,0.3", "--top-k", "1"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "search"
        assert parsed["top_k"] == 1
        assert len(parsed["results"]) == 1
        assert parsed["results"][0]["id"] == "a"

    def test_search_unified_snapshot(self) -> None:
        fake_manifest = MagicMock()
        fake_manifest.custom = {
            "vector": {
                "dim": 2,
                "metric": "l2",
                "unified": True,
            }
        }
        fake_ref = MagicMock()
        fake_ref.ref = "manifest.json"
        fake_store = MagicMock()
        fake_store.load_current.return_value = fake_ref
        fake_store.load_manifest.return_value = fake_manifest

        fake_response = MagicMock()
        fake_response.num_shards_queried = 2
        fake_response.latency_ms = 2.0
        fake_response.results = []

        with patch(
            "shardyfusion.cli.app._build_manifest_store", return_value=fake_store
        ):
            with patch(
                "shardyfusion.reader.unified_reader.UnifiedShardedReader"
            ) as MockReader:
                MockReader.return_value.__enter__ = lambda self: self
                MockReader.return_value.__exit__ = lambda *args: None
                MockReader.return_value.search.return_value = fake_response
                result = self._invoke_search(["search", "1.0,2.0"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["op"] == "search"
        assert parsed["num_shards_queried"] == 2

    def test_search_no_vector_metadata(self) -> None:
        fake_manifest = MagicMock()
        fake_manifest.custom = {}
        fake_ref = MagicMock()
        fake_ref.ref = "manifest.json"
        fake_store = MagicMock()
        fake_store.load_current.return_value = fake_ref
        fake_store.load_manifest.return_value = fake_manifest

        with patch(
            "shardyfusion.cli.app._build_manifest_store", return_value=fake_store
        ):
            result = self._invoke_search(["search", "0.1,0.2"])

        assert result.exit_code != 0
        assert "does not contain vector metadata" in result.output


class TestSearchUnifiedKvFactoryWiring:
    """``shardy search`` must honor --sqlite-mode/--sqlite-auto-* for the
    unified reader's KV factory; the unified reader's auto-dispatch otherwise
    ignores the CLI ``reader_cfg``.

    These tests verify ``_build_unified_kv_factory`` is invoked and the
    correct concrete factory is passed via ``reader_factory=`` to
    :class:`UnifiedShardedReader`.
    """

    def _make_store(self, vector_meta: dict[str, Any]) -> Any:
        fake_manifest = MagicMock()
        fake_manifest.custom = {"vector": vector_meta}
        fake_ref = MagicMock()
        fake_ref.ref = "manifest.json"
        fake_store = MagicMock()
        fake_store.load_current.return_value = fake_ref
        fake_store.load_manifest.return_value = fake_manifest
        return fake_store

    def _cfg(self, tmp_path: Any, sqlite_mode: str = "auto") -> Any:
        cfg_path = tmp_path / "reader.toml"
        cfg_path.write_text(
            f"""\
[reader]
current_url = "s3://bucket/prefix/_CURRENT"
reader_backend = "sqlite"
sqlite_mode = "{sqlite_mode}"
"""
        )
        return cfg_path

    def _invoke(
        self,
        args: list[str],
        *,
        store: Any,
        captured: dict[str, Any],
    ) -> Any:
        fake_response = MagicMock()
        fake_response.num_shards_queried = 1
        fake_response.latency_ms = 1.0
        fake_response.results = []

        def _capture_unified(**kwargs: Any) -> Any:
            captured.update(kwargs)
            mock_reader = MagicMock()
            mock_reader.search.return_value = fake_response
            mock_reader.close.return_value = None
            return mock_reader

        with (
            patch(
                "shardyfusion.cli.app._build_manifest_store", return_value=store
            ),
            patch(
                "shardyfusion.reader.unified_reader.UnifiedShardedReader",
                side_effect=_capture_unified,
            ),
            patch(
                "shardyfusion.cli.app._build_reader",
                return_value=_FakeReader(),
            ),
        ):
            runner = click.testing.CliRunner()
            return runner.invoke(cli, args, obj={})

    def test_sqlite_vec_auto_uses_adaptive_factory(self, tmp_path: Any) -> None:
        from shardyfusion.sqlite_vec_adapter import (
            AdaptiveSqliteVecReaderFactory,
            _ThresholdPolicy,
        )

        store = self._make_store(
            {"dim": 2, "metric": "l2", "unified": True, "backend": "sqlite-vec"}
        )
        captured: dict[str, Any] = {}
        cfg_path = self._cfg(tmp_path, sqlite_mode="auto")

        result = self._invoke(
            [
                "--config",
                str(cfg_path),
                "--sqlite-auto-per-shard-bytes",
                "1024",
                "--sqlite-auto-total-bytes",
                "8192",
                "search",
                "1.0,2.0",
            ],
            store=store,
            captured=captured,
        )

        assert result.exit_code == 0, result.output
        factory = captured.get("reader_factory")
        assert isinstance(factory, AdaptiveSqliteVecReaderFactory)
        # Thresholds from CLI propagate into the policy
        policy = factory._policy
        assert isinstance(policy, _ThresholdPolicy)
        assert policy.per_shard_threshold == 1024
        assert policy.total_budget == 8192

    def test_sqlite_vec_download_uses_concrete_download_factory(
        self, tmp_path: Any
    ) -> None:
        from shardyfusion.sqlite_vec_adapter import (
            AdaptiveSqliteVecReaderFactory,
            SqliteVecReaderFactory,
        )

        store = self._make_store(
            {"dim": 2, "metric": "l2", "unified": True, "backend": "sqlite-vec"}
        )
        captured: dict[str, Any] = {}
        cfg_path = self._cfg(tmp_path, sqlite_mode="download")

        result = self._invoke(
            ["--config", str(cfg_path), "search", "1.0,2.0"],
            store=store,
            captured=captured,
        )

        assert result.exit_code == 0, result.output
        factory = captured.get("reader_factory")
        assert isinstance(factory, SqliteVecReaderFactory)
        assert not isinstance(factory, AdaptiveSqliteVecReaderFactory)

    def test_lancedb_with_sqlite_kv_uses_composite_factory(
        self, tmp_path: Any
    ) -> None:
        from shardyfusion.composite_adapter import CompositeReaderFactory
        from shardyfusion.sqlite_adapter import AdaptiveSqliteReaderFactory

        store = self._make_store(
            {
                "dim": 2,
                "metric": "l2",
                "unified": True,
                "backend": "lancedb",
                "kv_backend": "sqlite",
            }
        )
        captured: dict[str, Any] = {}
        cfg_path = self._cfg(tmp_path, sqlite_mode="auto")

        result = self._invoke(
            ["--config", str(cfg_path), "search", "1.0,2.0"],
            store=store,
            captured=captured,
        )

        assert result.exit_code == 0, result.output
        factory = captured.get("reader_factory")
        assert isinstance(factory, CompositeReaderFactory)
        # The KV side must respect sqlite_mode = "auto"
        assert isinstance(factory.kv_factory, AdaptiveSqliteReaderFactory)

    def test_lancedb_with_slatedb_kv_falls_back_to_auto_dispatch(
        self, tmp_path: Any
    ) -> None:
        """When KV backend is SlateDB, sqlite_mode is irrelevant; pass None
        so UnifiedShardedReader's auto-dispatch picks the right factory."""
        store = self._make_store(
            {
                "dim": 2,
                "metric": "l2",
                "unified": True,
                "backend": "lancedb",
                "kv_backend": "slatedb",
            }
        )
        captured: dict[str, Any] = {}
        cfg_path = self._cfg(tmp_path, sqlite_mode="auto")

        result = self._invoke(
            ["--config", str(cfg_path), "search", "1.0,2.0"],
            store=store,
            captured=captured,
        )

        assert result.exit_code == 0, result.output
        # No SQLite involved → helper returns None → auto-dispatch path
        assert captured.get("reader_factory") is None
