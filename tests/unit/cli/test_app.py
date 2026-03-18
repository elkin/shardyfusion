"""Unit tests for the shardy Click CLI (app.py).

All tests mock the reader construction to avoid real S3 / SlateDB dependencies.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import click.testing

from shardyfusion.cli.app import _build_manifest_store, cli
from shardyfusion.cli.config import ManifestStoreConfig
from shardyfusion.reader import ShardDetail, SnapshotInfo
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
            "current_name": "_CURRENT",
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
                current_name="_CURRENT",
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
            "current_name": "_CURRENT",
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

    def test_comdb2_backend(self) -> None:
        store_cfg = ManifestStoreConfig(backend="comdb2", dsn="mydb")
        params: dict[str, Any] = {
            "s3_prefix": "s3://bucket/prefix",
            "current_name": "_CURRENT",
            "credential_provider": None,
            "s3_connection_options": {},
        }
        with patch(
            "shardyfusion.db_manifest_store.Comdb2ManifestStore", autospec=True
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
        assert captured_kwargs.get("pool_checkout_timeout") == 15.0


# ---------------------------------------------------------------------------
# Schema subcommand (no reader needed)
# ---------------------------------------------------------------------------


class TestSchemaSubcommand:
    def test_manifest_schema_default(self) -> None:
        runner = click.testing.CliRunner()
        result = runner.invoke(cli, ["schema"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert parsed["title"] == "SlateDB Sharded Manifest"
        assert "properties" in parsed

    def test_manifest_schema_explicit(self) -> None:
        runner = click.testing.CliRunner()
        result = runner.invoke(cli, ["schema", "--type", "manifest"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["title"] == "SlateDB Sharded Manifest"

    def test_current_pointer_schema(self) -> None:
        runner = click.testing.CliRunner()
        result = runner.invoke(cli, ["schema", "--type", "current-pointer"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert parsed["title"] == "SlateDB Sharded CURRENT Pointer"
        assert "manifest_ref" in parsed["properties"]

    def test_invalid_type_rejected(self) -> None:
        runner = click.testing.CliRunner()
        result = runner.invoke(cli, ["schema", "--type", "bogus"])
        assert result.exit_code != 0


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
