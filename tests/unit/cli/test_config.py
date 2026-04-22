"""Unit tests for shardyfusion.cli.config helpers."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from shardyfusion.cli.config import (
    ManifestStoreConfig,
    ReaderConfig,
    build_connection_factory,
    build_s3_config,
    coerce_cli_key,
    coerce_s3_option,
    load_reader_config,
    resolve_current_url,
    resolve_dsn,
    split_current_url,
)

# ---------------------------------------------------------------------------
# coerce_s3_option
# ---------------------------------------------------------------------------


class TestCoerceS3Option:
    def test_bool_true_variants(self) -> None:
        for val in ("true", "True", "1", "yes"):
            assert coerce_s3_option("verify_ssl", val) is True

    def test_bool_false_variants(self) -> None:
        for val in ("false", "False", "0", "no"):
            assert coerce_s3_option("verify_ssl", val) is False

    def test_bool_invalid(self) -> None:
        with pytest.raises(ValueError, match="Cannot coerce"):
            coerce_s3_option("verify_ssl", "maybe")

    def test_int_coercion(self) -> None:
        assert coerce_s3_option("connect_timeout", "30") == 30
        assert coerce_s3_option("read_timeout", "60") == 60

    def test_int_invalid(self) -> None:
        with pytest.raises(ValueError):
            coerce_s3_option("connect_timeout", "abc")

    def test_str_passthrough(self) -> None:
        assert coerce_s3_option("addressing_style", "path") == "path"

    def test_unknown_key_returns_str(self) -> None:
        assert coerce_s3_option("unknown_key", "value") == "value"


# ---------------------------------------------------------------------------
# split_current_url
# ---------------------------------------------------------------------------


class TestSplitCurrentUrl:
    def test_basic_split(self) -> None:
        prefix, name = split_current_url("s3://bucket/prefix/_CURRENT")
        assert prefix == "s3://bucket/prefix"
        assert name == "_CURRENT"

    def test_trailing_slash_stripped(self) -> None:
        prefix, name = split_current_url("s3://bucket/prefix/_CURRENT/")
        assert prefix == "s3://bucket/prefix"
        assert name == "_CURRENT"

    def test_no_slash_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot split"):
            split_current_url("noslash")


# ---------------------------------------------------------------------------
# resolve_current_url
# ---------------------------------------------------------------------------


class TestResolveCurrentUrl:
    def test_positional_wins(self) -> None:
        cfg = ReaderConfig(current_url="s3://from-config/_CURRENT")
        result = resolve_current_url("s3://positional/_CURRENT", cfg)
        assert result == "s3://positional/_CURRENT"

    def test_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHARDY_CURRENT", "s3://from-env/_CURRENT")
        result = resolve_current_url(None, ReaderConfig())
        assert result == "s3://from-env/_CURRENT"

    def test_config_fallback(self) -> None:
        cfg = ReaderConfig(current_url="s3://from-config/_CURRENT")
        result = resolve_current_url(None, cfg)
        assert result == "s3://from-config/_CURRENT"

    def test_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SHARDY_CURRENT", raising=False)
        with pytest.raises(SystemExit, match="CURRENT URL is required"):
            resolve_current_url(None, ReaderConfig())


# ---------------------------------------------------------------------------
# coerce_cli_key
# ---------------------------------------------------------------------------


class TestCoerceCliKey:
    def test_u64be_int_string(self) -> None:
        assert coerce_cli_key("42", "u64be") == 42

    def test_u64be_negative(self) -> None:
        assert coerce_cli_key("-1", "u64be") == -1

    def test_u64be_non_numeric_raises(self) -> None:
        with pytest.raises(ValueError, match="not a valid integer"):
            coerce_cli_key("abc", "u64be")

    def test_utf8_passthrough(self) -> None:
        assert coerce_cli_key("hello", "utf8") == "hello"

    def test_u32be_int_string(self) -> None:
        assert coerce_cli_key("42", "u32be") == 42

    def test_u32be_non_numeric_raises(self) -> None:
        with pytest.raises(ValueError, match="not a valid integer"):
            coerce_cli_key("abc", "u32be")

    def test_raw_encoding_returns_bytes(self) -> None:
        assert coerce_cli_key("42", "raw") == b"42"

    def test_utf8_encoding_passthrough(self) -> None:
        assert coerce_cli_key("hello", "utf8") == "hello"


# ---------------------------------------------------------------------------
# build_s3_config
# ---------------------------------------------------------------------------


class TestBuildS3Config:
    def test_no_profile(self) -> None:
        cred_provider, conn_opts = build_s3_config(None)
        assert cred_provider is None
        assert conn_opts == {}

    def test_overrides_applied(self) -> None:
        cred_provider, conn_opts = build_s3_config(None, {"connect_timeout": 99})
        assert cred_provider is None
        assert conn_opts["connect_timeout"] == 99


# ---------------------------------------------------------------------------
# ManifestStoreConfig
# ---------------------------------------------------------------------------


class TestManifestStoreConfig:
    def test_defaults(self) -> None:
        cfg = ManifestStoreConfig()
        assert cfg.backend == "s3"
        assert cfg.dsn is None
        assert cfg.table_name == "shardyfusion_manifests"
        assert cfg.ensure_table is True

    def test_postgres_backend(self) -> None:
        cfg = ManifestStoreConfig(backend="postgres", dsn="host=localhost dbname=mydb")
        assert cfg.backend == "postgres"
        assert cfg.dsn == "host=localhost dbname=mydb"

    def test_custom_table_name(self) -> None:
        cfg = ManifestStoreConfig(table_name="my_manifests")
        assert cfg.table_name == "my_manifests"

    def test_invalid_backend_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ManifestStoreConfig(backend="sqlite")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ReaderConfig new fields
# ---------------------------------------------------------------------------


class TestReaderConfigNewFields:
    def test_pool_checkout_timeout_default(self) -> None:
        cfg = ReaderConfig()
        assert cfg.pool_checkout_timeout == timedelta(seconds=30.0)

    def test_pool_checkout_timeout_custom(self) -> None:
        cfg = ReaderConfig(pool_checkout_timeout=timedelta(seconds=15.0))
        assert cfg.pool_checkout_timeout == timedelta(seconds=15.0)

    def test_pool_checkout_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            ReaderConfig(pool_checkout_timeout=timedelta(seconds=0))

    def test_pool_checkout_timeout_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReaderConfig(pool_checkout_timeout=timedelta(seconds=-1.0))

    def test_s3_prefix_default_none(self) -> None:
        cfg = ReaderConfig()
        assert cfg.s3_prefix is None

    def test_s3_prefix_set(self) -> None:
        cfg = ReaderConfig(s3_prefix="s3://bucket/prefix")
        assert cfg.s3_prefix == "s3://bucket/prefix"

    def test_s3_prefix_empty_string_becomes_none(self) -> None:
        cfg = ReaderConfig(s3_prefix="")
        assert cfg.s3_prefix is None

    def test_reader_backend_default_is_slatedb(self) -> None:
        cfg = ReaderConfig()
        assert cfg.reader_backend == "slatedb"

    def test_reader_backend_sqlite(self) -> None:
        cfg = ReaderConfig(reader_backend="sqlite")
        assert cfg.reader_backend == "sqlite"

    def test_invalid_reader_backend_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReaderConfig(reader_backend="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# load_reader_config with [manifest_store]
# ---------------------------------------------------------------------------


class TestLoadReaderConfigManifestStore:
    def test_defaults_without_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SHARDY_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)  # no reader.toml here
        reader_cfg, store_cfg, output_cfg = load_reader_config(None)
        assert store_cfg.backend == "s3"
        assert reader_cfg.pool_checkout_timeout == timedelta(seconds=30.0)

    def test_with_manifest_store_section(self, tmp_path: Path) -> None:
        toml_content = """\
[reader]
s3_prefix = "s3://bucket/prefix"
pool_checkout_timeout = 10.0

[manifest_store]
backend = "postgres"
dsn = "host=localhost dbname=test"
table_name = "my_table"
ensure_table = false
"""
        config_file = tmp_path / "reader.toml"
        config_file.write_text(toml_content)

        reader_cfg, store_cfg, output_cfg = load_reader_config(str(config_file))
        assert reader_cfg.s3_prefix == "s3://bucket/prefix"
        assert reader_cfg.pool_checkout_timeout == timedelta(seconds=10.0)
        assert store_cfg.backend == "postgres"
        assert store_cfg.dsn == "host=localhost dbname=test"
        assert store_cfg.table_name == "my_table"
        assert store_cfg.ensure_table is False

    def test_without_manifest_store_section(self, tmp_path: Path) -> None:
        toml_content = """\
[reader]
current_url = "s3://bucket/prefix/_CURRENT"
"""
        config_file = tmp_path / "reader.toml"
        config_file.write_text(toml_content)

        reader_cfg, store_cfg, output_cfg = load_reader_config(str(config_file))
        assert store_cfg.backend == "s3"
        assert store_cfg.dsn is None


# ---------------------------------------------------------------------------
# resolve_dsn
# ---------------------------------------------------------------------------


class TestResolveDsn:
    def test_config_dsn_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHARDY_MANIFEST_DSN", "from-env")
        cfg = ManifestStoreConfig(backend="postgres", dsn="from-config")
        assert resolve_dsn(cfg) == "from-config"

    def test_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHARDY_MANIFEST_DSN", "from-env")
        cfg = ManifestStoreConfig(backend="postgres")
        assert resolve_dsn(cfg) == "from-env"

    def test_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SHARDY_MANIFEST_DSN", raising=False)
        cfg = ManifestStoreConfig(backend="postgres")
        with pytest.raises(SystemExit, match="DSN is required"):
            resolve_dsn(cfg)


# ---------------------------------------------------------------------------
# build_connection_factory
# ---------------------------------------------------------------------------


class TestBuildConnectionFactory:
    def test_returns_callable(self) -> None:
        factory = build_connection_factory("host=localhost dbname=test")
        assert callable(factory)

    def test_missing_driver(self) -> None:
        factory = build_connection_factory("host=localhost")
        # The ImportError happens when calling the factory, not creating it
        with pytest.raises(ImportError, match="psycopg2"):
            factory()
