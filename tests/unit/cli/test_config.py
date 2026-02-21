"""Unit tests for slatedb_spark_sharded.cli.config helpers."""

from __future__ import annotations

import pytest

from slatedb_spark_sharded.cli.config import (
    ReaderConfig,
    build_s3_client_config,
    coerce_cli_key,
    coerce_s3_option,
    resolve_current_url,
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
        monkeypatch.setenv("SLATE_READER_CURRENT", "s3://from-env/_CURRENT")
        result = resolve_current_url(None, ReaderConfig())
        assert result == "s3://from-env/_CURRENT"

    def test_config_fallback(self) -> None:
        cfg = ReaderConfig(current_url="s3://from-config/_CURRENT")
        result = resolve_current_url(None, cfg)
        assert result == "s3://from-config/_CURRENT"

    def test_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SLATE_READER_CURRENT", raising=False)
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

    def test_other_encoding_passthrough(self) -> None:
        assert coerce_cli_key("42", "raw") == "42"


# ---------------------------------------------------------------------------
# build_s3_client_config
# ---------------------------------------------------------------------------


class TestBuildS3ClientConfig:
    def test_no_profile(self) -> None:
        cfg = build_s3_client_config(None)
        assert cfg == {}

    def test_overrides_applied(self) -> None:
        cfg = build_s3_client_config(None, {"connect_timeout": 99})
        assert cfg["connect_timeout"] == 99
