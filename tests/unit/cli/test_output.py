"""Unit tests for shardyfusion.cli.output formatters."""

from __future__ import annotations

import json

from shardyfusion.cli.config import OutputConfig
from shardyfusion.cli.output import (
    build_error_result,
    build_get_result,
    build_multiget_result,
    encode_value,
    format_result,
)

# ---------------------------------------------------------------------------
# encode_value
# ---------------------------------------------------------------------------


class TestEncodeValue:
    def test_base64(self) -> None:
        assert encode_value(b"\x00\x01\x02", "base64") == "AAEC"

    def test_hex(self) -> None:
        assert encode_value(b"\xde\xad", "hex") == "dead"

    def test_utf8(self) -> None:
        assert encode_value(b"hello", "utf8") == "hello"

    def test_unknown_falls_back_to_base64(self) -> None:
        assert encode_value(b"\x00", "unknown") == encode_value(b"\x00", "base64")


# ---------------------------------------------------------------------------
# build_get_result
# ---------------------------------------------------------------------------


class TestBuildGetResult:
    def test_found(self) -> None:
        cfg = OutputConfig(value_encoding="hex")
        result = build_get_result("mykey", b"\xab\xcd", cfg)
        assert result["op"] == "get"
        assert result["key"] == "mykey"
        assert result["found"] is True
        assert result["value"] == "abcd"

    def test_not_found(self) -> None:
        cfg = OutputConfig(null_repr="<nil>")
        result = build_get_result("mykey", None, cfg)
        assert result["found"] is False
        assert result["value"] == "<nil>"


# ---------------------------------------------------------------------------
# build_multiget_result
# ---------------------------------------------------------------------------


class TestBuildMultigetResult:
    def test_mixed_results(self) -> None:
        cfg = OutputConfig(value_encoding="utf8")
        values: dict[str, bytes | None] = {"a": b"val_a", "b": None}
        result = build_multiget_result(["a", "b"], values, cfg)
        assert result["op"] == "multiget"
        assert len(result["results"]) == 2
        assert result["results"][0]["found"] is True
        assert result["results"][1]["found"] is False

    def test_coerced_keys_int_lookup(self) -> None:
        """When values dict is keyed by int, coerced_keys enables correct lookup."""
        cfg = OutputConfig(value_encoding="utf8")
        values: dict[int, bytes | None] = {1: b"val_1", 2: b"val_2"}
        result = build_multiget_result(["1", "2"], values, cfg, coerced_keys=[1, 2])
        assert result["results"][0]["found"] is True
        assert result["results"][0]["key"] == "1"
        assert result["results"][1]["found"] is True
        assert result["results"][1]["key"] == "2"


# ---------------------------------------------------------------------------
# build_error_result
# ---------------------------------------------------------------------------


class TestBuildErrorResult:
    def test_with_key(self) -> None:
        result = build_error_result("get", "k1", "boom")
        assert result == {"op": "get", "error": "boom", "key": "k1"}

    def test_without_key(self) -> None:
        result = build_error_result("refresh", None, "fail")
        assert "key" not in result


# ---------------------------------------------------------------------------
# format_result
# ---------------------------------------------------------------------------


class TestFormatResult:
    def test_jsonl(self) -> None:
        result = {"op": "get", "key": "k", "found": True, "value": "v"}
        out = format_result(result, "jsonl")
        assert json.loads(out) == result

    def test_json(self) -> None:
        result = {"op": "get", "key": "k", "found": True, "value": "v"}
        out = format_result(result, "json")
        assert json.loads(out) == result
        assert "\n" in out, "json format should be pretty-printed (multi-line)"
        assert "  " in out, "json format should use 2-space indentation"

    def test_text_get(self) -> None:
        result = {"op": "get", "key": "k", "found": True, "value": "v"}
        assert format_result(result, "text") == "k=v"

    def test_text_get_not_found(self) -> None:
        result = {"op": "get", "key": "k", "found": False, "value": "null"}
        assert format_result(result, "text") == "k=null"

    def test_text_multiget(self) -> None:
        result = {
            "op": "multiget",
            "results": [
                {"key": "a", "found": True, "value": "va"},
                {"key": "b", "found": False},
            ],
        }
        out = format_result(result, "text")
        assert "a=va" in out
        assert "b=null" in out

    def test_text_refresh(self) -> None:
        result = {"op": "refresh", "changed": True}
        assert "changed=True" in format_result(result, "text")

    def test_text_info(self) -> None:
        result = {"op": "info", "run_id": "r1", "num_dbs": 4}
        out = format_result(result, "text")
        assert "run_id=r1" in out
        assert "num_dbs=4" in out

    def test_text_error(self) -> None:
        result = {"op": "unknown", "error": "boom"}
        assert "error: boom" in format_result(result, "text")

    def test_table_multiget(self) -> None:
        result = {
            "op": "multiget",
            "results": [
                {"key": "a", "found": True, "value": "va"},
                {"key": "b", "found": False},
            ],
        }
        out = format_result(result, "table")
        assert "KEY" in out
        assert "VALUE" in out
        assert "va" in out

    def test_table_fallback_to_json(self) -> None:
        result = {"op": "get", "key": "k"}
        out = format_result(result, "table")
        assert json.loads(out)["key"] == "k"

    def test_unknown_format_falls_back(self) -> None:
        result = {"op": "get", "key": "k"}
        out = format_result(result, "xml")
        assert json.loads(out)["key"] == "k"

    def test_text_shards(self) -> None:
        result = {
            "op": "shards",
            "shards": [
                {
                    "db_id": 0,
                    "row_count": 10,
                    "min_key": 0,
                    "max_key": 49,
                    "db_url": "s3://b/s0",
                },
                {
                    "db_id": 1,
                    "row_count": 20,
                    "min_key": None,
                    "max_key": None,
                    "db_url": "s3://b/s1",
                },
            ],
        }
        out = format_result(result, "text")
        assert "db_id=0" in out
        assert "rows=10" in out

    def test_text_route(self) -> None:
        result = {"op": "route", "key": "42", "db_id": 3}
        out = format_result(result, "text")
        assert "42 -> shard 3" in out

    def test_table_shards(self) -> None:
        result = {
            "op": "shards",
            "shards": [
                {
                    "db_id": 0,
                    "row_count": 10,
                    "min_key": 0,
                    "max_key": 49,
                    "db_url": "s3://b/s0",
                },
            ],
        }
        out = format_result(result, "table")
        assert "DB_ID" in out
        assert "ROWS" in out


# ---------------------------------------------------------------------------
# build_shards_result / build_route_result
# ---------------------------------------------------------------------------


class TestBuildShardsResult:
    def test_structure(self) -> None:
        from shardyfusion.cli.output import build_shards_result

        shards = [
            {
                "db_id": 0,
                "row_count": 5,
                "min_key": None,
                "max_key": None,
                "db_url": "s3://x",
            }
        ]
        result = build_shards_result(shards)
        assert result["op"] == "shards"
        assert result["shards"] == shards


class TestBuildRouteResult:
    def test_structure(self) -> None:
        from shardyfusion.cli.output import build_route_result

        result = build_route_result("42", 3)
        assert result == {"op": "route", "key": "42", "db_id": 3}
