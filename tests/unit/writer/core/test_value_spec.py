"""Unit tests for ValueSpec serialization strategies in serde.py."""

from __future__ import annotations

import json

import pytest

from shardyfusion.errors import ConfigValidationError
from shardyfusion.serde import ValueSpec


class TestBinaryCol:
    def test_bytes_value(self) -> None:
        spec = ValueSpec.binary_col("payload")
        row = {"payload": b"\x01\x02\x03"}
        assert spec.encode(row) == b"\x01\x02\x03"

    def test_none_returns_empty_bytes(self) -> None:
        spec = ValueSpec.binary_col("payload")
        row = {"payload": None}
        assert spec.encode(row) == b""

    def test_bytearray_converted(self) -> None:
        spec = ValueSpec.binary_col("payload")
        row = {"payload": bytearray(b"\xaa\xbb")}
        result = spec.encode(row)
        assert result == b"\xaa\xbb"
        assert isinstance(result, bytes)

    def test_string_encoded_as_utf8(self) -> None:
        spec = ValueSpec.binary_col("payload")
        row = {"payload": "hello"}
        assert spec.encode(row) == b"hello"

    def test_unsupported_type_raises(self) -> None:
        spec = ValueSpec.binary_col("payload")
        row = {"payload": 42}
        with pytest.raises(ConfigValidationError, match="binary_col expects"):
            spec.encode(row)

    def test_description(self) -> None:
        spec = ValueSpec.binary_col("data")
        assert "binary_col:data" in spec.description


class TestJsonCols:
    def test_selected_columns(self) -> None:
        spec = ValueSpec.json_cols(["name", "age"])
        row = {"name": "Alice", "age": 30, "extra": "ignored"}
        result = json.loads(spec.encode(row))
        assert result == {"name": "Alice", "age": 30}

    def test_all_columns(self) -> None:
        spec = ValueSpec.json_cols(None)
        row = {"a": 1, "b": "two"}
        result = json.loads(spec.encode(row))
        assert result == {"a": 1, "b": "two"}

    def test_none_value_in_column(self) -> None:
        spec = ValueSpec.json_cols(["x"])
        row = {"x": None}
        result = json.loads(spec.encode(row))
        assert result == {"x": None}

    def test_nested_dict(self) -> None:
        spec = ValueSpec.json_cols(["meta"])
        row = {"meta": {"nested": True}}
        result = json.loads(spec.encode(row))
        assert result == {"meta": {"nested": True}}

    def test_unicode(self) -> None:
        spec = ValueSpec.json_cols(["text"])
        row = {"text": "héllo wörld 你好"}
        result = spec.encode(row)
        parsed = json.loads(result)
        assert parsed["text"] == "héllo wörld 你好"

    def test_missing_col_returns_none(self) -> None:
        spec = ValueSpec.json_cols(["missing"])
        row = {"other": "val"}
        result = json.loads(spec.encode(row))
        assert result == {"missing": None}

    def test_description_selected(self) -> None:
        spec = ValueSpec.json_cols(["a", "b"])
        assert "json_cols:a,b" in spec.description

    def test_description_all(self) -> None:
        spec = ValueSpec.json_cols(None)
        assert "json_cols:all" in spec.description


class TestCallableEncoder:
    def test_basic_callable(self) -> None:
        spec = ValueSpec.callable_encoder(lambda row: b"fixed")
        assert spec.encode({"anything": True}) == b"fixed"

    def test_encoder_that_raises(self) -> None:
        def bad_encoder(row: object) -> bytes:
            raise ValueError("encode failed")

        spec = ValueSpec.callable_encoder(bad_encoder)
        with pytest.raises(ValueError, match="encode failed"):
            spec.encode({"x": 1})

    def test_description_uses_function_name(self) -> None:
        def my_custom_encoder(row: object) -> bytes:
            return b""

        spec = ValueSpec.callable_encoder(my_custom_encoder)
        assert "my_custom_encoder" in spec.description

    def test_description_fallback_for_lambda(self) -> None:
        spec = ValueSpec.callable_encoder(lambda r: b"")
        assert spec.description  # should not be empty
