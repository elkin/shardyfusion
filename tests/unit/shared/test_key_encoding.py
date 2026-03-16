"""Tests for UTF8 and RAW key encodings."""

import pytest

from shardyfusion.serde import make_key_encoder
from shardyfusion.sharding_types import KeyEncoding


class TestUtf8Encoding:
    def test_encode_string(self) -> None:
        encoder = make_key_encoder(KeyEncoding.UTF8)
        assert encoder("hello") == b"hello"

    def test_encode_unicode(self) -> None:
        encoder = make_key_encoder(KeyEncoding.UTF8)
        assert encoder("café") == "café".encode()

    def test_encode_bytes_passthrough(self) -> None:
        encoder = make_key_encoder(KeyEncoding.UTF8)
        assert encoder(b"raw") == b"raw"

    def test_reject_int(self) -> None:
        encoder = make_key_encoder(KeyEncoding.UTF8)
        with pytest.raises(Exception, match="utf8 encoding expects str or bytes"):
            encoder(42)


class TestRawEncoding:
    def test_encode_bytes(self) -> None:
        encoder = make_key_encoder(KeyEncoding.RAW)
        assert encoder(b"\x00\x01\x02") == b"\x00\x01\x02"

    def test_encode_string_to_bytes(self) -> None:
        encoder = make_key_encoder(KeyEncoding.RAW)
        assert encoder("hello") == b"hello"

    def test_reject_int(self) -> None:
        encoder = make_key_encoder(KeyEncoding.RAW)
        with pytest.raises(Exception, match="raw encoding expects bytes or str"):
            encoder(42)


class TestKeyEncodingEnum:
    def test_utf8_from_value(self) -> None:
        assert KeyEncoding.from_value("utf8") == KeyEncoding.UTF8

    def test_raw_from_value(self) -> None:
        assert KeyEncoding.from_value("raw") == KeyEncoding.RAW

    def test_cel_strategy_in_enum(self) -> None:
        from shardyfusion.sharding_types import ShardingStrategy

        assert ShardingStrategy.from_value("cel") == ShardingStrategy.CEL
