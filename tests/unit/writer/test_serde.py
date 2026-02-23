"""Tests for make_key_encoder factory and per-encoding functions."""

import pytest

from slatedb_spark_sharded.errors import ConfigValidationError
from slatedb_spark_sharded.serde import make_key_encoder
from slatedb_spark_sharded.sharding_types import KeyEncoding


class TestMakeKeyEncoderU64be:
    """Tests for the u64be key encoder returned by make_key_encoder."""

    def test_returns_8_bytes(self) -> None:
        encoder = make_key_encoder(KeyEncoding.U64BE)
        assert len(encoder(42)) == 8

    def test_zero(self) -> None:
        encoder = make_key_encoder(KeyEncoding.U64BE)
        assert encoder(0) == b"\x00" * 8

    def test_max_value(self) -> None:
        encoder = make_key_encoder(KeyEncoding.U64BE)
        result = encoder(2**64 - 1)
        assert result == b"\xff" * 8

    def test_big_endian_ordering(self) -> None:
        encoder = make_key_encoder(KeyEncoding.U64BE)
        assert encoder(1) < encoder(256)

    def test_rejects_non_int(self) -> None:
        encoder = make_key_encoder(KeyEncoding.U64BE)
        with pytest.raises(ConfigValidationError, match="u64be encoding expects int"):
            encoder("not_an_int")

    def test_rejects_negative(self) -> None:
        encoder = make_key_encoder(KeyEncoding.U64BE)
        with pytest.raises(ConfigValidationError, match="u64be encoding requires"):
            encoder(-1)

    def test_rejects_overflow(self) -> None:
        encoder = make_key_encoder(KeyEncoding.U64BE)
        with pytest.raises(ConfigValidationError, match="u64be encoding requires"):
            encoder(2**64)

    def test_known_value(self) -> None:
        encoder = make_key_encoder(KeyEncoding.U64BE)
        assert encoder(1) == b"\x00\x00\x00\x00\x00\x00\x00\x01"


class TestMakeKeyEncoderU32be:
    """Tests for the u32be key encoder returned by make_key_encoder."""

    def test_returns_4_bytes(self) -> None:
        encoder = make_key_encoder(KeyEncoding.U32BE)
        assert len(encoder(42)) == 4

    def test_zero(self) -> None:
        encoder = make_key_encoder(KeyEncoding.U32BE)
        assert encoder(0) == b"\x00" * 4

    def test_max_value(self) -> None:
        encoder = make_key_encoder(KeyEncoding.U32BE)
        result = encoder(2**32 - 1)
        assert result == b"\xff" * 4

    def test_big_endian_ordering(self) -> None:
        encoder = make_key_encoder(KeyEncoding.U32BE)
        assert encoder(1) < encoder(256)

    def test_rejects_non_int(self) -> None:
        encoder = make_key_encoder(KeyEncoding.U32BE)
        with pytest.raises(ConfigValidationError, match="u32be encoding expects int"):
            encoder("not_an_int")

    def test_rejects_negative(self) -> None:
        encoder = make_key_encoder(KeyEncoding.U32BE)
        with pytest.raises(ConfigValidationError, match="u32be encoding requires"):
            encoder(-1)

    def test_rejects_overflow(self) -> None:
        encoder = make_key_encoder(KeyEncoding.U32BE)
        with pytest.raises(ConfigValidationError, match="u32be encoding requires"):
            encoder(2**32)

    def test_known_value(self) -> None:
        encoder = make_key_encoder(KeyEncoding.U32BE)
        assert encoder(1) == b"\x00\x00\x00\x01"


class TestMakeKeyEncoderFactory:
    """Tests for factory-level behavior of make_key_encoder."""

    def test_returns_same_function_for_same_encoding(self) -> None:
        a = make_key_encoder(KeyEncoding.U64BE)
        b = make_key_encoder(KeyEncoding.U64BE)
        assert a is b  # module-level function reference

    def test_returns_different_functions_for_different_encodings(self) -> None:
        u64 = make_key_encoder(KeyEncoding.U64BE)
        u32 = make_key_encoder(KeyEncoding.U32BE)
        assert u64 is not u32

    def test_unsupported_encoding_raises(self) -> None:
        with pytest.raises(ConfigValidationError, match="Unsupported key encoding"):
            make_key_encoder("bogus")  # type: ignore[arg-type]
