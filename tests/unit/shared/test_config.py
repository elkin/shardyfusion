from __future__ import annotations

import pytest

from shardyfusion.config import WriteConfig
from shardyfusion.errors import ConfigValidationError
from shardyfusion.sharding_types import KeyEncoding

# ---------------------------------------------------------------------------
# WriteConfig tests
# ---------------------------------------------------------------------------


def test_write_config_accepts_valid_values() -> None:
    config = WriteConfig(
        num_dbs=4,
        s3_prefix="s3://bucket/prefix",
    )
    assert config.num_dbs == 4
    assert config.key_encoding == KeyEncoding.U64BE
    assert config.batch_size == 50_000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_dbs": 0},
        {"batch_size": 0},
        {"s3_prefix": "not-s3"},
        {"key_encoding": "u16be"},
        {"sharding": "invalid"},
    ],
)
def test_write_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    base = {
        "num_dbs": 4,
        "s3_prefix": "s3://bucket/prefix",
    }
    base.update(kwargs)
    with pytest.raises(ConfigValidationError):
        WriteConfig(**base)


def test_write_config_accepts_string_key_encoding() -> None:
    config = WriteConfig(
        num_dbs=4,
        s3_prefix="s3://bucket/prefix",
        key_encoding="u32be",
    )
    assert config.key_encoding == KeyEncoding.U32BE
