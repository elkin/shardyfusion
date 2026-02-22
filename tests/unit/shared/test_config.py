from __future__ import annotations

import pytest

from slatedb_spark_sharded.config import (
    EngineOptions,
    ManifestOptions,
    OutputOptions,
    ShardingOptions,
    SlateDbConfig,
    WriteConfig,
)
from slatedb_spark_sharded.errors import ConfigValidationError
from slatedb_spark_sharded.serde import ValueSpec
from slatedb_spark_sharded.sharding_types import KeyEncoding, ShardingSpec

# ---------------------------------------------------------------------------
# Legacy SlateDbConfig tests
# ---------------------------------------------------------------------------


def test_config_accepts_valid_values() -> None:
    config = SlateDbConfig(
        num_dbs=4,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
        sharding=ShardingOptions(),
    )
    assert config.num_dbs == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_dbs": 0},
        {"engine": EngineOptions(batch_size=0)},
        {"s3_prefix": "not-s3"},
        {"output": OutputOptions(tmp_prefix="../tmp")},
        {"manifest": ManifestOptions(manifest_name="bad/name")},
        {"sharding": ShardingSpec()},
        {"key_encoding": "u16be"},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    base = {
        "num_dbs": 4,
        "s3_prefix": "s3://bucket/prefix",
        "key_col": "id",
        "value_spec": ValueSpec.binary_col("payload"),
    }
    base.update(kwargs)

    with pytest.raises(ConfigValidationError):
        SlateDbConfig(**base)


def test_config_accepts_u32be_key_encoding() -> None:
    config = SlateDbConfig(
        num_dbs=4,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
        key_encoding="u32be",
    )
    assert config.key_encoding == KeyEncoding.U32BE


def test_config_defaults_to_u64be_key_encoding() -> None:
    config = SlateDbConfig(
        num_dbs=4,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
    )
    assert config.key_encoding == KeyEncoding.U64BE


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


def test_legacy_config_to_write_config() -> None:
    legacy = SlateDbConfig(
        num_dbs=4,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
        engine=EngineOptions(batch_size=1000),
    )
    wc = legacy.to_write_config()
    assert wc.num_dbs == 4
    assert wc.batch_size == 1000
    assert wc.key_encoding == KeyEncoding.U64BE
