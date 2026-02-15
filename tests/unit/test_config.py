from __future__ import annotations

import pytest

from slatedb_spark_sharded.config import EngineOptions, ManifestOptions, OutputOptions, SlateDbConfig
from slatedb_spark_sharded.errors import ConfigValidationError
from slatedb_spark_sharded.serde import ValueSpec


def test_config_accepts_valid_values() -> None:
    config = SlateDbConfig(
        num_dbs=4,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
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
