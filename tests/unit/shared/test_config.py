from __future__ import annotations

import pytest

from shardyfusion.config import (
    OutputOptions,
    VectorSpec,
    WriteConfig,
    _validate_segment,
    vector_metric_to_str,
)
from shardyfusion.errors import ConfigValidationError
from shardyfusion.sharding_types import KeyEncoding, ShardingSpec, ShardingStrategy

# ---------------------------------------------------------------------------
# vector_metric_to_str
# ---------------------------------------------------------------------------


def test_vector_metric_to_str_accepts_valid() -> None:
    assert vector_metric_to_str("cosine") == "cosine"
    assert vector_metric_to_str("l2") == "l2"
    assert vector_metric_to_str("dot_product") == "dot_product"


def test_vector_metric_to_str_rejects_invalid() -> None:
    with pytest.raises(ConfigValidationError, match="vector_spec.metric"):
        vector_metric_to_str("invalid")
    with pytest.raises(ConfigValidationError, match="vector_spec.metric"):
        vector_metric_to_str(123)


# ---------------------------------------------------------------------------
# VectorSpec tests
# ---------------------------------------------------------------------------


def test_vector_spec_post_init_coerces_metric() -> None:
    vs = VectorSpec(dim=4, metric="l2")
    assert vs.metric == "l2"


def test_vector_spec_rejects_invalid_metric() -> None:
    with pytest.raises(ConfigValidationError, match="vector_spec.metric"):
        VectorSpec(dim=4, metric="unknown")


def test_vector_spec_to_vector_index_config() -> None:
    vs = VectorSpec(dim=8, metric="cosine", index_type="ivf", quantization="fp16")
    config = vs.to_vector_index_config()
    assert config.dim == 8
    assert config.index_type == "ivf"
    assert config.quantization == "fp16"


def test_vector_spec_to_vector_sharding_spec() -> None:
    vs = VectorSpec(dim=8, metric="cosine")
    spec = vs.to_vector_sharding_spec()
    assert spec.strategy.value == vs.sharding.strategy


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
        {"output": OutputOptions(run_registry_prefix="bad/path")},
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


def test_write_config_accepts_custom_run_registry_prefix() -> None:
    config = WriteConfig(
        num_dbs=4,
        s3_prefix="s3://bucket/prefix",
        output=OutputOptions(run_registry_prefix="writer-runs"),
    )
    assert config.output.run_registry_prefix == "writer-runs"


def test_write_config_cel_rejects_num_dbs() -> None:
    with pytest.raises(ConfigValidationError, match="num_dbs must be None for CEL"):
        WriteConfig(
            num_dbs=4,
            s3_prefix="s3://bucket/prefix",
            sharding=ShardingSpec(
                strategy=ShardingStrategy.CEL,
                cel_expr="true",
                cel_columns=["col"],
            ),
        )


def test_write_config_max_keys_per_shard_rejects_num_dbs() -> None:
    with pytest.raises(
        ConfigValidationError, match="num_dbs must be None when max_keys_per_shard"
    ):
        WriteConfig(
            num_dbs=4,
            s3_prefix="s3://bucket/prefix",
            sharding=ShardingSpec(max_keys_per_shard=1000),
        )


def test_write_config_accepts_cel_without_num_dbs() -> None:
    config = WriteConfig(
        s3_prefix="s3://bucket/prefix",
        sharding=ShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr="true",
            cel_columns=["col"],
        ),
    )
    assert config.num_dbs is None


def test_write_config_vector_spec_dim_must_be_positive() -> None:
    with pytest.raises(ConfigValidationError, match="vector_spec.dim must be > 0"):
        WriteConfig(
            num_dbs=4,
            s3_prefix="s3://bucket/prefix",
            vector_spec=VectorSpec(dim=0, metric="cosine"),
        )


def test_write_config_vector_spec_invalid_metric() -> None:
    with pytest.raises(ConfigValidationError, match="vector_spec.metric"):
        WriteConfig(
            num_dbs=4,
            s3_prefix="s3://bucket/prefix",
            vector_spec=VectorSpec(dim=4, metric="bad"),
        )


def test_write_config_vector_spec_valid() -> None:
    config = WriteConfig(
        num_dbs=4,
        s3_prefix="s3://bucket/prefix",
        vector_spec=VectorSpec(dim=8, metric="dot_product"),
    )
    assert config.vector_spec is not None
    assert config.vector_spec.dim == 8
    assert config.vector_spec.metric == "dot_product"


# ---------------------------------------------------------------------------
# _validate_segment
# ---------------------------------------------------------------------------


def test_validate_segment_empty() -> None:
    with pytest.raises(ConfigValidationError, match="must be non-empty"):
        _validate_segment("", field_name="test")


def test_validate_segment_double_dot() -> None:
    with pytest.raises(ConfigValidationError, match="must not contain '\\.\\.'"):
        _validate_segment("foo..bar", field_name="test")


def test_validate_segment_path_separator() -> None:
    with pytest.raises(ConfigValidationError, match="single path segment"):
        _validate_segment("foo/bar", field_name="test")


def test_validate_segment_unsafe_chars() -> None:
    with pytest.raises(ConfigValidationError, match="unsupported characters"):
        _validate_segment("foo@bar", field_name="test")
