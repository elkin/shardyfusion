"""Tests for VectorSpec config and WriteConfig validation."""

from __future__ import annotations

from typing import Any

import pytest

from shardyfusion.config import (
    CelShardedWriteConfig,
    HashShardedWriteConfig,
    VectorSpec,
)
from shardyfusion.errors import ConfigValidationError


class TestVectorSpec:
    def test_defaults(self) -> None:
        vs = VectorSpec(dim=128)
        assert vs.dim == 128
        assert vs.vector_col is None
        assert vs.metric == "cosine"
        assert vs.index_type == "hnsw"
        assert vs.index_params == {}
        assert vs.quantization is None

    def test_custom_params(self) -> None:
        vs = VectorSpec(
            dim=64,
            vector_col="embedding",
            metric="l2",
            index_params={"M": 32},
            quantization="fp16",
        )
        assert vs.dim == 64
        assert vs.vector_col == "embedding"
        assert vs.metric == "l2"
        assert vs.index_params == {"M": 32}
        assert vs.quantization == "fp16"


class TestWriteConfigVectorValidation:
    def _cel_config(self, **kwargs: Any) -> CelShardedWriteConfig:
        return CelShardedWriteConfig(
            s3_prefix="s3://bucket/prefix",
            cel_expr="shard_hash(key) % 4u",
            cel_columns={"key": "int"},
            routing_values=[0, 1, 2, 3],
            **kwargs,
        )

    def test_vector_spec_accepted_with_hash_sharding(self) -> None:
        """Vector sharding is independent of KV sharding strategy."""
        config = HashShardedWriteConfig(
            num_dbs=4,
            s3_prefix="s3://bucket/prefix",
            vector_spec=VectorSpec(dim=128),
        )
        assert config.vector_spec is not None
        assert config.vector_spec.dim == 128

    def test_vector_spec_with_cel_valid(self) -> None:
        config = self._cel_config(vector_spec=VectorSpec(dim=128))
        assert config.vector_spec is not None
        assert config.vector_spec.dim == 128

    def test_vector_spec_dim_zero(self) -> None:
        with pytest.raises(ConfigValidationError, match="dim must be > 0"):
            self._cel_config(vector_spec=VectorSpec(dim=0))

    def test_vector_spec_dim_negative(self) -> None:
        with pytest.raises(ConfigValidationError, match="dim must be > 0"):
            self._cel_config(vector_spec=VectorSpec(dim=-1))

    def test_vector_spec_invalid_metric(self) -> None:
        with pytest.raises(ConfigValidationError, match="metric must be one of"):
            self._cel_config(vector_spec=VectorSpec(dim=128, metric="manhattan"))

    def test_vector_spec_all_valid_metrics(self) -> None:
        for metric, expected in (
            ("cosine", "cosine"),
            ("l2", "l2"),
            ("dot_product", "dot_product"),
        ):
            config = self._cel_config(vector_spec=VectorSpec(dim=128, metric=metric))
            assert config.vector_spec is not None
            assert config.vector_spec.metric == expected

    def test_no_vector_spec_is_valid(self) -> None:
        config = HashShardedWriteConfig(
            num_dbs=4,
            s3_prefix="s3://bucket/prefix",
        )
        assert config.vector_spec is None
