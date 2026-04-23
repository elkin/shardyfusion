"""Tests for vector configuration types."""

from __future__ import annotations

import numpy as np
import pytest

from shardyfusion.errors import ConfigValidationError
from shardyfusion.vector.config import (
    VectorIndexConfig,
    VectorShardingSpec,
    VectorWriteConfig,
)
from shardyfusion.vector.types import DistanceMetric, VectorShardingStrategy


class TestVectorIndexConfig:
    def test_defaults(self):
        cfg = VectorIndexConfig(dim=128)
        assert cfg.dim == 128
        assert cfg.metric == DistanceMetric.COSINE
        assert cfg.index_type == "hnsw"
        assert cfg.quantization is None
        assert cfg.index_params == {}

    def test_custom_params(self):
        cfg = VectorIndexConfig(
            dim=768,
            metric=DistanceMetric.L2,
            quantization="fp16",
            index_params={"M": 32, "ef_construction": 200},
        )
        assert cfg.dim == 768
        assert cfg.metric == DistanceMetric.L2
        assert cfg.quantization == "fp16"
        assert cfg.index_params["M"] == 32


class TestVectorShardingSpec:
    def test_defaults(self):
        spec = VectorShardingSpec()
        assert spec.strategy == VectorShardingStrategy.CLUSTER
        assert spec.num_probes == 1
        assert spec.centroids is None
        assert spec.train_centroids is False

    def test_cluster_with_centroids(self):
        centroids = np.eye(4, dtype=np.float32)
        spec = VectorShardingSpec(
            strategy=VectorShardingStrategy.CLUSTER,
            centroids=centroids,
            num_probes=2,
        )
        assert spec.centroids is not None
        assert spec.num_probes == 2

    def test_lsh_config(self):
        spec = VectorShardingSpec(
            strategy=VectorShardingStrategy.LSH,
            num_hash_bits=12,
            num_probes=3,
        )
        assert spec.num_hash_bits == 12
        assert spec.num_probes == 3

    def test_explicit(self):
        spec = VectorShardingSpec(strategy=VectorShardingStrategy.EXPLICIT)
        assert spec.strategy == VectorShardingStrategy.EXPLICIT


class TestVectorWriteConfig:
    def test_missing_index_config_raises(self):
        with pytest.raises(ConfigValidationError, match=r"index_config\.dim"):
            VectorWriteConfig()

    def test_custom_config(self):
        cfg = VectorWriteConfig(
            num_dbs=8,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=256),
            batch_size=5000,
            max_writes_per_second=100.0,
        )
        assert cfg.num_dbs == 8
        assert cfg.index_config.dim == 256
        assert cfg.batch_size == 5000
