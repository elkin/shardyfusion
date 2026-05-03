from __future__ import annotations

import pytest

from shardyfusion.config import (
    BaseShardedWriteConfig,
    CelShardedWriteConfig,
    HashShardedWriteConfig,
)
from shardyfusion.errors import ConfigValidationError
from shardyfusion.sharding_types import DB_ID_COL, VECTOR_DB_ID_COL
from shardyfusion.vector.config import (
    VectorIndexConfig,
    VectorShardedWriteConfig,
    VectorShardingConfig,
)


class TestShardIdColValidation:
    """Tests for shard_id_col validation on write configs."""

    def test_base_config_default_shard_id_col(self) -> None:
        config = BaseShardedWriteConfig(s3_prefix="s3://bucket/prefix")
        assert config.shard_id_col == DB_ID_COL

    def test_base_config_custom_shard_id_col(self) -> None:
        config = BaseShardedWriteConfig(
            s3_prefix="s3://bucket/prefix",
            shard_id_col="my_shard",
        )
        assert config.shard_id_col == "my_shard"

    def test_base_config_empty_shard_id_col_raises(self) -> None:
        with pytest.raises(
            ConfigValidationError, match="shard_id_col must be a non-empty string"
        ):
            BaseShardedWriteConfig(
                s3_prefix="s3://bucket/prefix",
                shard_id_col="",
            )

    def test_hash_config_inherits_shard_id_col(self) -> None:
        config = HashShardedWriteConfig(
            s3_prefix="s3://bucket/prefix",
            num_dbs=4,
            shard_id_col="custom_col",
        )
        assert config.shard_id_col == "custom_col"

    def test_cel_config_inherits_shard_id_col(self) -> None:
        config = CelShardedWriteConfig(
            s3_prefix="s3://bucket/prefix",
            cel_expr="shard_hash(key) % 4u",
            cel_columns={"key": "string"},
            shard_id_col="cel_shard",
        )
        assert config.shard_id_col == "cel_shard"

    def test_vector_config_default_shard_id_col(self) -> None:
        import numpy as np

        config = VectorShardedWriteConfig(
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=128),
            num_dbs=4,
            sharding=VectorShardingConfig(
                num_dbs=4,
                strategy="explicit",
                centroids=np.zeros((4, 128), dtype=np.float32),
            ),
        )
        assert config.shard_id_col == VECTOR_DB_ID_COL

    def test_vector_config_custom_shard_id_col(self) -> None:
        import numpy as np

        config = VectorShardedWriteConfig(
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=128),
            num_dbs=4,
            shard_id_col="vec_shard",
            sharding=VectorShardingConfig(
                num_dbs=4,
                strategy="explicit",
                centroids=np.zeros((4, 128), dtype=np.float32),
            ),
        )
        assert config.shard_id_col == "vec_shard"

    def test_vector_config_empty_shard_id_col_raises(self) -> None:
        import numpy as np

        with pytest.raises(
            ConfigValidationError, match="shard_id_col must be a non-empty string"
        ):
            VectorShardedWriteConfig(
                s3_prefix="s3://bucket/prefix",
                index_config=VectorIndexConfig(dim=128),
                num_dbs=4,
                shard_id_col="",
                sharding=VectorShardingConfig(
                    num_dbs=4,
                    strategy="explicit",
                    centroids=np.zeros((4, 128), dtype=np.float32),
                ),
            )
