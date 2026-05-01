"""Tests that Ray writer restores shuffle_strategy on failure."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import ray.data
from ray.data.context import DataContext

from shardyfusion.config import (
    HashWriteConfig,
    ManifestOptions,
    OutputOptions,
)
from shardyfusion.manifest_store import InMemoryManifestStore
from shardyfusion.sharding_types import HashShardingSpec
from shardyfusion.writer.ray import write_sharded_by_hash


def _make_config(num_dbs: int = 2) -> HashWriteConfig:
    return HashWriteConfig(
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/test",
        adapter_factory=lambda *, db_url, local_dir: MagicMock(),
        manifest=ManifestOptions(store=InMemoryManifestStore()),
        output=OutputOptions(run_id="shuffle-test"),
    )


def test_shuffle_strategy_restored_on_repartition_failure() -> None:
    """shuffle_strategy is restored even if repartition raises."""
    ctx = DataContext.get_current()
    original_strategy = ctx.shuffle_strategy

    # Create a mock dataset whose repartition() raises
    mock_ds = MagicMock()
    mock_ds.repartition.side_effect = RuntimeError("simulated repartition failure")

    config = _make_config()

    with (
        patch(
            "shardyfusion.writer.ray.writer.add_db_id_column_hash",
            return_value=mock_ds,
        ),
        patch(
            "shardyfusion.writer.ray.writer._verify_hash_routing_agreement",
        ),
    ):
        with pytest.raises(RuntimeError, match="simulated repartition failure"):
            write_sharded_by_hash(
                ds=MagicMock(spec=ray.data.Dataset),
                config=config,
                key_col="id",
                value_spec="val",
                verify_routing=False,
            )

    # The critical assertion: strategy must be restored
    assert ctx.shuffle_strategy == original_strategy
