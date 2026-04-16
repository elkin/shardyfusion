"""Tests for vector writer full flow: write_vector_sharded end-to-end with mocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from shardyfusion.config import OutputOptions
from shardyfusion.errors import ConfigValidationError
from shardyfusion.run_registry import InMemoryRunRegistry
from shardyfusion.vector.config import (
    VectorIndexConfig,
    VectorShardingSpec,
    VectorWriteConfig,
)
from shardyfusion.vector.types import VectorRecord, VectorShardingStrategy

# ---------------------------------------------------------------------------
# Mock adapter for testing without usearch
# ---------------------------------------------------------------------------


@dataclass
class _MockWriter:
    batches: list[tuple[np.ndarray, np.ndarray, list[dict[str, Any]] | None]] = field(
        default_factory=list
    )
    flushed: bool = False
    closed: bool = False
    _checkpoint_id: str = "ckpt-mock"

    def add_batch(
        self,
        ids: np.ndarray,
        vectors: np.ndarray,
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        self.batches.append((ids.copy(), vectors.copy(), payloads))

    def flush(self) -> None:
        self.flushed = True

    def checkpoint(self) -> str | None:
        return self._checkpoint_id

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> _MockWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _MockFactory:
    def __init__(self) -> None:
        self.writers: dict[str, _MockWriter] = {}

    def __call__(
        self, *, db_url: str, local_dir: Path, index_config: Any
    ) -> _MockWriter:
        w = _MockWriter()
        self.writers[db_url] = w
        return w


def _explicit_config(
    tmp_path: Path,
    num_dbs: int = 2,
    batch_size: int = 100,
    metrics_collector: Any = None,
    max_writes_per_second: float | None = None,
) -> VectorWriteConfig:
    return VectorWriteConfig(
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        index_config=VectorIndexConfig(dim=4),
        sharding=VectorShardingSpec(strategy=VectorShardingStrategy.EXPLICIT),
        output=OutputOptions(local_root=str(tmp_path)),
        batch_size=batch_size,
        metrics_collector=metrics_collector,
        max_writes_per_second=max_writes_per_second,
    )


def _cluster_config(
    tmp_path: Path,
    num_dbs: int = 2,
    centroids: np.ndarray | None = None,
    train_centroids: bool = False,
) -> VectorWriteConfig:
    return VectorWriteConfig(
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        index_config=VectorIndexConfig(dim=4),
        sharding=VectorShardingSpec(
            strategy=VectorShardingStrategy.CLUSTER,
            centroids=centroids,
            train_centroids=train_centroids,
        ),
        output=OutputOptions(local_root=str(tmp_path)),
    )


def _lsh_config(
    tmp_path: Path,
    num_dbs: int = 4,
) -> VectorWriteConfig:
    return VectorWriteConfig(
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        index_config=VectorIndexConfig(dim=4),
        sharding=VectorShardingSpec(
            strategy=VectorShardingStrategy.LSH,
            num_hash_bits=3,
        ),
        output=OutputOptions(local_root=str(tmp_path)),
    )


def _make_records(n: int, shard_id: int = 0) -> list[VectorRecord]:
    return [
        VectorRecord(
            id=i,
            vector=np.random.default_rng(i).standard_normal(4).astype(np.float32),
            shard_id=shard_id,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# write_vector_sharded full flow tests
# ---------------------------------------------------------------------------


class TestWriteVectorShardedExplicit:
    """Test write_vector_sharded with EXPLICIT sharding (no optional deps)."""

    @patch("shardyfusion.vector.writer.create_s3_client", return_value=MagicMock())
    @patch(
        "shardyfusion.vector._distributed.put_bytes",
    )
    def test_basic_flow(self, mock_put: Any, mock_s3: Any, tmp_path: Path) -> None:
        from shardyfusion.vector.writer import write_vector_sharded

        factory = _MockFactory()
        config = _explicit_config(tmp_path, num_dbs=2)
        config.adapter_factory = factory

        records = [
            VectorRecord(id=0, vector=np.zeros(4, dtype=np.float32), shard_id=0),
            VectorRecord(id=1, vector=np.ones(4, dtype=np.float32), shard_id=1),
            VectorRecord(id=2, vector=np.zeros(4, dtype=np.float32), shard_id=0),
        ]

        # Mock manifest store to avoid real S3
        mock_store = MagicMock()
        mock_store.publish.return_value = "s3://bucket/prefix/manifests/test/manifest"
        config.manifest.store = mock_store

        result = write_vector_sharded(records, config)

        assert result.run_id is not None
        assert result.manifest_ref == "s3://bucket/prefix/manifests/test/manifest"
        assert len(result.winners) == 2
        assert result.stats.rows_written == 3

    @patch("shardyfusion.vector.writer.create_s3_client", return_value=MagicMock())
    def test_empty_records(self, mock_s3: Any, tmp_path: Path) -> None:
        from shardyfusion.vector.writer import write_vector_sharded

        factory = _MockFactory()
        config = _explicit_config(tmp_path, num_dbs=2)
        config.adapter_factory = factory

        mock_store = MagicMock()
        mock_store.publish.return_value = "s3://manifest"
        config.manifest.store = mock_store

        result = write_vector_sharded([], config)

        assert result.stats.rows_written == 0
        assert len(result.winners) == 0

    @patch("shardyfusion.vector.writer.create_s3_client", return_value=MagicMock())
    def test_run_registry_populated(self, mock_s3: Any, tmp_path: Path) -> None:
        from shardyfusion.vector.writer import write_vector_sharded

        factory = _MockFactory()
        registry = InMemoryRunRegistry()
        config = _explicit_config(tmp_path, num_dbs=1)
        config.adapter_factory = factory
        config.run_registry = registry

        mock_store = MagicMock()
        mock_store.publish.return_value = "s3://manifest"
        config.manifest.store = mock_store

        result = write_vector_sharded(_make_records(1, shard_id=0), config)

        assert result.run_record_ref is not None
        record = registry.load(result.run_record_ref)
        assert record.status == "succeeded"
        assert record.manifest_ref == "s3://manifest"

    @patch("shardyfusion.vector.writer.create_s3_client", return_value=MagicMock())
    def test_metrics_collector_emits_events(self, mock_s3: Any, tmp_path: Path) -> None:
        from shardyfusion.vector.writer import write_vector_sharded

        factory = _MockFactory()
        mc = MagicMock()
        config = _explicit_config(tmp_path, num_dbs=1, metrics_collector=mc)
        config.adapter_factory = factory

        mock_store = MagicMock()
        mock_store.publish.return_value = "s3://manifest"
        config.manifest.store = mock_store

        records = [
            VectorRecord(id=0, vector=np.zeros(4, dtype=np.float32), shard_id=0),
        ]
        write_vector_sharded(records, config)

        # Should have emitted VECTOR_WRITE_STARTED, VECTOR_SHARD_WRITE_COMPLETED,
        # VECTOR_WRITE_COMPLETED
        event_names = [call[0][0] for call in mc.emit.call_args_list]
        from shardyfusion.metrics import MetricEvent

        assert MetricEvent.VECTOR_WRITE_STARTED in event_names
        assert MetricEvent.VECTOR_SHARD_WRITE_COMPLETED in event_names
        assert MetricEvent.VECTOR_WRITE_COMPLETED in event_names

    @patch("shardyfusion.vector.writer.create_s3_client", return_value=MagicMock())
    def test_rate_limiter_used(self, mock_s3: Any, tmp_path: Path) -> None:
        from shardyfusion.vector.writer import write_vector_sharded

        factory = _MockFactory()
        config = _explicit_config(
            tmp_path, num_dbs=1, batch_size=1, max_writes_per_second=1000.0
        )
        config.adapter_factory = factory

        mock_store = MagicMock()
        mock_store.publish.return_value = "s3://manifest"
        config.manifest.store = mock_store

        records = _make_records(3, shard_id=0)
        result = write_vector_sharded(records, config)
        assert result.stats.rows_written == 3

    @patch("shardyfusion.vector.writer.create_s3_client", return_value=MagicMock())
    def test_zero_row_shard_excluded_from_winners(
        self, mock_s3: Any, tmp_path: Path
    ) -> None:
        """Shards with row_count=0 should be excluded from winners."""
        from shardyfusion.vector.writer import write_vector_sharded

        factory = _MockFactory()
        config = _explicit_config(tmp_path, num_dbs=2)
        config.adapter_factory = factory

        mock_store = MagicMock()
        mock_store.publish.return_value = "s3://manifest"
        config.manifest.store = mock_store

        # Only write to shard 0
        records = [
            VectorRecord(id=0, vector=np.zeros(4, dtype=np.float32), shard_id=0),
        ]
        result = write_vector_sharded(records, config)

        assert len(result.winners) == 1
        assert result.winners[0].db_id == 0

    @patch("shardyfusion.vector.writer.create_s3_client", return_value=MagicMock())
    def test_default_factory_requires_usearch(
        self, mock_s3: Any, tmp_path: Path
    ) -> None:
        """Without adapter_factory, should try to import USearchWriterFactory."""
        import builtins

        from shardyfusion.vector.writer import write_vector_sharded

        config = _explicit_config(tmp_path, num_dbs=1)
        # Don't set adapter_factory — let it try the default

        mock_store = MagicMock()
        mock_store.publish.return_value = "s3://manifest"
        config.manifest.store = mock_store

        original_import = builtins.__import__

        def fail_usearch(name: str, *args: Any, **kwargs: Any) -> Any:
            if "usearch" in name:
                raise ImportError("no usearch")
            return original_import(name, *args, **kwargs)

        records = _make_records(1, shard_id=0)
        with patch("builtins.__import__", side_effect=fail_usearch):
            with pytest.raises(ImportError, match="usearch"):
                write_vector_sharded(records, config)


class TestWriteVectorShardedCluster:
    """Test write_vector_sharded with CLUSTER sharding."""

    @patch("shardyfusion.vector.writer.create_s3_client", return_value=MagicMock())
    @patch("shardyfusion.vector._distributed.put_bytes")
    def test_cluster_with_centroids(
        self, mock_put: Any, mock_s3: Any, tmp_path: Path
    ) -> None:
        from shardyfusion.vector.writer import write_vector_sharded

        centroids = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        factory = _MockFactory()
        config = _cluster_config(tmp_path, num_dbs=2, centroids=centroids)
        config.adapter_factory = factory

        mock_store = MagicMock()
        mock_store.publish.return_value = "s3://manifest"
        config.manifest.store = mock_store

        records = [
            VectorRecord(id=0, vector=np.array([0.9, 0.1, 0, 0], dtype=np.float32)),
            VectorRecord(id=1, vector=np.array([0.1, 0.9, 0, 0], dtype=np.float32)),
        ]

        result = write_vector_sharded(records, config)
        assert result.stats.rows_written == 2
        # Centroids should be uploaded
        assert any("centroids.npy" in str(call) for call in mock_put.call_args_list)

    @patch("shardyfusion.vector.writer.create_s3_client", return_value=MagicMock())
    @patch("shardyfusion.vector._distributed.put_bytes")
    def test_cluster_with_training(
        self, mock_put: Any, mock_s3: Any, tmp_path: Path
    ) -> None:
        from shardyfusion.vector.writer import write_vector_sharded

        factory = _MockFactory()
        config = _cluster_config(tmp_path, num_dbs=2, train_centroids=True)
        config.adapter_factory = factory

        mock_store = MagicMock()
        mock_store.publish.return_value = "s3://manifest"
        config.manifest.store = mock_store

        rng = np.random.default_rng(42)
        # Create enough records for training (need > num_dbs*5)
        records = [
            VectorRecord(
                id=i,
                vector=rng.standard_normal(4).astype(np.float32),
            )
            for i in range(20)
        ]

        result = write_vector_sharded(records, config)
        assert result.stats.rows_written == 20

    @patch("shardyfusion.vector.writer.create_s3_client", return_value=MagicMock())
    @patch("shardyfusion.vector._distributed.put_bytes")
    def test_cluster_infers_num_dbs_from_centroids(
        self, mock_put: Any, mock_s3: Any, tmp_path: Path
    ) -> None:
        from shardyfusion.vector.writer import write_vector_sharded

        centroids = np.eye(3, 4, dtype=np.float32)
        config = VectorWriteConfig(
            num_dbs=None,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=4),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CLUSTER,
                centroids=centroids,
            ),
            output=OutputOptions(local_root=str(tmp_path)),
        )
        factory = _MockFactory()
        config.adapter_factory = factory

        mock_store = MagicMock()
        mock_store.publish.return_value = "s3://manifest"
        config.manifest.store = mock_store

        records = [
            VectorRecord(id=0, vector=np.array([0.9, 0, 0, 0], dtype=np.float32)),
        ]
        result = write_vector_sharded(records, config)
        assert result.stats.rows_written == 1

    @patch("shardyfusion.vector.writer.create_s3_client", return_value=MagicMock())
    def test_cluster_mismatched_centroids_count_raises(
        self, mock_s3: Any, tmp_path: Path
    ) -> None:
        from shardyfusion.vector.writer import write_vector_sharded

        centroids = np.eye(3, 4, dtype=np.float32)
        config = VectorWriteConfig(
            num_dbs=5,  # != 3 centroids
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=4),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CLUSTER,
                centroids=centroids,
            ),
            output=OutputOptions(local_root=str(tmp_path)),
        )
        config.adapter_factory = _MockFactory()

        with pytest.raises(ConfigValidationError, match="centroids count"):
            write_vector_sharded([], config)

    @patch("shardyfusion.vector.writer.create_s3_client", return_value=MagicMock())
    def test_cluster_training_requires_num_dbs(
        self, mock_s3: Any, tmp_path: Path
    ) -> None:
        from shardyfusion.vector.writer import write_vector_sharded

        config = VectorWriteConfig(
            num_dbs=None,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=4),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CLUSTER,
                train_centroids=True,
            ),
            output=OutputOptions(local_root=str(tmp_path)),
        )
        config.adapter_factory = _MockFactory()

        records = _make_records(10, shard_id=0)
        with pytest.raises(ConfigValidationError, match="num_dbs must be provided"):
            write_vector_sharded(records, config)


class TestWriteVectorShardedLSH:
    """Test write_vector_sharded with LSH sharding."""

    @patch("shardyfusion.vector.writer.create_s3_client", return_value=MagicMock())
    @patch("shardyfusion.vector._distributed.put_bytes")
    def test_lsh_auto_generates_hyperplanes(
        self, mock_put: Any, mock_s3: Any, tmp_path: Path
    ) -> None:
        from shardyfusion.vector.writer import write_vector_sharded

        factory = _MockFactory()
        config = _lsh_config(tmp_path, num_dbs=4)
        config.adapter_factory = factory

        mock_store = MagicMock()
        mock_store.publish.return_value = "s3://manifest"
        config.manifest.store = mock_store

        records = [
            VectorRecord(
                id=i,
                vector=np.random.default_rng(i).standard_normal(4).astype(np.float32),
            )
            for i in range(10)
        ]

        result = write_vector_sharded(records, config)
        assert result.stats.rows_written == 10
        # Hyperplanes should be uploaded
        assert any("hyperplanes.npy" in str(call) for call in mock_put.call_args_list)

    @patch("shardyfusion.vector.writer.create_s3_client", return_value=MagicMock())
    def test_lsh_requires_num_dbs(self, mock_s3: Any, tmp_path: Path) -> None:
        from shardyfusion.vector.writer import write_vector_sharded

        config = VectorWriteConfig(
            num_dbs=None,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=4),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.LSH,
                num_hash_bits=3,
            ),
            output=OutputOptions(local_root=str(tmp_path)),
        )
        config.adapter_factory = _MockFactory()

        with pytest.raises(ConfigValidationError, match="num_dbs must be provided"):
            write_vector_sharded([], config)


class TestWriteVectorShardedNoNumDbs:
    """Test num_dbs inference failures."""

    @patch("shardyfusion.vector.writer.create_s3_client", return_value=MagicMock())
    def test_explicit_requires_num_dbs(self, mock_s3: Any, tmp_path: Path) -> None:
        from shardyfusion.vector.writer import write_vector_sharded

        config = VectorWriteConfig(
            num_dbs=None,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=4),
            sharding=VectorShardingSpec(strategy=VectorShardingStrategy.EXPLICIT),
            output=OutputOptions(local_root=str(tmp_path)),
        )
        config.adapter_factory = _MockFactory()

        with pytest.raises(ConfigValidationError, match="num_dbs must be provided"):
            write_vector_sharded([], config)


# ---------------------------------------------------------------------------
# _publish_vector_manifest
# ---------------------------------------------------------------------------


class TestPublishVectorManifest:
    def test_manifest_contains_vector_custom_fields(self, tmp_path: Path) -> None:
        from shardyfusion.manifest import RequiredShardMeta
        from shardyfusion.vector.writer import _publish_vector_manifest

        mock_store = MagicMock()
        mock_store.publish.return_value = "s3://bucket/manifests/test/manifest"

        config = _explicit_config(tmp_path, num_dbs=2)
        config.manifest.store = mock_store

        centroids_ref = "s3://bucket/vector_meta/centroids.npy"
        winners = [
            RequiredShardMeta(db_id=0, db_url="s3://a", attempt=0, row_count=5),
        ]

        ref = _publish_vector_manifest(
            config=config,
            run_id="test-run",
            num_dbs=2,
            winners=winners,
            total_vectors=5,
            centroids_ref=centroids_ref,
            hyperplanes_ref=None,
        )

        assert ref == "s3://bucket/manifests/test/manifest"
        # Verify custom fields passed to store.publish
        call_kwargs = mock_store.publish.call_args[1]
        custom = call_kwargs["custom"]
        assert "vector" in custom
        assert custom["vector"]["dim"] == 4
        assert custom["vector"]["total_vectors"] == 5
        assert custom["vector"]["centroids_ref"] == centroids_ref

    def test_manifest_with_hyperplanes(self, tmp_path: Path) -> None:
        from shardyfusion.vector.writer import _publish_vector_manifest

        mock_store = MagicMock()
        mock_store.publish.return_value = "s3://manifest"

        config = _lsh_config(tmp_path, num_dbs=4)
        config.manifest.store = mock_store

        hp_ref = "s3://bucket/vector_meta/hyperplanes.npy"

        _publish_vector_manifest(
            config=config,
            run_id="r1",
            num_dbs=4,
            winners=[],
            total_vectors=0,
            centroids_ref=None,
            hyperplanes_ref=hp_ref,
        )

        call_kwargs = mock_store.publish.call_args[1]
        vec_custom = call_kwargs["custom"]["vector"]
        assert vec_custom["hyperplanes_ref"] == hp_ref
        assert vec_custom["num_hash_bits"] == 3

    def test_manifest_without_explicit_store_creates_s3(self, tmp_path: Path) -> None:
        """When no store is provided, an S3ManifestStore is created."""
        from shardyfusion.vector.writer import _publish_vector_manifest

        config = _explicit_config(tmp_path, num_dbs=1)
        config.manifest.store = None

        with patch(
            "shardyfusion.vector._distributed.S3ManifestStore"
        ) as mock_s3_store_cls:
            mock_instance = MagicMock()
            mock_instance.publish.return_value = "s3://manifest"
            mock_s3_store_cls.return_value = mock_instance

            ref = _publish_vector_manifest(
                config=config,
                run_id="r1",
                num_dbs=1,
                winners=[],
                total_vectors=0,
                centroids_ref=None,
                hyperplanes_ref=None,
            )

        assert ref == "s3://manifest"
        mock_s3_store_cls.assert_called_once()


# ---------------------------------------------------------------------------
# _assign_shard — unknown strategy
# ---------------------------------------------------------------------------


class TestAssignShardUnknown:
    def test_unknown_strategy_raises(self) -> None:
        from shardyfusion.vector.writer import _assign_shard

        record = VectorRecord(id=1, vector=np.zeros(4, dtype=np.float32))
        with pytest.raises(ConfigValidationError, match="Unknown"):
            _assign_shard(
                record=record,
                strategy="bogus",  # type: ignore[arg-type]
                num_dbs=4,
                metric=None,
                centroids=None,
                hyperplanes=None,
            )
