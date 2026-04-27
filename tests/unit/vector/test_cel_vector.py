"""Tests for CEL sharding support in the vector module."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from shardyfusion.errors import ConfigValidationError
from shardyfusion.vector._distributed import ResolvedVectorRouting, assign_vector_shard
from shardyfusion.vector.config import (
    VectorIndexConfig,
    VectorShardingSpec,
    VectorWriteConfig,
)
from shardyfusion.vector.types import (
    DistanceMetric,
    VectorRecord,
    VectorShardingStrategy,
)
from shardyfusion.vector.writer import _validate_config

# ---------------------------------------------------------------------------
# Mock CEL helpers — avoids hard dependency on cel-expr-python for unit tests
# ---------------------------------------------------------------------------


class FakeCompiledCel:
    """Minimal stand-in for CompiledCel that returns a routing key from context."""

    def __init__(self, key_field: str = "region") -> None:
        self._key_field = key_field

    def evaluate(self, context: dict[str, Any]) -> int | str | bytes:
        return context[self._key_field]


# ---------------------------------------------------------------------------
# Writer: _validate_config with CEL
# ---------------------------------------------------------------------------


class TestValidateConfigCel:
    def test_cel_valid(self):
        cfg = VectorWriteConfig(
            num_dbs=3,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=128),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CEL,
                cel_expr="region",
                cel_columns={"region": "string"},
                routing_values=["us", "eu", "asia"],
            ),
        )
        _validate_config(cfg)  # should not raise

    def test_cel_missing_expr_raises(self):
        cfg = VectorWriteConfig(
            num_dbs=3,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=128),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CEL,
                cel_expr=None,
                cel_columns={"region": "string"},
            ),
        )
        with pytest.raises(ConfigValidationError, match="cel_expr"):
            _validate_config(cfg)

    def test_cel_missing_columns_raises(self):
        cfg = VectorWriteConfig(
            num_dbs=3,
            s3_prefix="s3://bucket/prefix",
            index_config=VectorIndexConfig(dim=128),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CEL,
                cel_expr="region",
                cel_columns=None,
            ),
        )
        with pytest.raises(ConfigValidationError, match="cel_columns"):
            _validate_config(cfg)


# ---------------------------------------------------------------------------
# Writer: assign_vector_shard with CEL
# ---------------------------------------------------------------------------


class TestAssignShardCel:
    def test_cel_categorical_routing(self):
        """CEL categorical routing maps routing_context to correct shard."""
        compiled = FakeCompiledCel(key_field="region")
        routing_values: list[int | str | bytes] = ["us", "eu", "asia"]
        cel_lookup = {v: i for i, v in enumerate(routing_values)}

        record = VectorRecord(
            id=1,
            vector=np.zeros(8, dtype=np.float32),
            routing_context={"region": "eu"},
        )

        with patch("shardyfusion.cel.route_cel") as mock_route_cel:
            mock_route_cel.return_value = 1  # "eu" → shard 1

            routing = ResolvedVectorRouting(
                strategy=VectorShardingStrategy.CEL,
                num_dbs=3,
                metric=DistanceMetric.COSINE,
                compiled_cel=compiled,
                routing_values=routing_values,
                cel_lookup=cel_lookup,
            )
            db_id = assign_vector_shard(
                vector=record.vector,
                routing=routing,
                routing_context=record.routing_context,
            )
            assert db_id == 1
            mock_route_cel.assert_called_once_with(
                compiled,
                {"region": "eu"},
                routing_values=routing_values,
                lookup=cel_lookup,
            )

    def test_cel_direct_integer_routing(self):
        """CEL direct integer routing (no routing_values)."""
        compiled = FakeCompiledCel(key_field="shard")
        record = VectorRecord(
            id=1,
            vector=np.zeros(8, dtype=np.float32),
            routing_context={"shard": 2},
        )

        with patch("shardyfusion.cel.route_cel") as mock_route_cel:
            mock_route_cel.return_value = 2
            routing = ResolvedVectorRouting(
                strategy=VectorShardingStrategy.CEL,
                num_dbs=4,
                metric=DistanceMetric.COSINE,
                compiled_cel=compiled,
                routing_values=None,
                cel_lookup=None,
            )
            db_id = assign_vector_shard(
                vector=record.vector,
                routing=routing,
                routing_context=record.routing_context,
            )
            assert db_id == 2

    def test_cel_out_of_range_shard_raises(self):
        """Writer rejects CEL shard IDs outside the configured shard count."""
        compiled = FakeCompiledCel(key_field="shard")
        record = VectorRecord(
            id=1,
            vector=np.zeros(8, dtype=np.float32),
            routing_context={"shard": 3},
        )

        with patch("shardyfusion.cel.route_cel") as mock_route_cel:
            mock_route_cel.return_value = 3
            routing = ResolvedVectorRouting(
                strategy=VectorShardingStrategy.CEL,
                num_dbs=3,
                metric=DistanceMetric.COSINE,
                compiled_cel=compiled,
                routing_values=None,
                cel_lookup=None,
            )
            with pytest.raises(ConfigValidationError, match="outside \\[0, 3\\)"):
                assign_vector_shard(
                    vector=record.vector,
                    routing=routing,
                    routing_context=record.routing_context,
                )

    def test_cel_missing_compiled_raises(self):
        record = VectorRecord(
            id=1,
            vector=np.zeros(8, dtype=np.float32),
            routing_context={"region": "us"},
        )
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.CEL,
            num_dbs=3,
            metric=DistanceMetric.COSINE,
            compiled_cel=None,
        )
        with pytest.raises(ConfigValidationError, match="compiled expression"):
            assign_vector_shard(
                vector=record.vector,
                routing=routing,
                routing_context=record.routing_context,
            )

    def test_cel_missing_routing_context_raises(self):
        compiled = FakeCompiledCel()
        record = VectorRecord(
            id=1,
            vector=np.zeros(8, dtype=np.float32),
            routing_context=None,
        )
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.CEL,
            num_dbs=3,
            metric=DistanceMetric.COSINE,
            compiled_cel=compiled,
        )
        with pytest.raises(ConfigValidationError, match="routing_context"):
            assign_vector_shard(
                vector=record.vector,
                routing=routing,
                routing_context=record.routing_context,
            )


# ---------------------------------------------------------------------------
# Sharding dispatch: route_vector_to_shards with CEL
# ---------------------------------------------------------------------------


class TestRouteVectorToShardsCel:
    def setup_method(self) -> None:
        from shardyfusion.cel import _compile_cel_cached

        _compile_cel_cached.cache_clear()

    def teardown_method(self) -> None:
        from shardyfusion.cel import _compile_cel_cached

        _compile_cel_cached.cache_clear()

    def test_cel_routes_to_single_shard(self):
        """CEL routing via routing_context resolves to a single shard."""
        with (
            patch("shardyfusion.cel.compile_cel") as mock_compile,
            patch("shardyfusion.cel.route_cel") as mock_route,
        ):
            mock_compile.return_value = MagicMock()
            mock_route.return_value = 1

            from shardyfusion.vector.sharding import route_vector_to_shards

            result = route_vector_to_shards(
                np.zeros(8, dtype=np.float32),
                strategy=VectorShardingStrategy.CEL,
                num_dbs=3,
                num_probes=1,
                routing_context={"region": "eu"},
                cel_expr="region",
                cel_columns={"region": "string"},
                routing_values=["us", "eu", "asia"],
            )
            assert result == [1]
            mock_compile.assert_called_once_with("region", {"region": "string"})
            mock_route.assert_called_once()

    def test_cel_reuses_cached_compilation(self):
        with (
            patch("shardyfusion.cel.compile_cel") as mock_compile,
            patch("shardyfusion.cel.route_cel") as mock_route,
        ):
            mock_compile.return_value = MagicMock()
            mock_route.return_value = 1

            from shardyfusion.vector.sharding import route_vector_to_shards

            for _ in range(2):
                result = route_vector_to_shards(
                    np.zeros(8, dtype=np.float32),
                    strategy=VectorShardingStrategy.CEL,
                    num_dbs=3,
                    num_probes=1,
                    routing_context={"region": "eu"},
                    cel_expr="region",
                    cel_columns={"region": "string"},
                    routing_values=["us", "eu", "asia"],
                )
                assert result == [1]

            mock_compile.assert_called_once_with("region", {"region": "string"})
            assert mock_route.call_count == 2

    def test_cel_missing_routing_context_raises(self):
        from shardyfusion.vector.sharding import route_vector_to_shards

        with pytest.raises(ConfigValidationError, match="routing_context"):
            route_vector_to_shards(
                np.zeros(8, dtype=np.float32),
                strategy=VectorShardingStrategy.CEL,
                num_dbs=3,
                num_probes=1,
                cel_expr="region",
                cel_columns={"region": "string"},
            )

    def test_cel_missing_expr_raises(self):
        from shardyfusion.vector.sharding import route_vector_to_shards

        with pytest.raises(ConfigValidationError, match="cel_expr"):
            route_vector_to_shards(
                np.zeros(8, dtype=np.float32),
                strategy=VectorShardingStrategy.CEL,
                num_dbs=3,
                num_probes=1,
                routing_context={"region": "eu"},
                cel_expr=None,
                cel_columns=None,
            )

    def test_shard_ids_override_cel(self):
        """Explicit shard_ids override should still work with CEL strategy."""
        from shardyfusion.vector.sharding import route_vector_to_shards

        result = route_vector_to_shards(
            np.zeros(8, dtype=np.float32),
            strategy=VectorShardingStrategy.CEL,
            num_dbs=3,
            num_probes=1,
            shard_ids=[0, 2],
        )
        assert result == [0, 2]


# ---------------------------------------------------------------------------
# Reader: search with routing_context
# ---------------------------------------------------------------------------


class TestReaderCelSearch:
    """Test that reader.search() passes routing_context through to routing."""

    def _make_cel_reader(self, tmp_path: Path):
        """Build a reader with CEL sharding metadata in the manifest."""
        from datetime import UTC, datetime

        from shardyfusion.manifest import (
            ManifestRef,
            ManifestShardingSpec,
            ParsedManifest,
            RequiredBuildMeta,
            RequiredShardMeta,
            WriterInfo,
        )
        from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy
        from shardyfusion.vector.reader import ShardedVectorReader
        from shardyfusion.vector.types import SearchResult

        shards = [
            RequiredShardMeta(
                db_id=i,
                db_url=f"s3://bucket/shards/db={i:05d}/attempt=00",
                attempt=0,
                row_count=100,
                writer_info=WriterInfo(),
                db_bytes=0,
            )
            for i in range(3)
        ]

        vector_custom = {
            "dim": 8,
            "metric": "cosine",
            "index_type": "hnsw",
            "quantization": None,
            "total_vectors": 300,
            "sharding_strategy": "cel",
            "num_probes": 1,
            "cel_expr": "region",
            "cel_columns": {"region": "string"},
            "routing_values": ["us", "eu", "asia"],
        }

        manifest = ParsedManifest(
            required_build=RequiredBuildMeta(
                run_id="test-run",
                created_at=datetime.now(UTC),
                num_dbs=3,
                s3_prefix="s3://bucket",
                key_col="_vector_id",
                sharding=ManifestShardingSpec(
                    strategy=ShardingStrategy.HASH,
                    hash_algorithm="xxh3_64",
                ),
                db_path_template="db={db_id:05d}",
                shard_prefix="shards",
                key_encoding=KeyEncoding.RAW,
            ),
            shards=shards,
            custom={"vector": vector_custom},
        )

        class MockStore:
            def load_current(self):
                return ManifestRef(
                    ref="s3://bucket/manifests/test/manifest",
                    run_id="test-run",
                    published_at=datetime.now(UTC),
                )

            def load_manifest(self, ref):
                return manifest

            def list_manifests(self, *, limit=10):
                return [self.load_current()]

            def publish(self, **kwargs):
                return "ref"

            def set_current(self, ref):
                pass

        class MockShardReader:
            def __init__(self, shard_id):
                self._shard_id = shard_id

            def search(self, query, top_k):
                return [
                    SearchResult(id=self._shard_id * 100 + i, score=float(i) * 0.1)
                    for i in range(min(top_k, 3))
                ]

            def close(self):
                pass

        class MockReaderFactory:
            def __init__(self):
                self.created = {}
                self._counter = 0

            def __call__(self, *, db_url, local_dir, index_config, **_kwargs):
                reader = MockShardReader(self._counter)
                self._counter += 1
                self.created[db_url] = reader
                return reader

        factory = MockReaderFactory()
        reader = ShardedVectorReader(
            s3_prefix="s3://bucket",
            local_root=str(tmp_path),
            reader_factory=factory,
            manifest_store=MockStore(),
        )
        return reader, factory

    def test_search_with_routing_context(self, tmp_path: Path):
        """search() with routing_context routes to CEL-determined shard."""
        reader, factory = self._make_cel_reader(tmp_path)

        # Patch route_vector_to_shards to avoid needing real CEL
        with patch("shardyfusion.vector.reader.route_vector_to_shards") as mock_route:
            mock_route.return_value = [1]  # CEL says shard 1

            query = np.zeros(8, dtype=np.float32)
            response = reader.search(
                query,
                top_k=3,
                routing_context={"region": "eu"},
            )

            assert response.num_shards_queried == 1
            assert len(response.results) <= 3

            # Verify routing_context was forwarded
            call_kwargs = mock_route.call_args.kwargs
            assert call_kwargs["routing_context"] == {"region": "eu"}
            assert call_kwargs["cel_expr"] == "region"
            assert call_kwargs["cel_columns"] == {"region": "string"}
            assert call_kwargs["routing_values"] == ["us", "eu", "asia"]

        reader.close()

    def test_cel_reader_loads_metadata_from_manifest(self, tmp_path: Path):
        """Reader parses CEL metadata from manifest custom fields."""
        reader, _ = self._make_cel_reader(tmp_path)

        assert reader._cel_expr == "region"
        assert reader._cel_columns == {"region": "string"}
        assert reader._routing_values == ["us", "eu", "asia"]
        assert reader._sharding_strategy == VectorShardingStrategy.CEL

        reader.close()

    def test_route_vector_with_routing_context(self, tmp_path: Path):
        """route_vector() also accepts routing_context for CEL."""
        reader, _ = self._make_cel_reader(tmp_path)

        with patch("shardyfusion.vector.reader.route_vector_to_shards") as mock_route:
            mock_route.return_value = [2]

            result = reader.route_vector(
                np.zeros(8, dtype=np.float32),
                routing_context={"region": "asia"},
            )
            assert result == [2]

            call_kwargs = mock_route.call_args.kwargs
            assert call_kwargs["routing_context"] == {"region": "asia"}

        reader.close()

    def test_snapshot_info_shows_cel_strategy(self, tmp_path: Path):
        reader, _ = self._make_cel_reader(tmp_path)
        info = reader.snapshot_info()
        assert info.sharding == VectorShardingStrategy.CEL
        reader.close()
