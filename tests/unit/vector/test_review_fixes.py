"""Regression tests for PR review comment fixes.

Each test class targets a specific fix to prevent regressions.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from shardyfusion.errors import ConfigValidationError, ManifestParseError
from shardyfusion.manifest import (
    ManifestRef,
    ManifestShardingSpec,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
    WriterInfo,
)
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy
from shardyfusion.vector.sharding import (
    lsh_assign,
    train_centroids_kmeans,
)


def _has_usearch() -> bool:
    try:
        import usearch  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manifest_ref(
    *, ref: str = "s3://b/manifests/test/manifest", run_id: str = "run-1"
) -> ManifestRef:
    return ManifestRef(ref=ref, run_id=run_id, published_at=datetime.now(UTC))


def _make_manifest(num_dbs: int = 2) -> ParsedManifest:
    shards = [
        RequiredShardMeta(
            db_id=i,
            db_url=f"s3://b/shards/db={i:05d}/attempt=00",
            attempt=0,
            row_count=10,
            writer_info=WriterInfo(),
        )
        for i in range(num_dbs)
    ]
    return ParsedManifest(
        required_build=RequiredBuildMeta(
            run_id="run-1",
            created_at=datetime.now(UTC),
            num_dbs=num_dbs,
            s3_prefix="s3://b",
            key_col="_vector_id",
            sharding=ManifestShardingSpec(strategy=ShardingStrategy.HASH),
            db_path_template="db={db_id:05d}",
            shard_prefix="shards",
            key_encoding=KeyEncoding.RAW,
        ),
        shards=shards,
        custom={
            "vector": {
                "dim": 8,
                "metric": "cosine",
                "index_type": "hnsw",
                "sharding_strategy": "explicit",
                "num_probes": 1,
            }
        },
    )


# ---------------------------------------------------------------------------
# 1. Manifest fallback iteration — vector/reader.py
# ---------------------------------------------------------------------------


class TestManifestFallbackIteration:
    """_load_initial_manifest should skip the failed ref and try older ones."""

    def test_fallback_skips_current_and_loads_previous(self, tmp_path: Path) -> None:
        from shardyfusion.vector.reader import ShardedVectorReader

        good_manifest = _make_manifest()
        good_ref = _make_manifest_ref(
            ref="s3://b/manifests/old/manifest", run_id="run-old"
        )
        bad_ref = _make_manifest_ref(
            ref="s3://b/manifests/new/manifest", run_id="run-new"
        )

        class FallbackStore:
            def load_current(self) -> ManifestRef:
                return bad_ref

            def load_manifest(self, ref: str) -> ParsedManifest:
                if ref == bad_ref.ref:
                    raise ManifestParseError("corrupt")
                return good_manifest

            def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
                # Newest first
                return [bad_ref, good_ref]

            def publish(self, **kw: Any) -> str:
                return ""

            def set_current(self, ref: str) -> None:
                pass

        factory = MagicMock()
        reader = ShardedVectorReader(
            s3_prefix="s3://b",
            local_root=str(tmp_path),
            reader_factory=factory,
            manifest_store=FallbackStore(),
            max_fallback_attempts=3,
        )
        # Should have loaded the good (old) manifest
        assert reader._manifest_ref is not None
        assert reader._manifest_ref.ref == good_ref.ref
        reader.close()

    def test_fallback_fails_when_no_valid_manifest(self, tmp_path: Path) -> None:
        from shardyfusion.vector.reader import ShardedVectorReader

        bad_ref1 = _make_manifest_ref(ref="s3://b/m/1", run_id="r1")
        bad_ref2 = _make_manifest_ref(ref="s3://b/m/2", run_id="r2")

        class AllBadStore:
            def load_current(self) -> ManifestRef:
                return bad_ref1

            def load_manifest(self, ref: str) -> ParsedManifest:
                raise ManifestParseError("corrupt")

            def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
                return [bad_ref1, bad_ref2]

            def publish(self, **kw: Any) -> str:
                return ""

            def set_current(self, ref: str) -> None:
                pass

        factory = MagicMock()
        with pytest.raises(ManifestParseError):
            ShardedVectorReader(
                s3_prefix="s3://b",
                local_root=str(tmp_path),
                reader_factory=factory,
                manifest_store=AllBadStore(),
                max_fallback_attempts=3,
            )

    def test_fallback_disabled_with_zero_attempts(self, tmp_path: Path) -> None:
        from shardyfusion.vector.reader import ShardedVectorReader

        bad_ref = _make_manifest_ref()

        class BadStore:
            def load_current(self) -> ManifestRef:
                return bad_ref

            def load_manifest(self, ref: str) -> ParsedManifest:
                raise ManifestParseError("corrupt")

            def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
                return [bad_ref]

            def publish(self, **kw: Any) -> str:
                return ""

            def set_current(self, ref: str) -> None:
                pass

        with pytest.raises(ManifestParseError):
            ShardedVectorReader(
                s3_prefix="s3://b",
                local_root=str(tmp_path),
                reader_factory=MagicMock(),
                manifest_store=BadStore(),
                max_fallback_attempts=0,
            )


# ---------------------------------------------------------------------------
# 2. String IDs in USearch adapter — usearch_adapter.py
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_usearch(), reason="usearch not installed")
class TestUSearchStringIds:
    """USearch add_batch should handle string IDs via id_map table."""

    def test_string_ids_written_to_id_map(self, tmp_path: Path) -> None:
        """Writer stores string→int mapping when string IDs are used."""
        from shardyfusion.vector.adapters.usearch_adapter import (
            PAYLOADS_FILENAME,
            USearchWriter,
        )
        from shardyfusion.vector.types import DistanceMetric

        writer = USearchWriter(
            db_url="s3://b/test",
            local_dir=tmp_path,
            dim=4,
            metric=DistanceMetric.COSINE,
        )

        ids = np.array(["abc", "def", "ghi"], dtype=object)
        vecs = np.random.default_rng(42).standard_normal((3, 4)).astype(np.float32)
        writer.add_batch(ids, vecs)
        writer.flush()

        # Check id_map table has the mappings
        conn = sqlite3.connect(str(tmp_path / PAYLOADS_FILENAME))
        rows = conn.execute(
            "SELECT internal_id, original_id FROM id_map ORDER BY internal_id"
        ).fetchall()
        conn.close()

        assert len(rows) == 3
        assert rows[0] == (0, "abc")
        assert rows[1] == (1, "def")
        assert rows[2] == (2, "ghi")

        with patch(
            "shardyfusion.vector.adapters.usearch_adapter.put_bytes",
            return_value=None,
        ):
            writer.close()

    def test_int_ids_no_id_map_rows(self, tmp_path: Path) -> None:
        """Writer does NOT create id_map rows for integer IDs."""
        from shardyfusion.vector.adapters.usearch_adapter import (
            PAYLOADS_FILENAME,
            USearchWriter,
        )
        from shardyfusion.vector.types import DistanceMetric

        writer = USearchWriter(
            db_url="s3://b/test",
            local_dir=tmp_path,
            dim=4,
            metric=DistanceMetric.COSINE,
        )

        ids = np.array([10, 20, 30], dtype=np.int64)
        vecs = np.random.default_rng(42).standard_normal((3, 4)).astype(np.float32)
        writer.add_batch(ids, vecs)
        writer.flush()

        conn = sqlite3.connect(str(tmp_path / PAYLOADS_FILENAME))
        rows = conn.execute("SELECT COUNT(*) FROM id_map").fetchone()
        conn.close()
        assert rows[0] == 0

        with patch(
            "shardyfusion.vector.adapters.usearch_adapter.put_bytes",
            return_value=None,
        ):
            writer.close()


# ---------------------------------------------------------------------------
# 3. Block parallel mode + vector_spec — writer/python/writer.py
# ---------------------------------------------------------------------------


class TestParallelVectorSpecBlocked:
    """parallel=True with vector_spec must raise ConfigValidationError."""

    def test_parallel_with_vector_spec_raises(self) -> None:
        from shardyfusion.config import VectorSpec, WriteConfig
        from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy
        from shardyfusion.writer.python.writer import write_sharded

        config = WriteConfig(
            s3_prefix="s3://bucket/prefix",
            sharding=ShardingSpec(
                strategy=ShardingStrategy.CEL,
                cel_expr="shard_hash(key) % 4u",
                cel_columns={"key": "int"},
                routing_values=[0, 1, 2, 3],
            ),
            vector_spec=VectorSpec(dim=8),
        )

        with pytest.raises(
            ConfigValidationError, match="Parallel mode is not supported"
        ):
            write_sharded(
                [],
                config,
                key_fn=lambda r: r,
                value_fn=lambda r: b"v",
                vector_fn=lambda r: (0, np.zeros(8), None),
                parallel=True,
            )


# ---------------------------------------------------------------------------
# 4. Zero-sum distances in k-means++ — vector/sharding.py
# ---------------------------------------------------------------------------


class TestKMeansPlusPlusZeroSum:
    """k-means++ should not crash when all vectors are identical."""

    def test_identical_vectors_no_crash(self) -> None:
        # All vectors the same → distances are all zero after first centroid
        vectors = np.ones((20, 4), dtype=np.float32) * 3.0
        centroids = train_centroids_kmeans(vectors, 3, seed=42)
        assert centroids.shape == (3, 4)
        # All centroids should be the same vector (since all inputs are identical)
        for i in range(3):
            np.testing.assert_allclose(centroids[i], vectors[0], atol=1e-6)

    def test_nearly_identical_vectors(self) -> None:
        """Vectors with very small variance should still train without error."""
        rng = np.random.default_rng(42)
        base = np.ones((50, 8), dtype=np.float32)
        noise = rng.standard_normal((50, 8)).astype(np.float32) * 1e-10
        vectors = base + noise
        centroids = train_centroids_kmeans(vectors, 3, seed=42)
        assert centroids.shape == (3, 8)


# ---------------------------------------------------------------------------
# 5. num_dbs=0 in LSH — vector/sharding.py
# ---------------------------------------------------------------------------


class TestLSHNumDbsValidation:
    """lsh_assign must reject num_dbs <= 0."""

    def test_num_dbs_zero_raises(self) -> None:
        hp = np.ones((4, 8), dtype=np.float32)
        vec = np.ones(8, dtype=np.float32)
        with pytest.raises(ConfigValidationError, match="num_dbs must be > 0"):
            lsh_assign(vec, hp, 0)

    def test_num_dbs_negative_raises(self) -> None:
        hp = np.ones((4, 8), dtype=np.float32)
        vec = np.ones(8, dtype=np.float32)
        with pytest.raises(ConfigValidationError, match="num_dbs must be > 0"):
            lsh_assign(vec, hp, -1)

    def test_num_dbs_one_ok(self) -> None:
        hp = np.ones((4, 8), dtype=np.float32)
        vec = np.ones(8, dtype=np.float32)
        shard = lsh_assign(vec, hp, 1)
        assert shard == 0


# ---------------------------------------------------------------------------
# 6. Bytes routing values round-trip — vector/writer.py + vector/reader.py
# ---------------------------------------------------------------------------


class TestBytesRoutingValuesRoundTrip:
    """bytes routing values must survive writer serialize → reader deserialize."""

    def test_bytes_serialized_with_marker(self) -> None:
        """Writer wraps bytes values in __bytes_hex__ dict."""
        routing_values: list[int | str | bytes] = [0, "hello", b"\xde\xad"]

        # Simulate writer serialization (from _publish_vector_manifest)
        serialized = [
            v if isinstance(v, (int, str)) else {"__bytes_hex__": v.hex()}
            for v in routing_values
        ]

        assert serialized[0] == 0
        assert serialized[1] == "hello"
        assert serialized[2] == {"__bytes_hex__": "dead"}

    def test_bytes_deserialized_by_reader(self) -> None:
        """Reader restores bytes from __bytes_hex__ wrapper."""
        raw_rv = [0, "hello", {"__bytes_hex__": "dead"}]

        # Simulate reader deserialization (from _apply_manifest)
        restored = [
            bytes.fromhex(v["__bytes_hex__"])
            if isinstance(v, dict) and "__bytes_hex__" in v
            else v
            for v in raw_rv
        ]

        assert restored[0] == 0
        assert restored[1] == "hello"
        assert restored[2] == b"\xde\xad"

    def test_full_round_trip(self) -> None:
        """Writer → JSON-like → Reader preserves all types."""
        original: list[int | str | bytes] = [42, "foo", b"\x01\x02\x03"]

        # Writer serializes
        serialized = [
            v if isinstance(v, (int, str)) else {"__bytes_hex__": v.hex()}
            for v in original
        ]

        # Reader deserializes
        restored = [
            bytes.fromhex(v["__bytes_hex__"])
            if isinstance(v, dict) and "__bytes_hex__" in v
            else v
            for v in serialized
        ]

        assert restored == original
