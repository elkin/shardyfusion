"""Tests for the vector-only sqlite-vec reader/writer factory shims
and for backend auto-dispatch in ``ShardedVectorReader``.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

try:
    import sqlite_vec
except ImportError:
    sqlite_vec = None  # type: ignore[assignment]

from shardyfusion.errors import ConfigValidationError
from shardyfusion.manifest import (
    ManifestRef,
    ManifestShardingSpec,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
    WriterInfo,
)
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy
from shardyfusion.vector.config import VectorIndexConfig
from shardyfusion.vector.reader import ShardedVectorReader
from shardyfusion.vector.types import (
    DistanceMetric,
    SearchResult,
    VectorShardReaderFactory,
)


def _is_sqlite_vec_available() -> bool:
    if sqlite_vec is None:
        return False
    try:
        conn = sqlite3.connect(":memory:")
        if hasattr(conn, "enable_load_extension"):
            conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return True
    except (sqlite3.OperationalError, AttributeError):
        return False


pytestmark = pytest.mark.vector_sqlite


# ---------------------------------------------------------------------------
# Manifest helpers (mirror the pattern in test_reader.py)
# ---------------------------------------------------------------------------


def _make_manifest(
    *, num_dbs: int = 2, backend: str | None = "sqlite-vec"
) -> ParsedManifest:
    shards = [
        RequiredShardMeta(
            db_id=db_id,
            db_url=f"s3://bucket/shards/db={db_id:05d}/attempt=00",
            attempt=0,
            row_count=10,
            writer_info=WriterInfo(),
        )
        for db_id in range(num_dbs)
    ]
    vector_custom: dict[str, Any] = {
        "dim": 4,
        "metric": "cosine",
        "index_type": "hnsw",
        "quantization": None,
        "total_vectors": num_dbs * 10,
        "sharding_strategy": "explicit",
        "num_probes": 1,
    }
    if backend is not None:
        vector_custom["backend"] = backend

    return ParsedManifest(
        required_build=RequiredBuildMeta(
            run_id="run-1",
            created_at=datetime.now(UTC),
            num_dbs=num_dbs,
            s3_prefix="s3://bucket",
            key_col="_vector_id",
            sharding=ManifestShardingSpec(strategy=ShardingStrategy.HASH),
            db_path_template="db={db_id:05d}",
            shard_prefix="shards",
            key_encoding=KeyEncoding.RAW,
        ),
        shards=shards,
        custom={"vector": vector_custom},
    )


class _FakeManifestStore:
    def __init__(self, manifest: ParsedManifest) -> None:
        self._manifest = manifest
        self._ref = ManifestRef(
            ref="s3://bucket/manifests/run-1/manifest",
            run_id="run-1",
            published_at=datetime.now(UTC),
        )

    def load_current(self) -> ManifestRef | None:
        return self._ref

    def load_manifest(self, ref: str) -> ParsedManifest:
        return self._manifest

    def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        return [self._ref]

    def publish(self, **kwargs: Any) -> str:
        return self._ref.ref

    def set_current(self, ref: str) -> None:
        pass


# ---------------------------------------------------------------------------
# Backend marker tests
# ---------------------------------------------------------------------------


def test_backend_name_on_factories() -> None:
    """Both USearch and sqlite-vec factories expose backend_name."""
    from shardyfusion.sqlite_vec_adapter import (
        SqliteVecReaderFactory,
    )
    from shardyfusion.vector.adapters.sqlite_vec_adapter import (
        SqliteVecVectorReaderFactory,
        SqliteVecVectorWriterFactory,
    )
    from shardyfusion.vector.adapters.usearch_adapter import (
        USearchReaderFactory,
        USearchWriterFactory,
    )

    assert USearchWriterFactory.backend_name == "usearch"
    assert USearchReaderFactory.backend_name == "usearch"
    assert SqliteVecReaderFactory().backend_name == "sqlite-vec"
    assert SqliteVecVectorWriterFactory().backend_name == "sqlite-vec"
    assert SqliteVecVectorReaderFactory().backend_name == "sqlite-vec"


# ---------------------------------------------------------------------------
# Factory protocol shape
# ---------------------------------------------------------------------------


def test_sqlite_vec_vector_reader_factory_is_protocol_compliant() -> None:
    """The new factory matches the VectorShardReaderFactory protocol."""
    from shardyfusion.vector.adapters.sqlite_vec_adapter import (
        SqliteVecVectorReaderFactory,
    )

    factory: VectorShardReaderFactory = SqliteVecVectorReaderFactory()
    # Static-typing level: if the above line type-checks, the shape is OK.
    # Runtime: ensure the call signature accepts the expected kwargs.
    import inspect

    sig = inspect.signature(factory.__call__)
    assert {"db_url", "local_dir", "index_config"}.issubset(sig.parameters)


# ---------------------------------------------------------------------------
# Auto-dispatch tests
# ---------------------------------------------------------------------------


class TestBackendAutoDispatch:
    def test_usearch_backend_selects_usearch_factory(self, tmp_path: Path) -> None:
        manifest = _make_manifest(backend="usearch")
        store = _FakeManifestStore(manifest)
        reader = ShardedVectorReader(
            s3_prefix="s3://bucket",
            local_root=str(tmp_path),
            manifest_store=store,
        )
        from shardyfusion.vector.adapters.usearch_adapter import USearchReaderFactory

        assert isinstance(reader._reader_factory, USearchReaderFactory)
        reader.close()

    def test_missing_backend_defaults_to_usearch(self, tmp_path: Path) -> None:
        """Manifests without a backend key default to usearch (legacy)."""
        manifest = _make_manifest(backend=None)
        store = _FakeManifestStore(manifest)
        reader = ShardedVectorReader(
            s3_prefix="s3://bucket",
            local_root=str(tmp_path),
            manifest_store=store,
        )
        from shardyfusion.vector.adapters.usearch_adapter import USearchReaderFactory

        assert isinstance(reader._reader_factory, USearchReaderFactory)
        reader.close()

    def test_sqlite_vec_backend_selects_sqlite_vec_factory(
        self, tmp_path: Path
    ) -> None:
        if not _is_sqlite_vec_available():
            pytest.skip("sqlite-vec not available")
        manifest = _make_manifest(backend="sqlite-vec")
        store = _FakeManifestStore(manifest)
        reader = ShardedVectorReader(
            s3_prefix="s3://bucket",
            local_root=str(tmp_path),
            manifest_store=store,
        )
        from shardyfusion.vector.adapters.sqlite_vec_adapter import (
            SqliteVecVectorReaderFactory,
        )

        assert isinstance(reader._reader_factory, SqliteVecVectorReaderFactory)
        reader.close()

    def test_user_override_skips_auto_dispatch(self, tmp_path: Path) -> None:
        """Explicit reader_factory= wins over manifest backend selection."""

        class _Custom:
            def __call__(
                self, *, db_url: str, local_dir: Path, index_config: Any
            ) -> Any:
                raise NotImplementedError

        custom = _Custom()
        manifest = _make_manifest(backend="sqlite-vec")
        store = _FakeManifestStore(manifest)
        reader = ShardedVectorReader(
            s3_prefix="s3://bucket",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=custom,
        )
        assert reader._reader_factory is custom
        reader.close()

    def test_unknown_backend_raises(self, tmp_path: Path) -> None:
        manifest = _make_manifest(backend="not-a-real-backend")
        store = _FakeManifestStore(manifest)
        with pytest.raises(ConfigValidationError, match="Unknown vector backend"):
            ShardedVectorReader(
                s3_prefix="s3://bucket",
                local_root=str(tmp_path),
                manifest_store=store,
            )


# ---------------------------------------------------------------------------
# End-to-end via the real sqlite-vec writer/reader (local-only, no S3)
# ---------------------------------------------------------------------------


class TestSqliteVecVectorRoundTrip:
    """Write a shard with SqliteVecVectorWriterFactory, read it back."""

    def _build_shard(
        self, local_dir: Path, db_url: str, index_config: VectorIndexConfig
    ) -> None:
        """Write vectors locally (upload to S3 is a no-op using a mock later)."""
        from shardyfusion.vector.adapters.sqlite_vec_adapter import (
            SqliteVecVectorWriterFactory,
        )

        # Monkey-patch: SqliteVecAdapter uploads to S3 on close.  For this
        # test we create the DB locally, keep a copy in `local_dir`, and
        # then intercept get_bytes for read-back.  Use the real writer.
        factory = SqliteVecVectorWriterFactory()
        writer = factory(db_url=db_url, local_dir=local_dir, index_config=index_config)
        ids = np.array([1, 2, 3, 4], dtype=np.int64)
        vectors = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        writer.add_batch(ids, vectors, payloads=[{"i": int(i)} for i in ids])
        writer.checkpoint()
        # Do NOT call writer.close() to avoid the S3 upload attempt; the
        # file on disk is enough for the reader test below after we
        # monkeypatch get_bytes.

    def test_end_to_end_with_local_db(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not _is_sqlite_vec_available():
            pytest.skip("sqlite-vec not available")

        from shardyfusion import sqlite_vec_adapter as unified_mod
        from shardyfusion.vector.adapters.sqlite_vec_adapter import (
            SqliteVecVectorReaderFactory,
        )

        shard_dir = tmp_path / "write"
        shard_dir.mkdir()
        db_url = "s3://bucket/shard"
        index_config = VectorIndexConfig(dim=4, metric=DistanceMetric.COSINE)

        self._build_shard(shard_dir, db_url, index_config)

        # The underlying reader calls `get_bytes(f"{db_url}/shard.db")` to
        # fetch the file.  Redirect that to the locally-written file.
        built_db = shard_dir / "shard.db"
        assert built_db.exists()

        def _fake_get_bytes(key: str, **_: Any) -> bytes:
            assert key.endswith("shard.db")
            return built_db.read_bytes()

        monkeypatch.setattr(unified_mod, "get_bytes", _fake_get_bytes)

        reader_factory = SqliteVecVectorReaderFactory()
        reader_dir = tmp_path / "read"
        reader_dir.mkdir()
        reader = reader_factory(
            db_url=db_url,
            local_dir=reader_dir,
            index_config=index_config,
        )
        try:
            query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            results: list[SearchResult] = reader.search(query, top_k=2)
            assert len(results) >= 1
            # The closest vector in cosine space to [1,0,0,0] is id=1.
            assert results[0].id == 1
            assert results[0].payload == {"i": 1}
        finally:
            reader.close()
