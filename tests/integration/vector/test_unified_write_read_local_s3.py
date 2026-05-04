"""Integration test for unified KV+vector write→read round-trip on moto S3.

Uses mock adapters that buffer KV + vector data in memory and store them
as JSON on S3, so no lancedb or sqlite-vec dependencies are needed.
"""

from __future__ import annotations

import json
import types as _types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np
import pytest

pytest.importorskip("cel_expr_python", reason="requires cel extra")

from shardyfusion.config import (
    CelShardedWriteConfig,
    VectorSpec,
    WriterManifestConfig,
    WriterOutputConfig,
)
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.manifest_store import S3ManifestStore
from shardyfusion.storage import ObstoreBackend, create_s3_store, parse_s3_url
from shardyfusion.type_defs import S3ConnectionOptions
from shardyfusion.vector.types import SearchResult

# ---------------------------------------------------------------------------
# Mock unified adapter — no lancedb / sqlite-vec needed
# ---------------------------------------------------------------------------


@dataclass
class FakeUnifiedAdapter:
    """Writes KV pairs and vectors to a JSON file on S3 for testing."""

    db_url: str
    local_dir: Path
    credential_provider: StaticCredentialProvider | None = None
    s3_connection_options: S3ConnectionOptions | None = None
    _kv: list[tuple[bytes, bytes]] = field(default_factory=list)
    _vec_ids: list[int | str] = field(default_factory=list)
    _vec_data: list[list[float]] = field(default_factory=list)
    _vec_payloads: list[dict[str, Any] | None] = field(default_factory=list)
    _closed: bool = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: _types.TracebackType | None,
    ) -> None:
        self.close()

    def write_batch(self, pairs: Any) -> None:
        self._kv.extend((k.hex(), v.hex()) for k, v in pairs)

    def write_vector_batch(
        self,
        ids: np.ndarray,
        vectors: np.ndarray,
        payloads: list[dict[str, Any] | None] | None = None,
    ) -> None:
        for i in range(len(ids)):
            self._vec_ids.append(int(ids[i]))
            self._vec_data.append(vectors[i].tolist())
            if payloads is not None:
                self._vec_payloads.append(payloads[i])

    def flush(self) -> None:
        pass

    def seal(self) -> None:
        return None

    def db_bytes(self) -> int:
        return 0

    def close(self) -> None:
        if self._closed:
            return
        s3_key = f"{self.db_url.rstrip('/')}/data.json"
        payload = json.dumps(
            {
                "kv": list(self._kv),
                "vec_ids": list(self._vec_ids),
                "vec_data": list(self._vec_data),
                "vec_payloads": list(self._vec_payloads),
            }
        )
        bucket, _ = parse_s3_url(s3_key)
        credentials = (
            self.credential_provider.resolve() if self.credential_provider else None
        )
        store = create_s3_store(
            bucket=bucket,
            credentials=credentials,
            connection_options=self.s3_connection_options,
        )
        backend = ObstoreBackend(store)
        backend.put(s3_key, payload.encode(), content_type="application/json")
        self._closed = True


class FakeUnifiedFactory:
    """Factory for FakeUnifiedAdapter — satisfies DbAdapterFactory protocol."""

    supports_vector_writes = True

    def __init__(
        self,
        credential_provider: StaticCredentialProvider | None = None,
        s3_connection_options: S3ConnectionOptions | None = None,
    ) -> None:
        self.credential_provider = credential_provider
        self.s3_connection_options = s3_connection_options

    def __call__(self, *, db_url: str, local_dir: Path) -> FakeUnifiedAdapter:
        local_dir.mkdir(parents=True, exist_ok=True)
        return FakeUnifiedAdapter(
            db_url=db_url,
            local_dir=local_dir,
            credential_provider=self.credential_provider,
            s3_connection_options=self.s3_connection_options,
        )


@dataclass
class FakeUnifiedReader:
    """Reads from FakeUnifiedAdapter's JSON on S3."""

    db_url: str
    credential_provider: StaticCredentialProvider | None = None
    s3_connection_options: S3ConnectionOptions | None = None
    _data: dict[str, Any] = field(default_factory=dict)
    _closed: bool = False

    def __post_init__(self) -> None:
        s3_key = f"{self.db_url.rstrip('/')}/data.json"
        bucket, _ = parse_s3_url(s3_key)
        credentials = (
            self.credential_provider.resolve() if self.credential_provider else None
        )
        store = create_s3_store(
            bucket=bucket,
            credentials=credentials,
            connection_options=self.s3_connection_options,
        )
        backend = ObstoreBackend(store)
        raw = backend.get(s3_key)
        self._data = json.loads(raw)

    def get(self, key: bytes) -> bytes | None:
        key_hex = key.hex()
        for k, v in self._data.get("kv", []):
            if k == key_hex:
                return bytes.fromhex(v)
        return None

    def search(self, query: np.ndarray, top_k: int, ef: int = 50) -> list[SearchResult]:
        results = []
        q = query.astype(np.float32)
        for i, vec_data in enumerate(self._data.get("vec_data", [])):
            vec = np.array(vec_data, dtype=np.float32)
            dist = float(np.linalg.norm(q - vec))
            vid = self._data["vec_ids"][i]
            payload = (
                self._data["vec_payloads"][i]
                if i < len(self._data.get("vec_payloads", []))
                else None
            )
            results.append(SearchResult(id=vid, score=dist, payload=payload))
        results.sort(key=lambda r: r.score)
        return results[:top_k]

    def close(self) -> None:
        self._closed = True


class FakeUnifiedReaderFactory:
    def __init__(
        self,
        credential_provider: StaticCredentialProvider | None = None,
        s3_connection_options: S3ConnectionOptions | None = None,
    ) -> None:
        self.credential_provider = credential_provider
        self.s3_connection_options = s3_connection_options

    def __call__(
        self, *, db_url: str, local_dir: Path, checkpoint_id: str | None = None
    ) -> FakeUnifiedReader:
        return FakeUnifiedReader(
            db_url=db_url,
            credential_provider=self.credential_provider,
            s3_connection_options=self.s3_connection_options,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def s3_prefix(s3_info: dict[str, Any]) -> str:
    return f"s3://{s3_info['bucket']}/unified-test"


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------


@dataclass
class SampleRecord:
    key: int
    value: str
    region: str
    embedding: list[float]


def _make_records(rng: np.random.Generator) -> list[SampleRecord]:
    regions = ["us", "eu", "ap"]
    return [
        SampleRecord(
            key=i,
            value=f"item-{i}",
            region=regions[i % 3],
            embedding=rng.standard_normal(8).astype(np.float32).tolist(),
        )
        for i in range(30)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUnifiedWriteReadRoundTrip:
    """Full unified KV+vector write → manifest → read round-trip."""

    def test_write_and_read_with_vector_fn(
        self,
        s3_prefix: str,
        cred_provider: StaticCredentialProvider,
        s3_conn_opts: S3ConnectionOptions,
        tmp_path: Path,
    ) -> None:
        from shardyfusion.reader.unified_reader import UnifiedShardedReader
        from shardyfusion.sqlite_vec_adapter import (
            SqliteVecFactory,
            SqliteVecReaderFactory,
        )
        from tests.helpers.writer_api import (
            write_python_cel_sharded as write_cel_sharded,
        )

        rng = np.random.default_rng(42)
        records = _make_records(rng)

        config = CelShardedWriteConfig(
            s3_prefix=s3_prefix,
            cel_expr="shard_hash(key) % 3u",
            cel_columns={"key": "int"},
            routing_values=[0, 1, 2],
            vector_spec=VectorSpec(dim=8),
            adapter_factory=SqliteVecFactory(
                vector_spec=VectorSpec(dim=8),
                credential_provider=cred_provider,
                s3_connection_options=s3_conn_opts,
            ),
            manifest=WriterManifestConfig(
                credential_provider=cred_provider,
                s3_connection_options=s3_conn_opts,
            ),
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
            output=WriterOutputConfig(local_root=str(tmp_path / "unified_writer")),
        )

        result = write_cel_sharded(
            records,
            config,
            key_fn=lambda r: r.key,
            value_fn=lambda r: r.value.encode(),
            columns_fn=lambda r: {"key": r.key, "region": r.region},
            vector_fn=lambda r: (r.key, np.array(r.embedding, dtype=np.float32), None),
        )

        assert result.manifest_ref is not None
        assert result.stats.rows_written == 30

        # Verify manifest has vector metadata
        credentials = cred_provider.resolve() if cred_provider else None
        bucket, _ = parse_s3_url(s3_prefix)
        store = create_s3_store(
            bucket=bucket, credentials=credentials, connection_options=s3_conn_opts
        )
        backend = ObstoreBackend(store)
        manifest_store = S3ManifestStore(backend, s3_prefix)
        current = manifest_store.load_current()
        assert current is not None
        manifest = manifest_store.load_manifest(current.ref)
        vec_meta = manifest.custom.get("vector")
        assert vec_meta is not None
        assert vec_meta["dim"] == 8
        assert vec_meta["unified"] is True
        assert vec_meta["backend"] == "sqlite-vec"

        # Read back
        reader = UnifiedShardedReader(
            s3_prefix=s3_prefix,
            local_root=str(tmp_path / "reader"),
            reader_factory=SqliteVecReaderFactory(
                credential_provider=cred_provider,
                s3_connection_options=s3_conn_opts,
            ),
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
        )
        try:
            # KV point lookup
            val = reader.get(0, routing_context={"key": 0})
            assert val == b"item-0"

            # Vector search
            query = np.array(records[0].embedding, dtype=np.float32)
            response = reader.search(query, top_k=5)
            assert len(response.results) <= 5
            assert response.num_shards_queried > 0
            # The record's own vector should be the best match (distance ≈ 0)
            assert response.results[0].score < 1e-5
        finally:
            reader.close()

    def test_write_with_vector_col_fallback(
        self,
        s3_prefix: str,
        cred_provider: StaticCredentialProvider,
        s3_conn_opts: S3ConnectionOptions,
        tmp_path: Path,
    ) -> None:
        """vector_col extracts vectors from columns_fn without explicit vector_fn."""
        from shardyfusion.sqlite_vec_adapter import SqliteVecFactory
        from tests.helpers.writer_api import (
            write_python_cel_sharded as write_cel_sharded,
        )

        rng = np.random.default_rng(123)
        records = _make_records(rng)

        config = CelShardedWriteConfig(
            s3_prefix=f"{s3_prefix}-col",
            cel_expr="shard_hash(key) % 3u",
            cel_columns={"key": "int"},
            routing_values=[0, 1, 2],
            vector_spec=VectorSpec(dim=8, vector_col="embedding"),
            adapter_factory=SqliteVecFactory(
                vector_spec=VectorSpec(dim=8, vector_col="embedding"),
                credential_provider=cred_provider,
                s3_connection_options=s3_conn_opts,
            ),
            manifest=WriterManifestConfig(
                credential_provider=cred_provider,
                s3_connection_options=s3_conn_opts,
            ),
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
            output=WriterOutputConfig(local_root=str(tmp_path / "unified_writer2")),
        )

        result = write_cel_sharded(
            records,
            config,
            key_fn=lambda r: r.key,
            value_fn=lambda r: r.value.encode(),
            columns_fn=lambda r: {
                "key": r.key,
                "embedding": np.array(r.embedding, dtype=np.float32),
            },
            # No vector_fn — should auto-extract from columns_fn["embedding"]
        )

        assert result.stats.rows_written == 30
        assert result.manifest_ref is not None
