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

from shardyfusion.config import ManifestOptions, VectorSpec, WriteConfig
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.manifest_store import S3ManifestStore
from shardyfusion.storage import get_bytes, put_bytes
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

    def checkpoint(self) -> str | None:
        return "fake-ckpt"

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
        put_bytes(s3_key, payload.encode(), content_type="application/json")
        self._closed = True


class FakeUnifiedFactory:
    """Factory for FakeUnifiedAdapter — satisfies DbAdapterFactory protocol."""

    supports_vector_writes = True

    def __call__(self, *, db_url: str, local_dir: Path) -> FakeUnifiedAdapter:
        local_dir.mkdir(parents=True, exist_ok=True)
        return FakeUnifiedAdapter(db_url=db_url, local_dir=local_dir)


@dataclass
class FakeUnifiedReader:
    """Reads from FakeUnifiedAdapter's JSON on S3."""

    db_url: str
    _data: dict[str, Any] = field(default_factory=dict)
    _closed: bool = False

    def __post_init__(self) -> None:
        s3_key = f"{self.db_url.rstrip('/')}/data.json"
        raw = get_bytes(s3_key)
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
    def __call__(
        self, *, db_url: str, local_dir: Path, checkpoint_id: str | None = None
    ) -> FakeUnifiedReader:
        return FakeUnifiedReader(db_url=db_url)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def s3_info(local_s3_service: dict[str, Any]) -> dict[str, Any]:
    return local_s3_service


@pytest.fixture
def s3_prefix(s3_info: dict[str, Any]) -> str:
    return f"s3://{s3_info['bucket']}/unified-test"


@pytest.fixture
def cred_provider(s3_info: dict[str, Any]) -> StaticCredentialProvider:
    return StaticCredentialProvider(
        access_key_id=s3_info["access_key_id"],
        secret_access_key=s3_info["secret_access_key"],
    )


@pytest.fixture
def s3_conn_opts(s3_info: dict[str, Any]) -> S3ConnectionOptions:
    return S3ConnectionOptions(
        endpoint_url=s3_info["endpoint_url"],
        region_name=s3_info["region_name"],
    )


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
        from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy
        from shardyfusion.writer.python.writer import write_sharded

        rng = np.random.default_rng(42)
        records = _make_records(rng)

        sharding = ShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr="shard_hash(key) % 3u",
            cel_columns={"key": "int"},
            routing_values=[0, 1, 2],
        )

        config = WriteConfig(
            s3_prefix=s3_prefix,
            sharding=sharding,
            vector_spec=VectorSpec(dim=8),
            adapter_factory=FakeUnifiedFactory(),
            manifest=ManifestOptions(
                credential_provider=cred_provider,
                s3_connection_options=s3_conn_opts,
            ),
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
        )

        result = write_sharded(
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
        store = S3ManifestStore(
            s3_prefix,
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
        )
        current = store.load_current()
        assert current is not None
        manifest = store.load_manifest(current.ref)
        vec_meta = manifest.custom.get("vector")
        assert vec_meta is not None
        assert vec_meta["dim"] == 8
        assert vec_meta["unified"] is True
        assert vec_meta["backend"] == "lancedb"

        # Read back
        reader = UnifiedShardedReader(
            s3_prefix=s3_prefix,
            local_root=str(tmp_path / "reader"),
            reader_factory=FakeUnifiedReaderFactory(),
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
        from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy
        from shardyfusion.writer.python.writer import write_sharded

        rng = np.random.default_rng(123)
        records = _make_records(rng)

        sharding = ShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr="shard_hash(key) % 3u",
            cel_columns={"key": "int"},
            routing_values=[0, 1, 2],
        )

        config = WriteConfig(
            s3_prefix=f"{s3_prefix}-col",
            sharding=sharding,
            vector_spec=VectorSpec(dim=8, vector_col="embedding"),
            adapter_factory=FakeUnifiedFactory(),
            manifest=ManifestOptions(
                credential_provider=cred_provider,
                s3_connection_options=s3_conn_opts,
            ),
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
        )

        result = write_sharded(
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
