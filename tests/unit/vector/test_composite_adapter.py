"""Tests for CompositeAdapter — unified KV + vector sidecar lifecycle."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Self

import numpy as np
import pytest

from shardyfusion.composite_adapter import (
    CompositeAdapter,
    CompositeAdapterError,
    CompositeShardReader,
)
from shardyfusion.vector.types import SearchResult

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeKvAdapter:
    """Minimal DbAdapter fake."""

    def __init__(self) -> None:
        self.batches: list[list[tuple[bytes, bytes]]] = []
        self.flushed = False
        self.checkpointed = False
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        self.close()

    def write_batch(self, pairs: Any) -> None:
        self.batches.append(list(pairs))

    def flush(self) -> None:
        self.flushed = True

    def checkpoint(self) -> str | None:
        self.checkpointed = True
        return "kv-ckpt-abc"

    def close(self) -> None:
        self.closed = True


class FakeVectorWriter:
    """Minimal VectorIndexWriter fake."""

    def __init__(self) -> None:
        self.added: list[tuple[np.ndarray, np.ndarray]] = []
        self.flushed = False
        self.checkpointed = False
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def add_batch(
        self,
        ids: np.ndarray,
        vectors: np.ndarray,
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        self.added.append((ids.copy(), vectors.copy()))

    def flush(self) -> None:
        self.flushed = True

    def checkpoint(self) -> str | None:
        self.checkpointed = True
        return "vec-ckpt-xyz"

    def close(self) -> None:
        self.closed = True


class FakeKvReader:
    def __init__(self, data: dict[bytes, bytes]) -> None:
        self._data = data
        self.closed = False

    def get(self, key: bytes) -> bytes | None:
        return self._data.get(key)

    def close(self) -> None:
        self.closed = True


class FakeVectorReader:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.closed = False

    def search(self, query: np.ndarray, top_k: int, ef: int = 50) -> list[SearchResult]:
        return self._results[:top_k]

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# CompositeAdapter tests
# ---------------------------------------------------------------------------


class TestCompositeAdapter:
    def test_write_batch_delegates_to_kv(self) -> None:
        kv = FakeKvAdapter()
        vec = FakeVectorWriter()
        adapter = CompositeAdapter(kv_adapter=kv, vector_writer=vec)

        pairs = [(b"k1", b"v1"), (b"k2", b"v2")]
        adapter.write_batch(pairs)

        assert len(kv.batches) == 1
        assert kv.batches[0] == pairs

    def test_write_vector_batch_delegates(self) -> None:
        kv = FakeKvAdapter()
        vec = FakeVectorWriter()
        adapter = CompositeAdapter(kv_adapter=kv, vector_writer=vec)

        ids = np.array([1, 2, 3])
        vectors = np.random.randn(3, 8).astype(np.float32)
        adapter.write_vector_batch(ids, vectors)

        assert len(vec.added) == 1
        np.testing.assert_array_equal(vec.added[0][0], ids)

    def test_flush_calls_both(self) -> None:
        kv = FakeKvAdapter()
        vec = FakeVectorWriter()
        adapter = CompositeAdapter(kv_adapter=kv, vector_writer=vec)

        adapter.flush()
        assert kv.flushed
        assert vec.flushed

    def test_checkpoint_calls_both_returns_kv(self) -> None:
        kv = FakeKvAdapter()
        vec = FakeVectorWriter()
        adapter = CompositeAdapter(kv_adapter=kv, vector_writer=vec)

        result = adapter.checkpoint()
        assert result == "kv-ckpt-abc"
        assert kv.checkpointed
        assert vec.checkpointed

    def test_close_calls_both(self) -> None:
        kv = FakeKvAdapter()
        vec = FakeVectorWriter()
        adapter = CompositeAdapter(kv_adapter=kv, vector_writer=vec)

        adapter.close()
        assert kv.closed
        assert vec.closed

    def test_close_idempotent(self) -> None:
        kv = FakeKvAdapter()
        vec = FakeVectorWriter()
        adapter = CompositeAdapter(kv_adapter=kv, vector_writer=vec)

        adapter.close()
        adapter.close()  # Should not raise

    def test_write_after_close_raises(self) -> None:
        kv = FakeKvAdapter()
        vec = FakeVectorWriter()
        adapter = CompositeAdapter(kv_adapter=kv, vector_writer=vec)

        adapter.close()
        with pytest.raises(CompositeAdapterError, match="closed"):
            adapter.write_batch([(b"k", b"v")])

    def test_write_vector_after_close_raises(self) -> None:
        kv = FakeKvAdapter()
        vec = FakeVectorWriter()
        adapter = CompositeAdapter(kv_adapter=kv, vector_writer=vec)

        adapter.close()
        with pytest.raises(CompositeAdapterError, match="closed"):
            adapter.write_vector_batch(
                np.array([1]), np.random.randn(1, 8).astype(np.float32)
            )

    def test_context_manager(self) -> None:
        kv = FakeKvAdapter()
        vec = FakeVectorWriter()
        with CompositeAdapter(kv_adapter=kv, vector_writer=vec) as adapter:
            adapter.write_batch([(b"k", b"v")])
        assert kv.closed
        assert vec.closed


# ---------------------------------------------------------------------------
# CompositeShardReader tests
# ---------------------------------------------------------------------------


class TestCompositeShardReader:
    def test_get_delegates_to_kv(self) -> None:
        kv = FakeKvReader({b"k1": b"v1"})
        vec = FakeVectorReader([])
        reader = CompositeShardReader(kv_reader=kv, vector_reader=vec)

        assert reader.get(b"k1") == b"v1"
        assert reader.get(b"missing") is None

    def test_search_delegates_to_vector(self) -> None:
        results = [SearchResult(id=1, score=0.5), SearchResult(id=2, score=0.8)]
        kv = FakeKvReader({})
        vec = FakeVectorReader(results)
        reader = CompositeShardReader(kv_reader=kv, vector_reader=vec)

        query = np.random.randn(8).astype(np.float32)
        got = reader.search(query, top_k=2)
        assert len(got) == 2
        assert got[0].id == 1

    def test_close_both(self) -> None:
        kv = FakeKvReader({})
        vec = FakeVectorReader([])
        reader = CompositeShardReader(kv_reader=kv, vector_reader=vec)

        reader.close()
        assert kv.closed
        assert vec.closed

    def test_context_manager(self) -> None:
        kv = FakeKvReader({})
        vec = FakeVectorReader([])
        with CompositeShardReader(kv_reader=kv, vector_reader=vec):
            pass
        assert kv.closed
        assert vec.closed


# ---------------------------------------------------------------------------
# Exception safety in close()
# ---------------------------------------------------------------------------


class TestCompositeCloseExceptionSafety:
    """Verify both adapters close even if one raises."""

    def test_vec_close_failure_still_closes_kv(self) -> None:
        kv = FakeKvAdapter()
        vec = FakeVectorWriter()
        vec.close = lambda: (_ for _ in ()).throw(RuntimeError("vec boom"))  # type: ignore[assignment]

        adapter = CompositeAdapter(kv_adapter=kv, vector_writer=vec)
        with pytest.raises(RuntimeError, match="vec boom"):
            adapter.close()

        assert kv.closed
        assert adapter._closed

    def test_kv_close_failure_still_marks_closed(self) -> None:
        kv = FakeKvAdapter()
        kv.close = lambda: (_ for _ in ()).throw(RuntimeError("kv boom"))  # type: ignore[assignment]
        vec = FakeVectorWriter()

        adapter = CompositeAdapter(kv_adapter=kv, vector_writer=vec)
        with pytest.raises(RuntimeError, match="kv boom"):
            adapter.close()

        assert vec.closed
        assert adapter._closed

    def test_reader_vec_close_failure_still_closes_kv(self) -> None:
        kv = FakeKvReader({})
        vec = FakeVectorReader([])
        vec.close = lambda: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[assignment]

        reader = CompositeShardReader(kv_reader=kv, vector_reader=vec)
        with pytest.raises(RuntimeError, match="boom"):
            reader.close()

        assert kv.closed


# ---------------------------------------------------------------------------
# CompositeFactory
# ---------------------------------------------------------------------------


class TestCompositeFactory:
    def test_factory_wires_up_adapters(self, tmp_path: Path) -> None:
        from shardyfusion.composite_adapter import CompositeFactory

        kv_created: list[dict[str, Any]] = []
        vec_created: list[dict[str, Any]] = []

        def kv_factory(*, db_url: str, local_dir: Path) -> FakeKvAdapter:
            kv_created.append({"db_url": db_url, "local_dir": local_dir})
            return FakeKvAdapter()

        def vec_factory(
            *, db_url: str, local_dir: Path, index_config: Any
        ) -> FakeVectorWriter:
            vec_created.append(
                {"db_url": db_url, "local_dir": local_dir, "config": index_config}
            )
            return FakeVectorWriter()

        from shardyfusion.config import VectorSpec

        vs = VectorSpec(dim=8, metric="cosine")
        factory = CompositeFactory(
            kv_factory=kv_factory,
            vector_factory=vec_factory,
            vector_spec=vs,
        )
        adapter = factory(db_url="s3://bucket/shard", local_dir=tmp_path)

        assert isinstance(adapter, CompositeAdapter)
        assert len(kv_created) == 1
        assert kv_created[0]["db_url"] == "s3://bucket/shard"
        assert len(vec_created) == 1
        assert vec_created[0]["db_url"] == "s3://bucket/shard/vector"
        assert vec_created[0]["config"].dim == 8

        adapter.close()


class TestCompositeReaderFactory:
    def test_factory_creates_composite_reader(self, tmp_path: Path) -> None:
        from shardyfusion.composite_adapter import CompositeReaderFactory

        def kv_factory(
            *, db_url: str, local_dir: Path, checkpoint_id: str | None = None
        ) -> FakeKvReader:
            return FakeKvReader({b"k": b"v"})

        def vec_factory(
            *, db_url: str, local_dir: Path, index_config: Any
        ) -> FakeVectorReader:
            return FakeVectorReader([SearchResult(id=1, score=0.5)])

        from shardyfusion.config import VectorSpec

        vs = VectorSpec(dim=8, metric="cosine")
        factory = CompositeReaderFactory(
            kv_factory=kv_factory,
            vector_factory=vec_factory,
            vector_spec=vs,
        )
        reader = factory(
            db_url="s3://bucket/shard",
            local_dir=tmp_path,
            checkpoint_id="abc",
        )

        assert isinstance(reader, CompositeShardReader)
        assert reader.get(b"k") == b"v"
        assert len(reader.search(np.zeros(8), top_k=1)) == 1
        reader.close()
