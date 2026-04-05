"""Tests for CompositeAdapter — unified KV + vector sidecar lifecycle."""

from __future__ import annotations

import types
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
