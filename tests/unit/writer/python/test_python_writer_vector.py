"""Tests for vector integration in the Python writer (auto_vector_fn, batch flush)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from shardyfusion.config import ManifestOptions, VectorSpec, WriteConfig
from shardyfusion.errors import ConfigValidationError
from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy


def _cel_sharding() -> ShardingSpec:
    return ShardingSpec(
        strategy=ShardingStrategy.CEL,
        cel_expr="shard_hash(key) % 2u",
        cel_columns={"key": "int"},
        routing_values=[0, 1],
    )


# ---------------------------------------------------------------------------
# _auto_vector_fn — exercised via write_sharded
# ---------------------------------------------------------------------------


@dataclass
class _Record:
    key: Any
    value: bytes
    embedding: list[float]


class TestAutoVectorFn:
    def test_vector_col_without_columns_fn_raises(self) -> None:
        from shardyfusion.writer.python.writer import write_sharded

        config = WriteConfig(
            s3_prefix="s3://bucket/prefix",
            sharding=_cel_sharding(),
            vector_spec=VectorSpec(dim=3, vector_col="embedding"),
        )
        with pytest.raises(ConfigValidationError, match="columns_fn is None"):
            write_sharded(
                records=[],
                config=config,
                key_fn=lambda r: r.key,
                value_fn=lambda r: r.value,
            )

    def test_auto_vector_fn_builds_and_runs(self) -> None:
        """When vector_col + columns_fn provided, auto vector_fn is built and used."""
        from shardyfusion.writer.python.writer import write_sharded

        config = WriteConfig(
            s3_prefix="s3://bucket/prefix",
            sharding=_cel_sharding(),
            vector_spec=VectorSpec(dim=3, vector_col="embedding"),
            manifest=ManifestOptions(custom_manifest_fields={}),
        )

        with (
            patch(
                "shardyfusion.writer.python.writer._write_single_process"
            ) as mock_write,
            patch(
                "shardyfusion.writer.python.writer._wrap_factory_for_vector",
                side_effect=lambda f, c: f,
            ),
            patch("shardyfusion.writer.python.writer._inject_vector_manifest_fields"),
            patch(
                "shardyfusion.writer.python.writer.publish_to_store",
                return_value="s3://manifest",
            ),
        ):
            mock_write.return_value = ([], 0)
            write_sharded(
                records=[],
                config=config,
                key_fn=lambda r: r.key,
                value_fn=lambda r: r.value,
                columns_fn=lambda r: {"embedding": r.embedding},
            )
            # Verify _write_single_process was called (auto_vector_fn path reached)
            mock_write.assert_called_once()
            # The vector_fn kwarg should not be None
            call_kwargs = mock_write.call_args[1]
            assert call_kwargs["vector_fn"] is not None

    def test_auto_vector_fn_converts_bytes_key(self) -> None:
        """auto_vector_fn converts bytes keys to hex strings for vector IDs."""
        from shardyfusion.writer.python.writer import write_sharded

        config = WriteConfig(
            s3_prefix="s3://bucket/prefix",
            sharding=_cel_sharding(),
            vector_spec=VectorSpec(dim=3, vector_col="emb"),
            manifest=ManifestOptions(custom_manifest_fields={}),
        )

        captured_fn = None

        def capture_write(*args: Any, **kwargs: Any) -> tuple[list, int]:
            nonlocal captured_fn
            captured_fn = kwargs.get("vector_fn")
            return ([], 0)

        with (
            patch(
                "shardyfusion.writer.python.writer._write_single_process",
                side_effect=capture_write,
            ),
            patch(
                "shardyfusion.writer.python.writer._wrap_factory_for_vector",
                side_effect=lambda f, c: f,
            ),
            patch("shardyfusion.writer.python.writer._inject_vector_manifest_fields"),
            patch(
                "shardyfusion.writer.python.writer.publish_to_store",
                return_value="s3://manifest",
            ),
        ):
            write_sharded(
                records=[],
                config=config,
                key_fn=lambda r: b"\xde\xad",
                value_fn=lambda r: b"v",
                columns_fn=lambda r: {"emb": [1.0, 2.0, 3.0]},
            )

        # Verify the captured function handles bytes keys
        assert captured_fn is not None
        rec = _Record(key=b"\xde\xad", value=b"v", embedding=[1.0, 2.0, 3.0])
        vec_id, vec, payload = captured_fn(rec)
        assert isinstance(vec_id, str)
        assert vec_id == "dead"
        assert vec == [1.0, 2.0, 3.0]
        assert payload is None


# ---------------------------------------------------------------------------
# _detect_kv_backend
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _wrap_factory_for_vector — usearch ImportError fallback
# ---------------------------------------------------------------------------


class TestWrapFactoryImportFallback:
    def test_usearch_import_error_raises(self) -> None:
        """When usearch is not installed, _wrap_factory_for_vector raises."""
        import builtins
        from unittest.mock import patch

        from shardyfusion.writer.python.writer import _wrap_factory_for_vector

        original_import = builtins.__import__

        def fail_usearch(name: str, *args: Any, **kwargs: Any) -> Any:
            if "usearch" in name:
                raise ImportError("no usearch")
            return original_import(name, *args, **kwargs)

        config = WriteConfig(
            s3_prefix="s3://bucket/prefix",
            sharding=_cel_sharding(),
            vector_spec=VectorSpec(dim=8),
        )
        mock_factory = MagicMock()

        with patch("builtins.__import__", side_effect=fail_usearch):
            with pytest.raises(ConfigValidationError, match="vector.*extra"):
                _wrap_factory_for_vector(mock_factory, config)


class TestDetectKvBackendImportFallback:
    def test_sqlite_adapter_import_error_falls_through(self) -> None:
        """When sqlite_adapter can't be imported, _detect_kv_backend returns slatedb."""
        import builtins
        from unittest.mock import patch

        from shardyfusion.writer.python.writer import _detect_kv_backend

        original_import = builtins.__import__

        def fail_sqlite_adapter(name: str, *args: Any, **kwargs: Any) -> Any:
            if "sqlite_adapter" in name:
                raise ImportError("no sqlite_adapter")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fail_sqlite_adapter):
            result = _detect_kv_backend(MagicMock(spec=[]))

        assert result == "slatedb"


class TestDetectKvBackend:
    def test_default_is_slatedb(self) -> None:
        from shardyfusion.writer.python.writer import _detect_kv_backend

        assert _detect_kv_backend(MagicMock(spec=[])) == "slatedb"

    def test_sqlite_vec_factory(self) -> None:
        from shardyfusion.sqlite_vec_adapter import SqliteVecFactory
        from shardyfusion.writer.python.writer import _detect_kv_backend

        f = SqliteVecFactory(vector_spec=MagicMock(dim=4))
        assert _detect_kv_backend(f) == "sqlite-vec"

    def test_sqlite_factory(self) -> None:
        from shardyfusion.sqlite_adapter import SqliteFactory
        from shardyfusion.writer.python.writer import _detect_kv_backend

        assert _detect_kv_backend(SqliteFactory()) == "sqlite"

    def test_composite_unwraps_kv(self) -> None:
        from shardyfusion.composite_adapter import CompositeFactory
        from shardyfusion.sqlite_adapter import SqliteFactory
        from shardyfusion.writer.python.writer import _detect_kv_backend

        cf = CompositeFactory(
            kv_factory=SqliteFactory(),
            vector_factory=MagicMock(),
            vector_spec=MagicMock(dim=4),
        )
        assert _detect_kv_backend(cf) == "sqlite"


# ---------------------------------------------------------------------------
# _flush_single_process_shard — vector batch flush path
# ---------------------------------------------------------------------------


class TestFlushSingleProcessShardVectorBatch:
    """Directly exercises the vector batch flush in _flush_single_process_shard."""

    def test_vector_batch_flushed_alongside_kv(self) -> None:
        """When adapter has write_vector_batch and vector data is buffered, flush it."""
        import contextlib

        from shardyfusion.writer.python.writer import (
            _flush_single_process_shard,
            _SingleProcessState,
        )

        # Create adapter with write_vector_batch
        vector_batches: list[tuple] = []

        class VectorAdapter:
            def __init__(self) -> None:
                self.kv_written: list = []

            def __enter__(self) -> VectorAdapter:
                return self

            def __exit__(self, *exc: object) -> None:
                pass

            def write_batch(self, pairs: Any) -> None:
                self.kv_written.extend(pairs)

            def write_vector_batch(self, ids: Any, vecs: Any, payloads: Any) -> None:
                vector_batches.append((list(ids), vecs.tolist(), payloads))

        adapter = VectorAdapter()
        state = _SingleProcessState()
        state.adapters[0] = adapter
        state.db_urls[0] = "s3://test"
        state.batches[0] = [(b"k1", b"v1"), (b"k2", b"v2")]
        state.batch_byte_sizes[0] = 8
        state.row_counts[0] = 2
        state.total_batched_items = 2
        state.total_batched_bytes = 8
        # Buffer vector data
        state.vector_ids[0] = [1, 2]
        state.vector_vecs[0] = [[0.1, 0.2], [0.3, 0.4]]
        state.vector_payloads[0] = [None, {"label": "a"}]

        config = WriteConfig(
            num_dbs=1,
            s3_prefix="s3://bucket/prefix",
        )

        with contextlib.ExitStack() as stack:
            _flush_single_process_shard(
                state,
                db_id=0,
                stack=stack,
                config=config,
                run_id="run-1",
                attempt=0,
                factory=MagicMock(),
            )

        # KV batch flushed
        assert adapter.kv_written == [(b"k1", b"v1"), (b"k2", b"v2")]
        # Vector batch flushed
        assert len(vector_batches) == 1
        assert vector_batches[0][0] == [1, 2]
        # Vector buffers cleared
        assert state.vector_ids[0] == []
        assert state.vector_vecs[0] == []
        assert state.vector_payloads[0] == []

    def test_vector_buffer_in_write_loop(self) -> None:
        """Exercises vector buffering (lines 766-769) via _write_single_process."""
        from shardyfusion.writer.python.writer import _write_single_process

        vector_batches: list[tuple] = []

        class VectorAdapter:
            def __init__(self) -> None:
                self.kv_written: list = []

            def __enter__(self) -> VectorAdapter:
                return self

            def __exit__(self, *exc: object) -> None:
                pass

            def write_batch(self, pairs: Any) -> None:
                self.kv_written.extend(pairs)

            def write_vector_batch(self, ids: Any, vecs: Any, payloads: Any) -> None:
                vector_batches.append((list(ids), payloads))

            def flush(self) -> None:
                pass

            def checkpoint(self) -> str:
                return "ckpt"

            def close(self) -> None:
                pass

        def factory(
            *, db_url: str, local_dir: Any, checkpoint_id: str | None = None
        ) -> VectorAdapter:
            return VectorAdapter()

        config = WriteConfig(
            num_dbs=1,
            s3_prefix="s3://bucket/prefix",
        )

        @dataclass
        class Item:
            key: int
            value: bytes

        records = [Item(key=i, value=b"v") for i in range(3)]

        attempts, num_dbs = _write_single_process(
            records=records,
            config=config,
            sharding=config.sharding,
            num_dbs=1,
            run_id="test-run",
            factory=factory,
            key_fn=lambda r: r.key,
            value_fn=lambda r: r.value,
            vector_fn=lambda r: (r.key, [float(r.key)] * 4, None),
            max_writes_per_second=None,
            max_write_bytes_per_second=None,
            max_total_batched_items=None,
            max_total_batched_bytes=None,
        )

        # Vector batch was flushed
        assert len(vector_batches) >= 1
        total_ids = sum(len(b[0]) for b in vector_batches)
        assert total_ids == 3

    def test_no_vector_flush_without_data(self) -> None:
        """No vector flush when vector_ids is empty."""
        import contextlib

        from shardyfusion.writer.python.writer import (
            _flush_single_process_shard,
            _SingleProcessState,
        )

        class PlainAdapter:
            def write_batch(self, pairs: Any) -> None:
                pass

            def write_vector_batch(self, ids: Any, vecs: Any, payloads: Any) -> None:
                raise AssertionError("Should not be called")

        adapter = PlainAdapter()
        state = _SingleProcessState()
        state.adapters[0] = adapter
        state.db_urls[0] = "s3://test"
        state.batches[0] = [(b"k", b"v")]
        state.batch_byte_sizes[0] = 4
        state.total_batched_items = 1
        state.total_batched_bytes = 4
        state.vector_ids[0] = []  # empty — no vector data
        state.vector_vecs[0] = []
        state.vector_payloads[0] = []

        config = WriteConfig(num_dbs=1, s3_prefix="s3://bucket/prefix")

        with contextlib.ExitStack() as stack:
            _flush_single_process_shard(
                state,
                db_id=0,
                stack=stack,
                config=config,
                run_id="run-1",
                attempt=0,
                factory=MagicMock(),
            )
