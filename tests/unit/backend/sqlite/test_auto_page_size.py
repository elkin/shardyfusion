"""Unit tests for the ``page_size="auto"`` adapter mode (post-write VACUUM)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shardyfusion.sqlite_adapter import (
    SqliteAdapter,
    _maybe_repage_to_auto,
)
from shardyfusion.sqlite_page_size import recommend_page_size


def _read_page_size_from_header(db_path: Path) -> int:
    raw = db_path.read_bytes()
    encoded = int.from_bytes(raw[16:18], "big")
    return 65536 if encoded == 1 else encoded


class TestAutoModePicksFromValues:
    def test_small_values_stay_at_default(self, tmp_path: Path) -> None:
        local_dir = tmp_path / "shard"
        with patch("shardyfusion.sqlite_adapter.ObstoreBackend") as mock:
            mock.return_value = MagicMock()
            adapter = SqliteAdapter(
                db_url="s3://bucket/x",
                local_dir=local_dir,
                page_size="auto",
                emit_sidecar=False,
            )
            adapter.write_batch([(i.to_bytes(8, "big"), b"v" * 50) for i in range(100)])
            adapter.seal()

        assert _read_page_size_from_header(local_dir / "shard.db") == 4096

    @pytest.mark.parametrize(
        "value_bytes,expected_page_size",
        [
            (1500, 8192),
            (4000, 16384),
            (8000, 32768),
            (12000, 65536),
        ],
    )
    def test_auto_picker_matches_recommend(
        self,
        tmp_path: Path,
        value_bytes: int,
        expected_page_size: int,
    ) -> None:
        # Sanity-check that the picker and the helper agree on the boundary.
        assert recommend_page_size(p95_value_bytes=value_bytes) == expected_page_size

        local_dir = tmp_path / "shard"
        with patch("shardyfusion.sqlite_adapter.ObstoreBackend") as mock:
            mock.return_value = MagicMock()
            adapter = SqliteAdapter(
                db_url="s3://bucket/x",
                local_dir=local_dir,
                page_size="auto",
                emit_sidecar=False,
            )
            adapter.write_batch(
                [(i.to_bytes(8, "big"), b"v" * value_bytes) for i in range(50)]
            )
            adapter.seal()

        assert _read_page_size_from_header(local_dir / "shard.db") == expected_page_size


class TestRepageHelperEdgeCases:
    def test_repage_noop_on_empty_kv(self, tmp_path: Path) -> None:
        # Empty kv → picker has nothing to sample → no rewrite.
        local_dir = tmp_path / "shard"
        with patch("shardyfusion.sqlite_adapter.ObstoreBackend") as mock:
            mock.return_value = MagicMock()
            adapter = SqliteAdapter(
                db_url="s3://bucket/x",
                local_dir=local_dir,
                page_size="auto",
                emit_sidecar=False,
            )
            adapter.seal()
        assert _read_page_size_from_header(local_dir / "shard.db") == 4096

    def test_repage_direct_call_is_idempotent(self, tmp_path: Path) -> None:
        """A second invocation against an already-repaged file is a no-op."""
        import sqlite3

        db_path = tmp_path / "tiny.db"
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        conn.execute("PRAGMA page_size = 16384")
        conn.execute(
            "CREATE TABLE kv (k BLOB PRIMARY KEY, v BLOB NOT NULL) WITHOUT ROWID"
        )
        conn.execute("INSERT INTO kv VALUES (?, ?)", (b"k", b"x" * 4000))
        conn.close()

        before = _read_page_size_from_header(db_path)
        _maybe_repage_to_auto(db_path, db_url="s3://test/x")
        after = _read_page_size_from_header(db_path)
        # Picker recommends 16384 for 4000-byte values; same as current.
        assert before == after == 16384


class TestSqliteVecAdapterAutoMode:
    """Auto-mode VACUUM must keep sqlite-vec virtual tables resolvable.

    SQLite VACUUM walks every schema object during the rewrite; the
    ``vec_index`` virtual table is unresolvable without the sqlite-vec
    extension loaded on the same connection.  The unified adapter wires
    ``_open_sqlite_vec_connection`` into :func:`_maybe_repage_to_auto`
    for exactly this reason — this test exercises the wire-up end to
    end.
    """

    def test_auto_mode_rewrites_unified_shard(self, tmp_path: Path) -> None:
        pytest.importorskip("sqlite_vec")
        import numpy as np

        from shardyfusion.config import VectorSpec
        from shardyfusion.sqlite_vec_adapter import SqliteVecAdapter

        local_dir = tmp_path / "vec_shard"
        spec = VectorSpec(dim=8, metric="cosine", index_type="flat")
        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend") as mock:
            mock.return_value = MagicMock()
            adapter = SqliteVecAdapter(
                db_url="s3://bucket/vec",
                local_dir=local_dir,
                vector_spec=spec,
                page_size="auto",
                emit_sidecar=False,
            )
            # ~8 KB KV values force a repage above 4 KB inline threshold.
            adapter.write_batch(
                [(i.to_bytes(8, "big"), b"v" * 8000) for i in range(20)]
            )
            ids = np.arange(20, dtype=np.int64)
            vecs = np.random.rand(20, 8).astype(np.float32)
            adapter.write_vector_batch(ids, vecs)
            adapter.seal()

        # VACUUM with the sqlite-vec extension loaded must succeed, and
        # the post-rewrite file must report the recommended page_size in
        # its SQLite header.
        assert _read_page_size_from_header(local_dir / "shard.db") == 32768

    def test_auto_mode_kv_only_workload(self, tmp_path: Path) -> None:
        """Vec adapter with empty vec_index still routes the repage through
        the sqlite-vec connection opener (the schema still has vec0)."""
        pytest.importorskip("sqlite_vec")

        from shardyfusion.config import VectorSpec
        from shardyfusion.sqlite_vec_adapter import SqliteVecAdapter

        local_dir = tmp_path / "vec_shard_kv_only"
        spec = VectorSpec(dim=8, metric="cosine", index_type="flat")
        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend") as mock:
            mock.return_value = MagicMock()
            adapter = SqliteVecAdapter(
                db_url="s3://bucket/vec2",
                local_dir=local_dir,
                vector_spec=spec,
                page_size="auto",
                emit_sidecar=False,
            )
            adapter.write_batch([(i.to_bytes(8, "big"), b"v" * 50) for i in range(50)])
            adapter.seal()

        # Tiny values fit inline at 4 KB; picker stays put.  VACUUM
        # still runs through the vec-aware opener because the vec0
        # schema object is present.
        assert _read_page_size_from_header(local_dir / "shard.db") == 4096

    def test_auto_mode_vec_only_workload(self, tmp_path: Path) -> None:
        """Vec-only workload (no kv writes) still picks a page size that
        fits the embedding inline.  Prior to the multi-cell refactor the
        repage helper short-circuited on empty kv, leaving vec-only
        files at the 4 KiB default with every embedding spilling to
        overflow."""
        pytest.importorskip("sqlite_vec")
        import numpy as np

        from shardyfusion.config import VectorSpec
        from shardyfusion.sqlite_vec_adapter import SqliteVecAdapter

        local_dir = tmp_path / "vec_only_shard"
        # dim=1024 → 4096-byte raw embedding.  At 4 KiB pages the
        # inline threshold is ~1002 B, so every embedding would overflow
        # — picker must bump us to a larger page even though kv is empty.
        spec = VectorSpec(dim=1024, metric="cosine", index_type="flat")
        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend") as mock:
            mock.return_value = MagicMock()
            adapter = SqliteVecAdapter(
                db_url="s3://bucket/vec_only",
                local_dir=local_dir,
                vector_spec=spec,
                page_size="auto",
                emit_sidecar=False,
            )
            ids = np.arange(8, dtype=np.int64)
            vecs = np.random.rand(8, 1024).astype(np.float32)
            adapter.write_vector_batch(ids, vecs)
            adapter.seal()

        # 4*1024 + 16 (vec0 overhead) + 9 (rowid) + 12 (cell) = 4117 B
        # required — exceeds 4096 (1002) and 8192 (2030); 16384 has
        # threshold 4086 (also too small); 32768 has 8198 → fits.
        assert _read_page_size_from_header(local_dir / "shard.db") == 32768

    def test_close_refuses_upload_when_seal_failed(self, tmp_path: Path) -> None:
        """If _maybe_repage_to_auto raises during seal(), close() must
        skip the upload AND raise so the writer learns the shard was
        not published.  Otherwise the un-repaged DB ships silently with
        no signal that the chosen strategy was abandoned."""
        pytest.importorskip("sqlite_vec")
        from shardyfusion.config import VectorSpec
        from shardyfusion.sqlite_vec_adapter import (
            SqliteVecAdapter,
            SqliteVecAdapterError,
        )

        local_dir = tmp_path / "fail_shard"
        spec = VectorSpec(dim=8, metric="cosine", index_type="flat")
        adapter = SqliteVecAdapter(
            db_url="s3://bucket/fail",
            local_dir=local_dir,
            vector_spec=spec,
            page_size="auto",
            emit_sidecar=False,
        )
        adapter.write_batch([(i.to_bytes(8, "big"), b"v" * 100) for i in range(10)])

        # Inject a failure in _maybe_repage_to_auto and verify close()
        # does NOT call ObstoreBackend.put() and raises.
        with patch(
            "shardyfusion.sqlite_vec_adapter._maybe_repage_to_auto",
            side_effect=RuntimeError("VACUUM failed"),
        ):
            with pytest.raises(RuntimeError, match="VACUUM failed"):
                adapter.seal()

        # adapter._sealed must be False, adapter._seal_attempted True.
        assert adapter._sealed is False
        assert adapter._seal_attempted is True

        with patch("shardyfusion.sqlite_vec_adapter.ObstoreBackend") as mock:
            mock.return_value = MagicMock()
            with pytest.raises(SqliteVecAdapterError, match="seal.*not complete"):
                adapter.close()
            # put() must NOT have been called.
            mock.return_value.put.assert_not_called()

    def test_close_refuses_upload_when_seal_failed_sqlite(self, tmp_path: Path) -> None:
        """SqliteAdapter (kv-only) has the same skip-and-raise contract."""
        from shardyfusion.sqlite_adapter import SqliteAdapter, SqliteAdapterError

        local_dir = tmp_path / "fail_shard_kv"
        adapter = SqliteAdapter(
            db_url="s3://bucket/fail-kv",
            local_dir=local_dir,
            page_size="auto",
            emit_sidecar=False,
        )
        adapter.write_batch([(i.to_bytes(8, "big"), b"v" * 100) for i in range(10)])

        with patch(
            "shardyfusion.sqlite_adapter._maybe_repage_to_auto",
            side_effect=RuntimeError("VACUUM failed"),
        ):
            with pytest.raises(RuntimeError, match="VACUUM failed"):
                adapter.seal()

        assert adapter._sealed is False
        assert adapter._seal_attempted is True

        with patch("shardyfusion.sqlite_adapter.ObstoreBackend") as mock:
            mock.return_value = MagicMock()
            with pytest.raises(SqliteAdapterError, match="seal.*not complete"):
                adapter.close()
            mock.return_value.put.assert_not_called()

    def test_context_manager_propagates_original_seal_error(
        self, tmp_path: Path
    ) -> None:
        """``with adapter:`` must surface the original seal exception when
        close() also raises — otherwise the seal failure would be buried
        under Python's __context__ chaining for any caller that only
        inspects the active exception."""
        from shardyfusion.sqlite_adapter import SqliteAdapter

        local_dir = tmp_path / "ctxmgr_shard"
        adapter = SqliteAdapter(
            db_url="s3://bucket/ctxmgr",
            local_dir=local_dir,
            page_size="auto",
            emit_sidecar=False,
        )

        with (
            patch(
                "shardyfusion.sqlite_adapter._maybe_repage_to_auto",
                side_effect=RuntimeError("VACUUM failed"),
            ),
            patch("shardyfusion.sqlite_adapter.ObstoreBackend") as mock,
        ):
            mock.return_value = MagicMock()
            with pytest.raises(Exception) as excinfo:
                with adapter:
                    adapter.write_batch(
                        [(i.to_bytes(8, "big"), b"v" * 50) for i in range(5)]
                    )
                    adapter.seal()

        # The original VACUUM failure must be reachable.  __exit__ runs
        # close() which raises SqliteAdapterError; the seal RuntimeError
        # ends up under __context__, not lost.
        exc = excinfo.value
        chain = []
        cur: BaseException | None = exc
        while cur is not None:
            chain.append(cur)
            cur = cur.__context__
        assert any(
            isinstance(c, RuntimeError) and "VACUUM failed" in str(c) for c in chain
        ), f"original seal RuntimeError missing from exception chain: {chain}"

    def test_composite_skips_vec_upload_when_kv_seal_fails(
        self, tmp_path: Path
    ) -> None:
        """Half-publish regression: when kv.seal() raises, CompositeAdapter
        close() must NOT call _vec.close() — otherwise LanceDB uploads its
        sidecar unconditionally (no _sealed gate), orphaning it in S3 with
        no matching kv .db."""
        from shardyfusion.composite_adapter import CompositeAdapter

        kv = MagicMock()
        kv.seal.side_effect = RuntimeError("kv seal failed")
        kv.close.return_value = None  # let kv.close() succeed for the test
        vec = MagicMock()

        adapter = CompositeAdapter(kv_adapter=kv, vector_writer=vec)
        with pytest.raises(RuntimeError, match="kv seal failed"):
            adapter.seal()
        assert adapter._seal_failed is True

        adapter.close()

        vec.seal.assert_not_called()
        vec.close.assert_not_called()
        kv.close.assert_called_once()
