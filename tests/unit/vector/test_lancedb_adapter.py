"""LanceDB adapter tests.

Tests for string/int ID handling and writer lifecycle.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from shardyfusion.vector.adapters.lancedb_adapter import (
    LanceDbWriter,
)
from shardyfusion.vector.types import DistanceMetric


class TestLanceDbStringIds:
    """LanceDb add_batch should handle string IDs."""

    def test_string_ids_written_to_table(self, tmp_path: Path) -> None:
        """Writer stores string->int mapping when string IDs are used."""
        writer = LanceDbWriter(
            db_url="s3://b/test",
            local_dir=tmp_path,
            dim=4,
            metric=DistanceMetric.COSINE,
        )

        ids = np.array(["abc", "def", "ghi"], dtype=object)
        vecs = np.random.default_rng(42).standard_normal((3, 4)).astype(np.float32)
        writer.add_batch(ids, vecs)
        writer.flush()

        assert writer._table is not None
        assert len(writer._table) == 3

        with patch(
            "shardyfusion.vector.adapters.lancedb_adapter.LanceDbWriter._upload_dir",
            return_value=None,
        ):
            writer.close()

    def test_int_ids_handled(self, tmp_path: Path) -> None:
        """Writer creates string representation for integer IDs."""
        writer = LanceDbWriter(
            db_url="s3://b/test",
            local_dir=tmp_path,
            dim=4,
            metric=DistanceMetric.COSINE,
        )

        ids = np.array([10, 20, 30], dtype=np.int64)
        vecs = np.random.default_rng(42).standard_normal((3, 4)).astype(np.float32)
        writer.add_batch(ids, vecs)
        writer.flush()

        assert writer._table is not None
        assert len(writer._table) == 3

        with patch(
            "shardyfusion.vector.adapters.lancedb_adapter.LanceDbWriter._upload_dir",
            return_value=None,
        ):
            writer.close()

    def test_payload_rows_use_original_ids(self, tmp_path: Path) -> None:
        writer = LanceDbWriter(
            db_url="s3://b/test",
            local_dir=tmp_path,
            dim=4,
            metric=DistanceMetric.COSINE,
        )

        ids = np.array([1, "1"], dtype=object)
        vecs = np.random.default_rng(21).standard_normal((2, 4)).astype(np.float32)
        payloads = [{"kind": "int"}, {"kind": "str"}]
        writer.add_batch(ids, vecs, payloads)
        writer.flush()

        assert writer._table is not None
        assert len(writer._table) == 2

        with patch(
            "shardyfusion.vector.adapters.lancedb_adapter.LanceDbWriter._upload_dir",
            return_value=None,
        ):
            writer.close()

    def test_close_marks_writer_closed_after_successful_upload(
        self, tmp_path: Path
    ) -> None:
        writer = LanceDbWriter(
            db_url="s3://b/test",
            local_dir=tmp_path,
            dim=4,
            metric=DistanceMetric.COSINE,
        )
        writer.add_batch(
            np.array([1], dtype=np.int64),
            np.random.default_rng(7).standard_normal((1, 4)).astype(np.float32),
        )

        with patch(
            "shardyfusion.vector.adapters.lancedb_adapter.LanceDbWriter._upload_dir",
            return_value=None,
        ):
            writer.close()

        assert writer._closed is True
