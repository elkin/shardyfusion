"""USearch adapter regression tests.

Tests for string/int ID handling, id_map table, and writer lifecycle.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from shardyfusion.errors import VectorIndexError
from shardyfusion.vector.adapters.usearch_adapter import (
    PAYLOADS_FILENAME,
    USearchWriter,
)
from shardyfusion.vector.types import DistanceMetric


class TestUSearchStringIds:
    """USearch add_batch should handle string IDs via id_map table."""

    def test_string_ids_written_to_id_map(self, tmp_path: Path) -> None:
        """Writer stores string->int mapping when string IDs are used."""
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

        conn = sqlite3.connect(str(tmp_path / PAYLOADS_FILENAME))
        rows = conn.execute(
            "SELECT internal_id, original_id FROM id_map ORDER BY internal_id"
        ).fetchall()
        conn.close()

        assert len(rows) == 3
        assert rows[0][0] == 0
        assert json.loads(rows[0][1]) == {"v": "abc", "t": "str"}
        assert rows[1][0] == 1
        assert json.loads(rows[1][1]) == {"v": "def", "t": "str"}
        assert rows[2][0] == 2
        assert json.loads(rows[2][1]) == {"v": "ghi", "t": "str"}

        with patch(
            "shardyfusion.vector.adapters.usearch_adapter.put_bytes",
            return_value=None,
        ):
            writer.close()

    def test_int_ids_have_typed_id_map_rows(self, tmp_path: Path) -> None:
        """Writer creates typed id_map rows for integer IDs too."""
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
        rows = conn.execute(
            "SELECT internal_id, original_id FROM id_map ORDER BY internal_id"
        ).fetchall()
        conn.close()
        assert len(rows) == 3
        assert json.loads(rows[0][1]) == {"v": 10, "t": "int"}
        assert json.loads(rows[1][1]) == {"v": 20, "t": "int"}
        assert json.loads(rows[2][1]) == {"v": 30, "t": "int"}

        with patch(
            "shardyfusion.vector.adapters.usearch_adapter.put_bytes",
            return_value=None,
        ):
            writer.close()

    def test_payload_rows_use_internal_ids_to_avoid_type_collisions(
        self, tmp_path: Path
    ) -> None:
        """Integer and string IDs with the same text should keep distinct payload rows."""
        writer = USearchWriter(
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

        conn = sqlite3.connect(str(tmp_path / PAYLOADS_FILENAME))
        rows = conn.execute("SELECT id, payload FROM payloads ORDER BY id").fetchall()
        conn.close()

        assert rows == [
            ("0", json.dumps({"kind": "int"})),
            ("1", json.dumps({"kind": "str"})),
        ]

        with patch(
            "shardyfusion.vector.adapters.usearch_adapter.put_bytes",
            return_value=None,
        ):
            writer.close()

    def test_close_marks_writer_closed_after_successful_upload(
        self, tmp_path: Path
    ) -> None:
        writer = USearchWriter(
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
            "shardyfusion.vector.adapters.usearch_adapter.put_bytes",
            return_value=None,
        ):
            writer.close()

        assert writer._closed is True

    def test_close_upload_failure_does_not_mark_writer_closed(
        self, tmp_path: Path
    ) -> None:
        writer = USearchWriter(
            db_url="s3://b/test",
            local_dir=tmp_path,
            dim=4,
            metric=DistanceMetric.COSINE,
        )
        writer.add_batch(
            np.array([1], dtype=np.int64),
            np.random.default_rng(9).standard_normal((1, 4)).astype(np.float32),
        )

        with patch(
            "shardyfusion.vector.adapters.usearch_adapter.put_bytes",
            side_effect=RuntimeError("upload failed"),
        ):
            with pytest.raises(VectorIndexError, match="Failed to upload"):
                writer.close()

        assert writer._closed is False
