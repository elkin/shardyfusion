from __future__ import annotations

import pytest

pytest.importorskip("lancedb")
pytest.importorskip("pyarrow")

from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa

from shardyfusion.vector.adapters.lancedb_adapter import (
    LanceDbShardReader,
)


@pytest.fixture
def sample_lancedb(tmp_path: Path) -> str:
    """Create a sample LanceDB dataset locally."""
    db_url = str(tmp_path / "test_db")
    db = lancedb.connect(db_url)

    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 4)),
            pa.field("payload", pa.string()),
        ]
    )

    table = db.create_table("vector_index", schema=schema)

    data = [
        {"id": "1", "vector": [0.1, 0.2, 0.3, 0.4], "payload": '{"key":"payload1"}'},
        {"id": "123", "vector": [0.4, 0.3, 0.2, 0.1], "payload": '{"key":"payload2"}'},
        {"id": "abc", "vector": [0.0, 0.0, 0.0, 1.0], "payload": '{"key":"payload3"}'},
    ]
    table.add(data)

    return db_url


class MockIndexConfig:
    pass


def test_lancedb_shard_reader_search(sample_lancedb: str, tmp_path: Path) -> None:
    reader = LanceDbShardReader(
        db_url=sample_lancedb,
        local_dir=tmp_path / "local",
        index_config=MockIndexConfig(),
    )

    query = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    results = reader.search(query, top_k=2)

    assert len(results) == 2
    assert results[0].id == "1"
    assert results[0].payload == {"key": "payload1"}

    # Check that "123" stays a string and isn't converted to int 123
    assert results[1].id == "123"

    reader.close()


def test_lancedb_shard_reader_search_empty(tmp_path: Path) -> None:
    db_url = str(tmp_path / "empty_db")
    db = lancedb.connect(db_url)
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 4)),
            pa.field("payload", pa.string()),
        ]
    )
    db.create_table("vector_index", schema=schema)

    reader = LanceDbShardReader(
        db_url=db_url, local_dir=tmp_path / "local", index_config=MockIndexConfig()
    )

    query = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    results = reader.search(query, top_k=5)

    assert len(results) == 0
    reader.close()
