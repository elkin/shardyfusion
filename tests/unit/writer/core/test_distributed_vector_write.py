"""Unit tests for distributed unified vector write helpers."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from shardyfusion._shard_writer import (
    ShardWriteParams,
    write_shard_core_distributed,
    write_shard_with_retry_distributed,
)
from shardyfusion._writer_core import VectorColumnMapping, resolve_distributed_vector_fn
from shardyfusion.config import VectorSpec, WriteConfig
from shardyfusion.errors import ConfigValidationError, ShardWriteError
from shardyfusion.serde import make_key_encoder
from shardyfusion.sharding_types import KeyEncoding, ShardingSpec, ShardingStrategy
from shardyfusion.type_defs import RetryConfig


class _VectorAdapter:
    def __init__(self) -> None:
        self.kv_batches: list[list[tuple[bytes, bytes]]] = []
        self.vector_batches: list[tuple[Any, Any, list[dict[str, Any] | None]]] = []

    def __enter__(self) -> _VectorAdapter:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def write_batch(self, batch: list[tuple[bytes, bytes]]) -> None:
        self.kv_batches.append(list(batch))

    def write_vector_batch(
        self,
        ids: Any,
        vectors: Any,
        payloads: list[dict[str, Any] | None] | None = None,
    ) -> None:
        ids_copy = ids.copy() if hasattr(ids, "copy") else ids
        vectors_copy = vectors.copy() if hasattr(vectors, "copy") else vectors
        self.vector_batches.append((ids_copy, vectors_copy, list(payloads or [])))

    def flush(self) -> None:
        pass

    def checkpoint(self) -> str:
        return "ckpt"

    def db_bytes(self) -> int:
        return 0


class _NoVectorAdapter(_VectorAdapter):
    write_vector_batch = None  # type: ignore[assignment]


class _Factory:
    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    def __call__(self, *, db_url: str, local_dir: Path) -> Any:
        local_dir.mkdir(parents=True, exist_ok=True)
        return self.adapter


def _cel_config(vector_spec: VectorSpec | None) -> WriteConfig:
    return WriteConfig(
        num_dbs=None,
        s3_prefix="s3://bucket/prefix",
        sharding=ShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr="key % 2",
            cel_columns={"key": "int"},
        ),
        vector_spec=vector_spec,
    )


def test_resolve_distributed_vector_fn_from_vector_columns() -> None:
    config = _cel_config(VectorSpec(dim=4, vector_col="embedding"))
    fn = resolve_distributed_vector_fn(
        config=config,
        key_col="key",
        vector_fn=None,
        vector_columns=VectorColumnMapping(
            vector_col="embedding", id_col="doc_id", payload_cols=["region"]
        ),
    )
    assert fn is not None
    vec_id, vec, payload = fn(
        {"key": 1, "doc_id": "id-1", "embedding": [1, 2], "region": "us"}
    )
    assert vec_id == "id-1"
    assert vec == [1, 2]
    assert payload == {"region": "us"}


def test_resolve_distributed_vector_fn_requires_vector_spec() -> None:
    with pytest.raises(ConfigValidationError, match="vector_fn requires"):
        resolve_distributed_vector_fn(
            config=WriteConfig(num_dbs=2, s3_prefix="s3://bucket/prefix"),
            key_col="key",
            vector_fn=lambda row: (1, row["embedding"], None),
            vector_columns=None,
        )


def test_resolve_distributed_vector_fn_uses_vector_spec_defaults() -> None:
    config = _cel_config(VectorSpec(dim=4, vector_col="embedding"))
    fn = resolve_distributed_vector_fn(
        config=config,
        key_col="doc_id",
        vector_fn=None,
        vector_columns=None,
    )
    assert fn is not None
    vec_id, vec, payload = fn(
        {"doc_id": 9, "embedding": [1.0, 2.0, 3.0, 4.0], "ignored": "v"}
    )
    assert vec_id == 9
    assert vec == [1.0, 2.0, 3.0, 4.0]
    assert payload is None


def test_resolve_distributed_vector_fn_requires_vector_source_when_missing() -> None:
    config = _cel_config(VectorSpec(dim=2, vector_col=None))
    with pytest.raises(ConfigValidationError, match="no vector_fn was provided"):
        resolve_distributed_vector_fn(
            config=config,
            key_col="key",
            vector_fn=None,
            vector_columns=None,
        )


def test_write_shard_core_distributed_writes_vectors(tmp_path: Path) -> None:
    adapter = _VectorAdapter()
    params = ShardWriteParams(
        db_id=0,
        attempt=0,
        run_id="r1",
        db_url="s3://bucket/prefix/db=00000/attempt=00",
        local_dir=tmp_path / "db=00000",
        factory=_Factory(adapter),
        key_encoder=make_key_encoder(KeyEncoding.U64BE),
        batch_size=2,
        ops_limiter=None,
        bytes_limiter=None,
        metrics_collector=None,
        started=0.0,
    )
    rows = [
        (1, b"v1", ("id-1", [0.1, 0.2], {"a": 1})),
        (2, b"v2", ("id-2", [0.3, 0.4], None)),
    ]
    result = write_shard_core_distributed(params, rows)
    assert result.row_count == 2
    assert len(adapter.kv_batches) == 1
    assert len(adapter.vector_batches) == 1
    ids, vectors, payloads = adapter.vector_batches[0]
    assert ids.tolist() == ["id-1", "id-2"]
    assert vectors.shape == (2, 2)
    assert payloads[0] == {"a": 1}


def test_write_shard_core_distributed_requires_vector_capable_adapter(
    tmp_path: Path,
) -> None:
    adapter = _NoVectorAdapter()
    params = ShardWriteParams(
        db_id=0,
        attempt=0,
        run_id="r1",
        db_url="s3://bucket/prefix/db=00000/attempt=00",
        local_dir=tmp_path / "db=00000",
        factory=_Factory(adapter),
        key_encoder=make_key_encoder(KeyEncoding.U64BE),
        batch_size=1,
        ops_limiter=None,
        bytes_limiter=None,
        metrics_collector=None,
        started=0.0,
    )
    with pytest.raises(ShardWriteError, match="Adapter does not support vector writes"):
        write_shard_core_distributed(params, [(1, b"v1", ("id-1", [1.0], None))])


def test_write_shard_core_distributed_ignores_rows_without_vectors(
    tmp_path: Path,
) -> None:
    adapter = _VectorAdapter()
    params = ShardWriteParams(
        db_id=0,
        attempt=0,
        run_id="r1",
        db_url="s3://bucket/prefix/db=00000/attempt=00",
        local_dir=tmp_path / "db=00000",
        factory=_Factory(adapter),
        key_encoder=make_key_encoder(KeyEncoding.U64BE),
        batch_size=2,
        ops_limiter=None,
        bytes_limiter=None,
        metrics_collector=None,
        started=0.0,
    )

    rows = [
        (1, b"v1", None),
        (2, b"v2", ("id-2", [0.2, 0.3], {"region": "us"})),
    ]
    result = write_shard_core_distributed(params, rows)

    assert result.row_count == 2
    assert len(adapter.kv_batches) == 1
    assert len(adapter.vector_batches) == 1
    ids, vectors, payloads = adapter.vector_batches[0]
    assert ids.tolist() == ["id-2"]
    assert vectors.shape == (1, 2)
    assert payloads == [{"region": "us"}]


@patch("shardyfusion._shard_writer.time.sleep")
def test_write_shard_with_retry_distributed_retries_and_tracks_attempt_urls(
    _mock_sleep: Any,
    tmp_path: Path,
) -> None:
    call_count = 0

    class _FailOnceVectorFactory:
        def __call__(self, *, db_url: str, local_dir: Path) -> Any:
            nonlocal call_count
            call_count += 1
            adapter = _VectorAdapter()
            if call_count == 1:

                def _fail_write_batch(_batch: list[tuple[bytes, bytes]]) -> None:
                    raise OSError("transient")

                adapter.write_batch = _fail_write_batch  # type: ignore[method-assign]
            return adapter

    result = write_shard_with_retry_distributed(
        db_id=0,
        rows_fn=lambda: iter([(1, b"v1", ("id-1", [0.1, 0.2], None))]),
        run_id="test-run",
        s3_prefix="s3://bucket/prefix",
        shard_prefix="shards",
        db_path_template="db={db_id:05d}",
        local_root=str(tmp_path),
        key_encoder=make_key_encoder(KeyEncoding.U64BE),
        batch_size=1,
        factory=_FailOnceVectorFactory(),
        max_writes_per_second=None,
        max_write_bytes_per_second=None,
        metrics_collector=None,
        started=0.0,
        retry_config=RetryConfig(
            max_retries=2, initial_backoff=timedelta(milliseconds=1)
        ),
    )

    assert result.attempt == 1
    assert result.row_count == 1
    assert call_count == 2
    assert result.all_attempt_urls == (
        "s3://bucket/prefix/shards/run_id=test-run/db=00000/attempt=00",
        "s3://bucket/prefix/shards/run_id=test-run/db=00000/attempt=01",
    )
