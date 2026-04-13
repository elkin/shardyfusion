"""Tests for resolve_distributed_vector_fn in _writer_core.py."""

from __future__ import annotations

import pytest

from shardyfusion._writer_core import VectorColumnMapping, resolve_distributed_vector_fn
from shardyfusion.config import VectorSpec, WriteConfig
from shardyfusion.errors import ConfigValidationError
from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy


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


class TestResolveDistributedVectorFn:
    def test_no_vector_spec_returns_none(self) -> None:
        config = WriteConfig(num_dbs=2, s3_prefix="s3://bucket/prefix")
        result = resolve_distributed_vector_fn(
            config=config, key_col="key", vector_fn=None, vector_columns=None
        )
        assert result is None

    def test_vector_fn_passed_through(self) -> None:
        config = _cel_config(VectorSpec(dim=4, vector_col="emb"))

        def sentinel_fn(row: dict) -> tuple:
            return (row["key"], row["emb"], None)

        result = resolve_distributed_vector_fn(
            config=config, key_col="key", vector_fn=sentinel_fn, vector_columns=None
        )
        assert result is sentinel_fn

    def test_vector_fn_without_vector_spec_raises(self) -> None:
        config = WriteConfig(num_dbs=2, s3_prefix="s3://bucket/prefix")
        with pytest.raises(ConfigValidationError, match="vector_fn requires"):
            resolve_distributed_vector_fn(
                config=config,
                key_col="key",
                vector_fn=lambda row: (1, [0.0], None),
                vector_columns=None,
            )

    def test_auto_fn_from_vector_columns(self) -> None:
        config = _cel_config(VectorSpec(dim=4, vector_col="emb"))
        mapping = VectorColumnMapping(
            vector_col="emb", id_col="doc_id", payload_cols=["region"]
        )
        fn = resolve_distributed_vector_fn(
            config=config, key_col="key", vector_fn=None, vector_columns=mapping
        )
        assert fn is not None
        vec_id, vec, payload = fn(
            {"key": 1, "doc_id": "abc", "emb": [1.0, 2.0], "region": "us"}
        )
        assert vec_id == "abc"
        assert vec == [1.0, 2.0]
        assert payload == {"region": "us"}

    def test_auto_fn_uses_vector_spec_defaults(self) -> None:
        config = _cel_config(VectorSpec(dim=4, vector_col="embedding"))
        fn = resolve_distributed_vector_fn(
            config=config, key_col="doc_id", vector_fn=None, vector_columns=None
        )
        assert fn is not None
        vec_id, vec, payload = fn({"doc_id": 42, "embedding": [1, 2, 3, 4]})
        assert vec_id == 42
        assert vec == [1, 2, 3, 4]
        assert payload is None

    def test_auto_fn_no_vector_col_raises(self) -> None:
        config = _cel_config(VectorSpec(dim=2, vector_col=None))
        with pytest.raises(ConfigValidationError, match="no vector_fn was provided"):
            resolve_distributed_vector_fn(
                config=config, key_col="key", vector_fn=None, vector_columns=None
            )

    def test_auto_fn_coerces_non_int_str_id(self) -> None:
        """Non-int, non-str IDs are cast to str."""
        config = _cel_config(VectorSpec(dim=2, vector_col="v"))
        fn = resolve_distributed_vector_fn(
            config=config, key_col="k", vector_fn=None, vector_columns=None
        )
        assert fn is not None
        vec_id, _, _ = fn({"k": 3.14, "v": [1.0, 2.0]})
        assert isinstance(vec_id, str)
        assert vec_id == "3.14"

    def test_auto_fn_payload_cols_from_mapping(self) -> None:
        config = _cel_config(VectorSpec(dim=2, vector_col="v"))
        mapping = VectorColumnMapping(
            vector_col="v", id_col=None, payload_cols=["color", "size"]
        )
        fn = resolve_distributed_vector_fn(
            config=config, key_col="k", vector_fn=None, vector_columns=mapping
        )
        assert fn is not None
        _, _, payload = fn({"k": 1, "v": [1.0], "color": "red", "size": 10})
        assert payload == {"color": "red", "size": 10}

    def test_auto_fn_id_col_falls_back_to_key_col(self) -> None:
        """When mapping.id_col is None, key_col is used for IDs."""
        config = _cel_config(VectorSpec(dim=2, vector_col="v"))
        mapping = VectorColumnMapping(vector_col="v", id_col=None, payload_cols=None)
        fn = resolve_distributed_vector_fn(
            config=config, key_col="my_key", vector_fn=None, vector_columns=mapping
        )
        assert fn is not None
        vec_id, _, _ = fn({"my_key": 99, "v": [1.0, 2.0]})
        assert vec_id == 99
