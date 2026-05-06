"""Tests for unified KV+vector writer integration — vector_fn, vector_col,
factory wrapping, and manifest metadata injection."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from shardyfusion.config import CelShardedWriteConfig, VectorSpec
from shardyfusion.errors import ConfigValidationError
from shardyfusion.sharding_types import CelShardingSpec


def _cel_sharding() -> CelShardingSpec:
    return CelShardingSpec(
        cel_expr="shard_hash(key) % 4u",
        cel_columns={"key": "int"},
        routing_values=[0, 1, 2, 3],
    )


def _cel_config(**kwargs: Any) -> CelShardedWriteConfig:
    defaults: dict[str, Any] = {
        "s3_prefix": "s3://bucket/prefix",
        "cel_expr": "shard_hash(key) % 4u",
        "cel_columns": {"key": "int"},
        "routing_values": [0, 1, 2, 3],
    }
    defaults.update(kwargs)
    return CelShardedWriteConfig(**defaults)


# ---------------------------------------------------------------------------
# vector_fn / vector_col validation in write_cel_sharded
# ---------------------------------------------------------------------------


class TestWriteShardedVectorValidation:
    """Tests that run before any actual writing — validation only."""

    def test_vector_fn_without_vector_spec_raises(self) -> None:
        from tests.helpers.writer_api import (
            write_python_cel_sharded as write_cel_sharded,
        )

        config = CelShardedWriteConfig(
            s3_prefix="s3://bucket/prefix",
            cel_expr="shard_hash(key) % 4u",
            cel_columns={"key": "int"},
            routing_values=[0, 1, 2, 3],
        )
        with pytest.raises(ConfigValidationError, match="vector_fn requires"):
            write_cel_sharded(
                [],
                config,
                key_fn=lambda r: r,
                value_fn=lambda r: b"v",
                vector_fn=lambda r: (0, np.zeros(8), None),
            )

    def test_vector_spec_without_vector_fn_or_col_raises(self) -> None:
        from tests.helpers.writer_api import (
            write_python_cel_sharded as write_cel_sharded,
        )

        config = _cel_config(vector_spec=VectorSpec(dim=8))
        with pytest.raises(ConfigValidationError, match="vector_fn.*vector_col"):
            write_cel_sharded(
                [],
                config,
                key_fn=lambda r: r,
                value_fn=lambda r: b"v",
            )

    def test_vector_col_without_columns_fn_raises(self) -> None:
        from tests.helpers.writer_api import (
            write_python_cel_sharded as write_cel_sharded,
        )

        config = _cel_config(vector_spec=VectorSpec(dim=8, vector_col="embedding"))
        with pytest.raises(ConfigValidationError, match="columns_fn is None"):
            write_cel_sharded(
                [],
                config,
                key_fn=lambda r: r,
                value_fn=lambda r: b"v",
            )


# ---------------------------------------------------------------------------
# _wrap_factory_for_vector
# ---------------------------------------------------------------------------


class TestWrapFactoryForVector:
    def test_sqlite_vec_factory_passthrough(self) -> None:
        from shardyfusion.sqlite_vec_adapter import SqliteVecFactory
        from shardyfusion.writer.python.writer import _wrap_factory_for_vector

        vs = VectorSpec(dim=8)
        config = _cel_config(vector_spec=vs)
        factory = SqliteVecFactory(vector_spec=vs)
        result = _wrap_factory_for_vector(factory, config)
        assert result is factory  # same object, not wrapped

    def test_non_sqlite_wraps_with_composite(self) -> None:
        from shardyfusion.composite_adapter import CompositeFactory
        from shardyfusion.writer.python.writer import _wrap_factory_for_vector

        vs = VectorSpec(dim=8)
        config = _cel_config(vector_spec=vs)
        mock_factory = MagicMock()

        with patch(
            "shardyfusion.writer.python.writer.LanceDbWriterFactory",
            create=True,
        ):
            # Patch the import inside _wrap_factory_for_vector
            with patch(
                "shardyfusion.vector.adapters.lancedb_adapter.LanceDbWriterFactory"
            ) as mock_lancedb:
                mock_lancedb.return_value = MagicMock()
                result = _wrap_factory_for_vector(mock_factory, config)

        assert isinstance(result, CompositeFactory)
        assert result.kv_factory is mock_factory

    def test_custom_vector_capable_factory_passthrough(self) -> None:
        from shardyfusion.writer.python.writer import _wrap_factory_for_vector

        vs = VectorSpec(dim=8)
        config = _cel_config(vector_spec=vs)
        mock_factory = MagicMock()
        mock_factory.supports_vector_writes = True

        result = _wrap_factory_for_vector(mock_factory, config)

        assert result is mock_factory


# ---------------------------------------------------------------------------
#         detect_vector_backend
# ---------------------------------------------------------------------------


class TestDetectVectorBackend:
    def test_sqlite_vec_detected(self) -> None:
        from shardyfusion.sqlite_vec_adapter import SqliteVecFactory
        from shardyfusion.writer.python.writer import detect_vector_backend

        factory = SqliteVecFactory(vector_spec=VectorSpec(dim=8))
        assert detect_vector_backend(factory) == "sqlite-vec"

    def test_other_factory_is_lancedb_sidecar(self) -> None:
        from shardyfusion.writer.python.writer import detect_vector_backend

        assert detect_vector_backend(MagicMock()) == "lancedb"


# ---------------------------------------------------------------------------
# _inject_vector_manifest_fields
# ---------------------------------------------------------------------------


class TestInjectVectorManifestFields:
    def test_injects_all_fields(self) -> None:
        from shardyfusion.writer.python.writer import inject_vector_manifest_fields

        config = _cel_config(
            vector_spec=VectorSpec(
                dim=128,
                metric="l2",
                index_type="hnsw",
                quantization="fp16",
                index_params={"M": 32},
            )
        )
        factory = MagicMock()
        inject_vector_manifest_fields(config, factory)

        vec = config.manifest.custom_manifest_fields["vector"]
        assert vec["dim"] == 128
        assert vec["metric"] == "l2"
        assert vec["index_type"] == "hnsw"
        assert vec["quantization"] == "fp16"
        assert vec["unified"] is True
        assert vec["backend"] == "lancedb"
        assert vec["index_params"] == {"M": 32}

    def test_injects_sqlite_vec_backend(self) -> None:
        from shardyfusion.sqlite_vec_adapter import SqliteVecFactory
        from shardyfusion.writer.python.writer import inject_vector_manifest_fields

        vs = VectorSpec(dim=64)
        config = _cel_config(vector_spec=vs)
        factory = SqliteVecFactory(vector_spec=vs)
        inject_vector_manifest_fields(config, factory)

        vec = config.manifest.custom_manifest_fields["vector"]
        assert vec["backend"] == "sqlite-vec"

    def test_no_index_params_when_empty(self) -> None:
        from shardyfusion.writer.python.writer import inject_vector_manifest_fields

        config = _cel_config(vector_spec=VectorSpec(dim=64))
        inject_vector_manifest_fields(config, MagicMock())

        vec = config.manifest.custom_manifest_fields["vector"]
        assert "index_params" not in vec

    def test_copies_custom_manifest_fields_before_injecting(self) -> None:
        from shardyfusion.writer.python.writer import inject_vector_manifest_fields

        original_custom = {"existing": {"keep": True}}
        config = _cel_config(vector_spec=VectorSpec(dim=64))
        config.manifest.custom_manifest_fields = original_custom

        inject_vector_manifest_fields(config, MagicMock())

        assert config.manifest.custom_manifest_fields is not original_custom
        assert original_custom == {"existing": {"keep": True}}
        assert config.manifest.custom_manifest_fields["existing"] == {"keep": True}


# ---------------------------------------------------------------------------
# inject_sqlite_btreemeta_manifest_field
# ---------------------------------------------------------------------------


class TestInjectSqliteBtreemetaManifestField:
    """The btree-metadata manifest hook records the snapshot-level flag
    so reader-side range mode can fetch the sidecar without per-shard
    HEAD probes. The hook must be safe to call universally — it's a
    no-op when the factory is not SQLite-backed.
    """

    def _field(self, config: CelShardedWriteConfig) -> object:
        return config.manifest.custom_manifest_fields.get("sqlite_btreemeta")

    def test_sqlite_factory_default_on_records_field(self) -> None:
        from shardyfusion._writer_core import inject_sqlite_btreemeta_manifest_field
        from shardyfusion.sqlite_adapter import SqliteFactory

        config = _cel_config()
        inject_sqlite_btreemeta_manifest_field(config, SqliteFactory())

        assert self._field(config) == {
            "format_version": 3,
            "page_size": 4096,
            "filename": "shard.btreemeta",
            "codec": "zstd",
        }

    def test_sqlite_factory_explicit_off_omits_field(self) -> None:
        from shardyfusion._writer_core import inject_sqlite_btreemeta_manifest_field
        from shardyfusion.sqlite_adapter import SqliteFactory

        config = _cel_config()
        inject_sqlite_btreemeta_manifest_field(
            config, SqliteFactory(emit_btree_metadata=False)
        )
        assert self._field(config) is None

    def test_sqlite_factory_records_custom_page_size(self) -> None:
        from shardyfusion._writer_core import inject_sqlite_btreemeta_manifest_field
        from shardyfusion.sqlite_adapter import SqliteFactory

        config = _cel_config()
        inject_sqlite_btreemeta_manifest_field(config, SqliteFactory(page_size=8192))
        field = self._field(config)
        assert isinstance(field, dict)
        assert field["page_size"] == 8192

    def test_sqlite_vec_factory_default_on_records_field(self) -> None:
        from shardyfusion._writer_core import inject_sqlite_btreemeta_manifest_field
        from shardyfusion.sqlite_vec_adapter import SqliteVecFactory

        vs = VectorSpec(dim=8)
        config = _cel_config(vector_spec=vs)
        inject_sqlite_btreemeta_manifest_field(config, SqliteVecFactory(vector_spec=vs))
        assert self._field(config) == {
            "format_version": 3,
            "page_size": 4096,
            "filename": "shard.btreemeta",
            "codec": "zstd",
        }

    def test_sqlite_vec_factory_explicit_off_omits_field(self) -> None:
        from shardyfusion._writer_core import inject_sqlite_btreemeta_manifest_field
        from shardyfusion.sqlite_vec_adapter import SqliteVecFactory

        vs = VectorSpec(dim=8)
        config = _cel_config(vector_spec=vs)
        inject_sqlite_btreemeta_manifest_field(
            config, SqliteVecFactory(vector_spec=vs, emit_btree_metadata=False)
        )
        assert self._field(config) is None

    def test_composite_factory_unwraps_inner_sqlite_kv(self) -> None:
        from shardyfusion._writer_core import inject_sqlite_btreemeta_manifest_field
        from shardyfusion.composite_adapter import CompositeFactory
        from shardyfusion.sqlite_adapter import SqliteFactory

        vs = VectorSpec(dim=8)
        # Vector factory can be any non-SQLite factory; we use a MagicMock
        # because we don't actually instantiate adapters here.
        composite = CompositeFactory(
            kv_factory=SqliteFactory(),
            vector_factory=MagicMock(),
            vector_spec=vs,
        )
        config = _cel_config(vector_spec=vs)
        inject_sqlite_btreemeta_manifest_field(config, composite)

        field = self._field(config)
        assert isinstance(field, dict)
        assert field["page_size"] == 4096

    def test_composite_factory_with_inner_sqlite_off_omits_field(self) -> None:
        from shardyfusion._writer_core import inject_sqlite_btreemeta_manifest_field
        from shardyfusion.composite_adapter import CompositeFactory
        from shardyfusion.sqlite_adapter import SqliteFactory

        vs = VectorSpec(dim=8)
        composite = CompositeFactory(
            kv_factory=SqliteFactory(emit_btree_metadata=False),
            vector_factory=MagicMock(),
            vector_spec=vs,
        )
        config = _cel_config(vector_spec=vs)
        inject_sqlite_btreemeta_manifest_field(config, composite)
        assert self._field(config) is None

    def test_non_sqlite_factory_omits_field(self) -> None:
        from shardyfusion._writer_core import inject_sqlite_btreemeta_manifest_field

        config = _cel_config()
        inject_sqlite_btreemeta_manifest_field(config, MagicMock())
        assert self._field(config) is None

    def test_preserves_other_custom_fields(self) -> None:
        from shardyfusion._writer_core import inject_sqlite_btreemeta_manifest_field
        from shardyfusion.sqlite_adapter import SqliteFactory

        config = _cel_config()
        config.manifest.custom_manifest_fields = {"existing": {"keep": True}}

        inject_sqlite_btreemeta_manifest_field(config, SqliteFactory())

        assert config.manifest.custom_manifest_fields["existing"] == {"keep": True}
        assert "sqlite_btreemeta" in config.manifest.custom_manifest_fields
