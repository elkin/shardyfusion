"""Mutual-exclusivity tests for the page_size selection mechanisms.

A writer config must specify at most one page_size strategy:

* Explicit ``page_size`` on the SQLite factory (any supported int).
* ``page_size="auto"`` on the factory (post-write VACUUM picks per shard).
* ``kv.profile_value_sizes_for_page_size=True`` on the writer config
  (distributed engines compute the percentile upstream).

The default — ``page_size=4096`` with the flag unset — is the
"no choice made" baseline.  Any other combination is a configuration
error raised at config construction.
"""

from __future__ import annotations

from typing import Any

import pytest

from shardyfusion.config import KeyValueWriteConfig
from shardyfusion.errors import ConfigValidationError
from shardyfusion.sqlite_adapter import SqliteFactory


class TestPageSizeMutualExclusion:
    def test_defaults_pass(self) -> None:
        KeyValueWriteConfig()

    def test_explicit_int_only(self) -> None:
        KeyValueWriteConfig(adapter_factory=SqliteFactory(page_size=8192))

    def test_auto_only(self) -> None:
        KeyValueWriteConfig(adapter_factory=SqliteFactory(page_size="auto"))

    def test_flag_with_default_factory(self) -> None:
        # Default factory has page_size=4096 which is the "no choice"
        # baseline, so adding the flag is allowed.
        KeyValueWriteConfig(
            adapter_factory=SqliteFactory(),
            profile_value_sizes_for_page_size=True,
        )

    def test_flag_with_explicit_default_int(self) -> None:
        # page_size=4096 passed explicitly is treated as the "no choice"
        # baseline just like the default — the profiling flag is allowed.
        KeyValueWriteConfig(
            adapter_factory=SqliteFactory(page_size=4096),
            profile_value_sizes_for_page_size=True,
        )

    def test_flag_with_explicit_int_rejected(self) -> None:
        with pytest.raises(ConfigValidationError) as excinfo:
            KeyValueWriteConfig(
                adapter_factory=SqliteFactory(page_size=16384),
                profile_value_sizes_for_page_size=True,
            )
        msg = str(excinfo.value)
        assert "profile_value_sizes_for_page_size" in msg
        assert "page_size=16384" in msg

    def test_flag_with_auto_rejected(self) -> None:
        with pytest.raises(ConfigValidationError) as excinfo:
            KeyValueWriteConfig(
                adapter_factory=SqliteFactory(page_size="auto"),
                profile_value_sizes_for_page_size=True,
            )
        msg = str(excinfo.value)
        assert "profile_value_sizes_for_page_size" in msg
        assert "page_size='auto'" in msg

    def test_factory_rejects_unsupported_int(self) -> None:
        with pytest.raises(ConfigValidationError):
            SqliteFactory(page_size=3000)

    def test_factory_rejects_unknown_string(self) -> None:
        with pytest.raises(ConfigValidationError):
            SqliteFactory(page_size="adaptive")  # type: ignore[arg-type]


class TestPythonWriterRejectsProfileFlag:
    """The Python writer cannot consume the distributed-engine flag."""

    def test_python_writer_rejects_flag(self) -> None:
        # Imports gated because the writer module pulls in a lot of
        # transitive deps; the rejection lives near the entry point.
        from shardyfusion.writer.python.writer import _reject_engine_profile_flag

        cfg = type(
            "_Stub",
            (),
            {"kv": KeyValueWriteConfig(profile_value_sizes_for_page_size=True)},
        )()
        with pytest.raises(ConfigValidationError) as excinfo:
            _reject_engine_profile_flag(cfg)
        msg = str(excinfo.value)
        assert "Python writer" in msg
        assert "page_size='auto'" in msg

    def test_python_writer_accepts_when_flag_unset(self) -> None:
        from shardyfusion.writer.python.writer import _reject_engine_profile_flag

        cfg = type("_Stub", (), {"kv": KeyValueWriteConfig()})()
        _reject_engine_profile_flag(cfg)  # no raise


class TestEnginePickerSurvivesRevalidation:
    """Regression: the engine substitution must leave the config in a
    state that survives a downstream ``validate_configs(config)`` call.

    `_common_runtime_kwargs` re-runs validation after the writer
    entrypoint, and an earlier version of the helper left the flag set
    next to a non-default factory page_size — which then tripped
    ``_validate_page_size_mutual_exclusion`` and aborted the write.
    """

    def test_revalidation_after_substitution(self) -> None:
        from shardyfusion.config import BaseShardedWriteConfig, WriterStorageConfig
        from shardyfusion.writer._engine_page_size import maybe_apply_engine_page_size

        config = BaseShardedWriteConfig(
            storage=WriterStorageConfig(s3_prefix="s3://bucket/x"),
            kv=KeyValueWriteConfig(
                adapter_factory=SqliteFactory(),
                profile_value_sizes_for_page_size=True,
            ),
        )
        maybe_apply_engine_page_size(
            config, value_byte_samples=[3500] * 100, writer_kind="test"
        )
        assert config.kv.adapter_factory.page_size == 16384
        assert config.kv.profile_value_sizes_for_page_size is False
        # Must not raise — the writer re-validates inside
        # _common_runtime_kwargs and would have crashed before the fix.
        config.validate()


class TestEnginePickerVecAware:
    """The engine picker must account for the vec_index embedding payload
    so unified KV+vector workloads with tiny KV values still pick a
    page size large enough to fit the embedding inline."""

    def test_vec_payload_bumps_recommendation(self) -> None:
        pytest.importorskip("sqlite_vec")
        from shardyfusion.config import (
            BaseShardedWriteConfig,
            VectorSpec,
            WriterStorageConfig,
        )
        from shardyfusion.sqlite_vec_adapter import SqliteVecFactory
        from shardyfusion.writer._engine_page_size import maybe_apply_engine_page_size

        spec = VectorSpec(dim=256, metric="cosine")  # 256 * 4 = 1024 B
        config = BaseShardedWriteConfig(
            storage=WriterStorageConfig(s3_prefix="s3://bucket/x"),
            vector=spec,
            kv=KeyValueWriteConfig(
                adapter_factory=SqliteVecFactory(vector_spec=spec),
                profile_value_sizes_for_page_size=True,
            ),
        )
        # KV values are tiny but the vec embedding payload is 1024 B,
        # which exceeds the 4 KiB-page inline threshold (~1002 B) once
        # key + cell overhead are added.
        maybe_apply_engine_page_size(
            config, value_byte_samples=[64] * 100, writer_kind="test"
        )
        assert config.kv.adapter_factory.page_size == 8192


class TestMutexUnwrapsCompositeFactory:
    """The mutex validator must traverse CompositeFactory to reach the
    inner KV factory's page_size — otherwise a wrapped explicit/auto
    page_size paired with the profile flag silently bypasses the check
    and is overridden at engine-pick time."""

    def _composite(self, *, page_size: Any) -> Any:
        lancedb = pytest.importorskip("shardyfusion.vector.adapters.lancedb_adapter")
        from shardyfusion.composite_adapter import CompositeFactory
        from shardyfusion.config import VectorSpec

        return CompositeFactory(
            kv_factory=SqliteFactory(page_size=page_size),
            vector_factory=lancedb.LanceDbWriterFactory(),
            vector_spec=VectorSpec(dim=128, metric="cosine"),
        )

    def test_composite_auto_with_flag_rejected(self) -> None:
        with pytest.raises(ConfigValidationError) as excinfo:
            KeyValueWriteConfig(
                adapter_factory=self._composite(page_size="auto"),
                profile_value_sizes_for_page_size=True,
            )
        assert "page_size='auto'" in str(excinfo.value)

    def test_composite_explicit_int_with_flag_rejected(self) -> None:
        with pytest.raises(ConfigValidationError) as excinfo:
            KeyValueWriteConfig(
                adapter_factory=self._composite(page_size=16384),
                profile_value_sizes_for_page_size=True,
            )
        assert "page_size=16384" in str(excinfo.value)

    def test_composite_default_int_with_flag_allowed(self) -> None:
        # page_size=4096 (default) is the "no choice" baseline.
        KeyValueWriteConfig(
            adapter_factory=self._composite(page_size=4096),
            profile_value_sizes_for_page_size=True,
        )


class TestFactoryVecPayload:
    """``vec_payload_bytes_in_kv_db()`` reports the per-row embedding
    cost the factory itself stores in the KV ``.db``.  Layout-aware:
    SqliteFactory returns 0 (KV-only); SqliteVecFactory returns ``4*dim
    + margin`` (embeddings live in vec_index in the same .db);
    CompositeFactory returns 0 regardless of inner because the composite
    always routes embeddings to the sidecar vector_factory."""

    def test_sqlite_factory_reports_zero(self) -> None:
        assert SqliteFactory().vec_payload_bytes_in_kv_db() == 0

    def test_sqlite_vec_factory_reports_payload(self) -> None:
        pytest.importorskip("sqlite_vec")
        from shardyfusion.config import VectorSpec
        from shardyfusion.sqlite_vec_adapter import (
            _VEC0_ROW_OVERHEAD_BYTES,
            SqliteVecFactory,
        )

        f = SqliteVecFactory(vector_spec=VectorSpec(dim=256, metric="cosine"))
        assert f.vec_payload_bytes_in_kv_db() == 4 * 256 + _VEC0_ROW_OVERHEAD_BYTES

    def test_composite_with_sidecar_reports_zero(self) -> None:
        lancedb = pytest.importorskip("shardyfusion.vector.adapters.lancedb_adapter")
        from shardyfusion.composite_adapter import CompositeFactory
        from shardyfusion.config import VectorSpec

        spec = VectorSpec(dim=512, metric="cosine")
        f = CompositeFactory(
            kv_factory=SqliteFactory(),
            vector_factory=lancedb.LanceDbWriterFactory(),
            vector_spec=spec,
        )
        # Embeddings live in the LanceDB sidecar, not the KV .db.
        assert f.vec_payload_bytes_in_kv_db() == 0

    def test_composite_returns_zero_regardless_of_inner(self) -> None:
        """CompositeAdapter.write_vector_batch routes embeddings to the
        sidecar vector_factory unconditionally — even if the inner
        kv_factory is itself vec-aware (e.g. SqliteVecFactory), its
        vec_index table is never populated under the composite.
        Returning the inner's payload would over-budget the page-size
        picker for an empty cell."""
        pytest.importorskip("sqlite_vec")
        lancedb = pytest.importorskip("shardyfusion.vector.adapters.lancedb_adapter")
        from shardyfusion.composite_adapter import CompositeFactory
        from shardyfusion.config import VectorSpec
        from shardyfusion.sqlite_vec_adapter import SqliteVecFactory

        spec = VectorSpec(dim=128, metric="cosine")
        f = CompositeFactory(
            kv_factory=SqliteVecFactory(vector_spec=spec),
            vector_factory=lancedb.LanceDbWriterFactory(),
            vector_spec=spec,
        )
        # Inner SqliteVecFactory's vec_index is never written under the
        # composite, so the page-size budget excludes it.
        assert f.vec_payload_bytes_in_kv_db() == 0


class TestFlagClearedOnAllReturnPaths:
    """``profile_value_sizes_for_page_size`` must be cleared on every
    return path so downstream re-validation never trips the mutex."""

    def _config(self) -> Any:
        from shardyfusion.config import BaseShardedWriteConfig, WriterStorageConfig

        return BaseShardedWriteConfig(
            storage=WriterStorageConfig(s3_prefix="s3://bucket/x"),
            kv=KeyValueWriteConfig(
                adapter_factory=SqliteFactory(),
                profile_value_sizes_for_page_size=True,
            ),
        )

    def test_empty_samples(self) -> None:
        from shardyfusion.writer._engine_page_size import maybe_apply_engine_page_size

        cfg = self._config()
        maybe_apply_engine_page_size(cfg, value_byte_samples=[], writer_kind="t")
        assert cfg.kv.profile_value_sizes_for_page_size is False

    def test_factory_has_no_page_size(self) -> None:
        from shardyfusion.config import BaseShardedWriteConfig, WriterStorageConfig
        from shardyfusion.writer._engine_page_size import maybe_apply_engine_page_size

        class _FactoryWithoutPageSize:
            pass

        cfg = BaseShardedWriteConfig(
            storage=WriterStorageConfig(s3_prefix="s3://bucket/x"),
            kv=KeyValueWriteConfig(
                adapter_factory=_FactoryWithoutPageSize(),  # type: ignore[arg-type]
                profile_value_sizes_for_page_size=True,
            ),
        )
        maybe_apply_engine_page_size(cfg, value_byte_samples=[100], writer_kind="t")
        assert cfg.kv.profile_value_sizes_for_page_size is False

    def test_current_equals_target(self, caplog: Any) -> None:
        from shardyfusion.writer._engine_page_size import maybe_apply_engine_page_size

        cfg = self._config()
        # samples produce target=4096 (small p95), factory is 4096 default
        with caplog.at_level("DEBUG"):
            maybe_apply_engine_page_size(
                cfg, value_byte_samples=[50] * 100, writer_kind="t"
            )
        assert cfg.kv.profile_value_sizes_for_page_size is False
        assert cfg.kv.adapter_factory.page_size == 4096
        # The no-change branch must still emit ``engine_page_size_picked``
        # so operators can distinguish "picker accepted existing size"
        # from "picker never ran".
        assert any(
            "engine_page_size_picked" in record.message for record in caplog.records
        )

    def test_flag_preserved_when_replace_raises(self) -> None:
        """If dataclasses.replace fails (factory is not a dataclass), the
        helper raises and MUST leave the flag set so a retry path with a
        dataclass-friendly factory can still profile."""
        from shardyfusion.config import BaseShardedWriteConfig, WriterStorageConfig
        from shardyfusion.errors import ConfigValidationError
        from shardyfusion.writer._engine_page_size import maybe_apply_engine_page_size

        class _NonDataclassFactoryWithPageSize:
            # has a page_size attribute but is not a dataclass — replace fails
            page_size = 4096

        cfg = BaseShardedWriteConfig(
            storage=WriterStorageConfig(s3_prefix="s3://bucket/x"),
            kv=KeyValueWriteConfig(
                adapter_factory=_NonDataclassFactoryWithPageSize(),  # type: ignore[arg-type]
                profile_value_sizes_for_page_size=True,
            ),
        )
        with pytest.raises(ConfigValidationError):
            maybe_apply_engine_page_size(
                cfg, value_byte_samples=[8000] * 100, writer_kind="t"
            )
        # Flag survives the raise.
        assert cfg.kv.profile_value_sizes_for_page_size is True


class TestMultiCellSizing:
    """``recommend_page_size_for_cells`` returns the max across cells —
    so unified workloads with small kv + medium embedding pick a page
    that fits the larger of the two cell shapes, not their over-budgeted
    union."""

    def test_picks_max_across_two_cells(self) -> None:
        from shardyfusion.sqlite_page_size import (
            VEC_INDEX_ROWID_MAX_BYTES,
            CellShape,
            recommend_page_size,
            recommend_page_size_for_cells,
        )

        # Small kv cell, larger vec cell.  Independent recommendations:
        # kv: 100 + 64 + 12 = 176 → 4096.
        # vec: 1500 + 9 + 12 = 1521 → 8192 (threshold 2030).
        # max = 8192.
        cells = [
            CellShape(payload_bytes=100, max_key_bytes=64),
            CellShape(payload_bytes=1500, max_key_bytes=VEC_INDEX_ROWID_MAX_BYTES),
        ]
        assert recommend_page_size_for_cells(cells) == 8192
        # Independent recommendations confirm:
        assert recommend_page_size(p95_value_bytes=100, max_key_bytes=64) == 4096
        assert (
            recommend_page_size(
                p95_value_bytes=1500, max_key_bytes=VEC_INDEX_ROWID_MAX_BYTES
            )
            == 8192
        )

    def test_empty_cells_raises(self) -> None:
        from shardyfusion.sqlite_page_size import recommend_page_size_for_cells

        with pytest.raises(ConfigValidationError):
            recommend_page_size_for_cells([])


class TestCollectValueByteSamplesHardening:
    """Type-checks ``value_spec.encode`` and warns when every sample
    encodes to a zero-length value (formerly silently no-op)."""

    def test_str_encoder_raises(self) -> None:
        from shardyfusion.writer._engine_page_size import collect_value_byte_samples

        class _BadEncoder:
            def encode(self, row: Any) -> Any:
                return "not bytes"

        with pytest.raises(TypeError, match="bytes-like"):
            collect_value_byte_samples(rows=[1, 2], value_spec=_BadEncoder())

    def test_all_empty_bytes_does_not_warn(self, caplog: Any) -> None:
        """Workloads that legitimately produce ``b""`` for every value
        (e.g. presence-only KV) must NOT trigger the "all samples failed"
        WARNING — zero bytes is itself a valid signal that lets the
        picker correctly pick the smallest page size."""
        from shardyfusion.writer._engine_page_size import collect_value_byte_samples

        class _EmptyEncoder:
            def encode(self, row: Any) -> bytes:
                return b""

        with caplog.at_level("WARNING"):
            sizes = collect_value_byte_samples(
                rows=[1, 2, 3], value_spec=_EmptyEncoder()
            )
        assert sizes == [0, 0, 0]
        assert not any(
            "engine_page_size_all_samples_failed" in record.message
            for record in caplog.records
        )

    def test_all_encode_failures_warn(self, caplog: Any) -> None:
        """When every row's encoder RAISES, the WARNING fires — the
        picker truly has no signal."""
        from shardyfusion.writer._engine_page_size import collect_value_byte_samples

        class _RaisingEncoder:
            def encode(self, row: Any) -> bytes:
                raise RuntimeError("encoder broken")

        with caplog.at_level("WARNING"):
            sizes = collect_value_byte_samples(
                rows=[1, 2, 3], value_spec=_RaisingEncoder()
            )
        assert sizes == []
        assert any(
            "engine_page_size_all_samples_failed" in record.message
            for record in caplog.records
        )

    def test_normal_path_no_warning(self, caplog: Any) -> None:
        from shardyfusion.writer._engine_page_size import collect_value_byte_samples

        class _OkEncoder:
            def encode(self, row: Any) -> bytes:
                return b"x" * 100

        with caplog.at_level("WARNING"):
            sizes = collect_value_byte_samples(rows=[1, 2], value_spec=_OkEncoder())
        assert sizes == [100, 100]
        assert not any(
            "engine_page_size_all_samples_failed" in record.message
            for record in caplog.records
        )


class TestSqliteVecAdapterDimCoercion:
    """``self._vec_dim`` must be a plain int even when ``vector_spec.dim``
    is a numpy scalar, so downstream log_event JSON serialisation does
    not choke on np.int64."""

    def test_numpy_int64_dim_coerced(self, tmp_path: Any) -> None:
        np = pytest.importorskip("numpy")
        pytest.importorskip("sqlite_vec")
        from shardyfusion.config import VectorSpec
        from shardyfusion.sqlite_vec_adapter import SqliteVecAdapter

        spec = VectorSpec(dim=int(np.int64(128)), metric="cosine")
        # Bypass VectorSpec construction-time validation by injecting
        # the numpy scalar after the fact.
        object.__setattr__(spec, "dim", np.int64(128))
        local_dir = tmp_path / "shard"
        adapter = SqliteVecAdapter(
            db_url="s3://b/x", local_dir=local_dir, vector_spec=spec
        )
        try:
            assert type(adapter._vec_dim) is int
            assert adapter._vec_dim == 128
        finally:
            # Avoid background uploads in unit tests.
            adapter._uploaded = True
            adapter.close()


class TestNearestRankPercentile:
    """Verify the picker uses true nearest-rank percentile, not a
    floor-based off-by-one.  Sample sizes where ``0.95 * N`` is non-
    integer trigger the regression (e.g. 21, 41, 73, 199)."""

    def test_engine_picker_uses_ceil_for_odd_sample_sizes(self) -> None:
        from shardyfusion.config import BaseShardedWriteConfig, WriterStorageConfig
        from shardyfusion.writer._engine_page_size import maybe_apply_engine_page_size

        config = BaseShardedWriteConfig(
            storage=WriterStorageConfig(s3_prefix="s3://bucket/x"),
            kv=KeyValueWriteConfig(
                adapter_factory=SqliteFactory(),
                profile_value_sizes_for_page_size=True,
            ),
        )
        # 21 sorted values: 20 small + 1 large at rank 21.  Nearest-rank
        # p95 = ceil(0.95 * 21) = rank 20 (1-indexed), value 100 → 4096.
        # Buggy floor formula would have picked rank 19, value 100 too,
        # so the difference is only visible at the boundary where the
        # largest values straddle a page-size threshold.
        samples = [100] * 19 + [1500] + [4000]
        maybe_apply_engine_page_size(
            config, value_byte_samples=samples, writer_kind="test"
        )
        # Nearest-rank p95 of N=21 picks the 20th smallest = 1500 →
        # picker bumps to 8192.  Floor formula would have picked rank
        # 19 (still 100) → stayed at 4096.
        assert config.kv.adapter_factory.page_size == 8192
