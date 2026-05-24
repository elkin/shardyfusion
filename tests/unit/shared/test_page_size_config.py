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
