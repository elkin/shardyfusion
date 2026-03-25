"""Tests for the Ray single-database writer."""

from __future__ import annotations

import pathlib
from datetime import timedelta

import pytest
import ray
import ray.data

from shardyfusion.config import (
    ManifestOptions,
    OutputOptions,
    WriteConfig,
)
from shardyfusion.errors import (
    ConfigValidationError,
    ShardyfusionError,
)
from shardyfusion.manifest import BuildResult
from shardyfusion.manifest_store import InMemoryManifestStore
from shardyfusion.run_registry import InMemoryRunRegistry
from shardyfusion.serde import ValueSpec, make_key_encoder
from shardyfusion.sharding_types import KeyEncoding
from shardyfusion.testing import (
    file_backed_adapter_factory,
    file_backed_load_db,
)
from shardyfusion.type_defs import RetryConfig
from shardyfusion.writer.ray.single_db_writer import (
    RayCacheContext,
    write_single_db,
)
from tests.helpers.run_record_assertions import (
    assert_success_run_record,
    load_in_memory_run_record,
)
from tests.helpers.tracking import (
    RecordingTokenBucket,
    TrackingAdapter,
    TrackingFactory,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    factory: TrackingFactory | None = None,
    batch_size: int = 50_000,
    key_encoding: KeyEncoding = KeyEncoding.U64BE,
    num_dbs: int = 1,
    run_registry: InMemoryRunRegistry | None = None,
) -> WriteConfig:
    return WriteConfig(
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_encoding=key_encoding,
        batch_size=batch_size,
        adapter_factory=factory or TrackingFactory(),
        output=OutputOptions(run_id="test-run"),
        manifest=ManifestOptions(store=InMemoryManifestStore()),
        run_registry=run_registry,
    )


def _make_ray_ds(
    records: list[dict[str, object]], parallelism: int = 2
) -> ray.data.Dataset:
    if not records:
        return ray.data.from_items([{"id": 0}]).filter(lambda r: False)
    return ray.data.from_items(records, override_num_blocks=parallelism)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_sorted_write() -> None:
    factory = TrackingFactory()
    config = _make_config(factory=factory, batch_size=100)

    # Unsorted keys
    records = [{"id": k} for k in [5, 3, 1, 4, 2]]
    ds = _make_ray_ds(records, parallelism=1)

    result = write_single_db(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert isinstance(result, BuildResult)
    assert len(result.winners) == 1
    assert result.winners[0].db_id == 0
    assert result.winners[0].row_count == 5
    assert result.stats.rows_written == 5

    # Verify keys are written in sorted order
    adapter = factory.adapters[0]
    all_keys = [pair[0] for call in adapter.write_calls for pair in call]
    assert all_keys == sorted(all_keys)


def test_basic_sorted_write_records_succeeded_run_record() -> None:
    registry = InMemoryRunRegistry()
    config = _make_config(batch_size=100, run_registry=registry)
    ds = _make_ray_ds([{"id": k} for k in [5, 3, 1, 4, 2]], parallelism=1)

    result = write_single_db(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    run_record = load_in_memory_run_record(registry, result)
    assert_success_run_record(
        run_record,
        result=result,
        writer_type="ray",
        s3_prefix=config.s3_prefix,
    )


def test_sort_keys_false() -> None:
    factory = TrackingFactory()
    config = _make_config(factory=factory, batch_size=100)

    records = [{"id": k} for k in [5, 3, 1, 4, 2]]
    ds = _make_ray_ds(records, parallelism=1)

    result = write_single_db(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        sort_keys=False,
    )

    assert result.winners[0].row_count == 5


def test_validates_num_dbs_1() -> None:
    config = _make_config(num_dbs=2)

    records = [{"id": 1}]
    ds = _make_ray_ds(records, parallelism=1)

    with pytest.raises(ConfigValidationError, match="num_dbs=1"):
        write_single_db(
            ds,
            config,
            key_col="id",
            value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        )


def test_batch_size_controls_write_calls() -> None:
    factory = TrackingFactory()
    config = _make_config(factory=factory, batch_size=2)

    records = [{"id": k} for k in range(7)]
    ds = _make_ray_ds(records, parallelism=1)

    write_single_db(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    adapter = factory.adapters[0]
    total_pairs = sum(len(call) for call in adapter.write_calls)
    assert total_pairs == 7
    # With batch_size=2 and 7 rows: at least 3 full batches + 1 partial
    assert len(adapter.write_calls) >= 4


def test_rate_limiting() -> None:
    factory = TrackingFactory()
    config = _make_config(factory=factory, batch_size=2)

    records = [{"id": k} for k in range(5)]
    ds = _make_ray_ds(records, parallelism=1)

    result = write_single_db(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        max_writes_per_second=1000.0,
    )

    assert result.stats.rows_written == 5


# ---------------------------------------------------------------------------
# Rate-limiter integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def _patch_token_bucket(monkeypatch: pytest.MonkeyPatch) -> list[RecordingTokenBucket]:
    RecordingTokenBucket.instances = []
    monkeypatch.setattr(
        "shardyfusion.writer.ray.single_db_writer.TokenBucket",
        RecordingTokenBucket,
    )
    return RecordingTokenBucket.instances


def test_rate_limiter_bucket_created_with_correct_rate(
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    config = _make_config(batch_size=50_000)
    records = [{"id": i} for i in range(5)]
    ds = _make_ray_ds(records, parallelism=1)

    write_single_db(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        max_writes_per_second=42.5,
    )

    assert len(_patch_token_bucket) == 1
    assert _patch_token_bucket[0].rate == 42.5


def test_rate_limiter_no_bucket_when_rate_is_none(
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    config = _make_config(batch_size=50_000)
    records = [{"id": i} for i in range(5)]
    ds = _make_ray_ds(records, parallelism=1)

    write_single_db(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert len(_patch_token_bucket) == 0


def test_bytes_rate_limiter_bucket_created_with_correct_rate(
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    config = _make_config(batch_size=50_000)
    records = [{"id": i} for i in range(5)]
    ds = _make_ray_ds(records, parallelism=1)

    write_single_db(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        max_write_bytes_per_second=512.0,
    )

    assert len(_patch_token_bucket) == 1
    assert _patch_token_bucket[0].rate == 512.0


def test_bytes_rate_limiter_no_bucket_when_none(
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    config = _make_config(batch_size=50_000)
    records = [{"id": i} for i in range(5)]
    ds = _make_ray_ds(records, parallelism=1)

    write_single_db(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert len(_patch_token_bucket) == 0


def test_bytes_rate_limiter_acquire_called_with_byte_counts(
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    # 7 rows / batch_size 3 → 2 full batches + 1 trailing partial (1 row)
    # value is b"val" (3 bytes), key is 8 bytes (u64be)
    config = _make_config(batch_size=3)
    records = [{"id": i} for i in range(7)]
    ds = _make_ray_ds(records, parallelism=1)

    write_single_db(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"val"),
        max_write_bytes_per_second=10_000.0,
    )

    assert len(_patch_token_bucket) == 1
    bucket = _patch_token_bucket[0]
    # Each pair is 8 (u64be key) + 3 (b"val") = 11 bytes
    # Batches: [3*11, 3*11, 1*11] = [33, 33, 11]
    assert sum(bucket.acquire_calls) == 7 * 11
    assert len(bucket.acquire_calls) == 3


def test_empty_dataset() -> None:
    factory = TrackingFactory()
    config = _make_config(factory=factory)

    ds = _make_ray_ds([], parallelism=1)

    result = write_single_db(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert result.winners[0].row_count == 0
    assert result.winners[0].min_key is None
    assert result.winners[0].max_key is None


def test_min_max_keys() -> None:
    factory = TrackingFactory()
    config = _make_config(factory=factory)

    records = [{"id": k} for k in [50, 10, 90, 30]]
    ds = _make_ray_ds(records, parallelism=1)

    result = write_single_db(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert result.winners[0].min_key == 10
    assert result.winners[0].max_key == 90


def test_manifest_structure() -> None:
    store = InMemoryManifestStore()
    config = WriteConfig(
        num_dbs=1,
        s3_prefix="s3://bucket/prefix",
        adapter_factory=TrackingFactory(),
        output=OutputOptions(run_id="test-manifest"),
        manifest=ManifestOptions(store=store),
    )

    records = [{"id": k} for k in range(5)]
    ds = _make_ray_ds(records, parallelism=1)

    result = write_single_db(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert result.manifest_ref.startswith("mem://manifests/")

    # Parse manifest content via store
    parsed = store.load_manifest(result.manifest_ref)
    assert parsed.required_build.num_dbs == 1
    assert len(parsed.shards) == 1
    assert parsed.shards[0].db_id == 0


def test_key_encoding_u32be() -> None:
    factory = TrackingFactory()
    config = _make_config(factory=factory, key_encoding=KeyEncoding.U32BE)

    records = [{"id": 1}, {"id": 256}]
    ds = _make_ray_ds(records, parallelism=1)

    write_single_db(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    adapter = factory.adapters[0]
    all_keys = [pair[0] for call in adapter.write_calls for pair in call]
    assert all(len(k) == 4 for k in all_keys)
    assert all_keys[0] == b"\x00\x00\x00\x01"
    assert all_keys[1] == b"\x00\x00\x01\x00"


def test_ray_cache_context_enabled_false() -> None:
    records = [{"id": i} for i in range(3)]
    ds = ray.data.from_items(records)

    with RayCacheContext(ds, enabled=False) as result_ds:
        assert result_ds is ds


def test_ray_cache_context_enabled_true() -> None:
    records = [{"id": i} for i in range(3)]
    ds = ray.data.from_items(records)

    with RayCacheContext(ds, enabled=True) as result_ds:
        # Materialized datasets are a different object
        assert result_ds is not ds


def test_checkpoint_id_in_result() -> None:
    factory = TrackingFactory()
    config = _make_config(factory=factory)

    records = [{"id": 1}]
    ds = _make_ray_ds(records, parallelism=1)

    result = write_single_db(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert result.winners[0].checkpoint_id == "fake-checkpoint"


def test_shard_duration_is_zero() -> None:
    """Single-db mode has no sharding phase; shard_duration_ms must be 0."""
    factory = TrackingFactory()
    config = _make_config(factory=factory)

    records = [{"id": k} for k in range(5)]
    ds = _make_ray_ds(records, parallelism=1)

    result = write_single_db(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert result.stats.durations.sharding_ms == 0


def test_data_integrity_file_backed(tmp_path: pathlib.Path) -> None:
    root_dir = str(tmp_path / "file_backed")
    config = WriteConfig(
        num_dbs=1,
        s3_prefix="s3://bucket/prefix",
        adapter_factory=file_backed_adapter_factory(root_dir),
        output=OutputOptions(
            run_id="test-run",
            local_root=str(tmp_path / "local"),
        ),
        manifest=ManifestOptions(store=InMemoryManifestStore()),
    )

    records = [{"id": i} for i in range(50)]
    ds = _make_ray_ds(records, parallelism=3)

    result = write_single_db(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(
            lambda row: f"value-{row['id']}".encode()
        ),
    )

    assert result.stats.rows_written == 50

    all_kv: dict[bytes, bytes] = {}
    for winner in result.winners:
        shard_data = file_backed_load_db(root_dir, winner.db_url)
        all_kv.update(shard_data)

    assert len(all_kv) == 50
    for i in range(50):
        key_bytes = make_key_encoder(config.key_encoding)(i)
        assert all_kv[key_bytes] == f"value-{i}".encode()


# ---------------------------------------------------------------------------
# Error-wrapping tests
# ---------------------------------------------------------------------------


class _FailingAdapter(TrackingAdapter):
    """Adapter that raises on write_batch."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self._error = error

    def write_batch(self, pairs):  # type: ignore[override]
        raise self._error


class _FailingFactory:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def __call__(self, *, db_url: str, local_dir: str) -> _FailingAdapter:
        return _FailingAdapter(self._error)


class _FailOnceAdapter(TrackingAdapter):
    def __init__(self, *, error: Exception | None = None) -> None:
        super().__init__()
        self._error = error

    def write_batch(self, pairs):  # type: ignore[override]
        if self._error is not None:
            raise self._error
        super().write_batch(pairs)


class _FailOnceFactory:
    def __init__(self) -> None:
        self.urls: list[str] = []
        self.calls = 0

    def __call__(self, *, db_url: str, local_dir: pathlib.Path) -> _FailOnceAdapter:
        self.urls.append(db_url)
        adapter = _FailOnceAdapter(
            error=RuntimeError("boom on first single-db attempt")
            if self.calls == 0
            else None
        )
        self.calls += 1
        return adapter


def test_single_db_retry_uses_new_attempt_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "shardyfusion.writer.ray.single_db_writer.cleanup_losers",
        lambda *args, **kwargs: 0,
    )
    factory = _FailOnceFactory()
    config = _make_config(factory=factory, batch_size=2)
    config.shard_retry = RetryConfig(
        max_retries=1,
        initial_backoff=timedelta(seconds=0),
    )

    records = [{"id": k} for k in range(5)]
    ds = _make_ray_ds(records, parallelism=1)

    result = write_single_db(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert result.winners[0].attempt == 1
    assert result.winners[0].db_url is not None
    assert result.winners[0].db_url.endswith("attempt=01")
    assert factory.urls == [
        "s3://bucket/prefix/shards/run_id=test-run/db=00000/attempt=00",
        "s3://bucket/prefix/shards/run_id=test-run/db=00000/attempt=01",
    ]


def test_single_db_retry_records_succeeded_run_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "shardyfusion.writer.ray.single_db_writer.cleanup_losers",
        lambda *args, **kwargs: 0,
    )
    registry = InMemoryRunRegistry()
    factory = _FailOnceFactory()
    config = _make_config(factory=factory, batch_size=2, run_registry=registry)
    config.shard_retry = RetryConfig(
        max_retries=1,
        initial_backoff=timedelta(seconds=0),
    )

    ds = _make_ray_ds([{"id": k} for k in range(5)], parallelism=1)
    result = write_single_db(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert result.winners[0].attempt == 1
    run_record = load_in_memory_run_record(registry, result)
    assert_success_run_record(
        run_record,
        result=result,
        writer_type="ray",
        s3_prefix=config.s3_prefix,
    )


def test_unexpected_error_wrapped_in_slatedb_error() -> None:
    """Generic exceptions from the adapter are wrapped in ShardyfusionError."""
    config = _make_config(factory=_FailingFactory(RuntimeError("boom")))  # type: ignore[arg-type]

    records = [{"id": 1}]
    ds = _make_ray_ds(records, parallelism=1)

    with pytest.raises(ShardyfusionError, match="boom") as exc_info:
        write_single_db(
            ds,
            config,
            key_col="id",
            value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_slatedb_error_passes_through() -> None:
    """ShardyfusionError from the adapter is not double-wrapped."""
    original = ShardyfusionError("original error")
    config = _make_config(factory=_FailingFactory(original))  # type: ignore[arg-type]

    records = [{"id": 1}]
    ds = _make_ray_ds(records, parallelism=1)

    with pytest.raises(ShardyfusionError, match="original error") as exc_info:
        write_single_db(
            ds,
            config,
            key_col="id",
            value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        )

    assert exc_info.value is original
