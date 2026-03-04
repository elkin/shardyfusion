"""Tests for the Ray single-database writer."""

from __future__ import annotations

import json
import pathlib

import pytest

ray_data = pytest.importorskip("ray.data")
import ray  # noqa: E402

from shardyfusion.config import (  # noqa: E402
    ManifestOptions,
    OutputOptions,
    WriteConfig,
)
from shardyfusion.errors import (  # noqa: E402
    ConfigValidationError,
    ShardyfusionError,
)
from shardyfusion.manifest import BuildResult  # noqa: E402
from shardyfusion.serde import ValueSpec, make_key_encoder  # noqa: E402
from shardyfusion.sharding_types import KeyEncoding  # noqa: E402
from shardyfusion.testing import (  # noqa: E402
    file_backed_adapter_factory,
    file_backed_load_db,
)
from shardyfusion.writer.ray.single_db_writer import (  # noqa: E402
    RayCacheContext,
    write_single_db,
)
from tests.helpers.tracking import (  # noqa: E402
    InMemoryPublisher,
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
) -> WriteConfig:
    return WriteConfig(
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_encoding=key_encoding,
        batch_size=batch_size,
        adapter_factory=factory or TrackingFactory(),
        output=OutputOptions(run_id="test-run"),
        manifest=ManifestOptions(publisher=InMemoryPublisher()),
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
    publisher = InMemoryPublisher()
    config = WriteConfig(
        num_dbs=1,
        s3_prefix="s3://bucket/prefix",
        adapter_factory=TrackingFactory(),
        output=OutputOptions(run_id="test-manifest"),
        manifest=ManifestOptions(publisher=publisher),
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
    assert result.current_ref == "mem://_CURRENT"

    manifest_artifact = publisher.objects[result.manifest_ref]
    manifest_data = json.loads(manifest_artifact.payload.decode("utf-8"))
    assert manifest_data["required"]["num_dbs"] == 1
    assert len(manifest_data["shards"]) == 1
    assert manifest_data["shards"][0]["db_id"] == 0


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
        manifest=ManifestOptions(publisher=InMemoryPublisher()),
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
