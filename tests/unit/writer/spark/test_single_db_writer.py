"""Tests for the Spark single-database writer."""

from __future__ import annotations

import json

import pytest
from pyspark.sql import SparkSession

from slatedb_spark_sharded.config import (
    ManifestOptions,
    OutputOptions,
    WriteConfig,
)
from slatedb_spark_sharded.errors import ConfigValidationError, SlatedbSparkShardedError
from slatedb_spark_sharded.manifest import BuildResult
from slatedb_spark_sharded.serde import ValueSpec
from slatedb_spark_sharded.sharding_types import KeyEncoding
from slatedb_spark_sharded.writer.spark.single_db_writer import write_single_db
from slatedb_spark_sharded.writer.spark.writer import DataFrameCacheContext
from tests.helpers.tracking import (
    InMemoryPublisher,
    RecordingTokenBucket,
    TrackingAdapter,
    TrackingFactory,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    return (
        SparkSession.builder.master("local[1]")
        .appName("test_single_db_writer")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.spark
def test_basic_sorted_write(spark: SparkSession) -> None:
    factory = TrackingFactory()
    config = _make_config(factory=factory, batch_size=100)

    # Unsorted keys
    df = spark.createDataFrame([(k, f"v{k}") for k in [5, 3, 1, 4, 2]], ["key", "val"])

    result = write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.binary_col("val"),
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


@pytest.mark.spark
def test_sorted_write_multiple_partitions(spark: SparkSession) -> None:
    """Regression: sort order must be preserved when coalesce produces >1 partition.

    Previously repartition() was used, which destroys global order via hash shuffle.
    coalesce() merges adjacent partitions without shuffling, preserving sorted order.
    """
    factory = TrackingFactory()
    # batch_size=2 with 20 rows → ceil(20/2)=10 partitions
    config = _make_config(factory=factory, batch_size=2)

    df = spark.createDataFrame(
        [
            (k, f"v{k}")
            for k in [
                19,
                7,
                13,
                2,
                18,
                5,
                11,
                1,
                15,
                9,
                17,
                3,
                14,
                6,
                12,
                0,
                16,
                8,
                10,
                4,
            ]
        ],
        ["key", "val"],
    )

    result = write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.binary_col("val"),
    )

    assert result.winners[0].row_count == 20

    # Verify ALL keys arrive in globally sorted order across all batches
    adapter = factory.adapters[0]
    all_keys = [pair[0] for call in adapter.write_calls for pair in call]
    assert all_keys == sorted(all_keys), (
        f"Keys not globally sorted across {len(adapter.write_calls)} batches"
    )


@pytest.mark.spark
def test_sort_keys_false(spark: SparkSession) -> None:
    factory = TrackingFactory()
    config = _make_config(factory=factory, batch_size=100)

    df = spark.createDataFrame([(k,) for k in [5, 3, 1, 4, 2]], ["key"])

    result = write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        sort_keys=False,
    )

    assert result.winners[0].row_count == 5


@pytest.mark.spark
def test_validates_num_dbs_1(spark: SparkSession) -> None:
    config = _make_config(num_dbs=2)
    df = spark.createDataFrame([(1,)], ["key"])

    with pytest.raises(ConfigValidationError, match="num_dbs=1"):
        write_single_db(
            df,
            config,
            key_col="key",
            value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        )


@pytest.mark.spark
def test_partition_sizing(spark: SparkSession) -> None:
    factory = TrackingFactory()
    # 10 rows, batch_size=3 → ceil(10/3)=4 partitions
    config = _make_config(factory=factory, batch_size=3)

    df = spark.createDataFrame([(k,) for k in range(10)], ["key"])

    result = write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert result.winners[0].row_count == 10


@pytest.mark.spark
def test_batch_size_controls_write_calls(spark: SparkSession) -> None:
    factory = TrackingFactory()
    config = _make_config(factory=factory, batch_size=2)

    df = spark.createDataFrame([(k,) for k in range(7)], ["key"])

    write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    adapter = factory.adapters[0]
    total_pairs = sum(len(call) for call in adapter.write_calls)
    assert total_pairs == 7
    # With batch_size=2 and 7 rows: at least 3 full batches + 1 partial
    assert len(adapter.write_calls) >= 4


@pytest.mark.spark
def test_rate_limiting(spark: SparkSession) -> None:
    factory = TrackingFactory()
    config = _make_config(factory=factory, batch_size=2)

    df = spark.createDataFrame([(k,) for k in range(5)], ["key"])

    result = write_single_db(
        df,
        config,
        key_col="key",
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
        "slatedb_spark_sharded.writer.spark.single_db_writer.TokenBucket",
        RecordingTokenBucket,
    )
    return RecordingTokenBucket.instances


@pytest.mark.spark
def test_rate_limiter_bucket_created_with_correct_rate(
    spark: SparkSession,
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    config = _make_config(batch_size=50_000)
    df = spark.createDataFrame([(k,) for k in range(5)], ["key"])

    write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        max_writes_per_second=42.5,
    )

    assert len(_patch_token_bucket) == 1
    assert _patch_token_bucket[0].rate == 42.5


@pytest.mark.spark
def test_rate_limiter_no_bucket_when_rate_is_none(
    spark: SparkSession,
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    config = _make_config(batch_size=50_000)
    df = spark.createDataFrame([(k,) for k in range(5)], ["key"])

    write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert len(_patch_token_bucket) == 0


@pytest.mark.spark
def test_rate_limiter_acquire_count_matches_batch_writes(
    spark: SparkSession,
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    config = _make_config(batch_size=3)
    df = spark.createDataFrame([(k,) for k in range(7)], ["key"])

    write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        max_writes_per_second=100.0,
    )

    assert len(_patch_token_bucket) == 1
    bucket = _patch_token_bucket[0]
    # 7 rows / batch_size 3 → batches of [3, 3, 1] → 3 acquire calls
    assert len(bucket.acquire_calls) == 3
    assert bucket.acquire_calls == [3, 3, 1]
    assert sum(bucket.acquire_calls) == 7


@pytest.mark.spark
def test_rate_limiter_single_batch_single_acquire(
    spark: SparkSession,
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    config = _make_config(batch_size=50_000)
    df = spark.createDataFrame([(k,) for k in range(5)], ["key"])

    write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        max_writes_per_second=100.0,
    )

    assert len(_patch_token_bucket) == 1
    bucket = _patch_token_bucket[0]
    # All 5 rows fit in one batch → single acquire for all 5
    assert bucket.acquire_calls == [5]


@pytest.mark.spark
def test_rate_limiter_exact_batch_boundary(
    spark: SparkSession,
    _patch_token_bucket: list[RecordingTokenBucket],
) -> None:
    config = _make_config(batch_size=3)
    df = spark.createDataFrame([(k,) for k in range(6)], ["key"])

    write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        max_writes_per_second=100.0,
    )

    assert len(_patch_token_bucket) == 1
    bucket = _patch_token_bucket[0]
    # 6 rows / batch_size 3 → exactly 2 full batches, no trailing partial
    assert bucket.acquire_calls == [3, 3]


@pytest.mark.spark
def test_empty_dataframe(spark: SparkSession) -> None:
    factory = TrackingFactory()
    config = _make_config(factory=factory)

    df = spark.createDataFrame([], "key: long")

    result = write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert result.winners[0].row_count == 0
    assert result.winners[0].min_key is None
    assert result.winners[0].max_key is None


@pytest.mark.spark
def test_min_max_keys(spark: SparkSession) -> None:
    factory = TrackingFactory()
    config = _make_config(factory=factory)

    df = spark.createDataFrame([(k,) for k in [50, 10, 90, 30]], ["key"])

    result = write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert result.winners[0].min_key == 10
    assert result.winners[0].max_key == 90


@pytest.mark.spark
def test_manifest_structure(spark: SparkSession) -> None:
    publisher = InMemoryPublisher()
    config = WriteConfig(
        num_dbs=1,
        s3_prefix="s3://bucket/prefix",
        adapter_factory=TrackingFactory(),
        output=OutputOptions(run_id="test-manifest"),
        manifest=ManifestOptions(publisher=publisher),
    )

    df = spark.createDataFrame([(k,) for k in range(5)], ["key"])

    result = write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert result.manifest_ref.startswith("mem://manifests/")
    assert result.current_ref == "mem://_CURRENT"

    # Parse manifest content
    manifest_artifact = publisher.objects[result.manifest_ref]
    manifest_data = json.loads(manifest_artifact.payload.decode("utf-8"))
    assert manifest_data["required"]["num_dbs"] == 1
    assert len(manifest_data["shards"]) == 1
    assert manifest_data["shards"][0]["db_id"] == 0


@pytest.mark.spark
def test_key_encoding_u64be(spark: SparkSession) -> None:
    factory = TrackingFactory()
    config = _make_config(factory=factory, key_encoding=KeyEncoding.U64BE)

    df = spark.createDataFrame([(1,), (256,)], ["key"])

    write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    adapter = factory.adapters[0]
    all_keys = [pair[0] for call in adapter.write_calls for pair in call]
    assert all(len(k) == 8 for k in all_keys)


@pytest.mark.spark
def test_key_encoding_u32be(spark: SparkSession) -> None:
    factory = TrackingFactory()
    config = _make_config(factory=factory, key_encoding=KeyEncoding.U32BE)

    df = spark.createDataFrame([(1,), (256,)], ["key"])

    write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    adapter = factory.adapters[0]
    all_keys = [pair[0] for call in adapter.write_calls for pair in call]
    assert all(len(k) == 4 for k in all_keys)
    assert all_keys[0] == b"\x00\x00\x00\x01"
    assert all_keys[1] == b"\x00\x00\x01\x00"


@pytest.mark.spark
def test_cache_input_false(spark: SparkSession) -> None:
    factory = TrackingFactory()
    config = _make_config(factory=factory)

    df = spark.createDataFrame([(k,) for k in range(5)], ["key"])

    result = write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        cache_input=False,
    )

    assert result.stats.rows_written == 5


@pytest.mark.spark
def test_dataframe_cache_context_enabled_false(spark: SparkSession) -> None:
    df = spark.createDataFrame([(1,)], ["key"])
    with DataFrameCacheContext(df, enabled=False) as result_df:
        # Should return original DataFrame without persist
        assert result_df is df


@pytest.mark.spark
def test_dataframe_cache_context_enabled_true(spark: SparkSession) -> None:
    df = spark.createDataFrame([(1,)], ["key"])
    with DataFrameCacheContext(df, enabled=True) as result_df:
        # Should return a cached DataFrame (storageLevel should be set)
        assert result_df.storageLevel.useMemory or result_df.storageLevel.useDisk


@pytest.mark.spark
def test_prefetch_disabled(spark: SparkSession) -> None:
    factory = TrackingFactory()
    config = _make_config(factory=factory)

    df = spark.createDataFrame([(k,) for k in range(5)], ["key"])

    result = write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        prefetch_partitions=False,
    )

    assert result.stats.rows_written == 5


@pytest.mark.spark
def test_checkpoint_id_in_result(spark: SparkSession) -> None:
    factory = TrackingFactory()
    config = _make_config(factory=factory)

    df = spark.createDataFrame([(1,)], ["key"])

    result = write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert result.winners[0].checkpoint_id == "fake-checkpoint"


@pytest.mark.spark
def test_explicit_num_partitions_skips_count(spark: SparkSession) -> None:
    """When num_partitions is provided, df.count() is not called."""
    factory = TrackingFactory()
    config = _make_config(factory=factory, batch_size=2)

    df = spark.createDataFrame([(k,) for k in range(10)], ["key"])

    result = write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        num_partitions=3,
    )

    assert result.winners[0].row_count == 10


@pytest.mark.spark
def test_shard_duration_is_zero(spark: SparkSession) -> None:
    """Single-db mode has no sharding phase; shard_duration_ms must be 0."""
    factory = TrackingFactory()
    config = _make_config(factory=factory)

    df = spark.createDataFrame([(k,) for k in range(5)], ["key"])

    result = write_single_db(
        df,
        config,
        key_col="key",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert result.stats.durations.sharding_ms == 0


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


@pytest.mark.spark
def test_unexpected_error_wrapped_in_slatedb_error(spark: SparkSession) -> None:
    """Generic exceptions from the adapter are wrapped in SlatedbSparkShardedError."""
    config = _make_config(factory=_FailingFactory(RuntimeError("boom")))  # type: ignore[arg-type]
    df = spark.createDataFrame([(1,)], ["key"])

    with pytest.raises(SlatedbSparkShardedError, match="boom") as exc_info:
        write_single_db(
            df,
            config,
            key_col="key",
            value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.spark
def test_slatedb_error_passes_through(spark: SparkSession) -> None:
    """SlatedbSparkShardedError from the adapter is not double-wrapped."""
    original = SlatedbSparkShardedError("original error")
    config = _make_config(factory=_FailingFactory(original))  # type: ignore[arg-type]
    df = spark.createDataFrame([(1,)], ["key"])

    with pytest.raises(SlatedbSparkShardedError, match="original error") as exc_info:
        write_single_db(
            df,
            config,
            key_col="key",
            value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        )

    assert exc_info.value is original
