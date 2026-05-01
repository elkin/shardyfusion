from __future__ import annotations

from types import SimpleNamespace

from shardyfusion.writer.spark.util import (
    DataFrameCacheContext,
    SparkConfOverrideContext,
)
from shardyfusion.writer.spark.writer import (
    write_sharded_by_hash,
)


class _FakeSparkConf:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = dict(values or {})

    def get(self, key: str, default=None):
        return self.values.get(key, default)

    def set(self, key: str, value: str) -> None:
        self.values[key] = value

    def unset(self, key: str) -> None:
        self.values.pop(key, None)


class _FakeSparkSession:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.conf = _FakeSparkConf(values)


class _FakeDataFrame:
    def __init__(self, spark_session: _FakeSparkSession) -> None:
        self.sparkSession = spark_session
        self.persist_calls: list[object | None] = []
        self.unpersist_calls: list[bool] = []

    def persist(self, storage_level=None):
        self.persist_calls.append(storage_level)
        return self

    def unpersist(self, *, blocking: bool = False):
        self.unpersist_calls.append(blocking)
        return self

    def count(self):
        return 10


def test_spark_conf_context_overrides_and_restores() -> None:
    spark = _FakeSparkSession({"spark.speculation": "true"})

    with SparkConfOverrideContext(
        spark,
        {"spark.speculation": "false", "spark.sql.shuffle.partitions": "8"},
    ):
        assert spark.conf.get("spark.speculation") == "false"
        assert spark.conf.get("spark.sql.shuffle.partitions") == "8"

    assert spark.conf.get("spark.speculation") == "true"
    assert spark.conf.get("spark.sql.shuffle.partitions") is None


def test_spark_conf_context_restores_on_exception() -> None:
    spark = _FakeSparkSession({"spark.speculation": "true"})

    try:
        with SparkConfOverrideContext(spark, {"spark.speculation": "false"}):
            assert spark.conf.get("spark.speculation") == "false"
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert spark.conf.get("spark.speculation") == "true"


def test_dataframe_cache_context_caches_and_unpersists() -> None:
    spark = _FakeSparkSession()
    df = _FakeDataFrame(spark)
    storage_level = object()

    with DataFrameCacheContext(df, storage_level=storage_level) as cached_df:
        assert cached_df is df
        assert df.persist_calls == [storage_level]

    assert df.unpersist_calls == [False]


def test_write_sharded_spark_uses_optional_spark_conf_overrides(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    fake_spark = _FakeSparkSession()
    fake_df = SimpleNamespace(sparkSession=fake_spark, count=lambda: 10)
    fake_config = SimpleNamespace(
        output=SimpleNamespace(run_id=None),
        num_dbs=4,
        max_keys_per_shard=None,
    )

    class _RecordingCtx:
        def __init__(self, spark, overrides):
            calls.append(("ctx_init", (spark, overrides)))

        def __enter__(self):
            calls.append(("ctx_enter", None))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("ctx_exit", None))

    def _fake_impl(
        *,
        df,
        config,
        run_id,
        started,
        key_col,
        value_spec,
        sort_within_partitions,
        max_writes_per_second,
        max_write_bytes_per_second,
        verify_routing,
        vector_fn,
        vector_columns,
    ):
        _ = (
            started,
            key_col,
            value_spec,
            sort_within_partitions,
            max_writes_per_second,
            max_write_bytes_per_second,
            verify_routing,
            vector_fn,
            vector_columns,
        )
        calls.append(("impl", (df, config, run_id)))
        return "result-sentinel"

    monkeypatch.setattr(
        "shardyfusion.writer.spark.writer.SparkConfOverrideContext",
        _RecordingCtx,
    )
    monkeypatch.setattr(
        "shardyfusion.writer.spark.writer._write_hash_sharded",
        _fake_impl,
    )
    monkeypatch.setattr(
        "shardyfusion.writer.spark.writer.resolve_num_dbs",
        lambda _config, _count_fn: 4,
    )

    from shardyfusion.serde import ValueSpec

    result = write_sharded_by_hash(
        fake_df,  # type: ignore[arg-type]
        fake_config,  # type: ignore[arg-type]
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
        spark_conf_overrides={"spark.speculation": "false"},
    )

    assert result == "result-sentinel"
    assert calls[0] == ("ctx_init", (fake_spark, {"spark.speculation": "false"}))
    assert calls[1][0] == "ctx_enter"
    assert calls[2][0] == "impl"
    assert calls[3][0] == "ctx_exit"


def test_write_sharded_spark_wraps_input_df_when_cache_enabled(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    fake_spark = _FakeSparkSession()
    fake_df = _FakeDataFrame(fake_spark)
    fake_config = SimpleNamespace(
        output=SimpleNamespace(run_id=None),
        num_dbs=4,
        max_keys_per_shard=None,
    )

    class _RecordingCtx:
        def __init__(self, spark, overrides):
            calls.append(("ctx_init", (spark, overrides)))

        def __enter__(self):
            calls.append(("ctx_enter", None))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("ctx_exit", None))

    def _fake_impl(
        *,
        df,
        config,
        run_id,
        started,
        key_col,
        value_spec,
        sort_within_partitions,
        max_writes_per_second,
        max_write_bytes_per_second,
        verify_routing,
        vector_fn,
        vector_columns,
    ):
        _ = (
            started,
            key_col,
            value_spec,
            sort_within_partitions,
            max_writes_per_second,
            max_write_bytes_per_second,
            verify_routing,
            vector_fn,
            vector_columns,
        )
        calls.append(("impl", (df, config, run_id)))
        return "result-sentinel"

    monkeypatch.setattr(
        "shardyfusion.writer.spark.writer.SparkConfOverrideContext",
        _RecordingCtx,
    )
    monkeypatch.setattr(
        "shardyfusion.writer.spark.writer._write_hash_sharded",
        _fake_impl,
    )
    monkeypatch.setattr(
        "shardyfusion.writer.spark.writer.resolve_num_dbs",
        lambda _config, _count_fn: 4,
    )

    from shardyfusion.serde import ValueSpec

    result1 = write_sharded_by_hash(
        fake_df,  # type: ignore[arg-type]
        fake_config,  # type: ignore[arg-type]
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
        cache_input=True,
        storage_level="test-level",  # type: ignore[arg-type]
    )
    result2 = write_sharded_by_hash(
        fake_df,  # type: ignore[arg-type]
        fake_config,  # type: ignore[arg-type]
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
        cache_input=True,
        storage_level="test-level",  # type: ignore[arg-type]
    )

    assert result1 == "result-sentinel"
    assert result2 == "result-sentinel"
    assert fake_df.persist_calls == ["test-level", "test-level"]
    assert fake_df.unpersist_calls == [False, False]
