from __future__ import annotations

from types import SimpleNamespace

from slatedb_spark_sharded.sharding import ShardingSpec, ShardingStrategy
from slatedb_spark_sharded.writer import (
    DataFrameCacheContext,
    SparkConfOverrideContext,
    _manifest_safe_sharding,
    write_sharded_slatedb,
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


def test_write_sharded_slatedb_uses_optional_spark_conf_overrides(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    fake_spark = _FakeSparkSession()
    fake_df = SimpleNamespace(sparkSession=fake_spark)
    fake_config = SimpleNamespace(output=SimpleNamespace(run_id=None))

    class _RecordingCtx:
        def __init__(self, spark, overrides):
            calls.append(("ctx_init", (spark, overrides)))

        def __enter__(self):
            calls.append(("ctx_enter", None))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("ctx_exit", None))

    def _fake_impl(*, df, config, run_id, started):
        _ = started
        calls.append(("impl", (df, config, run_id)))
        return "result-sentinel"

    monkeypatch.setattr(
        "slatedb_spark_sharded.writer.SparkConfOverrideContext", _RecordingCtx
    )
    monkeypatch.setattr(
        "slatedb_spark_sharded.writer._write_sharded_slatedb_impl", _fake_impl
    )

    result = write_sharded_slatedb(
        fake_df,  # type: ignore[arg-type]
        fake_config,  # type: ignore[arg-type]
        spark_conf_overrides={"spark.speculation": "false"},
    )

    assert result == "result-sentinel"
    assert calls[0] == ("ctx_init", (fake_spark, {"spark.speculation": "false"}))
    assert calls[1][0] == "ctx_enter"
    assert calls[2][0] == "impl"
    assert calls[3][0] == "ctx_exit"


def test_write_sharded_slatedb_wraps_input_df_when_cache_enabled(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    fake_spark = _FakeSparkSession()
    fake_df = _FakeDataFrame(fake_spark)
    fake_config = SimpleNamespace(output=SimpleNamespace(run_id=None))

    class _RecordingCtx:
        def __init__(self, spark, overrides):
            calls.append(("ctx_init", (spark, overrides)))

        def __enter__(self):
            calls.append(("ctx_enter", None))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("ctx_exit", None))

    def _fake_impl(*, df, config, run_id, started):
        _ = started
        calls.append(("impl", (df, config, run_id)))
        return "result-sentinel"

    monkeypatch.setattr(
        "slatedb_spark_sharded.writer.SparkConfOverrideContext", _RecordingCtx
    )
    monkeypatch.setattr(
        "slatedb_spark_sharded.writer._write_sharded_slatedb_impl", _fake_impl
    )

    result1 = write_sharded_slatedb(
        fake_df,  # type: ignore[arg-type]
        fake_config,  # type: ignore[arg-type]
        cache_input=True,
        storage_level="test-level",  # type: ignore[arg-type]
    )
    result2 = write_sharded_slatedb(
        fake_df,  # type: ignore[arg-type]
        fake_config,  # type: ignore[arg-type]
        cache_input=True,
        storage_level="test-level",  # type: ignore[arg-type]
    )

    assert result1 == "result-sentinel"
    assert result2 == "result-sentinel"
    assert fake_df.persist_calls == ["test-level", "test-level"]
    assert fake_df.unpersist_calls == [False, False]


def test_manifest_safe_sharding_preserves_boundaries_for_custom_expr() -> None:
    spec = ShardingSpec(
        strategy=ShardingStrategy.CUSTOM_EXPR,
        boundaries=[10, 20],
        custom_expr="id % 2",
    )

    manifest_spec = _manifest_safe_sharding(spec)

    assert manifest_spec.strategy == ShardingStrategy.CUSTOM_EXPR
    assert manifest_spec.boundaries == [10, 20]
    assert manifest_spec.custom_expr == "id % 2"
    assert manifest_spec.custom_column_builder is None
