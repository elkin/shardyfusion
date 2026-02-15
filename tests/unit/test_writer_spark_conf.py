from __future__ import annotations

from types import SimpleNamespace

from slatedb_spark_sharded.writer import (
    SparkConfOverrideContext,
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


def test_write_sharded_slatedb_uses_optional_spark_conf_overrides(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    fake_spark = _FakeSparkSession()
    fake_df = SimpleNamespace(sparkSession=fake_spark)
    fake_config = SimpleNamespace(run_id=None)

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

    monkeypatch.setattr("slatedb_spark_sharded.writer.SparkConfOverrideContext", _RecordingCtx)
    monkeypatch.setattr("slatedb_spark_sharded.writer._write_sharded_slatedb_impl", _fake_impl)

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
