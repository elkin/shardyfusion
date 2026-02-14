from __future__ import annotations

from slatedb_spark_sharded.writer import SparkConfOverrideContext


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
