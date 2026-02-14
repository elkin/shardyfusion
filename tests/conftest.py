from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def spark():
    pyspark = pytest.importorskip("pyspark")
    session = (
        pyspark.sql.SparkSession.builder.master("local[2]")
        .appName("slatedb_spark_sharded_tests")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()
