"""Shared fixtures for Spark writer e2e tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest

pytest.importorskip("pyspark", reason="requires writer-spark extra")

from pyspark.sql import SparkSession  # noqa: E402


@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession, None, None]:
    session = (
        SparkSession.builder.master("local[2]")
        .appName("shardyfusion_e2e")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()
