from __future__ import annotations

import sqlite3
from collections.abc import Generator
from typing import TYPE_CHECKING, Any, TypedDict

import pytest

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def _create_sqlite3_conn(*args: Any, **kwargs: Any) -> sqlite3.Connection:
    """Create a sqlite3 connection with extension loading enabled."""
    conn = sqlite3.connect(*args, **kwargs)
    if hasattr(conn, "enable_load_extension"):
        conn.enable_load_extension(True)
    return conn


@pytest.fixture
def sqlite3_conn() -> sqlite3.Connection:
    """Create a sqlite3 connection with extension loading enabled."""
    return _create_sqlite3_conn(":memory:")


def _is_sqlite_vec_available() -> bool:
    """Check if sqlite-vec is available and loadable."""
    try:
        import sqlite_vec

        conn = _create_sqlite3_conn(":memory:")
        try:
            sqlite_vec.load(conn)
            return True
        except (sqlite3.OperationalError, AttributeError):
            return False
        finally:
            conn.close()
    except ImportError:
        return False


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "vector_sqlite: tests requiring sqlite-vec extension (sqlite-vec)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip vector_sqlite tests if sqlite-vec is not available."""
    if _is_sqlite_vec_available():
        return

    skip_sqlite_vec = pytest.mark.skip(reason="sqlite-vec extension not loadable")
    for item in items:
        if "vector_sqlite" in item.keywords:
            item.add_marker(skip_sqlite_vec)


class LocalS3Service(TypedDict):
    endpoint_url: str
    region_name: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    client: Any  # boto3 S3 client (dynamically generated, no static type)


@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession, None, None]:
    pytest.importorskip("pyspark", reason="requires writer-spark-slatedb extra")
    import pandas as pd
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("shardyfusion_tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    try:
        # Spark 4 mapInPandas paths require Arrow runtime support. Some
        # environments ship incompatible Arrow/Netty binaries; skip Spark
        # suites in that case instead of failing unrelated tests.
        session.range(1).mapInPandas(
            lambda batches: (pd.DataFrame({"id": b["id"]}) for b in batches),
            "id long",
        ).count()
    except Exception as exc:
        session.stop()
        pytest.skip(f"Spark Arrow runtime unavailable in this environment: {exc}")
    yield session
    session.stop()
