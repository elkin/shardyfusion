from __future__ import annotations

import os
import socket
from collections.abc import Generator
from typing import TYPE_CHECKING, TypedDict
from uuid import uuid4

import pytest

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


class LocalS3Service(TypedDict):
    endpoint_url: str
    region_name: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    client: object


@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession, None, None]:
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("slatedb_spark_sharded_tests")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


@pytest.fixture(scope="session")
def local_s3_service() -> Generator[LocalS3Service, None, None]:
    """Start a local S3-compatible test service backed by moto."""

    import boto3
    import moto.server as moto_server

    ThreadedMotoServer = moto_server.ThreadedMotoServer

    port = _pick_free_port()
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=port, verbose=False)
    server.start()

    host, resolved_port = server.get_host_and_port()
    endpoint_url = f"http://{host}:{resolved_port}"
    region_name = "us-east-1"
    access_key = "test"
    secret_key = "test"
    bucket = f"slatedb-test-{uuid4().hex[:8]}"

    previous_env = {
        "SLATEDB_S3_ENDPOINT_URL": os.environ.get("SLATEDB_S3_ENDPOINT_URL"),
        "AWS_REGION": os.environ.get("AWS_REGION"),
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
    }
    os.environ["SLATEDB_S3_ENDPOINT_URL"] = endpoint_url
    os.environ["AWS_REGION"] = region_name
    os.environ["AWS_ACCESS_KEY_ID"] = access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region_name,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    client.create_bucket(Bucket=bucket)

    try:
        yield {
            "endpoint_url": endpoint_url,
            "region_name": region_name,
            "access_key_id": access_key,
            "secret_access_key": secret_key,
            "bucket": bucket,
            "client": client,
        }
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        server.stop()
