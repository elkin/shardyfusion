"""Garage-backed S3 fixtures for end-to-end tests (compose-only).

Reads Garage connection details from environment variables set by the
compose stack (docker/compose-e2e.yaml). Creates an API key and bucket
via the Garage Admin API, then yields a ``LocalS3Service`` dict
compatible with the shared test scenarios.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Generator
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import boto3
import pytest

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from tests.conftest import LocalS3Service

from slatedb_spark_sharded.type_defs import S3ClientConfig


def _admin_request(
    admin_url: str,
    path: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    token: str = "",
) -> Any:
    """Make an HTTP request to the Garage Admin API."""
    url = f"{admin_url}{path}"
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read()
        if raw:
            return json.loads(raw)
        return None


@pytest.fixture(scope="session")
def garage_s3_service() -> Generator[LocalS3Service, None, None]:
    """Create an API key and bucket on the compose-managed Garage instance."""

    endpoint_url = os.environ.get("GARAGE_E2E_ENDPOINT")
    admin_url = os.environ.get("GARAGE_E2E_ADMIN_URL")
    admin_token = os.environ.get("GARAGE_E2E_ADMIN_TOKEN", "")
    region_name = os.environ.get("GARAGE_E2E_REGION", "garage")

    if not endpoint_url or not admin_url:
        pytest.skip(
            "GARAGE_E2E_ENDPOINT / GARAGE_E2E_ADMIN_URL not set (run via 'just d-e2e')"
        )

    admin_v2 = f"{admin_url}/v2"
    bucket = f"slatedb-e2e-{uuid4().hex[:8]}"

    # Create API key
    key_resp = _admin_request(
        admin_v2,
        "/CreateKey",
        method="POST",
        body={"name": "e2e-test-key"},
        token=admin_token,
    )
    access_key_id = key_resp["accessKeyId"]
    secret_access_key = key_resp["secretAccessKey"]

    # Create bucket
    _admin_request(
        admin_v2,
        "/CreateBucket",
        method="POST",
        body={"globalAlias": bucket},
        token=admin_token,
    )

    # Get bucket ID for AllowBucketKey
    bucket_info = _admin_request(
        admin_v2,
        f"/GetBucketInfo?globalAlias={bucket}",
        token=admin_token,
    )
    bucket_id = bucket_info["id"]

    _admin_request(
        admin_v2,
        "/AllowBucketKey",
        method="POST",
        body={
            "bucketId": bucket_id,
            "accessKeyId": access_key_id,
            "permissions": {"read": True, "write": True, "owner": True},
        },
        token=admin_token,
    )

    # Set env vars for implicit boto3 / slatedb client construction
    os.environ["SLATEDB_S3_ENDPOINT_URL"] = endpoint_url
    os.environ["AWS_REGION"] = region_name
    os.environ["AWS_ACCESS_KEY_ID"] = access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_access_key

    # Create boto3 client with path-style addressing
    from botocore.config import Config as BotocoreConfig

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region_name,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=BotocoreConfig(s3={"addressing_style": "path"}),
    )

    yield {
        "endpoint_url": endpoint_url,
        "region_name": region_name,
        "access_key_id": access_key_id,
        "secret_access_key": secret_access_key,
        "bucket": bucket,
        "client": client,
    }


def s3_client_config_from_service(service: LocalS3Service) -> S3ClientConfig:
    """Build an S3ClientConfig with path-style addressing from a service dict."""
    return {
        "endpoint_url": service["endpoint_url"],
        "region_name": service["region_name"],
        "access_key_id": service["access_key_id"],
        "secret_access_key": service["secret_access_key"],
        "addressing_style": "path",
    }


@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession, None, None]:
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("slatedb_spark_sharded_e2e")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()
