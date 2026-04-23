"""Shared fixtures for vector integration tests against local S3."""

from __future__ import annotations

from typing import Any

import pytest

from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.type_defs import S3ConnectionOptions


@pytest.fixture
def s3_info(local_s3_service: dict[str, Any]) -> dict[str, Any]:
    return local_s3_service


@pytest.fixture
def cred_provider(s3_info: dict[str, Any]) -> StaticCredentialProvider:
    return StaticCredentialProvider(
        access_key_id=s3_info["access_key_id"],
        secret_access_key=s3_info["secret_access_key"],
    )


@pytest.fixture
def s3_conn_opts(s3_info: dict[str, Any]) -> S3ConnectionOptions:
    return S3ConnectionOptions(
        endpoint_url=s3_info["endpoint_url"],
        region_name=s3_info["region_name"],
    )
