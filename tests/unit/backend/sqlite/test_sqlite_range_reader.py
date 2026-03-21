"""Unit tests for the SQLite range-read backend."""

from unittest.mock import MagicMock, patch

import pytest

from shardyfusion.sqlite_adapter import _S3ReadOnlyFile


class TestS3ReadOnlyFile:
    """Test the core S3 range-read I/O layer."""

    @pytest.fixture()
    def s3_client(self) -> MagicMock:
        client = MagicMock()
        client.head_object.return_value = {"ContentLength": 8192}
        return client

    def test_size_from_head(self, s3_client: MagicMock) -> None:
        with patch("shardyfusion.storage.create_s3_client", return_value=s3_client):
            f = _S3ReadOnlyFile(bucket="b", key="k")
        assert f.size == 8192

    def test_read_sends_range_header(self, s3_client: MagicMock) -> None:
        body = MagicMock()
        body.read.return_value = b"\x00" * 4096
        s3_client.get_object.return_value = {"Body": body}

        with patch("shardyfusion.storage.create_s3_client", return_value=s3_client):
            f = _S3ReadOnlyFile(bucket="b", key="k")

        data = f.read(0, 4096)
        s3_client.get_object.assert_called_once_with(
            Bucket="b", Key="k", Range="bytes=0-4095"
        )
        assert len(data) == 4096

    def test_page_cache_hit(self, s3_client: MagicMock) -> None:
        body = MagicMock()
        body.read.return_value = b"\x01" * 100
        s3_client.get_object.return_value = {"Body": body}

        with patch("shardyfusion.storage.create_s3_client", return_value=s3_client):
            f = _S3ReadOnlyFile(bucket="b", key="k", page_cache_pages=10)

        # First read — cache miss
        data1 = f.read(0, 100)
        assert s3_client.get_object.call_count == 1

        # Second read — cache hit
        data2 = f.read(0, 100)
        assert s3_client.get_object.call_count == 1  # no new S3 call
        assert data1 == data2

    def test_page_cache_eviction(self, s3_client: MagicMock) -> None:
        body = MagicMock()
        body.read.return_value = b"\x00" * 10
        s3_client.get_object.return_value = {"Body": body}

        with patch("shardyfusion.storage.create_s3_client", return_value=s3_client):
            f = _S3ReadOnlyFile(bucket="b", key="k", page_cache_pages=2)

        # Fill cache: 2 pages
        f.read(0, 10)
        f.read(10, 10)
        assert s3_client.get_object.call_count == 2

        # Third unique read evicts first page
        f.read(20, 10)
        assert s3_client.get_object.call_count == 3

        # Re-read first page — cache miss
        f.read(0, 10)
        assert s3_client.get_object.call_count == 4
