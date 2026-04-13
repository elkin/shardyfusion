"""Tests for _sqlite_vfs.S3ReadOnlyFile with mocked S3."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shardyfusion._sqlite_vfs import (
    S3ReadOnlyFile,
    S3VfsError,
    _normalize_page_cache_pages,
)

# ---------------------------------------------------------------------------
# _normalize_page_cache_pages
# ---------------------------------------------------------------------------


class TestNormalizePageCachePages:
    def test_valid_value(self) -> None:
        assert _normalize_page_cache_pages(10) == 10

    def test_zero_allowed(self) -> None:
        assert _normalize_page_cache_pages(0) == 0

    def test_negative_raises(self) -> None:
        with pytest.raises(S3VfsError, match="page_cache_pages must be >= 0"):
            _normalize_page_cache_pages(-1)


# ---------------------------------------------------------------------------
# S3ReadOnlyFile
# ---------------------------------------------------------------------------


def _mock_s3_client(content_length: int = 1000) -> MagicMock:
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": content_length}

    def get_object(*, Bucket: str, Key: str, Range: str) -> dict:
        # Parse range header "bytes=start-end"
        _, range_spec = Range.split("=")
        start, end = range_spec.split("-")
        length = int(end) - int(start) + 1
        body = MagicMock()
        body.read.return_value = b"x" * length
        return {"Body": body}

    client.get_object.side_effect = get_object
    return client


class TestS3ReadOnlyFile:
    @patch("shardyfusion.storage.create_s3_client")
    def test_init_and_size(self, mock_create: MagicMock) -> None:
        mock_create.return_value = _mock_s3_client(500)
        f = S3ReadOnlyFile(bucket="b", key="k")
        assert f.size == 500

    @patch("shardyfusion.storage.create_s3_client")
    def test_read_basic(self, mock_create: MagicMock) -> None:
        mock_create.return_value = _mock_s3_client(100)
        f = S3ReadOnlyFile(bucket="b", key="k")
        data = f.read(0, 10)
        assert len(data) == 10
        assert data == b"x" * 10

    @patch("shardyfusion.storage.create_s3_client")
    def test_read_past_eof_returns_empty(self, mock_create: MagicMock) -> None:
        mock_create.return_value = _mock_s3_client(100)
        f = S3ReadOnlyFile(bucket="b", key="k")
        data = f.read(200, 10)
        assert data == b""

    @patch("shardyfusion.storage.create_s3_client")
    def test_read_cache_hit(self, mock_create: MagicMock) -> None:
        client = _mock_s3_client(100)
        mock_create.return_value = client
        f = S3ReadOnlyFile(bucket="b", key="k", page_cache_pages=10)

        # First read — cache miss
        f.read(0, 10)
        call_count_1 = client.get_object.call_count

        # Second read — cache hit, no new S3 call
        f.read(0, 10)
        assert client.get_object.call_count == call_count_1

    @patch("shardyfusion.storage.create_s3_client")
    def test_read_cache_disabled(self, mock_create: MagicMock) -> None:
        client = _mock_s3_client(100)
        mock_create.return_value = client
        f = S3ReadOnlyFile(bucket="b", key="k", page_cache_pages=0)

        f.read(0, 10)
        f.read(0, 10)
        # Without cache, every read calls S3
        assert client.get_object.call_count == 2

    @patch("shardyfusion.storage.create_s3_client")
    def test_lru_eviction(self, mock_create: MagicMock) -> None:
        client = _mock_s3_client(100)
        mock_create.return_value = client
        f = S3ReadOnlyFile(bucket="b", key="k", page_cache_pages=1)

        # Fill the cache with one entry
        f.read(0, 10)
        assert client.get_object.call_count == 1

        # Evict first entry by reading a different range
        f.read(20, 10)
        assert client.get_object.call_count == 2

        # The first entry was evicted — re-reading it requires S3
        f.read(0, 10)
        assert client.get_object.call_count == 3


# ---------------------------------------------------------------------------
# S3VfsError
# ---------------------------------------------------------------------------


class TestS3VfsError:
    def test_not_retryable(self) -> None:
        err = S3VfsError("test")
        assert err.retryable is False
