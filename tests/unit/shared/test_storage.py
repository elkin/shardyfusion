"""Tests for the new obstore-based storage layer in shardyfusion.storage."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from shardyfusion.storage import (
    MemoryBackend,
    ObstoreBackend,
    create_s3_store,
    join_s3,
    parse_s3_url,
)


class TestParseS3Url:
    def test_basic(self) -> None:
        bucket, key = parse_s3_url("s3://mybucket/path/to/key")
        assert bucket == "mybucket"
        assert key == "path/to/key"

    def test_invalid_scheme(self) -> None:
        with pytest.raises(ValueError, match="Invalid S3 URL"):
            parse_s3_url("http://bucket/key")

    def test_missing_key(self) -> None:
        with pytest.raises(ValueError, match="must include key path"):
            parse_s3_url("s3://bucket/")


class TestJoinS3:
    def test_basic(self) -> None:
        assert join_s3("s3://bucket/prefix", "a", "b") == "s3://bucket/prefix/a/b"

    def test_strips_redundant_slashes(self) -> None:
        assert join_s3("s3://bucket/prefix/", "/a/", "/b/") == "s3://bucket/prefix/a/b"


class TestMemoryBackend:
    def test_put_and_get(self) -> None:
        backend = MemoryBackend()
        backend.put("s3://bucket/key", b"hello", "text/plain")
        assert backend.get("s3://bucket/key") == b"hello"

    def test_try_get_missing(self) -> None:
        backend = MemoryBackend()
        assert backend.try_get("s3://bucket/missing") is None

    def test_list_prefixes(self) -> None:
        backend = MemoryBackend()
        backend.put("s3://bucket/prefix/a/obj1", b"x", "application/octet-stream")
        backend.put("s3://bucket/prefix/a/obj2", b"x", "application/octet-stream")
        backend.put("s3://bucket/prefix/b/obj1", b"x", "application/octet-stream")
        result = backend.list_prefixes("s3://bucket/prefix/")
        assert result == [
            "s3://bucket/prefix/a/",
            "s3://bucket/prefix/b/",
        ]

    def test_delete_prefix(self) -> None:
        backend = MemoryBackend()
        backend.put("s3://bucket/prefix/a/obj1", b"x", "application/octet-stream")
        backend.put("s3://bucket/prefix/b/obj1", b"x", "application/octet-stream")
        deleted = backend.delete_prefix("s3://bucket/prefix/a/")
        assert deleted == 1
        assert backend.try_get("s3://bucket/prefix/a/obj1") is None
        assert backend.get("s3://bucket/prefix/b/obj1") == b"x"

    def test_delete_prefix_empty(self) -> None:
        backend = MemoryBackend()
        assert backend.delete_prefix("s3://bucket/empty/") == 0


class TestObstoreBackend:
    """Tests for ObstoreBackend using a mocked obstore store."""

    def test_put_and_get(self, monkeypatch):
        store = MagicMock()
        backend = ObstoreBackend(store)

        # Mock obstore.put and obstore.get
        mock_put = MagicMock()
        mock_get_result = MagicMock()
        mock_get_result.bytes.return_value = b"payload"
        mock_get = MagicMock(return_value=mock_get_result)

        monkeypatch.setattr("obstore.put", mock_put)
        monkeypatch.setattr("obstore.get", mock_get)

        backend.put("s3://bucket/key", b"payload", "text/plain")
        mock_put.assert_called_once_with(store, "key", b"payload")

        result = backend.get("s3://bucket/key")
        mock_get.assert_called_once_with(store, "key")
        assert result == b"payload"

    def test_try_get_not_found(self, monkeypatch):
        store = MagicMock()
        backend = ObstoreBackend(store)

        class FakeNotFoundError(Exception):
            pass

        mock_get = MagicMock(side_effect=FakeNotFoundError())
        monkeypatch.setattr("obstore.get", mock_get)
        monkeypatch.setattr("obstore.exceptions.NotFoundError", FakeNotFoundError)

        assert backend.try_get("s3://bucket/missing") is None

    def test_list_prefixes(self, monkeypatch):
        store = MagicMock()
        backend = ObstoreBackend(store)

        mock_list = MagicMock(
            return_value={
                "common_prefixes": ["prefix/a/", "prefix/b/"],
                "objects": [],
            }
        )
        monkeypatch.setattr("obstore.list_with_delimiter", mock_list)

        result = backend.list_prefixes("s3://bucket/prefix/")
        assert result == [
            "s3://bucket/prefix/a/",
            "s3://bucket/prefix/b/",
        ]
        mock_list.assert_called_once_with(store, prefix="prefix/")

    def test_delete_prefix(self, monkeypatch):
        store = MagicMock()
        backend = ObstoreBackend(store)

        mock_list = MagicMock(
            return_value=iter([[{"path": "prefix/a/obj1"}, {"path": "prefix/a/obj2"}]])
        )
        mock_delete = MagicMock()
        monkeypatch.setattr("obstore.list", mock_list)
        monkeypatch.setattr("obstore.delete", mock_delete)

        deleted = backend.delete_prefix("s3://bucket/prefix/")
        assert deleted == 2
        mock_delete.assert_called_once_with(store, ["prefix/a/obj1", "prefix/a/obj2"])

    def test_put_fallback_retry_on_generic_error(self, monkeypatch):
        store = MagicMock()
        backend = ObstoreBackend(store)

        class FakeGenericError(Exception):
            pass

        calls = []

        def mock_put(store_arg, key, payload):
            calls.append((key, payload))
            if len(calls) == 1:
                raise FakeGenericError()

        monkeypatch.setattr("obstore.put", mock_put)
        monkeypatch.setattr("obstore.exceptions.GenericError", FakeGenericError)
        monkeypatch.setattr("time.sleep", lambda _: None)

        backend.put("s3://bucket/key", b"payload", "text/plain")
        assert len(calls) == 2
        assert calls[0] == ("key", b"payload")
        assert calls[1] == ("key", b"payload")


class TestCreateS3Store:
    """Tests for create_s3_store configuration mapping."""

    def test_maps_credentials(self, monkeypatch):
        mock_s3store = MagicMock()
        monkeypatch.setattr("obstore.store.S3Store", mock_s3store)

        from shardyfusion.credentials import S3Credentials

        creds = S3Credentials(
            access_key_id="ak",
            secret_access_key="sk",
            session_token="st",
        )
        create_s3_store("bucket", credentials=creds)

        mock_s3store.assert_called_once()
        kwargs = mock_s3store.call_args.kwargs
        assert kwargs["access_key_id"] == "ak"
        assert kwargs["secret_access_key"] == "sk"
        assert kwargs["session_token"] == "st"

    def test_maps_endpoint_and_region(self, monkeypatch):
        mock_s3store = MagicMock()
        monkeypatch.setattr("obstore.store.S3Store", mock_s3store)

        create_s3_store(
            "bucket",
            connection_options={
                "endpoint_url": "http://localhost:9000",
                "region_name": "us-west-2",
                "addressing_style": "path",
            },
        )

        kwargs = mock_s3store.call_args.kwargs
        assert kwargs["endpoint"] == "http://localhost:9000"
        assert kwargs["region"] == "us-west-2"
        assert kwargs["virtual_hosted_style_request"] is False
        client_opts = kwargs["client_options"]
        assert client_opts["allow_http"] is True

    def test_maps_timeout_options(self, monkeypatch):
        mock_s3store = MagicMock()
        monkeypatch.setattr("obstore.store.S3Store", mock_s3store)

        create_s3_store(
            "bucket",
            connection_options={
                "connect_timeout": 5,
                "read_timeout": 30,
            },
        )

        kwargs = mock_s3store.call_args.kwargs
        client_opts = kwargs["client_options"]
        assert client_opts["connect_timeout"] == "5s"
        assert client_opts["timeout"] == "30s"

    def test_uses_shardyfusion_retry_defaults(self, monkeypatch):
        mock_s3store = MagicMock()
        monkeypatch.setattr("obstore.store.S3Store", mock_s3store)

        create_s3_store("bucket")

        kwargs = mock_s3store.call_args.kwargs
        retry = kwargs["retry_config"]
        assert retry["max_retries"] == 5
        assert retry["backoff"]["init_backoff"] == timedelta(milliseconds=200)
        assert retry["backoff"]["max_backoff"] == timedelta(seconds=4)
        assert retry["retry_timeout"] == timedelta(seconds=30)

    def test_custom_max_attempts(self, monkeypatch):
        mock_s3store = MagicMock()
        monkeypatch.setattr("obstore.store.S3Store", mock_s3store)

        create_s3_store("bucket", connection_options={"max_attempts": 3})

        kwargs = mock_s3store.call_args.kwargs
        assert kwargs["retry_config"]["max_retries"] == 2
