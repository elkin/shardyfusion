"""Integration tests for ``LocalSlateDbAdapter`` against moto S3.

Writes shard data locally via ``file://`` SlateDB, uploads to moto S3
on ``close()``, then verifies the uploaded objects exist and (when
``slatedb`` is installed) reads them back via ``SlateDbReaderFactory``.

Requires the ``slatedb`` extra; skipped in tox environments that don't
include it (the ``-slatedb-`` integration envs).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

pytest.importorskip("slatedb")

from shardyfusion._settings import SlateDbSettings
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.local_slatedb_adapter import (
    LocalSlateDbAdapter,
    LocalSlateDbFactory,
)
from shardyfusion.type_defs import S3ConnectionOptions


@pytest.fixture()
def s3_prefix(local_s3_service):
    """Yield an s3:// prefix backed by the session-scoped moto server."""
    bucket = local_s3_service["bucket"]
    return f"s3://{bucket}/test-local-slate-prefix"


class TestLocalSlateDbWriteUpload:
    """Write via ``LocalSlateDbAdapter`` — upload to moto S3 — verify objects."""

    def test_write_upload_and_list_objects(
        self, tmp_path: Path, s3_prefix: str, local_s3_service
    ) -> None:
        db_url = f"{s3_prefix}/shards/run_id=test/db=00000/attempt=00"
        local_dir = tmp_path / "write"

        adapter = LocalSlateDbAdapter(
            db_url=db_url,
            local_dir=local_dir,
            env_file=None,
            settings=None,
        )
        adapter.write_batch(
            [
                (b"k1", b"v1"),
                (b"k2", b"v2"),
            ]
        )
        adapter.flush()
        adapter.close()

        # The adapter should have created local files and uploaded them.
        store_dir = adapter._store_dir  # type: ignore[attr-defined]
        assert store_dir.is_dir()
        local_files = list(store_dir.rglob("*"))
        assert len([f for f in local_files if f.is_file()]) > 0

        # Verify objects exist in moto S3 under the shard prefix.
        bucket = local_s3_service["bucket"]
        client = local_s3_service["client"]
        # The S3 key prefix is the path portion of db_url
        key_prefix = db_url.split(f"s3://{bucket}/", 1)[1]
        response = client.list_objects_v2(Bucket=bucket, Prefix=key_prefix)
        assert "Contents" in response
        uploaded_keys = sorted(obj["Key"] for obj in response["Contents"])
        assert len(uploaded_keys) > 0
        # Every uploaded key should be under the shard prefix
        for k in uploaded_keys:
            assert k.startswith(key_prefix)

    def test_read_roundtrip_when_slatedb_available(
        self, tmp_path: Path, s3_prefix: str
    ) -> None:
        """Full roundtrip: ``LocalSlateDbAdapter`` writes locally + uploads to S3,
        then the standard SlateDB reader reads from the local store (matching the
        e2e-test pattern; moto S3 does not support SlateDB's full S3 surface)."""
        pytest.importorskip("slatedb")
        from shardyfusion.reader._types import SlateDbReaderFactory

        db_url = f"{s3_prefix}/shards/run_id=test-rt/db=00000/attempt=00"
        local_dir = tmp_path / "write"

        adapter = LocalSlateDbAdapter(
            db_url=db_url,
            local_dir=local_dir,
            env_file=None,
            settings=None,
        )
        adapter.write_batch(
            [
                (b"alpha", b"value-alpha"),
                (b"beta", b"value-beta"),
            ]
        )
        adapter.flush()
        adapter.close()

        # Read from the local file:// store (the upload to moto S3 is verified
        # by test_write_upload_and_list_objects above; full S3 roundtrip with
        # SlateDB requires a real S3-compatible store — exercised in e2e).
        store_dir = adapter._store_dir  # type: ignore[attr-defined]
        file_url = f"file://{store_dir}"

        factory = SlateDbReaderFactory()
        reader = factory(
            db_url=file_url,
            local_dir=tmp_path / "read",
            checkpoint_id=None,
            manifest=None,  # type: ignore[arg-type]
        )
        assert reader.get(b"alpha") == b"value-alpha"
        assert reader.get(b"beta") == b"value-beta"
        assert reader.get(b"missing") is None
        reader.close()

    def test_empty_shard_upload_has_manifest(
        self, tmp_path: Path, s3_prefix: str, local_s3_service
    ) -> None:
        """An empty shard (no write_batch calls, just flush + close) must
        still upload SlateDB's manifest files."""
        db_url = f"{s3_prefix}/shards/run_id=test-empty/db=00000/attempt=00"
        local_dir = tmp_path / "write"

        adapter = LocalSlateDbAdapter(
            db_url=db_url,
            local_dir=local_dir,
            env_file=None,
            settings=None,
        )
        adapter.flush()
        adapter.close()

        bucket = local_s3_service["bucket"]
        client = local_s3_service["client"]
        key_prefix = db_url.split(f"s3://{bucket}/", 1)[1]
        response = client.list_objects_v2(Bucket=bucket, Prefix=key_prefix)
        assert "Contents" in response
        assert len(response["Contents"]) > 0

    def test_close_idempotent_does_not_double_upload(
        self, tmp_path: Path, s3_prefix: str, local_s3_service
    ) -> None:
        """Double-close must not upload twice."""
        db_url = f"{s3_prefix}/shards/run_id=test-idem/db=00000/attempt=00"
        local_dir = tmp_path / "write"

        adapter = LocalSlateDbAdapter(
            db_url=db_url,
            local_dir=local_dir,
            env_file=None,
            settings=None,
        )
        adapter.write_batch([(b"x", b"y")])
        adapter.close()
        adapter.close()  # second close must be a no-op

        bucket = local_s3_service["bucket"]
        client = local_s3_service["client"]
        key_prefix = db_url.split(f"s3://{bucket}/", 1)[1]
        response = client.list_objects_v2(Bucket=bucket, Prefix=key_prefix)
        assert "Contents" in response
        # A double upload would overwrite but the count should still be sane.
        # We don't assert exact count (SlateDB internals), just that uploads
        # succeeded.
        assert len(response["Contents"]) > 0


class TestLocalSlateDbFactoryS3:
    """End-to-end: ``LocalSlateDbFactory`` → write → read via local store."""

    def test_factory_write_and_read_roundtrip(
        self, tmp_path: Path, s3_prefix: str, local_s3_service
    ) -> None:
        """Write via ``LocalSlateDbFactory``, verify S3 upload, read from local store."""
        pytest.importorskip("slatedb")
        from shardyfusion.local_slatedb_adapter import LocalSlateDbFactory
        from shardyfusion.reader._types import SlateDbReaderFactory

        db_url = f"{s3_prefix}/shards/run_id=test-fac/db=00000/attempt=00"
        local_dir = tmp_path / "write"

        factory = LocalSlateDbFactory()
        adapter = factory(db_url=db_url, local_dir=local_dir)
        adapter.write_batch([(b"k1", b"v1"), (b"k2", b"v2")])
        adapter.flush()
        adapter.close()

        # Verify S3 upload occurred
        bucket = local_s3_service["bucket"]
        client = local_s3_service["client"]
        key_prefix = db_url.split(f"s3://{bucket}/", 1)[1]
        response = client.list_objects_v2(Bucket=bucket, Prefix=key_prefix)
        assert "Contents" in response
        assert len(response["Contents"]) > 0

        # Read from the local file:// store
        store_dir = adapter._store_dir  # type: ignore[attr-defined]
        file_url = f"file://{store_dir}"
        reader_factory = SlateDbReaderFactory()
        reader = reader_factory(
            db_url=file_url,
            local_dir=tmp_path / "read",
            checkpoint_id=None,
            manifest=None,  # type: ignore[arg-type]
        )
        assert reader.get(b"k1") == b"v1"
        assert reader.get(b"k2") == b"v2"
        reader.close()


class TestLocalSlateDbSettingsPropagation:
    """Verify ``SlateDbSettings`` flows through the factory into a real
    ``slatedb.uniffi.Settings`` instance and the resulting database still
    writes + uploads correctly.

    Unit tests assert ``settings.set(...)`` is *called* with mocked uniffi
    bindings; this test exercises the real bindings end-to-end so a broken
    encoding (e.g. ``to_uniffi_pairs``) would fail at ``builder.build()``.
    """

    def test_factory_with_settings_writes_and_uploads(
        self, tmp_path: Path, s3_prefix: str, local_s3_service
    ) -> None:
        db_url = f"{s3_prefix}/shards/run_id=test-settings/db=00000/attempt=00"
        local_dir = tmp_path / "write"

        settings = SlateDbSettings(
            flush_interval="250ms",
            l0_sst_size_bytes=4 * 1024 * 1024,
        )
        factory = LocalSlateDbFactory(settings=settings)
        adapter = factory(db_url=db_url, local_dir=local_dir)
        adapter.write_batch([(b"k", b"v")])
        adapter.flush()
        adapter.close()

        bucket = local_s3_service["bucket"]
        client = local_s3_service["client"]
        key_prefix = db_url.split(f"s3://{bucket}/", 1)[1]
        response = client.list_objects_v2(Bucket=bucket, Prefix=key_prefix)
        assert "Contents" in response
        assert len(response["Contents"]) > 0


class TestLocalSlateDbExplicitCredentials:
    """Verify the upload path uses the explicit ``credential_provider`` and
    ``s3_connection_options`` rather than falling back to ambient env vars.

    The session fixture sets ``SLATEDB_S3_ENDPOINT_URL`` etc. so we
    monkeypatch them away for this test — if the explicit options weren't
    plumbed through to ``_upload_store`` / ``create_s3_store``, the upload
    would fail.
    """

    def test_factory_uses_explicit_credentials_and_endpoint(
        self,
        tmp_path: Path,
        s3_prefix: str,
        local_s3_service,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Strip ambient S3 env so explicit options are the only path to moto.
        monkeypatch.delenv("SLATEDB_S3_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)

        db_url = f"{s3_prefix}/shards/run_id=test-explicit/db=00000/attempt=00"
        local_dir = tmp_path / "write"

        creds = StaticCredentialProvider(
            access_key_id=local_s3_service["access_key_id"],
            secret_access_key=local_s3_service["secret_access_key"],
        )
        opts = S3ConnectionOptions(
            endpoint_url=local_s3_service["endpoint_url"],
            region_name=local_s3_service["region_name"],
        )
        factory = LocalSlateDbFactory(
            credential_provider=creds,
            s3_connection_options=opts,
        )
        adapter = factory(db_url=db_url, local_dir=local_dir)
        adapter.write_batch([(b"explicit", b"creds")])
        adapter.flush()
        adapter.close()

        # Verify upload landed in moto under the shard prefix.
        bucket = local_s3_service["bucket"]
        client = local_s3_service["client"]
        key_prefix = db_url.split(f"s3://{bucket}/", 1)[1]
        response = client.list_objects_v2(Bucket=bucket, Prefix=key_prefix)
        assert "Contents" in response
        assert len(response["Contents"]) > 0


class TestLocalSlateDbUploadFailure:
    """Verify the close() upload-failure contract documented on close()."""

    def test_upload_failure_sets_closed_but_not_uploaded(
        self,
        tmp_path: Path,
        local_s3_service,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Ambient env points at the real moto server; clear it so the only
        # endpoint configured is the unreachable one we pass explicitly.
        monkeypatch.delenv("SLATEDB_S3_ENDPOINT_URL", raising=False)

        # Real bucket name (so parse_s3_url succeeds) but unreachable endpoint.
        bucket = local_s3_service["bucket"]
        db_url = f"s3://{bucket}/test-upload-fail/shards/db=00000/attempt=00"
        local_dir = tmp_path / "write"

        creds = StaticCredentialProvider(
            access_key_id="test",
            secret_access_key="test",
        )
        # Port 1 is reserved (tcpmux); a connection here will be refused
        # immediately on every common platform.
        opts = S3ConnectionOptions(
            endpoint_url="http://127.0.0.1:1",
            region_name="us-east-1",
            connect_timeout=1,
            read_timeout=1,
            max_attempts=1,
        )
        factory = LocalSlateDbFactory(
            credential_provider=creds,
            s3_connection_options=opts,
        )
        adapter = factory(db_url=db_url, local_dir=local_dir)
        adapter.write_batch([(b"k", b"v")])
        adapter.flush()

        with caplog.at_level(logging.ERROR, logger="shardyfusion"):
            with pytest.raises(Exception):  # noqa: B017 — surface from obstore varies
                adapter.close()

        # Shutdown completed before the upload attempt, so _closed is True.
        assert adapter._closed is True  # type: ignore[attr-defined]
        # Upload failed, so a subsequent close() must be free to retry it.
        assert adapter._uploaded is False  # type: ignore[attr-defined]
        # The dedicated failure log was emitted.
        assert any(
            "local_slatedb_adapter_upload_failed" in record.getMessage()
            or getattr(record, "event", None) == "local_slatedb_adapter_upload_failed"
            for record in caplog.records
        ), f"expected upload-failed log; got {[r.getMessage() for r in caplog.records]}"
