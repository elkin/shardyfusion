from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest

from shardyfusion.config import HashWriteConfig, ManifestOptions
from shardyfusion.credentials import S3Credentials, StaticCredentialProvider
from shardyfusion.run_registry import (
    InMemoryRunRegistry,
    RunRecord,
    RunRecordLifecycle,
    RunStatus,
    S3RunRegistry,
    resolve_run_registry,
)
from shardyfusion.storage import MemoryBackend, StorageBackend


class _RecordingBackend:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put(
        self,
        url: str,
        data: bytes,
        content_type: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.objects[url] = data

    def get(self, url: str) -> bytes:
        return self.objects[url]

    def try_get(self, url: str) -> bytes | None:
        return self.objects.get(url)

    def delete(self, url: str) -> None:
        self.objects.pop(url, None)

    def list(self, prefix_url: str) -> list[str]:
        return [url for url in self.objects if url.startswith(prefix_url)]

    def list_prefixes(self, prefix_url: str, delimiter: str = "/") -> list[str]:
        return []


class _RecordingRegistry:
    def __init__(self) -> None:
        self.records: dict[str, RunRecord] = {}

    def create(self, record: RunRecord) -> str:
        ref = f"mem://runs/{record.run_id}"
        self.update(ref, record)
        return ref

    def update(self, ref: str, record: RunRecord) -> None:
        self.records[ref] = record.model_copy(deep=True)

    def load(self, ref: str) -> RunRecord:
        return self.records[ref].model_copy(deep=True)


def _config(*, registry: InMemoryRunRegistry | None = None) -> HashWriteConfig:
    return HashWriteConfig(
        num_dbs=1,
        s3_prefix="s3://bucket/prefix",
        run_registry=registry,
    )


def test_s3_run_registry_builds_expected_urls() -> None:
    backend = MemoryBackend()
    store = S3RunRegistry(backend, "s3://bucket/prefix", run_registry_prefix="runs")
    ref = store.create(
        RunRecord(
            run_id="run-123",
            writer_type="python",
            status=RunStatus.RUNNING,
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
            lease_expires_at=datetime(2026, 1, 1, 0, 5, tzinfo=UTC),
            s3_prefix="s3://bucket/prefix",
            shard_prefix="shards",
            db_path_template="db={db_id:05d}",
        )
    )

    assert ref.startswith("s3://bucket/prefix/runs/")
    assert ref.endswith("/run.yaml")
    assert backend.try_get(ref) is not None


def test_run_record_lifecycle_marks_succeeded() -> None:
    registry = InMemoryRunRegistry()
    cfg = _config(registry=registry)

    with RunRecordLifecycle.start(
        config=cfg, run_id="run-success", writer_type="python"
    ) as rr:
        rr.set_manifest_ref("mem://manifests/run-success")
        rr.mark_succeeded()

    assert rr.run_record_ref is not None
    record = registry.load(rr.run_record_ref)
    assert record.status is RunStatus.SUCCEEDED
    assert record.manifest_ref == "mem://manifests/run-success"
    assert record.error_type is None


def test_run_record_lifecycle_marks_failed_on_exception() -> None:
    registry = InMemoryRunRegistry()
    cfg = _config(registry=registry)

    with pytest.raises(RuntimeError, match="boom"):
        with RunRecordLifecycle.start(
            config=cfg, run_id="run-failed", writer_type="python"
        ) as rr:
            raise RuntimeError("boom")

    assert rr.run_record_ref is not None
    record = registry.load(rr.run_record_ref)
    assert record.status is RunStatus.FAILED
    assert record.error_type == "RuntimeError"
    assert record.error_message == "boom"


def test_run_record_lifecycle_is_context_managed() -> None:
    registry = _RecordingRegistry()
    cfg = _config(registry=registry)

    lifecycle = RunRecordLifecycle.start(
        config=cfg,
        run_id="run-context-manager",
        writer_type="python",
    )

    assert isinstance(lifecycle, RunRecordLifecycle)
    with lifecycle as rr:
        assert rr is lifecycle
        rr.mark_succeeded()

    assert lifecycle.run_record_ref is not None
    record = registry.load(lifecycle.run_record_ref)
    assert record.status is RunStatus.SUCCEEDED


def test_run_record_lifecycle_heartbeat_refreshes_lease() -> None:
    registry = InMemoryRunRegistry()
    cfg = _config(registry=registry)

    lifecycle = RunRecordLifecycle.start(
        config=cfg,
        run_id="run-heartbeat",
        writer_type="python",
        lease_duration=timedelta(milliseconds=80),
        heartbeat_interval=timedelta(milliseconds=20),
    )
    assert lifecycle.run_record_ref is not None
    initial = registry.load(lifecycle.run_record_ref)

    time.sleep(0.08)
    refreshed = registry.load(lifecycle.run_record_ref)
    lifecycle.mark_failed(RuntimeError("done"))

    assert refreshed.updated_at > initial.updated_at
    assert refreshed.lease_expires_at > initial.lease_expires_at


def test_resolve_run_registry_uses_only_writer_s3_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, S3Credentials | None, dict[str, object] | None]] = []
    backend = _RecordingBackend()

    def fake_create_s3_store(
        bucket: str,
        credentials: S3Credentials | None = None,
        connection_options: dict[str, object] | None = None,
    ) -> object:
        calls.append((bucket, credentials, connection_options))
        return object()

    def fake_obstore_backend(store: object) -> StorageBackend:
        return backend

    monkeypatch.setattr(
        "shardyfusion.run_registry.create_s3_store", fake_create_s3_store
    )
    monkeypatch.setattr(
        "shardyfusion.run_registry.ObstoreBackend", fake_obstore_backend
    )

    cfg = HashWriteConfig(
        num_dbs=1,
        s3_prefix="s3://bucket/prefix",
        credential_provider=StaticCredentialProvider(
            access_key_id="writer-key",
            secret_access_key="writer-secret",
        ),
        s3_connection_options={"endpoint_url": "http://writer.example"},
        manifest=ManifestOptions(
            credential_provider=StaticCredentialProvider(
                access_key_id="manifest-key",
                secret_access_key="manifest-secret",
            ),
            s3_connection_options={"endpoint_url": "http://manifest.example"},
        ),
    )

    registry = resolve_run_registry(cfg)
    record = RunRecord(
        run_id="run-settings",
        writer_type="python",
        status=RunStatus.RUNNING,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        lease_expires_at=datetime(2026, 1, 1, 0, 5, tzinfo=UTC),
        s3_prefix="s3://bucket/prefix",
        shard_prefix="shards",
        db_path_template="db={db_id:05d}",
    )
    ref = registry.create(record)

    assert calls == [
        (
            "bucket",
            S3Credentials(
                access_key_id="writer-key",
                secret_access_key="writer-secret",
            ),
            {"endpoint_url": "http://writer.example"},
        )
    ]
    assert ref.startswith("s3://bucket/prefix/runs/")
    assert backend.try_get(ref) is not None


def test_resolve_run_registry_does_not_use_manifest_s3_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, S3Credentials | None, dict[str, object] | None]] = []

    def fake_create_s3_store(
        bucket: str,
        credentials: S3Credentials | None = None,
        connection_options: dict[str, object] | None = None,
    ) -> object:
        calls.append((bucket, credentials, connection_options))
        return object()

    monkeypatch.setattr(
        "shardyfusion.run_registry.create_s3_store", fake_create_s3_store
    )
    monkeypatch.setattr(
        "shardyfusion.run_registry.ObstoreBackend", lambda store: _RecordingBackend()
    )

    cfg = HashWriteConfig(
        num_dbs=1,
        s3_prefix="s3://bucket/prefix",
        manifest=ManifestOptions(
            credential_provider=StaticCredentialProvider(
                access_key_id="manifest-key",
                secret_access_key="manifest-secret",
            ),
            s3_connection_options={"endpoint_url": "http://manifest.example"},
        ),
    )

    resolve_run_registry(cfg)

    assert calls == [("bucket", None, None)]
