from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest

from shardyfusion.config import WriteConfig
from shardyfusion.run_registry import (
    InMemoryRunRegistry,
    RunRecord,
    RunRecordLifecycle,
    RunStatus,
    S3RunRegistry,
    managed_run_record,
)


def _config(*, registry: InMemoryRunRegistry | None = None) -> WriteConfig:
    return WriteConfig(
        num_dbs=1,
        s3_prefix="s3://bucket/prefix",
        run_registry=registry,
    )


def test_s3_run_registry_builds_expected_urls(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_create_s3_client(_creds=None, _opts=None):
        return object()

    def fake_put_bytes(
        url,
        payload,
        content_type,
        headers=None,
        *,
        s3_client=None,
        metrics_collector=None,
        **kwargs,
    ):
        calls.append(
            {
                "url": url,
                "payload": payload,
                "content_type": content_type,
                "headers": headers,
                "s3_client": s3_client,
            }
        )

    monkeypatch.setattr(
        "shardyfusion.run_registry.create_s3_client", fake_create_s3_client
    )
    monkeypatch.setattr("shardyfusion.run_registry.put_bytes", fake_put_bytes)

    store = S3RunRegistry("s3://bucket/prefix", run_registry_prefix="runs")
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
    assert len(calls) == 1
    assert calls[0]["url"] == ref
    assert calls[0]["content_type"] == "application/x-yaml"


def test_managed_run_record_marks_succeeded() -> None:
    registry = InMemoryRunRegistry()
    cfg = _config(registry=registry)

    with managed_run_record(
        config=cfg, run_id="run-success", writer_type="python"
    ) as rr:
        rr.set_manifest_ref("mem://manifests/run-success")
        rr.mark_succeeded()

    assert rr.run_record_ref is not None
    record = registry.load(rr.run_record_ref)
    assert record.status is RunStatus.SUCCEEDED
    assert record.manifest_ref == "mem://manifests/run-success"
    assert record.error_type is None


def test_managed_run_record_marks_failed_on_exception() -> None:
    registry = InMemoryRunRegistry()
    cfg = _config(registry=registry)

    with pytest.raises(RuntimeError, match="boom"):
        with managed_run_record(
            config=cfg, run_id="run-failed", writer_type="python"
        ) as rr:
            raise RuntimeError("boom")

    assert rr.run_record_ref is not None
    record = registry.load(rr.run_record_ref)
    assert record.status is RunStatus.FAILED
    assert record.error_type == "RuntimeError"
    assert record.error_message == "boom"


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
