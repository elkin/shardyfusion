from __future__ import annotations

from typing import TYPE_CHECKING

from shardyfusion.run_registry import (
    RUN_RECORD_FORMAT_VERSION,
    RunRecord,
    RunStatus,
    parse_run_record,
)

if TYPE_CHECKING:
    from shardyfusion.manifest import BuildResult
    from shardyfusion.run_registry import InMemoryRunRegistry
    from tests.conftest import LocalS3Service


def assert_success_run_record(
    record: RunRecord,
    *,
    result: BuildResult,
    writer_type: str,
    s3_prefix: str,
    shard_prefix: str = "shards",
    db_path_template: str = "db={db_id:05d}",
) -> None:
    assert record.format_version == RUN_RECORD_FORMAT_VERSION
    assert record.run_id == result.run_id
    assert record.writer_type == writer_type
    assert record.status is RunStatus.SUCCEEDED
    assert record.started_at.tzinfo is not None
    assert record.updated_at.tzinfo is not None
    assert record.lease_expires_at.tzinfo is not None
    assert record.started_at <= record.updated_at <= record.lease_expires_at
    assert record.s3_prefix == s3_prefix
    assert record.shard_prefix == shard_prefix
    assert record.db_path_template == db_path_template
    assert record.manifest_ref == result.manifest_ref
    assert record.error_type is None
    assert record.error_message is None


def load_in_memory_run_record(
    registry: InMemoryRunRegistry,
    result: BuildResult,
) -> RunRecord:
    assert result.run_record_ref is not None
    return registry.load(result.run_record_ref)


def load_s3_run_record(
    local_s3_service: LocalS3Service,
    run_record_ref: str,
) -> RunRecord:
    bucket = local_s3_service["bucket"]
    run_record_key = run_record_ref.split(f"s3://{bucket}/", 1)[1]
    run_record_obj = local_s3_service["client"].get_object(
        Bucket=bucket,
        Key=run_record_key,
    )
    return parse_run_record(run_record_obj["Body"].read())
