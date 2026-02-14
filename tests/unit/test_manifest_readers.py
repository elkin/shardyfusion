from __future__ import annotations

import json

import pytest

from slatedb_spark_sharded.manifest_readers import DefaultS3ManifestReader, parse_json_manifest


def test_parse_json_manifest_round_trip() -> None:
    payload = {
        "required": {
            "run_id": "run-1",
            "created_at": "2026-01-01T00:00:00+00:00",
            "num_dbs": 2,
            "s3_prefix": "s3://bucket/prefix",
            "key_col": "id",
            "key_encoding": "u64be",
            "sharding": {"strategy": "hash"},
            "db_path_template": "db={db_id:05d}",
            "tmp_prefix": "_tmp",
            "format_version": 1,
        },
        "shards": [
            {
                "db_id": 0,
                "db_url": "s3://bucket/prefix/db=00000",
                "attempt": 0,
                "row_count": 1,
                "min_key": 1,
                "max_key": 1,
                "checkpoint_id": None,
                "writer_info": {},
            },
            {
                "db_id": 1,
                "db_url": "s3://bucket/prefix/db=00001",
                "attempt": 0,
                "row_count": 1,
                "min_key": 2,
                "max_key": 2,
                "checkpoint_id": None,
                "writer_info": {},
            },
        ],
        "custom": {"env": "test"},
    }

    parsed = parse_json_manifest(json.dumps(payload).encode("utf-8"))

    assert parsed.required_build.num_dbs == 2
    assert [shard.db_id for shard in parsed.shards] == [0, 1]
    assert parsed.custom == {"env": "test"}


def test_parse_json_manifest_rejects_bad_shard_coverage() -> None:
    payload = {
        "required": {
            "run_id": "run-1",
            "created_at": "2026-01-01T00:00:00+00:00",
            "num_dbs": 2,
            "s3_prefix": "s3://bucket/prefix",
            "key_col": "id",
            "key_encoding": "u64be",
            "sharding": {"strategy": "hash"},
            "db_path_template": "db={db_id:05d}",
            "tmp_prefix": "_tmp",
            "format_version": 1,
        },
        "shards": [
            {
                "db_id": 0,
                "db_url": "s3://bucket/prefix/db=00000",
                "attempt": 0,
                "row_count": 1,
                "min_key": 1,
                "max_key": 1,
                "checkpoint_id": None,
                "writer_info": {},
            }
        ],
        "custom": {},
    }

    with pytest.raises(ValueError, match="shard count mismatch"):
        parse_json_manifest(json.dumps(payload).encode("utf-8"))


def test_default_reader_rejects_non_json_manifest_content_type() -> None:
    reader = DefaultS3ManifestReader("s3://bucket/prefix")
    with pytest.raises(ValueError, match="supports only application/json"):
        reader.load_manifest("s3://bucket/prefix/manifest", "application/x-custom")
