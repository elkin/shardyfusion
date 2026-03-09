"""Tests for manifest and CURRENT pointer parse error paths."""

from __future__ import annotations

import json

import pytest

from shardyfusion.errors import ManifestParseError
from shardyfusion.manifest_store import parse_json_manifest


def _build_valid_manifest(num_dbs: int = 2) -> dict:
    """Build a valid manifest dict for testing."""
    return {
        "required": {
            "run_id": "test-run",
            "created_at": "2026-01-01T00:00:00+00:00",
            "num_dbs": num_dbs,
            "s3_prefix": "s3://bucket/prefix",
            "key_col": "id",
            "key_encoding": "u64be",
            "sharding": {"strategy": "hash"},
            "db_path_template": "db={db_id:05d}",
            "tmp_prefix": "_tmp",
        },
        "shards": [
            {
                "db_id": i,
                "db_url": f"s3://bucket/prefix/_tmp/db={i:05d}",
                "attempt": 0,
                "row_count": 100,
                "writer_info": {},
            }
            for i in range(num_dbs)
        ],
    }


def test_valid_manifest_parses() -> None:
    data = _build_valid_manifest()
    payload = json.dumps(data).encode()
    result = parse_json_manifest(payload)
    assert result.required_build.num_dbs == 2
    assert len(result.shards) == 2


def test_truncated_json() -> None:
    payload = b'{"required": {"run_id": "test"'  # truncated
    with pytest.raises(ManifestParseError):
        parse_json_manifest(payload)


def test_invalid_utf8() -> None:
    payload = b"\xff\xfe invalid bytes"
    with pytest.raises(ManifestParseError):
        parse_json_manifest(payload)


def test_empty_payload() -> None:
    with pytest.raises(ManifestParseError):
        parse_json_manifest(b"")


def test_missing_required_fields() -> None:
    payload = json.dumps({"required": {"run_id": "test"}, "shards": []}).encode()
    with pytest.raises(ManifestParseError):
        parse_json_manifest(payload)


def test_shard_count_mismatch() -> None:
    data = _build_valid_manifest(num_dbs=8)
    # Only include 5 shards instead of 8
    data["shards"] = data["shards"][:5]
    payload = json.dumps(data).encode()
    with pytest.raises(ManifestParseError, match="shard count mismatch"):
        parse_json_manifest(payload)


def test_non_contiguous_shard_ids() -> None:
    data = _build_valid_manifest(num_dbs=3)
    # Make shard IDs non-contiguous: [0, 1, 5] instead of [0, 1, 2]
    data["shards"][2]["db_id"] = 5
    payload = json.dumps(data).encode()
    with pytest.raises(ManifestParseError, match="coverage mismatch"):
        parse_json_manifest(payload)


def test_duplicate_shard_ids() -> None:
    data = _build_valid_manifest(num_dbs=2)
    # Both shards have db_id=0
    data["shards"][1]["db_id"] = 0
    payload = json.dumps(data).encode()
    with pytest.raises(ManifestParseError, match="coverage mismatch"):
        parse_json_manifest(payload)


def test_invalid_sharding_strategy() -> None:
    data = _build_valid_manifest(num_dbs=1)
    data["required"]["sharding"] = {"strategy": "nonexistent_strategy"}
    payload = json.dumps(data).encode()
    with pytest.raises(ManifestParseError):
        parse_json_manifest(payload)


def test_not_a_json_object() -> None:
    payload = json.dumps([1, 2, 3]).encode()
    with pytest.raises(ManifestParseError):
        parse_json_manifest(payload)
