"""Tests for manifest and CURRENT pointer parse error paths."""

from __future__ import annotations

import pytest
import yaml

from shardyfusion.errors import ManifestParseError
from shardyfusion.manifest_store import parse_manifest


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
            "shard_prefix": "shards",
        },
        "shards": [
            {
                "db_id": i,
                "db_url": f"s3://bucket/prefix/shards/db={i:05d}",
                "attempt": 0,
                "row_count": 100,
                "writer_info": {},
            }
            for i in range(num_dbs)
        ],
    }


def _to_yaml(data: dict) -> bytes:
    return yaml.safe_dump(data, sort_keys=True).encode()


def test_valid_manifest_parses() -> None:
    data = _build_valid_manifest()
    payload = _to_yaml(data)
    result = parse_manifest(payload)
    assert result.required_build.num_dbs == 2
    assert len(result.shards) == 2


def test_truncated_yaml() -> None:
    payload = b"required:\n  run_id: test\n  - invalid"  # malformed YAML
    with pytest.raises(ManifestParseError):
        parse_manifest(payload)


def test_invalid_utf8() -> None:
    payload = b"\xff\xfe invalid bytes"
    with pytest.raises(ManifestParseError):
        parse_manifest(payload)


def test_empty_payload() -> None:
    with pytest.raises(ManifestParseError):
        parse_manifest(b"")


def test_missing_required_fields() -> None:
    payload = _to_yaml({"required": {"run_id": "test"}, "shards": []})
    with pytest.raises(ManifestParseError):
        parse_manifest(payload)


def test_shard_count_exceeds_num_dbs() -> None:
    data = _build_valid_manifest(num_dbs=2)
    # Add extra shards beyond num_dbs
    data["shards"].append(
        {
            "db_id": 2,
            "db_url": "s3://x",
            "attempt": 0,
            "row_count": 1,
            "writer_info": {},
        }
    )
    payload = _to_yaml(data)
    with pytest.raises(ManifestParseError, match="exceeds num_dbs"):
        parse_manifest(payload)


def test_sparse_shards_accepted() -> None:
    """Manifests with fewer shards than num_dbs are valid (empty shards omitted)."""
    data = _build_valid_manifest(num_dbs=8)
    # Only include 5 shards — the other 3 are implicitly empty
    data["shards"] = data["shards"][:5]
    payload = _to_yaml(data)
    result = parse_manifest(payload)
    assert len(result.shards) == 5


def test_out_of_range_shard_ids() -> None:
    data = _build_valid_manifest(num_dbs=3)
    # Make shard ID out of range: db_id=5 with num_dbs=3
    data["shards"][2]["db_id"] = 5
    payload = _to_yaml(data)
    with pytest.raises(ManifestParseError, match="out of range"):
        parse_manifest(payload)


def test_duplicate_shard_ids() -> None:
    data = _build_valid_manifest(num_dbs=2)
    # Both shards have db_id=0
    data["shards"][1]["db_id"] = 0
    payload = _to_yaml(data)
    with pytest.raises(ManifestParseError, match="duplicate"):
        parse_manifest(payload)


def test_invalid_sharding_strategy() -> None:
    data = _build_valid_manifest(num_dbs=1)
    data["required"]["sharding"] = {"strategy": "nonexistent_strategy"}
    payload = _to_yaml(data)
    with pytest.raises(ManifestParseError):
        parse_manifest(payload)


def test_routing_values_require_format_version_3() -> None:
    data = _build_valid_manifest(num_dbs=2)
    data["required"]["format_version"] = 2
    data["required"]["sharding"] = {
        "strategy": "cel",
        "cel_expr": "region",
        "cel_columns": {"region": "string"},
        "routing_values": ["ap", "eu"],
    }
    payload = _to_yaml(data)
    with pytest.raises(ManifestParseError, match="format_version >= 3"):
        parse_manifest(payload)


def test_manifest_rejects_removed_boundaries_field() -> None:
    data = _build_valid_manifest(num_dbs=3)
    data["required"]["sharding"] = {
        "strategy": "cel",
        "cel_expr": "region",
        "cel_columns": {"region": "string"},
        "boundaries": ["eu", "us"],
    }
    payload = _to_yaml(data)
    with pytest.raises(ManifestParseError, match="boundaries"):
        parse_manifest(payload)


def test_manifest_rejects_unsupported_categorical_token_types() -> None:
    data = _build_valid_manifest(num_dbs=2)
    data["required"]["format_version"] = 3
    data["required"]["sharding"] = {
        "strategy": "cel",
        "cel_expr": "region",
        "cel_columns": {"region": "string"},
        "routing_values": [1.5, 2.5],
    }
    payload = _to_yaml(data)
    with pytest.raises(ManifestParseError, match="routing_values"):
        parse_manifest(payload)


def test_categorical_manifest_num_dbs_must_match_routing_values() -> None:
    data = _build_valid_manifest(num_dbs=3)
    data["required"]["format_version"] = 3
    data["required"]["sharding"] = {
        "strategy": "cel",
        "cel_expr": "region",
        "cel_columns": {"region": "string"},
        "routing_values": ["ap", "eu"],
    }
    payload = _to_yaml(data)
    with pytest.raises(ManifestParseError, match="routing_values cardinality"):
        parse_manifest(payload)


def test_not_a_yaml_object() -> None:
    payload = yaml.safe_dump([1, 2, 3]).encode()
    with pytest.raises(ManifestParseError):
        parse_manifest(payload)
