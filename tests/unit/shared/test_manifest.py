from __future__ import annotations

from shardyfusion.manifest import (
    ParsedManifest,
)

_REQUIRED_BUILD_DATA = {
    "run_id": "r1",
    "created_at": "2026-01-01T00:00:00+00:00",
    "num_dbs": 1,
    "s3_prefix": "s3://b/p",
    "key_col": "id",
    "sharding": {"strategy": "hash", "hash_algorithm": "xxh3_64"},
    "db_path_template": "db={db_id:05d}",
    "shard_prefix": "shards",
}

_SHARD_DATA = [
    {"db_id": 0, "db_url": "s3://b/p/db=00000", "attempt": 0, "row_count": 1}
]


def test_parsed_manifest_accepts_wire_key() -> None:
    """ParsedManifest accepts 'required' (the wire name) via validation_alias."""
    parsed = ParsedManifest.model_validate(
        {"required": _REQUIRED_BUILD_DATA, "shards": _SHARD_DATA}
    )
    assert parsed.required_build.run_id == "r1"


def test_parsed_manifest_accepts_python_field_name() -> None:
    """ParsedManifest still accepts 'required_build' via populate_by_name."""
    parsed = ParsedManifest.model_validate(
        {"required_build": _REQUIRED_BUILD_DATA, "shards": _SHARD_DATA}
    )
    assert parsed.required_build.run_id == "r1"


def test_parsed_manifest_schema_uses_wire_name() -> None:
    """model_json_schema(mode='serialization') emits 'required' as property."""
    schema = ParsedManifest.model_json_schema(mode="serialization")
    assert "required" in schema["properties"]
    assert "required_build" not in schema["properties"]
