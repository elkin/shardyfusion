from __future__ import annotations

import json

import pytest
import yaml

from shardyfusion.errors import ManifestParseError
from shardyfusion.manifest_store import (
    S3ManifestStore,
    parse_manifest,
)


def _to_yaml(data: dict) -> bytes:
    return yaml.safe_dump(data, sort_keys=True).encode()


def test_parse_manifest_round_trip() -> None:
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
            "shard_prefix": "shards",
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

    parsed = parse_manifest(_to_yaml(payload))

    assert parsed.required_build.num_dbs == 2
    assert [shard.db_id for shard in parsed.shards] == [0, 1]
    assert parsed.custom == {"env": "test"}


def test_parse_manifest_rejects_bad_shard_coverage() -> None:
    payload = {
        "required": {
            "run_id": "run-1",
            "created_at": "2026-01-01T00:00:00+00:00",
            "num_dbs": 1,
            "s3_prefix": "s3://bucket/prefix",
            "key_col": "id",
            "key_encoding": "u64be",
            "sharding": {"strategy": "hash"},
            "db_path_template": "db={db_id:05d}",
            "shard_prefix": "shards",
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
        "custom": {},
    }

    with pytest.raises(ManifestParseError, match="exceeds num_dbs"):
        parse_manifest(_to_yaml(payload))


def test_parse_manifest_rejects_corrupt_yaml() -> None:
    with pytest.raises(ManifestParseError, match="Manifest validation failed"):
        parse_manifest(b"not-yaml{{{")


def test_parse_manifest_rejects_missing_required_field() -> None:
    payload = _to_yaml({"shards": [], "custom": {}})
    with pytest.raises(ManifestParseError, match="Manifest validation failed"):
        parse_manifest(payload)


def test_load_current_rejects_corrupt_json(monkeypatch) -> None:
    def fake_try_get_bytes(url, *, s3_client=None, metrics_collector=None, **kwargs):
        return b"not-json{{{"

    monkeypatch.setattr("shardyfusion.manifest_store.try_get_bytes", fake_try_get_bytes)
    store = S3ManifestStore("s3://bucket/prefix")
    with pytest.raises(ManifestParseError, match="validation failed"):
        store.load_current()


def test_load_current_rejects_missing_manifest_ref(monkeypatch) -> None:
    def fake_try_get_bytes(url, *, s3_client=None, metrics_collector=None, **kwargs):
        return json.dumps(
            {
                "manifest_content_type": "application/x-yaml",
                "run_id": "run-1",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        ).encode("utf-8")

    monkeypatch.setattr("shardyfusion.manifest_store.try_get_bytes", fake_try_get_bytes)
    store = S3ManifestStore("s3://bucket/prefix")
    with pytest.raises(ManifestParseError, match="manifest_ref"):
        store.load_current()
