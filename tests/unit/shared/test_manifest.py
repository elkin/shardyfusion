from __future__ import annotations

import json

from slatedb_spark_sharded.manifest import (
    JsonManifestBuilder,
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from slatedb_spark_sharded.sharding import ShardingStrategy


def test_json_manifest_builder_includes_required_shards_and_custom() -> None:
    builder = JsonManifestBuilder()
    builder.add_custom_field("source", "unit-test")

    required = RequiredBuildMeta(
        run_id="r1",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=2,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        sharding=ManifestShardingSpec(strategy=ShardingStrategy.HASH),
        db_path_template="db={db_id:05d}",
        tmp_prefix="_tmp",
    )
    shards = [
        RequiredShardMeta(
            db_id=0,
            db_url="s3://bucket/prefix/db=00000",
            attempt=0,
            row_count=3,
            min_key=1,
            max_key=3,
            checkpoint_id=None,
            writer_info={},
        )
    ]

    artifact = builder.build(
        required_build=required,
        shards=shards,
        custom_fields={"env": "test"},
    )

    payload = json.loads(artifact.payload.decode("utf-8"))
    assert artifact.content_type == "application/json"
    assert set(payload.keys()) == {"required", "shards", "custom"}
    assert payload["required"]["run_id"] == "r1"
    assert payload["custom"] == {"source": "unit-test", "env": "test"}
