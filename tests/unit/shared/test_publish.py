from __future__ import annotations

from shardyfusion.manifest import (
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from shardyfusion.manifest_store import S3ManifestStore
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy
from shardyfusion.storage import MemoryBackend


def test_s3_manifest_store_builds_expected_urls() -> None:
    backend = MemoryBackend()
    store = S3ManifestStore(backend, "s3://bucket/prefix")

    required_build = RequiredBuildMeta(
        run_id="run123",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=1,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding=KeyEncoding.U64BE,
        sharding=ManifestShardingSpec(
            strategy=ShardingStrategy.HASH,
            hash_algorithm="xxh3_64",
        ),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
    )
    shards = [
        RequiredShardMeta(
            db_id=0,
            db_url="s3://bucket/prefix/db=00000",
            attempt=0,
            row_count=10,
            checkpoint_id=None,
            writer_info={},
            db_bytes=0,
        ),
    ]

    manifest_ref = store.publish(
        run_id="run123",
        required_build=required_build,
        shards=shards,
        custom={},
    )

    # Timestamp-prefixed path: manifests/{timestamp}_run_id=run123/manifest
    assert "run_id=run123/manifest" in manifest_ref
    assert manifest_ref.startswith("s3://bucket/prefix/manifests/")
    # Two puts: manifest + CURRENT
    assert backend.try_get(manifest_ref) is not None
    assert backend.try_get("s3://bucket/prefix/_CURRENT") is not None
