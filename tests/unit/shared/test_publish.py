from __future__ import annotations

from shardyfusion.manifest import (
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from shardyfusion.manifest_store import S3ManifestStore
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy


def test_s3_manifest_store_builds_expected_urls(monkeypatch) -> None:
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
        "shardyfusion.manifest_store.create_s3_client", fake_create_s3_client
    )
    monkeypatch.setattr("shardyfusion.manifest_store.put_bytes", fake_put_bytes)

    store = S3ManifestStore("s3://bucket/prefix")

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
    # Two put_bytes calls: manifest + CURRENT
    assert len(calls) == 2
    assert calls[0]["url"] == manifest_ref
    assert calls[1]["url"] == "s3://bucket/prefix/_CURRENT"
