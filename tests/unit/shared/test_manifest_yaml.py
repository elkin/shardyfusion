"""Tests for YAML manifest serialization and parsing."""

from datetime import UTC, datetime

import yaml

from shardyfusion.manifest import (
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
    YamlManifestBuilder,
)
from shardyfusion.manifest_store import parse_manifest
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy


class TestYamlManifestBuilder:
    def _make_required_build(self, **overrides):
        defaults = dict(
            run_id="test-run",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            num_dbs=4,
            s3_prefix="s3://bucket/prefix",
            key_col="_key",
            sharding=ManifestShardingSpec(strategy=ShardingStrategy.HASH),
            db_path_template="db={db_id:05d}",
            shard_prefix="shards",
            key_encoding=KeyEncoding.U64BE,
        )
        defaults.update(overrides)
        return RequiredBuildMeta(**defaults)

    def _make_shard(self, db_id=0, **overrides):
        defaults = dict(
            db_id=db_id,
            db_url=f"s3://bucket/prefix/shards/db={db_id:05d}",
            attempt=0,
            row_count=100,
        )
        defaults.update(overrides)
        return RequiredShardMeta(**defaults)

    def test_build_produces_yaml(self) -> None:
        builder = YamlManifestBuilder()
        rb = self._make_required_build()
        shards = [self._make_shard(0)]
        artifact = builder.build(required_build=rb, shards=shards, custom_fields={})
        assert artifact.content_type == "application/x-yaml"

        # Should parse as YAML
        parsed = yaml.safe_load(artifact.payload)
        assert "required" in parsed
        assert "shards" in parsed
        assert parsed["required"]["run_id"] == "test-run"

    def test_roundtrip_via_parse_manifest(self) -> None:
        builder = YamlManifestBuilder()
        rb = self._make_required_build()
        shards = [self._make_shard(i) for i in range(4)]
        artifact = builder.build(
            required_build=rb, shards=shards, custom_fields={"my_field": "val"}
        )

        parsed = parse_manifest(artifact.payload)
        assert parsed.required_build.run_id == "test-run"
        assert parsed.required_build.num_dbs == 4
        assert len(parsed.shards) == 4
        assert parsed.custom["my_field"] == "val"

    def test_cel_fields_in_manifest(self) -> None:
        builder = YamlManifestBuilder()
        sharding = ManifestShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr="key % 1000",
            cel_columns={"key": "int"},
            boundaries=[250, 500, 750],
        )
        rb = self._make_required_build(sharding=sharding)
        shards = [self._make_shard(i) for i in range(4)]
        artifact = builder.build(required_build=rb, shards=shards, custom_fields={})

        parsed = parse_manifest(artifact.payload)
        assert parsed.required_build.sharding.strategy == ShardingStrategy.CEL
        assert parsed.required_build.sharding.cel_expr == "key % 1000"
        assert parsed.required_build.sharding.cel_columns == {"key": "int"}
        assert parsed.required_build.sharding.boundaries == [250, 500, 750]

    def test_custom_fields_merged(self) -> None:
        builder = YamlManifestBuilder()
        builder.add_custom_field("builder_field", "from_builder")
        rb = self._make_required_build()
        artifact = builder.build(
            required_build=rb,
            shards=[self._make_shard(0)],
            custom_fields={"call_field": "from_call"},
        )

        parsed = parse_manifest(artifact.payload)
        assert parsed.custom["builder_field"] == "from_builder"
        assert parsed.custom["call_field"] == "from_call"

    def test_sorted_keys(self) -> None:
        builder = YamlManifestBuilder()
        rb = self._make_required_build()
        artifact = builder.build(required_build=rb, shards=[], custom_fields={})
        data = yaml.safe_load(artifact.payload)
        keys = list(data.keys())
        assert keys == sorted(keys)


class TestParseManifestRejectsInvalid:
    """Verify parse_manifest rejects non-YAML payloads."""

    def test_rejects_binary_garbage(self) -> None:
        import pytest

        from shardyfusion.errors import ManifestParseError

        with pytest.raises(ManifestParseError):
            parse_manifest(b"\xff\xfe\x00\x01")
