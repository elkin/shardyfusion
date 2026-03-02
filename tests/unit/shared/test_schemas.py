"""Tests for shardyfusion.schemas package resource helpers."""

from __future__ import annotations

from shardyfusion.schemas import load_current_pointer_schema, load_manifest_schema


class TestLoadManifestSchema:
    def test_returns_dict(self) -> None:
        schema = load_manifest_schema()
        assert isinstance(schema, dict)

    def test_has_json_schema_fields(self) -> None:
        schema = load_manifest_schema()
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["title"] == "SlateDB Sharded Manifest"
        assert "properties" in schema

    def test_has_required_section(self) -> None:
        schema = load_manifest_schema()
        props = schema["properties"]
        assert "required" in props
        assert "shards" in props


class TestLoadCurrentPointerSchema:
    def test_returns_dict(self) -> None:
        schema = load_current_pointer_schema()
        assert isinstance(schema, dict)

    def test_has_json_schema_fields(self) -> None:
        schema = load_current_pointer_schema()
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["title"] == "SlateDB Sharded CURRENT Pointer"
        assert "properties" in schema

    def test_has_manifest_ref_property(self) -> None:
        schema = load_current_pointer_schema()
        assert "manifest_ref" in schema["properties"]
