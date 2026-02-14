from __future__ import annotations

from slatedb_spark_sharded.manifest import ManifestArtifact
from slatedb_spark_sharded.publish import DefaultS3Publisher


def test_default_s3_publisher_builds_expected_urls(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_create_s3_client(_cfg=None):
        return object()

    def fake_put_bytes(url, payload, content_type, headers=None, *, s3_client=None):
        calls.append(
            {
                "url": url,
                "payload": payload,
                "content_type": content_type,
                "headers": headers,
                "s3_client": s3_client,
            }
        )

    monkeypatch.setattr("slatedb_spark_sharded.publish.create_s3_client", fake_create_s3_client)
    monkeypatch.setattr("slatedb_spark_sharded.publish.put_bytes", fake_put_bytes)

    publisher = DefaultS3Publisher("s3://bucket/prefix")
    artifact = ManifestArtifact(payload=b"{}", content_type="application/json")

    manifest_ref = publisher.publish_manifest(name="manifest", artifact=artifact, run_id="run123")
    current_ref = publisher.publish_current(name="_CURRENT", artifact=artifact)

    assert manifest_ref == "s3://bucket/prefix/manifests/run_id=run123/manifest"
    assert current_ref == "s3://bucket/prefix/_CURRENT"
    assert calls[0]["url"] == manifest_ref
    assert calls[1]["url"] == current_ref
