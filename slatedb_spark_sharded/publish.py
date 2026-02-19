"""Manifest publishing protocols and default S3 implementation."""

from __future__ import annotations

from typing import Protocol

from .manifest import ManifestArtifact
from .storage import create_s3_client, put_bytes
from .type_defs import S3ClientConfig


class ManifestPublisher(Protocol):
    """Protocol for writing manifest and CURRENT pointer artifacts."""

    def publish_manifest(
        self, *, name: str, artifact: ManifestArtifact, run_id: str
    ) -> str:
        """Persist manifest and return a reference."""
        ...

    def publish_current(self, *, name: str, artifact: ManifestArtifact) -> str | None:
        """Persist CURRENT pointer and return a reference if applicable."""
        ...


class DefaultS3Publisher:
    """Default publisher storing artifacts under a configured S3 prefix."""

    def __init__(
        self,
        s3_prefix: str,
        *,
        manifest_name: str = "manifest",
        current_name: str = "_CURRENT",
        s3_client_config: S3ClientConfig | None = None,
    ) -> None:
        self.s3_prefix = s3_prefix.rstrip("/")
        self.manifest_name = manifest_name
        self.current_name = current_name
        self._s3_client = create_s3_client(s3_client_config)

    def publish_manifest(
        self, *, name: str, artifact: ManifestArtifact, run_id: str
    ) -> str:
        manifest_name = name or self.manifest_name
        manifest_url = _join_s3(
            self.s3_prefix,
            "manifests",
            f"run_id={run_id}",
            manifest_name,
        )
        put_bytes(
            manifest_url,
            artifact.payload,
            artifact.content_type,
            artifact.headers,
            s3_client=self._s3_client,
        )
        return manifest_url

    def publish_current(self, *, name: str, artifact: ManifestArtifact) -> str | None:
        current_name = name or self.current_name
        current_url = _join_s3(self.s3_prefix, current_name)
        put_bytes(
            current_url,
            artifact.payload,
            artifact.content_type,
            artifact.headers,
            s3_client=self._s3_client,
        )
        return current_url


def _join_s3(base: str, *parts: str) -> str:
    clean = [base.rstrip("/")]
    clean.extend(part.strip("/") for part in parts if part)
    return "/".join(clean)
