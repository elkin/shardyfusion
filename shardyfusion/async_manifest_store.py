"""Async manifest store protocol and implementations.

``AsyncManifestStore`` is the read-only async counterpart of
``ManifestStore``.  ``AsyncS3ManifestStore`` uses obstore's native async
S3 I/O (no aiobotocore required).
"""

from __future__ import annotations

from typing import Protocol

from .logging import FailureSeverity, get_logger, log_failure
from .manifest import ManifestRef, ParsedManifest
from .metrics import MetricsCollector
from .storage import AsyncStorageBackend, join_s3

_logger = get_logger(__name__)


class AsyncManifestStore(Protocol):
    """Read-only async manifest loading (no publish — that's writer-side)."""

    async def load_current(self) -> ManifestRef | None: ...

    async def load_manifest(self, ref: str) -> ParsedManifest: ...

    async def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]: ...


class AsyncS3ManifestStore:
    """Native async S3 manifest store using obstore.

    Each operation delegates to the injected :class:`AsyncStorageBackend`.
    obstore's Rust retry layer handles transient S3 errors internally.
    """

    def __init__(
        self,
        backend: AsyncStorageBackend,
        s3_prefix: str,
        *,
        manifest_name: str = "manifest",
        current_pointer_key: str = "_CURRENT",
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        self._backend = backend
        self.s3_prefix = s3_prefix.rstrip("/")
        self.manifest_name = manifest_name
        self.current_pointer_key = current_pointer_key
        self._metrics = metrics_collector

    async def load_current(self) -> ManifestRef | None:
        from .manifest_store import parse_current_pointer_to_ref

        current_url = f"{self.s3_prefix}/{self.current_pointer_key}"
        payload = await self._backend.try_get(current_url)
        if payload is None:
            return None
        return parse_current_pointer_to_ref(payload)

    async def load_manifest(self, ref: str) -> ParsedManifest:
        try:
            payload = await self._backend.get(ref)
        except Exception as exc:
            log_failure(
                "manifest_s3_load_failed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=exc,
                manifest_ref=ref,
            )
            raise
        from .manifest_store import parse_manifest_payload

        return parse_manifest_payload(payload)

    async def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        from .manifest_store import parse_manifest_dir_entry

        manifests_prefix = join_s3(self.s3_prefix, "manifests") + "/"

        refs: list[ManifestRef] = []
        prefix_urls = await self._backend.list_prefixes(manifests_prefix)
        for url in prefix_urls:
            dir_name = url[len(manifests_prefix) :]
            ref = parse_manifest_dir_entry(dir_name, self.s3_prefix, self.manifest_name)
            if ref is not None:
                refs.append(ref)

        refs.sort(key=lambda r: r.published_at, reverse=True)
        return refs[:limit]
