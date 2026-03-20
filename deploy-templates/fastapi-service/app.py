"""FastAPI service for serving lookups from a shardyfusion snapshot.

Provides single-key and batch lookups, health checks, Prometheus metrics,
and background refresh.  Uses AsyncShardedReader for native asyncio support.

Configure via environment variables:
    S3_PREFIX            s3://bucket/prefix (required)
    LOCAL_ROOT           /tmp/shardyfusion (default)
    MAX_READS_PER_SEC    0 = unlimited (default)
    REFRESH_INTERVAL_S   300 (default, 0 = disable)
    STALENESS_THRESHOLD_S  3600 (default, for health check degradation)

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from shardyfusion import AsyncShardedReader, TokenBucket

logger = logging.getLogger("shardyfusion-service")

# ── Config ───────────────────────────────────────────────────────────────────

S3_PREFIX = os.environ.get("S3_PREFIX", "")
LOCAL_ROOT = os.environ.get("LOCAL_ROOT", "/tmp/shardyfusion")
MAX_READS_PER_SEC = float(os.environ.get("MAX_READS_PER_SEC", "0"))
REFRESH_INTERVAL_S = float(os.environ.get("REFRESH_INTERVAL_S", "300"))
STALENESS_THRESHOLD_S = float(os.environ.get("STALENESS_THRESHOLD_S", "3600"))

# ── State ────────────────────────────────────────────────────────────────────

reader: AsyncShardedReader | None = None
_refresh_task: asyncio.Task[None] | None = None
_prometheus_registry: Any = None


# ── Background refresh ──────────────────────────────────────────────────────


async def _periodic_refresh(interval: float) -> None:
    """Periodically refresh the reader to pick up new snapshots."""
    while True:
        await asyncio.sleep(interval)
        if reader is not None:
            try:
                changed = await reader.refresh()
                if changed:
                    logger.info("Snapshot refreshed to latest version")
            except Exception:
                logger.exception("Background refresh failed")


# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    global reader, _refresh_task, _prometheus_registry

    if not S3_PREFIX:
        raise RuntimeError("S3_PREFIX environment variable is required")

    # Optional Prometheus metrics
    collector = None
    try:
        from prometheus_client import CollectorRegistry

        from shardyfusion.metrics.prometheus import PrometheusCollector

        _prometheus_registry = CollectorRegistry()
        collector = PrometheusCollector(registry=_prometheus_registry)
        logger.info("Prometheus metrics enabled")
    except ImportError:
        logger.info("Prometheus not available (install shardyfusion[metrics-prometheus])")

    # Optional rate limiter
    rate_limiter = TokenBucket(rate=MAX_READS_PER_SEC) if MAX_READS_PER_SEC > 0 else None

    reader = await AsyncShardedReader.open(
        s3_prefix=S3_PREFIX,
        local_root=LOCAL_ROOT,
        metrics_collector=collector,
        rate_limiter=rate_limiter,
    )
    logger.info("Reader opened: %s", S3_PREFIX)

    # Start background refresh
    if REFRESH_INTERVAL_S > 0:
        _refresh_task = asyncio.create_task(_periodic_refresh(REFRESH_INTERVAL_S))
        logger.info("Background refresh every %.0fs", REFRESH_INTERVAL_S)

    yield

    # Shutdown
    if _refresh_task is not None:
        _refresh_task.cancel()
        try:
            await _refresh_task
        except asyncio.CancelledError:
            pass

    if reader is not None:
        await reader.close()
        logger.info("Reader closed")


app = FastAPI(title="shardyfusion-service", lifespan=lifespan)


# ── Request / Response models ────────────────────────────────────────────────


class MultiGetRequest(BaseModel):
    keys: list[str | int]


class MultiGetResponse(BaseModel):
    results: dict[str, str | None]


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/api/v1/get/{key}")
async def get_key(key: str) -> dict[str, str | None]:
    """Look up a single key.  Returns base64-encoded value."""
    if reader is None:
        raise HTTPException(status_code=503, detail="Reader not initialized")

    # Auto-coerce to int if the key looks numeric
    lookup_key: str | int = int(key) if key.lstrip("-").isdigit() else key

    value = await reader.get(lookup_key)
    encoded = base64.b64encode(value).decode("ascii") if value is not None else None
    return {"key": key, "value": encoded}


@app.post("/api/v1/multi-get")
async def multi_get_keys(req: MultiGetRequest) -> MultiGetResponse:
    """Batch lookup.  Returns base64-encoded values."""
    if reader is None:
        raise HTTPException(status_code=503, detail="Reader not initialized")

    results = await reader.multi_get(req.keys)
    encoded = {
        str(k): base64.b64encode(v).decode("ascii") if v is not None else None
        for k, v in results.items()
    }
    return MultiGetResponse(results=encoded)


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check — returns reader status."""
    if reader is None:
        return {"status": "unhealthy", "reason": "Reader not initialized"}

    h = reader.health(staleness_threshold_s=STALENESS_THRESHOLD_S)
    return {
        "status": h.status,
        "manifest_ref": h.manifest_ref,
        "manifest_age_seconds": h.manifest_age_seconds,
        "num_shards": h.num_shards,
        "is_closed": h.is_closed,
    }


@app.get("/info")
async def snapshot_info() -> dict[str, Any]:
    """Current snapshot metadata."""
    if reader is None:
        raise HTTPException(status_code=503, detail="Reader not initialized")

    info = reader.snapshot_info()
    return {
        "run_id": info.run_id,
        "num_shards": info.num_dbs,
        "row_count": info.row_count,
        "key_encoding": str(info.key_encoding),
        "sharding": str(info.sharding),
        "manifest_ref": info.manifest_ref,
    }


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    if _prometheus_registry is None:
        raise HTTPException(
            status_code=503,
            detail="Prometheus not configured (install shardyfusion[metrics-prometheus])",
        )
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(
        content=generate_latest(_prometheus_registry),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/admin/refresh")
async def manual_refresh() -> dict[str, bool]:
    """Manually trigger a snapshot refresh."""
    if reader is None:
        raise HTTPException(status_code=503, detail="Reader not initialized")

    changed = await reader.refresh()
    return {"changed": changed}
