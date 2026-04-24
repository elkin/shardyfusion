"""Output formatters for the reader CLI."""

from __future__ import annotations

import base64
import dataclasses
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .config import OutputConfig

if TYPE_CHECKING:
    from .._writer_core import CleanupAction
    from ..manifest import ManifestRef
    from ..reader import (
        ConcurrentShardedReader,
        ReaderHealth,
        ShardDetail,
        ShardedReader,
    )

# ---------------------------------------------------------------------------
# Value encoding
# ---------------------------------------------------------------------------


def encode_value(value: bytes, encoding: str) -> str:
    """Encode raw bytes to a string using the configured encoding."""
    if encoding == "base64":
        return base64.b64encode(value).decode("ascii")
    if encoding == "hex":
        return value.hex()
    if encoding == "utf8":
        return value.decode("utf-8", errors="replace")
    return base64.b64encode(value).decode("ascii")


# ---------------------------------------------------------------------------
# Result builders
# ---------------------------------------------------------------------------


def build_get_result(
    key: str,
    value: bytes | None,
    cfg: OutputConfig,
) -> dict[str, Any]:
    """Build a dict representing a single `get` result."""
    result: dict[str, Any] = {"op": "get", "key": key}
    if value is None:
        result["found"] = False
        result["value"] = cfg.null_repr
    else:
        result["found"] = True
        result["value"] = encode_value(value, cfg.value_encoding)
    return result


def build_multiget_result(
    keys: list[str],
    values: dict[Any, bytes | None],
    cfg: OutputConfig,
    *,
    coerced_keys: list[Any] | None = None,
) -> dict[str, Any]:
    """Build a dict representing a `multiget` result.

    When *coerced_keys* is provided, values are looked up by coerced key
    (e.g. ``int``) while the original string key is used for display.
    """
    lookup_keys = coerced_keys if coerced_keys is not None else keys
    results_list = []
    for display_key, lookup_key in zip(keys, lookup_keys, strict=True):
        raw = values.get(lookup_key)
        if raw is None:
            results_list.append({"key": display_key, "found": False})
        else:
            results_list.append(
                {
                    "key": display_key,
                    "found": True,
                    "value": encode_value(raw, cfg.value_encoding),
                }
            )
    return {"op": "multiget", "results": results_list}


def build_refresh_result(changed: bool) -> dict[str, Any]:
    return {"op": "refresh", "changed": changed}


def build_info_result(
    reader: ShardedReader | ConcurrentShardedReader,
) -> dict[str, Any]:
    """Extract manifest metadata from a reader instance."""
    info = reader.snapshot_info()
    d = dataclasses.asdict(info)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return {"op": "info", **d}


def build_shards_result(shards: list[ShardDetail]) -> dict[str, Any]:
    """Build a dict representing per-shard details."""
    return {"op": "shards", "shards": [dataclasses.asdict(s) for s in shards]}


def build_route_result(key: str, db_id: int) -> dict[str, Any]:
    """Build a dict representing a route lookup result."""
    return {"op": "route", "key": key, "db_id": db_id}


def build_history_result(refs: list[ManifestRef]) -> dict[str, Any]:
    """Build a dict representing manifest history listing."""
    entries = []
    for i, ref in enumerate(refs):
        entries.append(
            {
                "offset": i,
                "run_id": ref.run_id,
                "published_at": ref.published_at.isoformat(),
                "ref": ref.ref,
            }
        )
    return {"op": "history", "manifests": entries}


def build_cleanup_result(
    actions: list[CleanupAction],
    *,
    dry_run: bool,
    run_id: str,
) -> dict[str, Any]:
    """Build a dict representing cleanup results."""
    stale = []
    old_runs = []
    total_objects = 0
    total_prefixes = 0

    for action in actions:
        total_objects += action.objects_deleted
        total_prefixes += 1
        if action.kind == "stale_attempt":
            stale.append(
                {
                    "db_id": action.db_id,
                    "prefix_url": action.prefix_url,
                    "objects_deleted": action.objects_deleted,
                }
            )
        elif action.kind == "old_run":
            old_runs.append(
                {
                    "run_id": action.run_id,
                    "prefix_url": action.prefix_url,
                    "objects_deleted": action.objects_deleted,
                }
            )

    result: dict[str, Any] = {
        "op": "cleanup",
        "dry_run": dry_run,
        "run_id": run_id,
        "stale_attempts": stale,
        "total_objects_deleted": total_objects,
        "total_prefixes_removed": total_prefixes,
    }
    if old_runs:
        result["old_runs"] = old_runs
    return result


def build_health_result(health: ReaderHealth) -> dict[str, Any]:
    """Build a dict representing reader health diagnostics."""
    return {
        "op": "health",
        "status": health.status,
        "manifest_ref": health.manifest_ref,
        "manifest_age_seconds": round(health.manifest_age.total_seconds(), 1),
        "num_shards": health.num_shards,
        "is_closed": health.is_closed,
    }


def build_error_result(op: str, key_hint: str | None, error: str) -> dict[str, Any]:
    result: dict[str, Any] = {"op": op, "error": error}
    if key_hint is not None:
        result["key"] = key_hint
    return result


def build_search_result(
    response: Any,
    top_k: int,
) -> dict[str, Any]:
    """Build a dict representing a vector search result."""
    return {
        "op": "search",
        "top_k": top_k,
        "num_shards_queried": response.num_shards_queried,
        "latency_ms": round(response.latency_ms, 2),
        "results": [
            {
                "id": r.id,
                "score": round(float(r.score), 6),
                "payload": r.payload,
            }
            for r in response.results
        ],
    }


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def format_result(result: dict[str, Any], fmt: str) -> str:
    """Render a result dict to a string in the requested format."""
    if fmt == "json":
        return json.dumps(result, ensure_ascii=False, indent=2)

    if fmt == "jsonl":
        return json.dumps(result, ensure_ascii=False)

    if fmt == "text":
        op = result.get("op", "?")
        if op == "get":
            found = result.get("found", False)
            key = result.get("key", "")
            value = result.get("value", "")
            return f"{key}={'null' if not found else value}"
        if op == "multiget":
            lines = []
            for item in result.get("results", []):
                k = item.get("key", "")
                v = item.get("value", "null") if item.get("found") else "null"
                lines.append(f"{k}={v}")
            return "\n".join(lines)
        if op == "refresh":
            return f"changed={result.get('changed', False)}"
        if op == "info":
            return "\n".join(f"{k}={v}" for k, v in result.items() if k != "op")
        if op == "shards":
            lines = []
            for s in result.get("shards", []):
                parts = [f"db_id={s['db_id']}", f"rows={s['row_count']}"]
                if s.get("min_key") is not None:
                    parts.append(f"min={s['min_key']}")
                if s.get("max_key") is not None:
                    parts.append(f"max={s['max_key']}")
                lines.append("  ".join(parts))
            return "\n".join(lines)
        if op == "history":
            lines = []
            for m in result.get("manifests", []):
                lines.append(
                    f"[{m['offset']}] {m['published_at']}  run_id={m['run_id']}"
                )
            return "\n".join(lines) if lines else "(no manifests)"
        if op == "route":
            return f"{result.get('key', '')} -> shard {result.get('db_id', '?')}"
        if op == "search":
            lines = [
                f"top_k={result.get('top_k', '?')}  "
                f"shards={result.get('num_shards_queried', 0)}  "
                f"latency_ms={result.get('latency_ms', 0)}"
            ]
            for r in result.get("results", []):
                payload = r.get("payload")
                payload_str = json.dumps(payload) if payload is not None else ""
                lines.append(
                    f"id={r.get('id', '?')}  score={r.get('score', 0):.6f}  {payload_str}"
                )
            return "\n".join(lines)
        if op == "health":
            parts = [f"status={result.get('status', '?')}"]
            parts.append(f"manifest_age={result.get('manifest_age_seconds', 0)}s")
            parts.append(f"shards={result.get('num_shards', 0)}")
            if result.get("is_closed"):
                parts.append("CLOSED")
            return "  ".join(parts)
        if op == "cleanup":
            dry = "[DRY RUN] " if result.get("dry_run") else ""
            lines = [f"{dry}Cleanup for run_id={result.get('run_id', '?')}"]
            for s in result.get("stale_attempts", []):
                lines.append(
                    f"  stale attempt  db_id={s['db_id']}  "
                    f"objects={s['objects_deleted']}  {s['prefix_url']}"
                )
            for r in result.get("old_runs", []):
                lines.append(
                    f"  old run  run_id={r['run_id']}  "
                    f"objects={r['objects_deleted']}  {r['prefix_url']}"
                )
            lines.append(
                f"Total: {result.get('total_prefixes_removed', 0)} prefixes, "
                f"{result.get('total_objects_deleted', 0)} objects"
            )
            return "\n".join(lines)
        if "error" in result:
            return f"error: {result['error']}"
        return json.dumps(result, ensure_ascii=False)

    if fmt == "table":
        op = result.get("op", "?")
        if op == "multiget":
            rows = result.get("results", [])
            col_key = max((len(r.get("key", "")) for r in rows), default=3)
            col_key = max(col_key, 3)
            col_val = max(
                (len(r.get("value", "")) if r.get("found") else 4 for r in rows),
                default=5,
            )
            col_val = max(col_val, 5)
            header = f"{'KEY':<{col_key}}  {'VALUE':<{col_val}}"
            sep = "-" * len(header)
            lines = [header, sep]
            for row in rows:
                k = row.get("key", "")
                v = row.get("value", "null") if row.get("found") else "null"
                lines.append(f"{k:<{col_key}}  {v:<{col_val}}")
            return "\n".join(lines)
        if op == "shards":
            shards = result.get("shards", [])
            header = f"{'DB_ID':>5}  {'ROWS':>8}  {'MIN_KEY':>10}  {'MAX_KEY':>10}  URL"
            sep = "-" * len(header)
            lines = [header, sep]
            for s in shards:
                min_k = str(s.get("min_key") or "")
                max_k = str(s.get("max_key") or "")
                lines.append(
                    f"{s['db_id']:>5}  {s['row_count']:>8}  {min_k:>10}  {max_k:>10}  {s['db_url']}"
                )
            return "\n".join(lines)
        if op == "history":
            manifests = result.get("manifests", [])
            header = f"{'#':>3}  {'PUBLISHED_AT':>26}  {'RUN_ID'}"
            sep = "-" * len(header)
            lines = [header, sep]
            for m in manifests:
                lines.append(
                    f"{m['offset']:>3}  {m['published_at']:>26}  {m['run_id']}"
                )
            return "\n".join(lines)
        if op == "search":
            rows = result.get("results", [])
            header = f"{'ID':<20}  {'SCORE':>12}  PAYLOAD"
            sep = "-" * len(header)
            lines = [header, sep]
            for r in rows:
                payload = r.get("payload")
                payload_str = json.dumps(payload) if payload is not None else ""
                lines.append(
                    f"{str(r.get('id', '?')):<20}  {r.get('score', 0):>12.6f}  {payload_str}"
                )
            return "\n".join(lines)
        if op == "health":
            # Reuse text format for table mode — health is a single record
            return format_result(result, "text")
        if op == "cleanup":
            # Reuse text format for table mode — cleanup data is hierarchical, not tabular
            return format_result(result, "text")
        # Fall back to JSON for other ops in table mode
        return json.dumps(result, ensure_ascii=False, indent=2)

    # Unknown format: fall back to jsonl
    return json.dumps(result, ensure_ascii=False)


def emit(result: dict[str, Any], cfg: OutputConfig, file: Any = None) -> None:
    """Format and print a result; uses stdout when file is None."""
    import sys

    out = file or sys.stdout
    print(format_result(result, cfg.format), file=out)
