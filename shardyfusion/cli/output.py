"""Output formatters for the reader CLI."""

import base64
import json
from typing import Any

from .config import OutputConfig

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
) -> dict[str, Any]:
    """Build a dict representing a `multiget` result."""
    results_list = []
    for key in keys:
        raw = values.get(key)
        if raw is None:
            results_list.append({"key": key, "found": False})
        else:
            results_list.append(
                {
                    "key": key,
                    "found": True,
                    "value": encode_value(raw, cfg.value_encoding),
                }
            )
    return {"op": "multiget", "results": results_list}


def build_refresh_result(changed: bool) -> dict[str, Any]:
    return {"op": "refresh", "changed": changed}


def build_info_result(reader: Any) -> dict[str, Any]:
    """Extract manifest metadata from a reader instance."""
    info = reader.snapshot_info()
    return {"op": "info", **info}


def build_error_result(op: str, key_hint: str | None, error: str) -> dict[str, Any]:
    result: dict[str, Any] = {"op": op, "error": error}
    if key_hint is not None:
        result["key"] = key_hint
    return result


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def format_result(result: dict[str, Any], fmt: str) -> str:
    """Render a result dict to a string in the requested format."""
    if fmt in ("jsonl", "json"):
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
        # Fall back to JSON for non-multiget ops in table mode
        return json.dumps(result, ensure_ascii=False, indent=2)

    # Unknown format: fall back to jsonl
    return json.dumps(result, ensure_ascii=False)


def emit(result: dict[str, Any], cfg: OutputConfig, file: Any = None) -> None:
    """Format and print a result; uses stdout when file is None."""
    import sys

    out = file or sys.stdout
    print(format_result(result, cfg.format), file=out)
