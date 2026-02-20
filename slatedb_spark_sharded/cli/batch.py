"""Batch / script execution for the slate-reader CLI."""

from __future__ import annotations

import sys
from typing import IO, Any

from .config import OutputConfig
from .output import (
    build_error_result,
    build_get_result,
    build_info_result,
    build_multiget_result,
    build_refresh_result,
    emit,
)


def load_script(script_path: str) -> dict[str, Any]:
    """Parse a YAML script file and return its content as a dict."""
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "pyyaml is required for batch script execution. "
            "Install it with: pip install 'slatedb_spark_sharded[cli]'"
        ) from exc

    with open(script_path) as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"Script file must be a YAML mapping, got: {type(data)!r}")

    commands = data.get("commands")
    if not isinstance(commands, list):
        raise ValueError(
            "Script file must have a 'commands' key with a list of operations"
        )

    return data


def run_script(
    reader: Any,
    script_path: str,
    output_cfg: OutputConfig,
    output_file: IO[str] | None = None,
) -> int:
    """Execute all commands in a YAML script file.

    Returns the number of errors encountered.
    """
    out = output_file or sys.stdout

    # Batch mode uses jsonl by default
    batch_cfg = OutputConfig(
        format=output_cfg.format if output_cfg.format != "json" else "jsonl",
        value_encoding=output_cfg.value_encoding,
        null_repr=output_cfg.null_repr,
    )

    data = load_script(script_path)
    commands: list[Any] = data.get("commands", [])
    on_error: str = data.get("on_error", "stop")

    error_count = 0

    for i, cmd in enumerate(commands):
        if not isinstance(cmd, dict):
            result = build_error_result(
                "unknown", None, f"Command #{i} is not a mapping"
            )
            emit(result, batch_cfg, file=out)
            error_count += 1
            if on_error == "stop":
                break
            continue

        op = cmd.get("op")

        try:
            result = _execute_command(reader, op, cmd, batch_cfg)
        except Exception as exc:
            key_hint = cmd.get("key") or (
                str(cmd.get("keys", "")) if cmd.get("keys") else None
            )
            result = build_error_result(str(op), key_hint, str(exc))
            error_count += 1
            emit(result, batch_cfg, file=out)
            if on_error == "stop":
                break
            continue

        emit(result, batch_cfg, file=out)

    return error_count


def _execute_command(
    reader: Any,
    op: str | None,
    cmd: dict[str, Any],
    cfg: OutputConfig,
) -> dict[str, Any]:
    """Dispatch one script command to the appropriate reader method."""
    if op == "get":
        key = cmd.get("key")
        if key is None:
            raise ValueError("'get' command requires a 'key' field")
        key = str(key)
        value = reader.get(key)
        return build_get_result(key, value, cfg)

    if op == "multiget":
        keys_raw = cmd.get("keys")
        if not isinstance(keys_raw, list) or not keys_raw:
            raise ValueError("'multiget' command requires a non-empty 'keys' list")
        keys = [str(k) for k in keys_raw]
        values = reader.multi_get(keys)
        return build_multiget_result(keys, values, cfg)

    if op == "refresh":
        changed = reader.refresh()
        return build_refresh_result(changed)

    if op == "info":
        return build_info_result(reader)

    raise ValueError(f"Unknown op: {op!r}. Supported: get, multiget, refresh, info")
