"""Interactive REPL for the slate-reader CLI, backed by ConcurrentShardedReader."""

import cmd
import shlex
import sys
from typing import Any

from .config import OutputConfig, coerce_cli_key
from .output import (
    build_error_result,
    build_get_result,
    build_info_result,
    build_multiget_result,
    build_refresh_result,
    emit,
)


class SlateReaderRepl(cmd.Cmd):
    """cmd.Cmd REPL that wraps a single reader instance."""

    intro = ""
    prompt = "slate> "

    def __init__(self, reader: Any, output_cfg: OutputConfig) -> None:
        super().__init__()
        self._reader = reader
        self._output_cfg = output_cfg
        # Use json format in interactive mode for pretty output, unless
        # the user explicitly configured a different format.
        self._interactive_cfg = OutputConfig(
            format=output_cfg.format if output_cfg.format != "jsonl" else "json",
            value_encoding=output_cfg.value_encoding,
            null_repr=output_cfg.null_repr,
        )

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def do_get(self, line: str) -> None:
        """get KEY — Look up a single key."""
        parts = shlex.split(line)
        if len(parts) != 1:
            self._error("get", None, "Usage: get KEY")
            return
        raw_key = parts[0]
        try:
            coerced = coerce_cli_key(raw_key, self._reader.key_encoding)
            value = self._reader.get(coerced)
            result = build_get_result(raw_key, value, self._interactive_cfg)
            emit(result, self._interactive_cfg)
        except Exception as exc:
            self._error("get", raw_key, str(exc))

    def do_multiget(self, line: str) -> None:
        """multiget KEY [KEY …] — Look up multiple keys."""
        parts = shlex.split(line)
        if not parts:
            self._error("multiget", None, "Usage: multiget KEY [KEY …]")
            return
        try:
            coerced = [coerce_cli_key(k, self._reader.key_encoding) for k in parts]
            values = self._reader.multi_get(coerced)
            result = build_multiget_result(
                parts, values, self._interactive_cfg, coerced_keys=coerced
            )
            emit(result, self._interactive_cfg)
        except Exception as exc:
            self._error("multiget", None, str(exc))

    def do_refresh(self, line: str) -> None:
        """refresh — Reload CURRENT and manifest."""
        try:
            changed = self._reader.refresh()
            result = build_refresh_result(changed)
            emit(result, self._interactive_cfg)
        except Exception as exc:
            self._error("refresh", None, str(exc))

    def do_info(self, line: str) -> None:
        """info — Show manifest metadata."""
        try:
            result = build_info_result(self._reader)
            emit(result, self._interactive_cfg)
        except Exception as exc:
            self._error("info", None, str(exc))

    def do_quit(self, line: str) -> bool:
        """quit — Exit the REPL."""
        return True

    def do_exit(self, line: str) -> bool:
        """exit — Exit the REPL."""
        return True

    def do_EOF(self, line: str) -> bool:
        print()  # newline after ^D
        return True

    # ------------------------------------------------------------------
    # Error helper
    # ------------------------------------------------------------------

    def _error(self, op: str, key: str | None, message: str) -> None:
        result = build_error_result(op, key, message)
        emit(result, self._interactive_cfg, file=sys.stderr)

    # ------------------------------------------------------------------
    # Startup banner
    # ------------------------------------------------------------------

    def print_banner(self) -> None:
        """Print the startup banner showing manifest metadata."""
        try:
            info = self._reader.snapshot_info()
            print(
                f"Loaded manifest run_id={info.run_id}  "
                f"({info.num_dbs} shards, {info.sharding} sharding)"
            )
        except Exception:
            pass
