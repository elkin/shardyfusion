"""Interactive REPL for the shardy CLI, backed by ConcurrentShardedReader."""

from __future__ import annotations

import cmd
import shlex
import sys
from datetime import timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..reader import ConcurrentShardedReader, ShardedReader

from .config import OutputConfig, coerce_cli_key
from .output import (
    build_error_result,
    build_get_result,
    build_health_result,
    build_info_result,
    build_multiget_result,
    build_refresh_result,
    build_route_result,
    build_shards_result,
    emit,
)


class ShardyRepl(cmd.Cmd):
    """cmd.Cmd REPL that wraps a single reader instance."""

    intro = ""
    prompt = "shardy> "

    def __init__(
        self, reader: ShardedReader | ConcurrentShardedReader, output_cfg: OutputConfig
    ) -> None:
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
        """get KEY [--routing-context k=v ...] — Look up a single key."""
        parts = shlex.split(line)
        raw_key, routing_context = self._parse_key_and_context(parts)
        if raw_key is None:
            self._error("get", None, "Usage: get KEY [--routing-context k=v ...]")
            return
        try:
            coerced = coerce_cli_key(raw_key, self._reader.key_encoding)
            value = self._reader.get(coerced, routing_context=routing_context)
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
        """refresh — Reload CURRENT and manifest, clearing any session pin."""
        from ..cli.app import _PinnedManifestStore

        try:
            # Clear any session pin so refresh loads the true latest
            store = self._reader._manifest_store
            if isinstance(store, _PinnedManifestStore):
                self._reader._manifest_store = store._inner
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

    def do_shards(self, line: str) -> None:
        """shards — Show per-shard details."""
        try:
            shards = self._reader.shard_details()
            result = build_shards_result(shards)
            emit(result, self._interactive_cfg)
        except Exception as exc:
            self._error("shards", None, str(exc))

    def do_health(self, line: str) -> None:
        """health [STALENESS_THRESHOLD] — Show reader health status."""
        parts = shlex.split(line)
        threshold: timedelta | None = None
        if parts:
            try:
                threshold = timedelta(seconds=float(parts[0]))
            except ValueError:
                self._error("health", None, f"Invalid threshold: {parts[0]!r}")
                return
        try:
            health = self._reader.health(staleness_threshold=threshold)
            result = build_health_result(health)
            emit(result, self._interactive_cfg)
        except Exception as exc:
            self._error("health", None, str(exc))

    def do_route(self, line: str) -> None:
        """route KEY [--routing-context k=v ...] — Show which shard a key routes to."""
        parts = shlex.split(line)
        raw_key, routing_context = self._parse_key_and_context(parts)
        if raw_key is None:
            self._error("route", None, "Usage: route KEY [--routing-context k=v ...]")
            return
        try:
            coerced = coerce_cli_key(raw_key, self._reader.key_encoding)
            db_id = self._reader.route_key(coerced, routing_context=routing_context)
            result = build_route_result(raw_key, db_id)
            emit(result, self._interactive_cfg)
        except Exception as exc:
            self._error("route", raw_key, str(exc))

    def do_schema(self, line: str) -> None:
        """schema [manifest|current-pointer] — Print JSON Schema (default: manifest)."""
        import json

        from ..manifest import CurrentPointer, ParsedManifest

        schema_type = line.strip().lower() if line.strip() else "manifest"
        if schema_type not in ("manifest", "current-pointer"):
            self._error("schema", None, "Usage: schema [manifest|current-pointer]")
            return

        if schema_type == "manifest":
            schema = {
                **ParsedManifest.model_json_schema(mode="serialization"),
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": "https://github.com/elkin/shardyfusion/schemas/manifest.schema.json",
                "title": "SlateDB/SQLite Sharded Manifest",
                "description": "JSON manifest published to S3 by the sharded writer.",
            }
        else:
            schema = {
                **CurrentPointer.model_json_schema(),
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": "https://github.com/elkin/shardyfusion/schemas/current-pointer.schema.json",
                "title": "SlateDB/SQLite Sharded CURRENT Pointer",
                "description": "JSON pointer published to S3 at _CURRENT.",
            }

        print(json.dumps(schema, indent=2))

    def do_history(self, line: str) -> None:
        """history [LIMIT] — List recent published manifests."""
        from .output import build_history_result

        parts = shlex.split(line)
        limit = int(parts[0]) if parts else 10
        try:
            store = self._reader._manifest_store
            refs = store.list_manifests(limit=limit)
            result = build_history_result(refs)
            emit(result, self._interactive_cfg)
        except Exception as exc:
            self._error("history", None, str(exc))

    def do_use(self, line: str) -> None:
        """use --offset N | --ref REF | --latest — Switch to a different manifest.

        Session-local: does not mutate _CURRENT. Only `rollback` does that.
        """
        from ..cli.app import (
            _PinnedManifestStore,
            _resolve_manifest_ref_obj,
        )

        parts = shlex.split(line)
        if not parts:
            self._error("use", None, "Usage: use --offset N | --ref REF | --latest")
            return

        try:
            store = self._reader._manifest_store
            # Unwrap any existing pin to access the real store
            real_store = (
                store._inner if isinstance(store, _PinnedManifestStore) else store
            )

            if parts[0] == "--latest":
                # Clear any session pin and refresh from the real store
                self._reader._manifest_store = real_store
                self._reader.refresh()
                info = self._reader.snapshot_info()
                print(f"Switched to latest manifest run_id={info.run_id}")
            elif parts[0] in ("--offset", "--ref") and len(parts) == 2:
                ref_arg = None if parts[0] == "--offset" else parts[1]
                offset_arg = int(parts[1]) if parts[0] == "--offset" else None
                pinned = _resolve_manifest_ref_obj(
                    real_store, ref=ref_arg, offset=offset_arg
                )
                if pinned is not None:
                    # Pin the reader's store session-locally without mutating _CURRENT
                    self._reader._manifest_store = _PinnedManifestStore(
                        real_store, pinned
                    )
                self._reader.refresh()
                info = self._reader.snapshot_info()
                print(
                    f"Switched to manifest run_id={info.run_id} ({info.created_at.isoformat()})"
                )
            else:
                self._error("use", None, "Usage: use --offset N | --ref REF | --latest")
        except Exception as exc:
            self._error("use", None, str(exc))

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
    # Routing context helper
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_key_and_context(
        parts: list[str],
    ) -> tuple[str | None, dict[str, object] | None]:
        """Extract key and optional --routing-context from command args."""
        from .config import parse_routing_context

        if not parts:
            return None, None
        key = parts[0]
        ctx_pairs: list[str] = []
        i = 1
        while i < len(parts):
            if parts[i] == "--routing-context" and i + 1 < len(parts):
                ctx_pairs.append(parts[i + 1])
                i += 2
            else:
                i += 1
        routing_context = parse_routing_context(tuple(ctx_pairs)) if ctx_pairs else None
        return key, routing_context

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
