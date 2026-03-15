"""Click CLI application for shardy."""

import sys
from typing import Any

import click

from .batch import run_script
from .config import (
    ManifestStoreConfig,
    OutputConfig,
    ReaderConfig,
    build_connection_factory,
    build_s3_config,
    coerce_cli_key,
    coerce_s3_option,
    load_credentials_profile,
    load_reader_config,
    resolve_current_url,
    resolve_dsn,
    split_current_url,
)
from .output import (
    build_error_result,
    build_get_result,
    build_info_result,
    build_multiget_result,
    build_route_result,
    build_shards_result,
    emit,
)

# ---------------------------------------------------------------------------
# Shared context keys
# ---------------------------------------------------------------------------

_CTX_INIT_PARAMS = "shardy_init_params"


def _build_manifest_store(
    store_cfg: ManifestStoreConfig,
    params: dict[str, Any],
) -> Any:
    """Create the manifest store for the configured backend."""
    if store_cfg.backend == "s3":
        from ..manifest_store import S3ManifestStore

        return S3ManifestStore(
            params["s3_prefix"],
            current_name=params["current_name"],
            credential_provider=params["credential_provider"],
            s3_connection_options=params["s3_connection_options"],
        )

    # DB backends (postgres / comdb2)
    dsn = resolve_dsn(store_cfg)
    conn_factory = build_connection_factory(store_cfg.backend, dsn)  # type: ignore[arg-type]

    if store_cfg.backend == "postgres":
        from ..db_manifest_store import PostgresManifestStore

        return PostgresManifestStore(
            conn_factory,
            table_name=store_cfg.table_name,
            ensure_table=store_cfg.ensure_table,
        )

    from ..db_manifest_store import Comdb2ManifestStore

    return Comdb2ManifestStore(
        conn_factory,
        table_name=store_cfg.table_name,
        ensure_table=store_cfg.ensure_table,
    )


def _resolve_manifest_ref(
    store: Any,
    *,
    ref: str | None,
    offset: int | None,
) -> str | None:
    """Resolve --ref / --offset to a manifest ref string, or None for default."""
    if ref is not None and offset is not None:
        raise click.UsageError("--ref and --offset are mutually exclusive")
    if ref is not None:
        return ref
    if offset is not None:
        if offset < 0:
            raise click.UsageError("--offset must be >= 0")
        refs = store.list_manifests(limit=offset + 1)
        if offset >= len(refs):
            raise click.ClickException(
                f"Offset {offset} out of range (only {len(refs)} manifests available)"
            )
        return refs[offset].ref
    return None


def _build_reader(ctx: click.Context) -> Any:
    """Construct a ConcurrentShardedReader from the parameters stored in ctx.obj."""
    params = ctx.obj[_CTX_INIT_PARAMS]
    reader_cfg: ReaderConfig = params["reader_cfg"]
    store_cfg: ManifestStoreConfig = params["store_cfg"]

    from ..reader import ConcurrentShardedReader

    manifest_store = _build_manifest_store(store_cfg, params)

    # If --ref or --offset was given, point the store at that manifest first
    target_ref = _resolve_manifest_ref(
        manifest_store,
        ref=params.get("manifest_ref"),
        offset=params.get("manifest_offset"),
    )
    if target_ref is not None:
        manifest_store.set_current(target_ref)

    try:
        return ConcurrentShardedReader(
            s3_prefix=params["s3_prefix"],
            local_root=reader_cfg.local_root,
            manifest_store=manifest_store,
            current_name=params.get("current_name", "_CURRENT"),
            slate_env_file=reader_cfg.slate_env_file,
            credential_provider=params["credential_provider"],
            s3_connection_options=params["s3_connection_options"],
            thread_safety=reader_cfg.thread_safety,
            pool_checkout_timeout=reader_cfg.pool_checkout_timeout,
            max_workers=reader_cfg.max_workers,
        )
    except Exception as exc:
        raise click.ClickException(f"Failed to initialise reader: {exc}") from exc


def _get_output_cfg(ctx: click.Context) -> OutputConfig:
    return ctx.obj[_CTX_INIT_PARAMS]["output_cfg"]


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group(
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(package_name="shardyfusion", prog_name="shardy")
@click.option(
    "--current-url",
    "current_url",
    default=None,
    metavar="URL",
    help=(
        "S3 URL to the _CURRENT pointer (overrides SHARDY_CURRENT env and reader.toml)."
    ),
)
@click.option(
    "--config",
    "config_path",
    default=None,
    metavar="PATH",
    help="Path to reader.toml (overrides SHARDY_CONFIG env and default search).",
)
@click.option(
    "--credentials",
    "credentials_path",
    default=None,
    metavar="PATH",
    help=(
        "Path to credentials.toml "
        "(overrides SHARDY_CREDENTIALS env and default search)."
    ),
)
@click.option(
    "--s3-option",
    "s3_options",
    multiple=True,
    metavar="KEY=VALUE",
    help="Override an S3 connection option (e.g. addressing_style=path). Repeatable.",
)
@click.option(
    "--output-format",
    "output_format",
    default=None,
    metavar="FORMAT",
    type=click.Choice(["json", "jsonl", "table", "text"], case_sensitive=False),
    help="Output format: json | jsonl | table | text.",
)
@click.option(
    "--ref",
    "manifest_ref",
    default=None,
    metavar="REF",
    help="Load a specific manifest by ref (mutually exclusive with --offset).",
)
@click.option(
    "--offset",
    "manifest_offset",
    default=None,
    type=int,
    metavar="N",
    help="Load the Nth previous manifest (0=latest, 1=previous, etc.).",
)
@click.pass_context
def cli(
    ctx: click.Context,
    current_url: str | None,
    config_path: str | None,
    credentials_path: str | None,
    s3_options: tuple[str, ...],
    output_format: str | None,
    manifest_ref: str | None,
    manifest_offset: int | None,
) -> None:
    """shardy — interactive and batch lookups for sharded SlateDB snapshots.

    \b
    CURRENT_URL may be supplied via --current-url, the SHARDY_CURRENT
    environment variable, or as current_url in reader.toml.

    \b
    When no subcommand is given the tool enters interactive mode.
    """
    ctx.ensure_object(dict)

    # schema subcommand needs no reader/S3 setup
    if ctx.invoked_subcommand == "schema":
        return

    # Parse --s3-option KEY=VALUE pairs
    s3_overrides: dict[str, bool | int | str] = {}
    for opt in s3_options:
        if "=" not in opt:
            raise click.UsageError(f"--s3-option must be KEY=VALUE, got: {opt!r}")
        key, _, raw_value = opt.partition("=")
        try:
            s3_overrides[key.strip()] = coerce_s3_option(key.strip(), raw_value.strip())
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc

    # Load reader + manifest-store + output config
    reader_cfg, store_cfg, output_cfg = load_reader_config(config_path)

    # Apply --output-format override
    if output_format:
        output_cfg = OutputConfig(
            format=output_format,
            value_encoding=output_cfg.value_encoding,
            null_repr=output_cfg.null_repr,
        )

    # Load credentials profile
    profile = load_credentials_profile(
        profile_name=reader_cfg.credentials_profile,
        credentials_path=credentials_path,
    )

    # Build credential provider + connection options from profile + overrides
    credential_provider, s3_connection_options = build_s3_config(profile, s3_overrides)

    # Resolve S3 prefix and CURRENT name based on backend
    if store_cfg.backend == "s3":
        resolved_url = resolve_current_url(current_url, reader_cfg)
        s3_prefix, current_name = split_current_url(resolved_url)
    else:
        # DB backends: s3_prefix is required in config (shard DBs are still on S3)
        if current_url:
            click.echo(
                "Warning: --current-url is ignored for DB manifest store backends.",
                err=True,
            )
        if not reader_cfg.s3_prefix:
            raise click.UsageError(
                f"s3_prefix is required in [reader] when using "
                f"'{store_cfg.backend}' manifest store."
            )
        s3_prefix = reader_cfg.s3_prefix
        current_name = "_CURRENT"

    # Store init parameters in context for lazy reader construction in subcommands
    ctx.obj[_CTX_INIT_PARAMS] = {
        "s3_prefix": s3_prefix,
        "current_name": current_name,
        "reader_cfg": reader_cfg,
        "store_cfg": store_cfg,
        "output_cfg": output_cfg,
        "credential_provider": credential_provider,
        "s3_connection_options": s3_connection_options,
        "manifest_ref": manifest_ref,
        "manifest_offset": manifest_offset,
    }

    # If no subcommand was invoked, enter interactive mode
    if ctx.invoked_subcommand is None:
        from .interactive import ShardyRepl

        with _build_reader(ctx) as reader:
            repl = ShardyRepl(reader, output_cfg)
            repl.print_banner()
            repl.cmdloop()


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@cli.command("get")
@click.argument("key")
@click.pass_context
def get_cmd(ctx: click.Context, key: str) -> None:
    """Look up a single KEY."""
    output_cfg = _get_output_cfg(ctx)
    with _build_reader(ctx) as reader:
        try:
            coerced = coerce_cli_key(key, reader.key_encoding)
            value = reader.get(coerced)
            result = build_get_result(key, value, output_cfg)
            emit(result, output_cfg)
        except Exception as exc:
            result = build_error_result("get", key, str(exc))
            emit(result, output_cfg, file=sys.stderr)
            sys.exit(1)


@cli.command("multiget")
@click.argument("keys", nargs=-1, required=True)
@click.pass_context
def multiget_cmd(ctx: click.Context, keys: tuple[str, ...]) -> None:
    """Look up multiple KEYS (space-separated).

    Pass a single '-' to read keys from stdin, one per line.
    """
    # Support reading keys from stdin when '-' is the sole argument
    if keys == ("-",):
        stdin_keys = [line.strip() for line in sys.stdin if line.strip()]
        if not stdin_keys:
            raise click.UsageError("No keys provided on stdin")
        keys = tuple(stdin_keys)

    output_cfg = _get_output_cfg(ctx)
    with _build_reader(ctx) as reader:
        try:
            coerced = [coerce_cli_key(k, reader.key_encoding) for k in keys]
            values = reader.multi_get(coerced)
            display_keys = list(keys)
            result = build_multiget_result(
                display_keys, values, output_cfg, coerced_keys=coerced
            )
            emit(result, output_cfg)
        except Exception as exc:
            result = build_error_result("multiget", None, str(exc))
            emit(result, output_cfg, file=sys.stderr)
            sys.exit(1)


@cli.command("info")
@click.pass_context
def info_cmd(ctx: click.Context) -> None:
    """Show manifest metadata."""
    output_cfg = _get_output_cfg(ctx)
    with _build_reader(ctx) as reader:
        try:
            result = build_info_result(reader)
            emit(result, output_cfg)
        except Exception as exc:
            result = build_error_result("info", None, str(exc))
            emit(result, output_cfg, file=sys.stderr)
            sys.exit(1)


@cli.command("shards")
@click.pass_context
def shards_cmd(ctx: click.Context) -> None:
    """Show per-shard details from the manifest."""
    output_cfg = _get_output_cfg(ctx)
    with _build_reader(ctx) as reader:
        try:
            shards = reader.shard_details()
            result = build_shards_result(shards)
            emit(result, output_cfg)
        except Exception as exc:
            result = build_error_result("shards", None, str(exc))
            emit(result, output_cfg, file=sys.stderr)
            sys.exit(1)


@cli.command("route")
@click.argument("key")
@click.pass_context
def route_cmd(ctx: click.Context, key: str) -> None:
    """Show which shard a KEY routes to (without performing a lookup)."""
    output_cfg = _get_output_cfg(ctx)
    with _build_reader(ctx) as reader:
        try:
            coerced = coerce_cli_key(key, reader.key_encoding)
            db_id = reader.route_key(coerced)
            result = build_route_result(key, db_id)
            emit(result, output_cfg)
        except Exception as exc:
            result = build_error_result("route", key, str(exc))
            emit(result, output_cfg, file=sys.stderr)
            sys.exit(1)


@cli.command("exec")
@click.option(
    "--script",
    "script_path",
    required=True,
    metavar="FILE",
    help="Path to YAML script file.",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    metavar="FILE",
    help="Write results to FILE instead of stdout.",
)
@click.pass_context
def exec_cmd(ctx: click.Context, script_path: str, output_path: str | None) -> None:
    """Execute a batch YAML script file against the snapshot."""
    output_cfg = _get_output_cfg(ctx)

    out_file = None
    with _build_reader(ctx) as reader:
        try:
            if output_path:
                out_file = open(output_path, "w", encoding="utf-8")
            error_count = run_script(
                reader, script_path, output_cfg, output_file=out_file
            )
        except (OSError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        finally:
            if out_file is not None:
                out_file.close()

    if error_count:
        sys.exit(1)


@cli.command("history")
@click.option("--limit", default=10, type=int, help="Maximum manifests to list.")
@click.pass_context
def history_cmd(ctx: click.Context, limit: int) -> None:
    """List recent published manifests."""
    from .output import build_history_result

    output_cfg = _get_output_cfg(ctx)
    params = ctx.obj[_CTX_INIT_PARAMS]
    store_cfg: ManifestStoreConfig = params["store_cfg"]
    manifest_store = _build_manifest_store(store_cfg, params)

    try:
        refs = manifest_store.list_manifests(limit=limit)
        result = build_history_result(refs)
        emit(result, output_cfg)
    except Exception as exc:
        result = build_error_result("history", None, str(exc))
        emit(result, output_cfg, file=sys.stderr)
        sys.exit(1)


@cli.command("rollback")
@click.option(
    "--ref",
    "target_ref",
    default=None,
    metavar="REF",
    help="Manifest ref to roll back to.",
)
@click.option(
    "--run-id",
    "target_run_id",
    default=None,
    metavar="RUN_ID",
    help="Run ID to roll back to.",
)
@click.option(
    "--offset",
    "target_offset",
    default=None,
    type=int,
    metavar="N",
    help="Roll back N versions (e.g. 1 = previous).",
)
@click.pass_context
def rollback_cmd(
    ctx: click.Context,
    target_ref: str | None,
    target_run_id: str | None,
    target_offset: int | None,
) -> None:
    """Roll back the current pointer to a previous manifest."""
    output_cfg = _get_output_cfg(ctx)
    params = ctx.obj[_CTX_INIT_PARAMS]
    store_cfg: ManifestStoreConfig = params["store_cfg"]
    manifest_store = _build_manifest_store(store_cfg, params)

    provided = sum(x is not None for x in (target_ref, target_run_id, target_offset))
    if provided != 1:
        raise click.UsageError(
            "Exactly one of --ref, --run-id, or --offset is required"
        )

    try:
        ref = _resolve_manifest_ref(
            manifest_store, ref=target_ref, offset=target_offset
        )
        if ref is None and target_run_id is not None:
            # --run-id: find the ref by scanning history
            refs = manifest_store.list_manifests(limit=100)
            matching = [r for r in refs if r.run_id == target_run_id]
            if not matching:
                raise click.ClickException(
                    f"No manifest found with run_id={target_run_id}"
                )
            ref = matching[0].ref

        assert ref is not None
        manifest_store.set_current(ref)
        click.echo(f"Rolled back _CURRENT to: {ref}")
    except click.ClickException:
        raise
    except Exception as exc:
        result = build_error_result("rollback", None, str(exc))
        emit(result, output_cfg, file=sys.stderr)
        sys.exit(1)


@cli.command("schema")
@click.option(
    "--type",
    "schema_type",
    default="manifest",
    type=click.Choice(["manifest", "current-pointer"], case_sensitive=False),
    help="Which schema to print (default: manifest).",
)
def schema_cmd(schema_type: str) -> None:
    """Print the JSON Schema for manifest or current-pointer formats."""
    import json

    from ..manifest import CurrentPointer, ParsedManifest

    if schema_type == "manifest":
        schema = {
            **ParsedManifest.model_json_schema(mode="serialization"),
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://github.com/slatedb/shardyfusion/schemas/manifest.schema.json",
            "title": "SlateDB Sharded Manifest",
            "description": "JSON manifest published to S3 by the sharded writer.",
        }
    else:
        schema = {
            **CurrentPointer.model_json_schema(),
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://github.com/slatedb/shardyfusion/schemas/current-pointer.schema.json",
            "title": "SlateDB Sharded CURRENT Pointer",
            "description": "JSON pointer published to S3 at _CURRENT.",
        }

    click.echo(json.dumps(schema, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cli()
