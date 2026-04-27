"""Click CLI application for shardy."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..manifest import ManifestRef, ParsedManifest
    from ..manifest_store import ManifestStore
    from ..reader import ConcurrentShardedReader

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
    build_health_result,
    build_info_result,
    build_multiget_result,
    build_route_result,
    build_search_result,
    build_shards_result,
    emit,
)

# ---------------------------------------------------------------------------
# Shared context keys
# ---------------------------------------------------------------------------

_CTX_INIT_PARAMS = "shardy_init_params"
_CTX_RAW_PARAMS = "shardy_raw_params"


def _ensure_init_params(ctx: click.Context) -> dict[str, Any]:
    """Resolve global CLI options into reader construction parameters once."""
    ctx.ensure_object(dict)
    if _CTX_INIT_PARAMS in ctx.obj:
        return ctx.obj[_CTX_INIT_PARAMS]

    raw_params = ctx.obj.get(_CTX_RAW_PARAMS)
    if raw_params is None:
        raise click.UsageError("CLI context was not initialised")

    current_url: str | None = raw_params["current_url"]
    config_path: str | None = raw_params["config_path"]
    credentials_path: str | None = raw_params["credentials_path"]
    s3_options: tuple[str, ...] = raw_params["s3_options"]
    output_format: str | None = raw_params["output_format"]
    manifest_ref: str | None = raw_params["manifest_ref"]
    manifest_offset: int | None = raw_params["manifest_offset"]

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

    # Apply --sqlite-mode and threshold overrides
    sqlite_overrides: dict[str, Any] = {}
    sqlite_mode_override = raw_params.get("sqlite_mode")
    if sqlite_mode_override is not None:
        sqlite_overrides["sqlite_mode"] = sqlite_mode_override.lower()
    per_shard_override = raw_params.get("sqlite_auto_per_shard_threshold_bytes")
    if per_shard_override is not None:
        sqlite_overrides["sqlite_auto_per_shard_threshold_bytes"] = per_shard_override
    total_override = raw_params.get("sqlite_auto_total_budget_bytes")
    if total_override is not None:
        sqlite_overrides["sqlite_auto_total_budget_bytes"] = total_override
    if sqlite_overrides:
        # model_copy(update=...) skips validators, so use model_validate to
        # enforce ge=0 on the threshold fields. Surface validation errors as
        # UsageError so they show up cleanly on the CLI.
        from pydantic import ValidationError

        try:
            reader_cfg = ReaderConfig.model_validate(
                {**reader_cfg.model_dump(), **sqlite_overrides}
            )
        except ValidationError as exc:
            raise click.UsageError(
                f"Invalid SQLite override(s): {exc.errors()}"
            ) from exc

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
        s3_prefix, current_pointer_key = split_current_url(resolved_url)
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
        current_pointer_key = "_CURRENT"

    params = {
        "s3_prefix": s3_prefix,
        "current_pointer_key": current_pointer_key,
        "reader_cfg": reader_cfg,
        "store_cfg": store_cfg,
        "output_cfg": output_cfg,
        "credential_provider": credential_provider,
        "s3_connection_options": s3_connection_options,
        "manifest_ref": manifest_ref,
        "manifest_offset": manifest_offset,
    }
    ctx.obj[_CTX_INIT_PARAMS] = params
    return params


def _build_manifest_store(
    store_cfg: ManifestStoreConfig,
    params: dict[str, Any],
) -> ManifestStore:
    """Create the manifest store for the configured backend."""
    if store_cfg.backend == "s3":
        from ..manifest_store import S3ManifestStore

        return S3ManifestStore(
            params["s3_prefix"],
            current_pointer_key=params["current_pointer_key"],
            credential_provider=params["credential_provider"],
            s3_connection_options=params["s3_connection_options"],
        )

    # Postgres backend
    dsn = resolve_dsn(store_cfg)
    conn_factory = build_connection_factory(dsn)

    from ..db_manifest_store import PostgresManifestStore

    return PostgresManifestStore(
        conn_factory,
        table_name=store_cfg.table_name,
        ensure_table=store_cfg.ensure_table,
    )


def _resolve_manifest_ref(
    store: ManifestStore,
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


def _resolve_manifest_ref_obj(
    store: ManifestStore,
    *,
    ref: str | None,
    offset: int | None,
) -> ManifestRef | None:
    """Like _resolve_manifest_ref but returns the full ManifestRef when available.

    Used by _PinnedManifestStore to avoid a redundant list_manifests call.
    """
    if ref is not None and offset is not None:
        raise click.UsageError("--ref and --offset are mutually exclusive")
    if offset is not None:
        if offset < 0:
            raise click.UsageError("--offset must be >= 0")
        refs = store.list_manifests(limit=offset + 1)
        if offset >= len(refs):
            raise click.ClickException(
                f"Offset {offset} out of range (only {len(refs)} manifests available)"
            )
        return refs[offset]  # full ManifestRef, not just .ref
    if ref is not None:
        # Only have a string ref — need to look up the full ManifestRef
        for r in store.list_manifests(limit=100):
            if r.ref == ref:
                return r
        # Fallback: construct minimal ManifestRef
        from datetime import UTC, datetime

        from ..manifest import ManifestRef

        return ManifestRef(ref=ref, run_id="unknown", published_at=datetime.now(UTC))
    return None


class _PinnedManifestStore:
    """Wrapper that pins load_current() to a specific ref without mutating the backing store.

    The pinned ManifestRef is resolved once at construction time and cached.
    set_current() is a deliberate passthrough — rollback is the one CLI command
    that should mutate _CURRENT, and it operates on the real store directly.
    """

    def __init__(self, inner: ManifestStore, pinned_manifest_ref: ManifestRef) -> None:
        self._inner = inner
        self._cached_ref = pinned_manifest_ref

    def load_current(self) -> ManifestRef | None:
        return self._cached_ref

    def load_manifest(self, ref: str) -> ParsedManifest:
        return self._inner.load_manifest(ref)

    def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        return self._inner.list_manifests(limit=limit)

    def set_current(self, ref: str) -> None:
        self._inner.set_current(ref)

    def publish(self, **kwargs: Any) -> str:
        return self._inner.publish(**kwargs)


def _build_reader(ctx: click.Context) -> ConcurrentShardedReader:
    """Construct a ConcurrentShardedReader from the parameters stored in ctx.obj."""
    params = _ensure_init_params(ctx)
    reader_cfg: ReaderConfig = params["reader_cfg"]
    store_cfg: ManifestStoreConfig = params["store_cfg"]

    from ..reader import ConcurrentShardedReader

    manifest_store = _build_manifest_store(store_cfg, params)

    # If --ref or --offset was given, pin the store to that manifest
    # without mutating _CURRENT (only rollback should do that).
    pinned_ref = _resolve_manifest_ref_obj(
        manifest_store,
        ref=params.get("manifest_ref"),
        offset=params.get("manifest_offset"),
    )
    if pinned_ref is not None:
        manifest_store = _PinnedManifestStore(manifest_store, pinned_ref)

    reader_factory = None
    if reader_cfg.reader_backend == "sqlite":
        mode = reader_cfg.sqlite_mode
        if mode == "download":
            from ..sqlite_adapter import SqliteReaderFactory

            reader_factory = SqliteReaderFactory(
                credential_provider=params["credential_provider"],
                s3_connection_options=params["s3_connection_options"],
            )
        elif mode == "range":
            from ..sqlite_adapter import SqliteRangeReaderFactory

            reader_factory = SqliteRangeReaderFactory(
                credential_provider=params["credential_provider"],
                s3_connection_options=params["s3_connection_options"],
            )
        else:  # "auto"
            from ..sqlite_adapter import AdaptiveSqliteReaderFactory

            reader_factory = AdaptiveSqliteReaderFactory(
                per_shard_threshold=reader_cfg.sqlite_auto_per_shard_threshold_bytes,
                total_budget=reader_cfg.sqlite_auto_total_budget_bytes,
                credential_provider=params["credential_provider"],
                s3_connection_options=params["s3_connection_options"],
            )

    try:
        return ConcurrentShardedReader(
            s3_prefix=params["s3_prefix"],
            local_root=reader_cfg.local_root,
            manifest_store=manifest_store,
            current_pointer_key=params.get("current_pointer_key", "_CURRENT"),
            slate_env_file=reader_cfg.slate_env_file,
            credential_provider=params["credential_provider"],
            s3_connection_options=params["s3_connection_options"],
            thread_safety=reader_cfg.thread_safety,
            pool_checkout_timeout=reader_cfg.pool_checkout_timeout,
            max_workers=reader_cfg.max_workers,
            reader_factory=reader_factory,
        )
    except Exception as exc:
        raise click.ClickException(f"Failed to initialise reader: {exc}") from exc


def _get_output_cfg(ctx: click.Context) -> OutputConfig:
    return _ensure_init_params(ctx)["output_cfg"]


def _build_unified_kv_factory(
    vector_meta: dict[str, Any],
    *,
    reader_cfg: ReaderConfig,
    credential_provider: Any,
    s3_connection_options: Any,
) -> Any:
    """Build a reader factory honoring --sqlite-mode for unified search.

    Returns ``None`` to fall back to the unified reader's auto-dispatch when
    the snapshot doesn't use SQLite, when ``reader_backend`` is not ``sqlite``,
    or when the user did not request a non-default mode/threshold.

    For ``sqlite-vec`` snapshots, we build an explicit ``SqliteVec*`` factory.
    For composite (LanceDB sidecar with SQLite KV) snapshots, we build a
    composite factory that wires the chosen SQLite KV factory together with
    the LanceDB vector factory.
    """
    backend = vector_meta.get("backend")
    kv_backend = vector_meta.get("kv_backend") or "slatedb"

    # Only act when a SQLite tier is actually involved.
    sqlite_in_play = backend == "sqlite-vec" or (
        backend == "lancedb" and kv_backend == "sqlite"
    )
    if not sqlite_in_play:
        return None

    # Don't override unless user explicitly asked for sqlite reader behavior.
    # The CLI's --sqlite-mode flag is global (and 'auto' is the default), so
    # we always honor reader_cfg here when SQLite is involved.
    mode = reader_cfg.sqlite_mode

    if backend == "sqlite-vec":
        if mode == "auto":
            from ..sqlite_vec_adapter import AdaptiveSqliteVecReaderFactory

            return AdaptiveSqliteVecReaderFactory(
                per_shard_threshold=reader_cfg.sqlite_auto_per_shard_threshold_bytes,
                total_budget=reader_cfg.sqlite_auto_total_budget_bytes,
                credential_provider=credential_provider,
                s3_connection_options=s3_connection_options,
            )
        from ..sqlite_vec_adapter import make_sqlite_vec_reader_factory

        return make_sqlite_vec_reader_factory(
            mode=mode,
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
        )

    # backend == "lancedb" with kv_backend == "sqlite": build composite factory
    from ..composite_adapter import CompositeReaderFactory
    from ..config import VectorSpec, vector_metric_to_str
    from ..storage import create_s3_client
    from ..vector.adapters.lancedb_adapter import LanceDbReaderFactory

    if mode == "auto":
        from ..sqlite_adapter import AdaptiveSqliteReaderFactory

        kv_factory: Any = AdaptiveSqliteReaderFactory(
            per_shard_threshold=reader_cfg.sqlite_auto_per_shard_threshold_bytes,
            total_budget=reader_cfg.sqlite_auto_total_budget_bytes,
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
        )
    else:
        from ..sqlite_adapter import make_sqlite_reader_factory

        kv_factory = make_sqlite_reader_factory(
            mode=mode,
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
        )

    vs = VectorSpec(
        dim=vector_meta["dim"],
        metric=vector_metric_to_str(vector_meta["metric"]),
        index_type=vector_meta.get("index_type"),
        index_params=vector_meta.get("index_params"),
        quantization=vector_meta.get("quantization"),
    )
    credentials = credential_provider.resolve() if credential_provider else None
    s3_client = create_s3_client(credentials, s3_connection_options)
    vector_factory = LanceDbReaderFactory(
        s3_client=s3_client,
        s3_connection_options=s3_connection_options,
        credential_provider=credential_provider,
    )
    return CompositeReaderFactory(
        kv_factory=kv_factory,
        vector_factory=vector_factory,
        vector_spec=vs,
    )


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
@click.option(
    "--sqlite-mode",
    "sqlite_mode",
    default=None,
    type=click.Choice(["download", "range", "auto"], case_sensitive=False),
    help=(
        "SQLite shard access mode: 'download' (fetch full DB to local disk), "
        "'range' (S3 range-read VFS), or 'auto' (per-snapshot decision based on "
        "shard sizes). Overrides reader.toml; only consulted when "
        "reader_backend='sqlite'."
    ),
)
@click.option(
    "--sqlite-auto-per-shard-bytes",
    "sqlite_auto_per_shard_threshold_bytes",
    default=None,
    type=int,
    metavar="BYTES",
    help=(
        "Threshold (bytes): in 'auto' mode, switch to range-read when any shard's "
        "db_bytes is at or above this value. Default 16 MiB."
    ),
)
@click.option(
    "--sqlite-auto-total-bytes",
    "sqlite_auto_total_budget_bytes",
    default=None,
    type=int,
    metavar="BYTES",
    help=(
        "Threshold (bytes): in 'auto' mode, switch to range-read when the cumulative "
        "shard footprint is at or above this value. Default 2 GiB."
    ),
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
    sqlite_mode: str | None,
    sqlite_auto_per_shard_threshold_bytes: int | None,
    sqlite_auto_total_budget_bytes: int | None,
) -> None:
    """shardy — interactive and batch lookups for sharded SlateDB snapshots.

    \b
    CURRENT_URL may be supplied via --current-url, the SHARDY_CURRENT
    environment variable, or as current_url in reader.toml.

    \b
    When no subcommand is given the tool enters interactive mode.
    """
    ctx.ensure_object(dict)
    ctx.obj[_CTX_RAW_PARAMS] = {
        "current_url": current_url,
        "config_path": config_path,
        "credentials_path": credentials_path,
        "s3_options": s3_options,
        "output_format": output_format,
        "manifest_ref": manifest_ref,
        "manifest_offset": manifest_offset,
        "sqlite_mode": sqlite_mode,
        "sqlite_auto_per_shard_threshold_bytes": (
            sqlite_auto_per_shard_threshold_bytes
        ),
        "sqlite_auto_total_budget_bytes": sqlite_auto_total_budget_bytes,
    }

    # If no subcommand was invoked, enter interactive mode
    if ctx.invoked_subcommand is None:
        from .interactive import ShardyRepl

        params = _ensure_init_params(ctx)
        with _build_reader(ctx) as reader:
            repl = ShardyRepl(reader, params["output_cfg"])
            repl.print_banner()
            repl.cmdloop()


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@cli.command("get")
@click.argument("key")
@click.option(
    "--routing-context",
    "routing_ctx_pairs",
    multiple=True,
    metavar="KEY=VALUE",
    help="Routing context for CEL split mode (repeatable).",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Exit with code 1 when the key is not found.",
)
@click.pass_context
def get_cmd(
    ctx: click.Context,
    key: str,
    routing_ctx_pairs: tuple[str, ...],
    strict: bool,
) -> None:
    """Look up a single KEY."""
    from .config import parse_routing_context

    output_cfg = _get_output_cfg(ctx)
    routing_context = (
        parse_routing_context(routing_ctx_pairs) if routing_ctx_pairs else None
    )
    with _build_reader(ctx) as reader:
        try:
            coerced = coerce_cli_key(key, reader.key_encoding)
            value = reader.get(coerced, routing_context=routing_context)
            result = build_get_result(key, value, output_cfg)
            emit(result, output_cfg)
            if strict and value is None:
                sys.exit(1)
        except Exception as exc:
            result = build_error_result("get", key, str(exc))
            emit(result, output_cfg, file=sys.stderr)
            sys.exit(1)


@cli.command("multiget")
@click.argument("keys", nargs=-1, required=True)
@click.option(
    "--routing-context",
    "routing_ctx_pairs",
    multiple=True,
    metavar="KEY=VALUE",
    help="Routing context for CEL split mode (repeatable).",
)
@click.pass_context
def multiget_cmd(
    ctx: click.Context, keys: tuple[str, ...], routing_ctx_pairs: tuple[str, ...]
) -> None:
    """Look up multiple KEYS (space-separated).

    Pass a single '-' to read keys from stdin, one per line.
    """
    from .config import parse_routing_context

    # Support reading keys from stdin when '-' is the sole argument
    if keys == ("-",):
        stdin_keys = [line.strip() for line in sys.stdin if line.strip()]
        if not stdin_keys:
            raise click.UsageError("No keys provided on stdin")
        keys = tuple(stdin_keys)

    output_cfg = _get_output_cfg(ctx)
    routing_context = (
        parse_routing_context(routing_ctx_pairs) if routing_ctx_pairs else None
    )
    with _build_reader(ctx) as reader:
        try:
            coerced = [coerce_cli_key(k, reader.key_encoding) for k in keys]
            values = reader.multi_get(coerced, routing_context=routing_context)
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


@cli.command("health")
@click.option(
    "--staleness-threshold",
    default=None,
    type=float,
    metavar="SECONDS",
    help="Manifest age threshold in seconds; exceeding it marks the reader as degraded.",
)
@click.pass_context
def health_cmd(ctx: click.Context, staleness_threshold: float | None) -> None:
    """Show reader health status.

    \b
    Exit codes:  0 = healthy,  1 = degraded,  2 = unhealthy.
    """
    output_cfg = _get_output_cfg(ctx)
    with _build_reader(ctx) as reader:
        try:
            threshold_td = (
                timedelta(seconds=staleness_threshold)
                if staleness_threshold is not None
                else None
            )
            health = reader.health(staleness_threshold=threshold_td)
            result = build_health_result(health)
            emit(result, output_cfg)
            if health.status == "degraded":
                sys.exit(1)
            if health.status == "unhealthy":
                sys.exit(2)
        except Exception as exc:
            result = build_error_result("health", None, str(exc))
            emit(result, output_cfg, file=sys.stderr)
            sys.exit(2)


@cli.command("route")
@click.argument("key")
@click.option(
    "--routing-context",
    "routing_ctx_pairs",
    multiple=True,
    metavar="KEY=VALUE",
    help="Routing context for CEL split mode (repeatable).",
)
@click.pass_context
def route_cmd(ctx: click.Context, key: str, routing_ctx_pairs: tuple[str, ...]) -> None:
    """Show which shard a KEY routes to (without performing a lookup)."""
    from .config import parse_routing_context

    output_cfg = _get_output_cfg(ctx)
    routing_context = (
        parse_routing_context(routing_ctx_pairs) if routing_ctx_pairs else None
    )
    with _build_reader(ctx) as reader:
        try:
            coerced = coerce_cli_key(key, reader.key_encoding)
            db_id = reader.route_key(coerced, routing_context=routing_context)
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
    params = _ensure_init_params(ctx)
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
    params = _ensure_init_params(ctx)
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


def _parse_duration(value: str) -> timedelta:
    """Parse a duration string like ``7d`` or ``24h`` into a timedelta."""

    value = value.strip()
    if not value:
        raise click.BadParameter("Duration must not be empty")
    suffix = value[-1].lower()
    try:
        amount = int(value[:-1])
    except ValueError as exc:
        raise click.BadParameter(
            f"Invalid duration: {value!r} (expected e.g. 7d, 24h)"
        ) from exc
    if suffix == "d":
        return timedelta(days=amount)
    if suffix == "h":
        return timedelta(hours=amount)
    raise click.BadParameter(
        f"Unknown duration unit {suffix!r} in {value!r} (use 'd' or 'h')"
    )


@cli.command("cleanup")
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be deleted without deleting.",
)
@click.option(
    "--include-old-runs",
    is_flag=True,
    default=False,
    help="Also remove data from runs not referenced by any manifest.",
)
@click.option(
    "--older-than",
    default=None,
    metavar="DURATION",
    help="Delete shard data for runs older than DURATION (e.g. 7d, 24h).",
)
@click.option(
    "--keep-last",
    default=None,
    type=int,
    metavar="N",
    help="Keep shard data for only the N most recent runs.",
)
@click.option(
    "--max-retries",
    default=3,
    type=int,
    help="Retry count for transient S3 errors (default 3).",
)
@click.pass_context
def cleanup_cmd(
    ctx: click.Context,
    dry_run: bool,
    include_old_runs: bool,
    older_than: str | None,
    keep_last: int | None,
    max_retries: int,
) -> None:
    """Delete stale attempt directories and optionally old run data from S3."""
    from .._writer_core import cleanup_old_runs, cleanup_stale_attempts
    from ..storage import create_s3_client
    from ..type_defs import RetryConfig
    from .output import build_cleanup_result

    output_cfg = _get_output_cfg(ctx)
    params = _ensure_init_params(ctx)
    store_cfg: ManifestStoreConfig = params["store_cfg"]
    manifest_store = _build_manifest_store(store_cfg, params)

    try:
        # Resolve target manifest without mutating _CURRENT
        target_ref = _resolve_manifest_ref(
            manifest_store,
            ref=params.get("manifest_ref"),
            offset=params.get("manifest_offset"),
        )
        if target_ref is not None:
            manifest = manifest_store.load_manifest(target_ref)
        else:
            current_ref = manifest_store.load_current()
            if current_ref is None:
                raise click.ClickException("No current manifest found")
            manifest = manifest_store.load_manifest(current_ref.ref)

        cred_provider = params.get("credential_provider")
        credentials = cred_provider.resolve() if cred_provider else None
        client = create_s3_client(
            credentials=credentials,
            connection_options=params.get("s3_connection_options"),
        )
        retry_config = RetryConfig(max_retries=max_retries)

        # 1. Always clean stale attempts for the current manifest's run
        all_actions = cleanup_stale_attempts(
            manifest, s3_client=client, dry_run=dry_run, retry_config=retry_config
        )

        # 2. Optionally clean old runs
        wants_old_runs = (
            include_old_runs or older_than is not None or keep_last is not None
        )
        if wants_old_runs:
            refs = manifest_store.list_manifests(limit=10_000)
            current_run_id = manifest.required_build.run_id

            # Compute protected set: each active flag contributes a set, then intersect
            criteria_sets: list[set[str]] = []

            if include_old_runs:
                criteria_sets.append({r.run_id for r in refs})

            if older_than is not None:
                delta = _parse_duration(older_than)
                cutoff = datetime.now(UTC) - delta
                criteria_sets.append(
                    {r.run_id for r in refs if r.published_at >= cutoff}
                )

            if keep_last is not None:
                if keep_last < 1:
                    raise click.BadParameter("--keep-last must be >= 1")
                criteria_sets.append({r.run_id for r in refs[:keep_last]})

            protected: set[str] = criteria_sets[0] if criteria_sets else set()
            for s in criteria_sets[1:]:
                protected = protected & s

            # Current manifest's run is always protected
            protected.add(current_run_id)

            old_run_actions = cleanup_old_runs(
                manifest.required_build.s3_prefix,
                manifest.required_build.shard_prefix,
                protected_run_ids=protected,
                s3_client=client,
                dry_run=dry_run,
                retry_config=retry_config,
            )
            all_actions.extend(old_run_actions)

        result = build_cleanup_result(
            all_actions, dry_run=dry_run, run_id=manifest.required_build.run_id
        )
        emit(result, output_cfg)
    except click.ClickException:
        raise
    except Exception as exc:
        result = build_error_result("cleanup", None, str(exc))
        emit(result, output_cfg, file=sys.stderr)
        sys.exit(1)


@cli.command("search")
@click.argument("query", required=False)
@click.option(
    "--vector-file",
    "vector_file",
    type=click.Path(exists=True),
    help="Path to a .npy file containing the query vector.",
)
@click.option("--top-k", default=10, type=int, help="Number of results to return.")
@click.option(
    "--shard-ids",
    "shard_ids_str",
    default=None,
    help="Comma-separated list of shard IDs to search.",
)
@click.pass_context
def search_cmd(
    ctx: click.Context,
    query: str | None,
    vector_file: str | None,
    top_k: int,
    shard_ids_str: str | None,
) -> None:
    """Search a vector snapshot by approximate nearest-neighbor search.

    Provide either a positional QUERY as comma-separated floats
    (e.g. ``0.1,0.2,0.3``) or a ``--vector-file`` pointing to a .npy file.
    """
    output_cfg = _get_output_cfg(ctx)
    params = _ensure_init_params(ctx)
    store_cfg: ManifestStoreConfig = params["store_cfg"]

    try:
        import numpy as np
    except ImportError as exc:
        raise click.ClickException(
            "Vector search requires numpy. "
            "Install a supported vector extra, for example: "
            "pip install 'shardyfusion[vector]' or "
            "'shardyfusion[vector-lancedb]' or "
            "'shardyfusion[vector-sqlite]'"
        ) from exc

    # Parse query vector
    if vector_file is not None:
        query_vector = np.load(vector_file)
        if query_vector.ndim != 1:
            raise click.ClickException(
                f"Query vector must be 1-D, got shape {query_vector.shape}"
            )
    elif query is not None:
        try:
            parts = [float(x.strip()) for x in query.split(",")]
        except ValueError as exc:
            raise click.ClickException(
                "QUERY must be comma-separated floats, e.g. '0.1,0.2,0.3'"
            ) from exc
        query_vector = np.array(parts, dtype=np.float32)
    else:
        raise click.UsageError("Provide either a positional QUERY or --vector-file")

    shard_ids: list[int] | None = None
    if shard_ids_str is not None:
        try:
            shard_ids = [int(x.strip()) for x in shard_ids_str.split(",")]
        except ValueError as exc:
            raise click.ClickException(
                "--shard-ids must be comma-separated integers"
            ) from exc

    manifest_store = _build_manifest_store(store_cfg, params)

    pinned_ref = _resolve_manifest_ref_obj(
        manifest_store,
        ref=params.get("manifest_ref"),
        offset=params.get("manifest_offset"),
    )
    if pinned_ref is not None:
        manifest_store = _PinnedManifestStore(manifest_store, pinned_ref)

    try:
        ref = manifest_store.load_current()
        if ref is None:
            raise click.ClickException("No CURRENT manifest found")
        manifest = manifest_store.load_manifest(ref.ref)
    except Exception as exc:
        raise click.ClickException(f"Failed to load manifest: {exc}") from exc

    vector_meta = manifest.custom.get("vector")
    if not isinstance(vector_meta, dict) or not vector_meta:
        raise click.ClickException(
            "Snapshot does not contain vector metadata. "
            "Use a vector-enabled writer to build searchable snapshots."
        )

    s3_prefix = params["s3_prefix"]
    local_root = params["reader_cfg"].local_root
    cred_provider = params["credential_provider"]
    s3_conn_opts = params["s3_connection_options"]
    reader_cfg: ReaderConfig = params["reader_cfg"]

    # Honor --sqlite-mode / --sqlite-auto-* overrides for the unified reader's
    # KV factory.  The unified reader's auto-dispatch otherwise constructs the
    # adaptive factory with default thresholds and ignores reader_cfg.
    unified_kv_factory = _build_unified_kv_factory(
        vector_meta,
        reader_cfg=reader_cfg,
        credential_provider=cred_provider,
        s3_connection_options=s3_conn_opts,
    )

    try:
        if vector_meta.get("unified"):
            from ..reader.unified_reader import UnifiedShardedReader

            reader = UnifiedShardedReader(
                s3_prefix=s3_prefix,
                local_root=local_root,
                manifest_store=manifest_store,
                credential_provider=cred_provider,
                s3_connection_options=s3_conn_opts,
                reader_factory=unified_kv_factory,
            )
            try:
                response = reader.search(query_vector, top_k=top_k, shard_ids=shard_ids)
                result = build_search_result(response, top_k)
                emit(result, output_cfg)
            finally:
                reader.close()
        else:
            from ..vector.reader import ShardedVectorReader

            reader = ShardedVectorReader(
                s3_prefix=s3_prefix,
                local_root=local_root,
                manifest_store=manifest_store,
                credential_provider=cred_provider,
                s3_connection_options=s3_conn_opts,
            )
            try:
                response = reader.search(query_vector, top_k=top_k, shard_ids=shard_ids)
                result = build_search_result(response, top_k)
                emit(result, output_cfg)
            finally:
                reader.close()
    except ImportError as exc:
        raise click.ClickException(
            "Vector search requires vector extras. "
            "Install a supported vector extra, for example: "
            "pip install 'shardyfusion[vector]' or "
            "'shardyfusion[vector-lancedb]' or "
            "'shardyfusion[vector-sqlite]'"
        ) from exc
    except Exception as exc:
        result = build_error_result("search", None, str(exc))
        emit(result, output_cfg, file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cli()
