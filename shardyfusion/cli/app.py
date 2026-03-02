"""Click CLI application for slate-reader."""

import sys
from typing import Any

import click

from .batch import run_script
from .config import (
    OutputConfig,
    ReaderConfig,
    build_s3_client_config,
    coerce_cli_key,
    coerce_s3_option,
    load_credentials_profile,
    load_reader_config,
    resolve_current_url,
    split_current_url,
)
from .interactive import SlateReaderRepl
from .output import (
    build_error_result,
    build_get_result,
    build_info_result,
    build_multiget_result,
    build_refresh_result,
    emit,
)

# ---------------------------------------------------------------------------
# Shared context keys
# ---------------------------------------------------------------------------

_CTX_INIT_PARAMS = "slate_init_params"


def _build_reader(ctx: click.Context) -> Any:
    """Construct a SlateShardedReader from the parameters stored in ctx.obj."""
    params = ctx.obj[_CTX_INIT_PARAMS]
    s3_prefix: str = params["s3_prefix"]
    current_name: str = params["current_name"]
    reader_cfg: ReaderConfig = params["reader_cfg"]
    s3_client_config = params["s3_client_config"]

    from ..manifest_readers import DefaultS3ManifestReader
    from ..reader import SlateShardedReader

    manifest_reader = DefaultS3ManifestReader(
        s3_prefix,
        current_name=current_name,
        s3_client_config=s3_client_config,
    )

    try:
        return SlateShardedReader(
            s3_prefix=s3_prefix,
            local_root=reader_cfg.local_root,
            manifest_reader=manifest_reader,
            current_name=current_name,
            slate_env_file=reader_cfg.slate_env_file,
            thread_safety=reader_cfg.thread_safety,
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
@click.option(
    "--current-url",
    "current_url",
    default=None,
    metavar="URL",
    help=(
        "S3 URL to the _CURRENT pointer "
        "(overrides SLATE_READER_CURRENT env and reader.toml)."
    ),
)
@click.option(
    "--config",
    "config_path",
    default=None,
    metavar="PATH",
    help="Path to reader.toml (overrides SLATE_READER_CONFIG env and default search).",
)
@click.option(
    "--credentials",
    "credentials_path",
    default=None,
    metavar="PATH",
    help=(
        "Path to credentials.toml "
        "(overrides SLATE_READER_CREDENTIALS env and default search)."
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
@click.pass_context
def cli(
    ctx: click.Context,
    current_url: str | None,
    config_path: str | None,
    credentials_path: str | None,
    s3_options: tuple[str, ...],
    output_format: str | None,
) -> None:
    """slate-reader — interactive and batch lookups for sharded SlateDB snapshots.

    \b
    CURRENT_URL may be supplied via --current-url, the SLATE_READER_CURRENT
    environment variable, or as current_url in reader.toml.

    \b
    When no subcommand is given the tool enters interactive mode.
    """
    ctx.ensure_object(dict)

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

    # Load reader + output config
    reader_cfg, output_cfg = load_reader_config(config_path)

    # Apply --output-format override
    if output_format:
        output_cfg = OutputConfig(
            format=output_format,
            value_encoding=output_cfg.value_encoding,
            null_repr=output_cfg.null_repr,
        )

    # Resolve CURRENT URL
    resolved_url = resolve_current_url(current_url, reader_cfg)
    s3_prefix, current_name = split_current_url(resolved_url)

    # Load credentials profile
    profile = load_credentials_profile(
        profile_name=reader_cfg.credentials_profile,
        credentials_path=credentials_path,
    )

    # Build S3 client config (credentials + connection options + per-invocation overrides)
    s3_client_config = build_s3_client_config(profile, s3_overrides)

    # Store init parameters in context for lazy reader construction in subcommands
    ctx.obj[_CTX_INIT_PARAMS] = {
        "s3_prefix": s3_prefix,
        "current_name": current_name,
        "reader_cfg": reader_cfg,
        "output_cfg": output_cfg,
        "s3_client_config": s3_client_config,
    }

    # If no subcommand was invoked, enter interactive mode
    if ctx.invoked_subcommand is None:
        with _build_reader(ctx) as reader:
            repl = SlateReaderRepl(reader, output_cfg)
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
    """Look up multiple KEYS (space-separated)."""
    output_cfg = _get_output_cfg(ctx)
    with _build_reader(ctx) as reader:
        try:
            coerced = [coerce_cli_key(k, reader.key_encoding) for k in keys]
            values = reader.multi_get(coerced)
            display_keys = list(keys)
            result = build_multiget_result(display_keys, values, output_cfg)
            emit(result, output_cfg)
        except Exception as exc:
            result = build_error_result("multiget", None, str(exc))
            emit(result, output_cfg, file=sys.stderr)
            sys.exit(1)


@cli.command("refresh")
@click.pass_context
def refresh_cmd(ctx: click.Context) -> None:
    """Reload CURRENT and manifest."""
    output_cfg = _get_output_cfg(ctx)
    with _build_reader(ctx) as reader:
        try:
            changed = reader.refresh()
            result = build_refresh_result(changed)
            emit(result, output_cfg)
        except Exception as exc:
            result = build_error_result("refresh", None, str(exc))
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cli()
