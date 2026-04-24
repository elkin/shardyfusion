# Operate snapshots from the command line

Use the **`shardy` CLI** to inspect, query, and manage published snapshots.

## When to use

- Quick lookups during development or incident response.
- Operational tasks: rollback, cleanup, schema inspection.
- Scripted batches via `exec`.

## When NOT to use

- High-QPS production reads — embed the reader in your service.
- Programmatic batch jobs that don't fit the YAML batch DSL — call the Python API directly.

## Install

```bash
uv add 'shardyfusion[cli,read]'
```

The `cli` extra pulls only `click>=8.0`. `pyyaml>=6.0` is a base dependency.

Entry point: `shardy = "shardyfusion.cli.app:main"`.

## Configuration

Unlike CLIs that take per-invocation snapshot flags, `shardy` reads its snapshot location and reader settings from a TOML config plus an optional `--current-url` override:

| Source | What it provides |
|---|---|
| `reader.toml` (`--config PATH` or `SHARDY_CONFIG` env or default search) | `current_url`, `local_root`, `thread_safety`, `max_workers`, `slate_env_file`, `credentials_profile`, output options. |
| `credentials.toml` (`--credentials PATH` or `SHARDY_CREDENTIALS` env or default search) | Per-profile S3 endpoint, region, access keys, addressing style, timeouts. |
| `--current-url URL` | Overrides `reader.current_url`. Also accepts `SHARDY_CURRENT` env. |
| `--s3-option KEY=VALUE` | Repeatable override of any S3 connection option (e.g. `addressing_style=path`). |
| `--output-format {json,jsonl,table,text}` | Overrides `[output].format` from `reader.toml`. |
| `--ref REF` / `--offset N` | Pin the reader to a specific manifest (mutually exclusive). |

Example `reader.toml`:

```toml
[reader]
current_url = "s3://bucket/prefix/_CURRENT"
thread_safety = "lock"
max_workers = 4
credentials_profile = "default"

[output]
format = "jsonl"
value_encoding = "base64"
null_repr = "null"
```

Example `credentials.toml`:

```toml
[default]
endpoint_url      = "https://s3.amazonaws.com"
region            = "us-east-1"
access_key_id     = "AKID…"
secret_access_key = "SECRET…"
addressing_style  = "virtual"
```

## Minimal example

```bash
# Point lookup (current_url comes from reader.toml or SHARDY_CURRENT env)
shardy get user-123

# Override the snapshot inline
shardy --current-url s3://bucket/snapshots/users/_CURRENT info

# Pin to a previous manifest
shardy --offset 1 get user-123

# Interactive mode — invoked by omitting any subcommand
shardy
```

## Subcommands

Top-level group at `shardyfusion/cli/app.py:209`. Subcommands:

| Command | Purpose |
|---|---|
| `get <key>` | Single key lookup. `--routing-context KEY=VALUE` (repeatable) for CEL split mode. `--strict` exits 1 if the key is missing. |
| `multiget <keys>...` | Multi-key lookup. Pass `-` to read keys from stdin. |
| `info` | Manifest summary. |
| `shards` | Per-shard details. |
| `health [--staleness-threshold SECONDS]` | Reader health. Exit codes `0`/`1`/`2` for healthy/degraded/unhealthy. |
| `route <key>` | Show which shard a key routes to (no lookup). |
| `exec --script FILE [--output FILE]` | Run a YAML batch script. |
| `history [--limit N]` | List recent published manifests (default 10). |
| `rollback (--ref REF \| --run-id RUN_ID \| --offset N)` | Atomically swap `_CURRENT` to a previous manifest. Exactly one selector required. |
| `cleanup [--dry-run] [--include-old-runs] [--older-than DURATION] [--keep-last N] [--max-retries N]` | Delete stale attempt directories and optionally old run data. `DURATION` is e.g. `7d`, `24h`. |
| `schema [--type {manifest,current-pointer,sqlite-manifest}]` | Print the JSON Schema for the manifest, current pointer, or SQLite manifest DDL. |

There is **no `publish`/`delete`** subcommand — publishing happens through writers; deletion goes through `cleanup`.

## Output formats

`cli/output.py` supports `json`, `jsonl`, `table`, `text`. Default comes from `[output].format` in `reader.toml` (`jsonl` in the shipped example). Default value encoding is `base64`.

```bash
shardy --output-format json get user-123
```

## Batch mode

`exec --script script.yaml` runs a sequence of read-only ops. Supported: `get, multiget, refresh, info, shards, route, health, history`.

## Interactive mode

Invoking `shardy` with **no subcommand** drops you into a `cmd.Cmd`-based REPL (`shardyfusion/cli/interactive.py`). Available commands include `get`, `multiget`, `refresh`, `info`, `shards`, `health`, `route`, `schema`, `history`, `use`, and `quit`/`exit`. The REPL `use` command is **session-local** — it does **not** mutate `_CURRENT` on the bucket.

## Functional / Non-functional properties

- Read-side commands honor the same routing / refresh model as `ShardedReader`.
- `--ref` / `--offset` pin the reader to a specific manifest **without** mutating `_CURRENT`.
- `rollback` and `cleanup` are the only state-mutating commands.

## Guarantees

- `rollback` performs the same atomic `_CURRENT` swap as a normal publish.
- `cleanup --dry-run` lists what would be deleted without mutating S3.

## Weaknesses

- No async CLI; reads use the sync reader.
- No publish-from-CLI flow.

## Failure modes & recovery

| Failure | Surface | Recovery |
|---|---|---|
| Bad config | `click.UsageError` / non-zero exit, stderr | Fix flags / TOML / env. |
| Missing `_CURRENT` pointer | `click.ClickException("No current manifest found")` for `cleanup`; `ReaderStateError("CURRENT pointer not found")` for read ops | Verify writer published; check `current_url`. |
| `--ref` and `--offset` both set | `click.UsageError("--ref and --offset are mutually exclusive")` | Choose one. |
| `--offset` out of range | `click.ClickException("Offset N out of range …")` | Use `history` to see what's available. |
| Cleanup transient S3 error | logged warning, retried up to `--max-retries` | Rerun. |

## See also

- [History & rollback](history-rollback.md)
- [Prometheus metrics](prometheus-metrics.md)
- [OTel metrics](otel-metrics.md)
- [Reference → CLI](../reference/cli.md)
