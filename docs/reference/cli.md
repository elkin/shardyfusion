# CLI reference

Entry point: `shardy = "shardyfusion.cli.app:main"`. Top-level group at `shardyfusion/cli/app.py:209`.

Snapshot location and reader settings come from `reader.toml` + `credentials.toml`. Example configs ship in `shardyfusion/cli/example_config/`. See [`use-cases/operate-cli.md`](../use-cases/operate-cli.md) for the full configuration walkthrough.

## Global flags

| Flag | Purpose |
|---|---|
| `--current-url URL` | S3 URL to the `_CURRENT` pointer. Overrides `reader.toml` and `SHARDY_CURRENT` env. |
| `--config PATH` | Path to `reader.toml`. Overrides `SHARDY_CONFIG` env and default search. |
| `--credentials PATH` | Path to `credentials.toml`. Overrides `SHARDY_CREDENTIALS` env and default search. |
| `--s3-option KEY=VALUE` | Override an S3 connection option (e.g. `addressing_style=path`). Repeatable. |
| `--output-format {json,jsonl,table,text}` | Override `[output].format` from `reader.toml`. |
| `--ref REF` | Pin the reader to a specific manifest ref. Mutually exclusive with `--offset`. |
| `--offset N` | Pin the reader to the Nth previous manifest (`0` = latest). Mutually exclusive with `--ref`. |
| `-h`, `--help` | Show help. |
| `--version` | Show shardyfusion version. |

## Subcommands

| Command | Purpose | Mutates? |
|---|---|---|
| `get <key>` | Single-key lookup. `--routing-context KEY=VALUE` (repeatable). `--strict` exits 1 if key missing. | no |
| `multiget <keys>...` | Multi-key lookup. Pass `-` to read keys from stdin. `--routing-context` repeatable. | no |
| `info` | Manifest summary. | no |
| `shards` | Per-shard details. | no |
| `health [--staleness-threshold SECONDS]` | Reader health (exit `0`/`1`/`2` = healthy/degraded/unhealthy). | no |
| `route <key>` | Show which shard a key routes to. `--routing-context` repeatable. | no |
| `history [--limit N]` | List recent published manifests (default 10). | no |
| `schema [--type {manifest,current-pointer,sqlite-manifest}]` | Print JSON Schema or SQLite manifest DDL. | no |
| `exec --script FILE [--output FILE]` | Run a YAML batch script. | no |
| `rollback (--ref REF \| --run-id RUN_ID \| --offset N)` | Atomically swap `_CURRENT` to a previous manifest. Exactly one selector required. | **yes** |
| `cleanup [--dry-run] [--include-old-runs] [--older-than DURATION] [--keep-last N] [--max-retries N]` | Delete stale attempts and optionally old runs. `DURATION` = `7d`, `24h`. | **yes** (without `--dry-run`) |

There is **no** `publish`, `delete`, or `repl` subcommand. Interactive mode is entered by invoking `shardy` with **no subcommand**.

## Batch DSL

`exec --script script.yaml` runs a list of read-only ops. Supported ops: `get, multiget, refresh, info, shards, route, health, history`.

Example:

```yaml
- op: refresh
- op: get
  key: user-1
- op: multiget
  keys: [user-1, user-2]
```

Use `--output FILE` to redirect results to a file instead of stdout.

## Interactive mode

Invoke `shardy` (or `shardy --current-url …`) with **no subcommand** to enter the `cmd.Cmd`-based REPL (`shardyfusion/cli/interactive.py`). Available REPL commands: `get`, `multiget`, `refresh`, `info`, `shards`, `health`, `route`, `schema`, `history`, `use`, `quit`/`exit`.

The REPL `use` command is **session-local** — it changes which snapshot the REPL reads from. It does **not** mutate `_CURRENT` on the bucket.

## Output

`shardyfusion/cli/output.py`:

- `json` — single JSON document.
- `jsonl` — one JSON object per line.
- `table` — fixed-width table.
- `text` — plain text.

The default comes from `[output].format` in `reader.toml` (`jsonl` in the shipped example). Value encoding is set per-config via `[output].value_encoding` (`base64` | `utf8` | `hex`).

## Exit codes

- `0` — success.
- `1` — error (or `health` returned `degraded`; or `get --strict` and key missing).
- `2` — `health` returned `unhealthy`.

## See also

- [`use-cases/operate-cli.md`](../use-cases/operate-cli.md)
- [`use-cases/operate-manifest-history-and-rollback.md`](../use-cases/operate-manifest-history-and-rollback.md)
