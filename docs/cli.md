# CLI (`shardy`)

The `shardy` command-line tool wraps `ConcurrentShardedReader` for interactive
lookups, batch script execution, and manifest inspection — no Python code needed.

## Installation

```bash
# CLI only (click + pyyaml, no Spark/Java required)
uv sync --extra cli

# Or as part of the full dev environment
uv sync --all-extras --dev
```

## Running Locally

During development, run via `uv run` so the tool picks up the local package
without a separate install step:

```bash
uv run shardy --help
```

After `uv sync`, the entry point is also available directly if the virtualenv
is activated:

```bash
shardy --help
```

## Quick Start

Every invocation needs a `_CURRENT` pointer URL. Supply it via `--current-url`,
the `SHARDY_CURRENT` environment variable, or a `reader.toml` config file
(see [Configuration](#configuration) below).

```bash
# Single key lookup
uv run shardy --current-url s3://bucket/prefix/_CURRENT get 42

# Multiple keys
uv run shardy --current-url s3://bucket/prefix/_CURRENT multiget 1 2 3

# Multiple keys from stdin
echo -e "1\n2\n3" | uv run shardy --current-url s3://bucket/prefix/_CURRENT multiget -

# Manifest info
uv run shardy --current-url s3://bucket/prefix/_CURRENT info

# Per-shard details
uv run shardy --current-url s3://bucket/prefix/_CURRENT shards

# Check which shard a key routes to
uv run shardy --current-url s3://bucket/prefix/_CURRENT route 42

# Reload manifest
uv run shardy --current-url s3://bucket/prefix/_CURRENT refresh

# Interactive REPL (no subcommand)
uv run shardy --current-url s3://bucket/prefix/_CURRENT
```

### Using an Environment Variable

To avoid repeating the URL on every call:

```bash
export SHARDY_CURRENT=s3://bucket/prefix/_CURRENT

uv run shardy get 42
uv run shardy multiget 1 2 3
uv run shardy info
uv run shardy shards
uv run shardy route 42
```

## Global Options

| Option | Description |
|---|---|
| `--current-url URL` | S3 URL to the `_CURRENT` pointer (overrides env / config) |
| `--config PATH` | Path to `reader.toml` (overrides `SHARDY_CONFIG` env) |
| `--credentials PATH` | Path to `credentials.toml` (overrides `SHARDY_CREDENTIALS` env) |
| `--s3-option KEY=VALUE` | Override S3 connection option (repeatable) |
| `--ref REF` | Load a specific manifest by reference (mutually exclusive with `--offset`) |
| `--offset N` | Load a manifest at position N in history — 0 is latest, 1 is previous, etc. (mutually exclusive with `--ref`) |
| `--output-format FORMAT` | Output format: `json`, `jsonl` (default), `table`, `text` |
| `--version` | Show version and exit |

## Subcommands

| Subcommand | Arguments | Description |
|---|---|---|
| `get` | `KEY` | Look up a single key |
| `multiget` | `KEY [KEY ...]` or `-` | Look up multiple keys; pass `-` to read from stdin |
| `info` | — | Show manifest metadata (run_id, num_dbs, sharding, key_encoding, row_count) |
| `shards` | — | Show per-shard details (db_id, row_count, min/max key, URL) |
| `route` | `KEY` | Show which shard a key routes to (without performing a lookup) |
| `refresh` | — | Reload `_CURRENT` and manifest |
| `history` | `[--limit N]` | List recent published manifests |
| `rollback` | `--ref REF \| --run-id RUN_ID \| --offset N` | Roll back the current pointer to a previous manifest |
| `schema` | `[--type manifest\|current-pointer]` | Print the JSON Schema for manifest or current-pointer formats |
| `exec` | `--script FILE [--output FILE]` | Execute a YAML batch script |

## Key Coercion

CLI keys are always strings. When the manifest uses an integer key encoding
(`u64be` or `u32be`), keys are automatically coerced to `int` before lookup.
For other encodings (e.g. `utf8`), keys are passed as-is.

The active key encoding is visible via `info` (the `key_encoding` field) and
affects `get`, `multiget`, and `route` commands.

## Configuration

### CURRENT URL Resolution

The URL is resolved in priority order:

1. `--current-url` CLI option
2. `SHARDY_CURRENT` environment variable
3. `current_url` in the `[reader]` section of `reader.toml`

### `reader.toml`

Searched in order: `./reader.toml`, `~/.config/shardy/reader.toml`,
or via `SHARDY_CONFIG` env / `--config` flag.

```toml
[reader]
current_url = "s3://bucket/prefix/_CURRENT"
local_root = "/tmp/shardy"
thread_safety = "lock"        # "lock" or "pool"
max_workers = 4
slate_env_file = "/path/to/env"
credentials_profile = "default"

[output]
format = "jsonl"              # json | jsonl | table | text
value_encoding = "base64"     # base64 | hex | utf8
null_repr = "null"
```

### `credentials.toml`

Searched in order: `./credentials.toml`, `~/.config/shardy/credentials.toml`,
or via `SHARDY_CREDENTIALS` env / `--credentials` flag.
The tool warns if file permissions are wider than `0600`.

```toml
[default]
endpoint_url = "http://localhost:9000"
region = "us-east-1"
access_key_id = "..."
secret_access_key = "..."
addressing_style = "path"
verify_ssl = true
connect_timeout = 10
read_timeout = 30
max_attempts = 3
```

### Inline S3 Option Overrides

Override individual S3 settings without editing config files:

```bash
uv run shardy --current-url s3://bucket/prefix/_CURRENT \
  --s3-option addressing_style=path \
  --s3-option verify_ssl=false \
  get 42
```

## Interactive REPL

When no subcommand is given, the CLI enters a `cmd.Cmd` REPL:

```
$ uv run shardy --current-url s3://bucket/prefix/_CURRENT

Loaded manifest run_id=2024-01-15T12:00:00Z  (4 shards, hash sharding)
shardy> get 42
{
  "op": "get",
  "key": "42",
  "found": true,
  "value": "aGVsbG8="
}
shardy> info
{
  "op": "info",
  "run_id": "2024-01-15T12:00:00Z",
  "num_dbs": 4,
  "sharding": "hash",
  "key_encoding": "u64be",
  "row_count": 1000000,
  ...
}
shardy> route 42
{
  "op": "route",
  "key": "42",
  "db_id": 2
}
shardy> shards
{
  "op": "shards",
  "shards": [...]
}
shardy> refresh
{
  "op": "refresh",
  "changed": false
}
shardy> quit
```

Interactive mode defaults to `json` (pretty-printed with indentation) output instead of `jsonl`.

REPL commands: `get KEY`, `multiget KEY [KEY ...]`, `info`, `shards`, `route KEY`,
`refresh`, `history [LIMIT]`, `use (--offset N | --ref REF | --latest)`,
`quit`/`exit`/`Ctrl-D`.

### REPL-Only Commands

#### `history [LIMIT]`

List recent manifests within the REPL:

```
shardy> history
shardy> history 5
```

#### `use`

Switch to a different manifest without restarting the REPL:

```
shardy> use --offset 1       # switch to the previous manifest
shardy> use --ref s3://...   # switch to a specific manifest ref
shardy> use --latest         # switch back to the latest manifest
```

After `use`, all subsequent lookups operate against the selected manifest until `use --latest` or `refresh` is called.

## Manifest History

List recent published manifests to see the history of snapshot builds:

```bash
uv run shardy --current-url s3://bucket/prefix/_CURRENT history
uv run shardy --current-url s3://bucket/prefix/_CURRENT history --limit 5
```

Output shows each manifest's reference, run ID, and publication timestamp in reverse chronological order.

## Rollback

Roll back the current pointer to a previous manifest. Exactly one targeting option is required:

```bash
# Roll back by manifest reference
uv run shardy rollback --ref s3://bucket/prefix/manifests/2026-03-13.../manifest

# Roll back by run ID
uv run shardy rollback --run-id abc123

# Roll back by offset (1 = previous, 2 = two versions ago)
uv run shardy rollback --offset 1
```

Rollback calls `set_current()` on the manifest store, updating the `_CURRENT` pointer. The old manifest data remains intact — rollback only changes which manifest is "current".

!!! warning
    Rollback affects all readers loading from this `_CURRENT` pointer. Readers will pick up the rolled-back manifest on their next `refresh()`.

## Schema

Print the JSON Schema for manifest or current-pointer formats. Generated at runtime from the Pydantic models:

```bash
# Manifest schema (default)
uv run shardy schema

# Current-pointer schema
uv run shardy schema --type current-pointer
```

The schemas are useful for validating external tools that produce or consume shardyfusion manifests.

## Loading a Specific Manifest

The `--ref` and `--offset` global options let you target a specific manifest for any subcommand:

```bash
# Look up a key against the previous manifest
uv run shardy --offset 1 get 42

# Inspect a specific manifest's metadata
uv run shardy --ref s3://bucket/prefix/manifests/.../manifest info
```

These options are mutually exclusive. The reader loads the targeted manifest instead of following the `_CURRENT` pointer.

## Batch Scripts

Execute a YAML file containing multiple commands:

```bash
uv run shardy --current-url s3://bucket/prefix/_CURRENT \
  exec --script commands.yaml --output results.jsonl
```

### Script Format

```yaml
on_error: continue    # stop (default) | continue
commands:
  - op: get
    key: 42
  - op: multiget
    keys: [1, 2, 3]
  - op: info
  - op: shards
  - op: route
    key: 42
  - op: refresh
```

Batch mode defaults output to `jsonl` — one JSON object per command, streamed
immediately.

## Output Formats

| Format | Best for |
|---|---|
| `jsonl` | Machine processing, piping, batch scripts (default for `exec` and one-shot) |
| `json` | Pretty-printed; interactive exploration (default for REPL) |
| `table` | Human-readable multiget results in a terminal |
| `text` | Plain `KEY=VALUE`; simplest shell scripting |

Set via `--output-format`, `[output] format` in `reader.toml`, or mode defaults.

## Local Development with MinIO

For local testing with a MinIO or S3-compatible store:

```bash
# Start MinIO (example)
docker run -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"

# Create credentials.toml for local MinIO
cat > credentials.toml << 'EOF'
[default]
endpoint_url = "http://localhost:9000"
region = "us-east-1"
access_key_id = "minioadmin"
secret_access_key = "minioadmin"
addressing_style = "path"
verify_ssl = false
EOF
chmod 600 credentials.toml

# Use it
uv run shardy \
  --current-url s3://my-bucket/prefix/_CURRENT \
  --credentials credentials.toml \
  info
```
