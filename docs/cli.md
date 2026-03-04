# CLI (`slate-reader`)

The `slate-reader` command-line tool wraps `ConcurrentShardedReader` for interactive
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
uv run slate-reader --help
```

After `uv sync`, the entry point is also available directly if the virtualenv
is activated:

```bash
slate-reader --help
```

## Quick Start

Every invocation needs a `_CURRENT` pointer URL. Supply it via `--current-url`,
the `SLATE_READER_CURRENT` environment variable, or a `reader.toml` config file
(see [Configuration](#configuration) below).

```bash
# Single key lookup
uv run slate-reader --current-url s3://bucket/prefix/_CURRENT get 42

# Multiple keys
uv run slate-reader --current-url s3://bucket/prefix/_CURRENT multiget 1 2 3

# Manifest info
uv run slate-reader --current-url s3://bucket/prefix/_CURRENT info

# Reload manifest
uv run slate-reader --current-url s3://bucket/prefix/_CURRENT refresh

# Interactive REPL (no subcommand)
uv run slate-reader --current-url s3://bucket/prefix/_CURRENT
```

### Using an Environment Variable

To avoid repeating the URL on every call:

```bash
export SLATE_READER_CURRENT=s3://bucket/prefix/_CURRENT

uv run slate-reader get 42
uv run slate-reader multiget 1 2 3
uv run slate-reader info
```

## Global Options

| Option | Description |
|---|---|
| `--current-url URL` | S3 URL to the `_CURRENT` pointer (overrides env / config) |
| `--config PATH` | Path to `reader.toml` (overrides `SLATE_READER_CONFIG` env) |
| `--credentials PATH` | Path to `credentials.toml` (overrides `SLATE_READER_CREDENTIALS` env) |
| `--s3-option KEY=VALUE` | Override S3 connection option (repeatable) |
| `--output-format FORMAT` | Output format: `json`, `jsonl` (default), `table`, `text` |

## Subcommands

| Subcommand | Arguments | Description |
|---|---|---|
| `get` | `KEY` | Look up a single key |
| `multiget` | `KEY [KEY ...]` | Look up multiple keys (space-separated) |
| `info` | — | Show manifest metadata (run_id, num_dbs, sharding strategy) |
| `refresh` | — | Reload `_CURRENT` and manifest |
| `exec` | `--script FILE [--output FILE]` | Execute a YAML batch script |

## Key Coercion

CLI keys are always strings. When the manifest uses an integer key encoding
(`u64be` or `u32be`), keys are automatically coerced to `int` before lookup.
For other encodings (e.g. `utf8`), keys are passed as-is.

## Configuration

### CURRENT URL Resolution

The URL is resolved in priority order:

1. `--current-url` CLI option
2. `SLATE_READER_CURRENT` environment variable
3. `current_url` in the `[reader]` section of `reader.toml`

### `reader.toml`

Searched in order: `./reader.toml`, `~/.config/slatefusion/reader.toml`,
or via `SLATE_READER_CONFIG` env / `--config` flag.

```toml
[reader]
current_url = "s3://bucket/prefix/_CURRENT"
local_root = "/tmp/slatefusion"
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

Searched in order: `./credentials.toml`, `~/.config/slatefusion/credentials.toml`,
or via `SLATE_READER_CREDENTIALS` env / `--credentials` flag.
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
uv run slate-reader --current-url s3://bucket/prefix/_CURRENT \
  --s3-option addressing_style=path \
  --s3-option verify_ssl=false \
  get 42
```

## Interactive REPL

When no subcommand is given, the CLI enters a `cmd.Cmd` REPL:

```
$ uv run slate-reader --current-url s3://bucket/prefix/_CURRENT

Loaded manifest run_id=2024-01-15T12:00:00Z  (4 shards, hash sharding)
slate> get 42
{"op":"get","key":"42","found":true,"value":"aGVsbG8="}
slate> multiget 1 2 3
{"op":"multiget","results":[{"key":"1","found":false},{"key":"2","found":true,"value":"d29ybGQ="}]}
slate> info
{"op":"info","run_id":"2024-01-15T12:00:00Z","num_dbs":4,"sharding":"hash"}
slate> refresh
{"op":"refresh","changed":false}
slate> quit
```

Interactive mode defaults to `json` (pretty-printed) output instead of `jsonl`.
REPL commands: `get KEY`, `multiget KEY [KEY ...]`, `info`, `refresh`,
`quit`/`exit`/`Ctrl-D`.

## Batch Scripts

Execute a YAML file containing multiple commands:

```bash
uv run slate-reader --current-url s3://bucket/prefix/_CURRENT \
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
uv run slate-reader \
  --current-url s3://my-bucket/prefix/_CURRENT \
  --credentials credentials.toml \
  info
```
