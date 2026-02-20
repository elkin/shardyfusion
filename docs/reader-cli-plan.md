# Reader CLI Utility — Design Plan

## Overview

A standalone CLI tool (`slate-reader`) that wraps `SlateShardedReader` for
interactive ad-hoc lookups, batch script execution, and manifest inspection.
All operations reuse a single in-memory manifest load; credentials are kept in
a dedicated file separate from the operational config.

---

## Commands

The CLI maps directly to `SlateShardedReader` methods:

| CLI command | Library method |
|---|---|
| `get KEY` | `reader.get(key)` |
| `multiget KEY [KEY …]` | `reader.multi_get(keys)` |
| `refresh` | `reader.refresh()` |
| `info` | manifest metadata (constructor state) |
| `exec --script FILE` | batch runner (multiple ops, single load) |

---

## Required Argument

The full S3 URL of the CURRENT pointer is the **required first positional argument**:

```
slate-reader s3://bucket/prefix/_CURRENT [OPTIONS] [COMMAND]
```

- `s3_prefix` is derived by stripping the last path segment.
- `current_name` is the last path segment (e.g. `_CURRENT`).
- If no COMMAND is given the tool enters **interactive mode**.

This argument may also be supplied via the `SLATE_READER_CURRENT` environment
variable or as `current_url` in `reader.toml`, but the positional argument
always wins.

---

## Configuration

### Non-sensitive: `reader.toml`

```toml
[reader]
local_root           = "/tmp/slatefusion"
thread_safety        = "lock"       # "lock" | "pool"
max_workers          = 4
slate_env_file       = ""           # path to slatedb env file, optional
credentials_profile  = "default"    # profile name from credentials.toml

[output]
format         = "jsonl"       # json | jsonl | table | text
value_encoding = "base64"      # base64 | utf8 | hex — how to emit raw bytes
null_repr      = "null"        # string representation for missing keys
```

### Sensitive: `credentials.toml`

```toml
[default]
endpoint_url      = "https://s3.amazonaws.com"   # omit for real AWS
region            = "us-east-1"
access_key_id     = "AKID…"
secret_access_key = "SECRET…"
# session_token   = ""   # optional

[ceph-prod]
endpoint_url      = "https://ceph.prod.example.com"
region            = "us-east-1"
access_key_id     = "…"
secret_access_key = "…"
```

**Why a separate file:**

- Can carry strict OS permissions (0600); the tool warns when permissions are
  wider on UNIX.
- Shared across every dataset regardless of which bucket it lives in — each
  dataset's `reader.toml` references a profile name, the credentials file
  holds all storage endpoints.
- Clean secret-scanning boundary: `reader.toml` is safe to commit, credentials
  are not.
- **Multi-bucket / multi-cluster growth path:** as datasets spread across
  different S3-compatible clusters, add a new profile block; no change to any
  `reader.toml` is needed unless the mapping changes.

### Config search order (highest → lowest priority)

| Priority | reader config | credentials |
|---|---|---|
| 1 | `--config PATH` CLI flag | `--credentials PATH` CLI flag |
| 2 | `SLATE_READER_CONFIG` env var | `SLATE_READER_CREDENTIALS` env var |
| 3 | `./reader.toml` | `./credentials.toml` |
| 4 | `~/.config/slatefusion/reader.toml` | `~/.config/slatefusion/credentials.toml` |

If no credentials file is found the tool falls back silently to the standard
boto3 chain (env vars, `~/.aws/credentials`, IAM role metadata).

---

## Usage Examples

### One-shot commands

```bash
# Single key lookup
slate-reader s3://bucket/prefix/_CURRENT get 12345

# Batch lookup
slate-reader s3://bucket/prefix/_CURRENT multiget 10 20 30

# Refresh the manifest
slate-reader s3://bucket/prefix/_CURRENT refresh

# Show manifest metadata
slate-reader s3://bucket/prefix/_CURRENT info

# Explicit config paths
slate-reader s3://bucket/prefix/_CURRENT \
  --config ~/datasets/orders/reader.toml \
  --credentials ~/.config/slatefusion/credentials.toml \
  get 42
```

### Interactive mode

Invoked when no subcommand is provided. The reader is initialised **once** at
startup; all subsequent commands reuse the in-memory manifest.

```
$ slate-reader s3://bucket/prefix/_CURRENT

Loaded manifest run_id=2024-01-15T12:00:00Z  (4 shards, hash sharding)
slate> get 12345
{"op":"get","key":"12345","found":true,"value":"aGVsbG8="}
slate> multiget 10 20 30
{"op":"multiget","results":[{"key":"10","found":false},{"key":"20","found":true,"value":"d29ybGQ="},{"key":"30","found":true,"value":"Zm9v"}]}
slate> refresh
{"op":"refresh","changed":false}
slate> info
{"op":"info","run_id":"2024-01-15T12:00:00Z","num_dbs":4,"sharding":"hash","created_at":"2024-01-15T12:00:00Z"}
slate> quit
```

The REPL is implemented with Python's stdlib `cmd.Cmd` (no extra dependency).
`readline` integration provides command history automatically.

### Batch / script mode

```bash
slate-reader s3://bucket/prefix/_CURRENT exec \
  --script commands.yaml \
  --output results.jsonl
```

The manifest is loaded once; all commands in the script run against the same
snapshot.

#### Script format: YAML

YAML was chosen over JSON (no comments), TOML (awkward heterogeneous lists),
and CSV (insufficient structure).

```yaml
# commands.yaml
on_error: stop    # stop | continue  (default: stop)

commands:
  - op: get
    key: "12345"

  - op: multiget
    keys:
      - "10"
      - "20"
      - "30"

  - op: refresh

  - op: get
    key: "42"
```

#### Output: JSONL (default for batch)

One JSON object per command result, streamed immediately:

```jsonl
{"op":"get","key":"12345","found":true,"value":"aGVsbG8="}
{"op":"multiget","results":[{"key":"10","found":false},{"key":"20","found":true,"value":"d29ybGQ="}]}
{"op":"refresh","changed":false}
{"op":"get","key":"42","found":true,"value":"Zm9v"}
```

JSONL is the right default for batch because results are streamed (no buffering
a full JSON array), there is a strict one-to-one mapping between input commands
and output lines, and it is trivially processable with `jq`.

---

## Output Formats

| Format | Best for |
|---|---|
| `jsonl` | Machine processing, piping, batch scripts (default for `exec`) |
| `json` | Pretty-printed; interactive exploration (default for interactive mode) |
| `table` | Human-readable multiget results in a terminal |
| `text` | Plain `KEY=VALUE`; simplest shell scripting |

Configurable via `[output] format` in `reader.toml` or the `--output-format`
flag per invocation.

---

## Error Handling

- Non-zero exit code on any unhandled error.
- `on_error: continue` in batch scripts: emit an error object and proceed.
  ```jsonl
  {"op":"get","key":"bad","error":"ReaderStateError: CURRENT pointer not found"}
  ```
- In interactive mode errors print to stderr; the REPL continues.

---

## Module Layout

```
slatedb_spark_sharded/
└── cli/
    ├── __init__.py          # exports main()
    ├── app.py               # click top-level group + subcommands (get, multiget,
    │                        #   refresh, info, exec)
    ├── config.py            # ReaderConfig + CredentialsConfig dataclasses,
    │                        #   TOML loaders, search-order resolution
    ├── interactive.py       # cmd.Cmd REPL backed by SlateShardedReader
    ├── batch.py             # YAML script loader + sequential executor
    ├── output.py            # formatters: json, jsonl, table, text
    └── example_config/
        ├── reader.toml      # annotated example (safe to commit)
        └── credentials.toml # annotated template (never commit real values)
```

---

## `pyproject.toml` Additions

```toml
[project.optional-dependencies]
cli = [
  "click>=8.0",
  "pyyaml>=6.0",
]

[project.scripts]
slate-reader = "slatedb_spark_sharded.cli.app:main"
```

All new dependencies go under the `cli` extra so the core library and existing
`read` / `writer` extras remain unaffected.

---

## New Dependencies

| Package | Purpose | Python built-in? |
|---|---|---|
| `click>=8.0` | CLI framework (argument parsing, subcommands, help) | No |
| `pyyaml>=6.0` | YAML script file parsing | No |

`tomllib` (TOML parsing) ships with the Python 3.11 stdlib, so no third-party
backfill is needed given the minimum Python version is now **3.11**.

---

## Key Design Decisions

### CURRENT URL as the primary entry point
Providing the full CURRENT pointer URL as a positional argument keeps every
invocation self-contained — the dataset identity is explicit. The `s3_prefix`
and `current_name` are derived, not configured separately.

### TOML for configs
`pyproject.toml` is already TOML; `tomllib` ships with the Python 3.11 stdlib
(the project minimum), so TOML config parsing requires no extra dependency. The
format is human-friendly with good comment support.

### YAML for scripts
Comments, multi-line strings, and readable heterogeneous lists make YAML the
most writable format for hand-authored scripts. Machine-generated scripts can
still use YAML (emit with PyYAML's safe dumper). JSONL would be the alternative
for fully machine-generated workloads; both can coexist via a `--script-format`
flag if needed.

### `cmd.Cmd` for the REPL
No extra dependency. Provides readline history out of the box on platforms where
`readline` is available. If richer UX is needed later (`prompt_toolkit` for
auto-complete, syntax highlighting) it can be swapped in without changing the
command implementations.

### Separate credentials file
Mirrors the pattern already established by `~/.aws/credentials`. One file per
user/machine, multiple named profiles for multiple storage clusters. Each
dataset config references a profile name, so rotating or adding credentials
never requires touching dataset configs.
