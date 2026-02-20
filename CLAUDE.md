# Claude Code — Project Notes

> General project structure, commands, and conventions are in `AGENTS.md`.
> This file records Claude-specific context and active work.

## Python Version

Minimum is **Python 3.11**. `AGENTS.md` and `tox.ini` still reference 3.10 in
some places — treat those as stale; 3.10 support has been dropped. Practically:
- Use `tomllib` (stdlib, 3.11+) for TOML parsing — no `tomli` backfill needed.
- Type checking target version and tox matrix should be updated to 3.11 when
  touched.

## Active Work: Reader CLI

Design plan lives at `docs/reader-cli-plan.md` on branch
`claude/plan-readers-cli-OExTm`. Key decisions recorded there:

- Entry point: `slate-reader <CURRENT-S3-URL> [OPTIONS] [COMMAND]`
- CURRENT URL is the required first positional argument; `s3_prefix` and
  `current_name` are derived from it.
- Two config files: non-sensitive `reader.toml` + sensitive `credentials.toml`
  (separate to allow strict file permissions and multi-cluster credential
  sharing).
- S3 connection options (`addressing_style`, `signature_version`, `verify_ssl`,
  `connect_timeout`, `read_timeout`, `max_attempts`) live inside each
  credentials profile — they are endpoint properties, not dataset properties.
  The `--s3-option KEY=VALUE` CLI flag covers per-invocation overrides.
- Interactive mode (no subcommand): `cmd.Cmd` REPL, manifest loaded once on
  startup.
- Batch mode (`exec --script FILE`): YAML command file, JSONL streamed output.
- New CLI optional-dep group: `click>=8.0`, `pyyaml>=6.0` (no `tomli` needed).
- Module layout: `slatedb_spark_sharded/cli/{app,config,interactive,batch,output}.py`

## Key Files

| File | Role |
|---|---|
| `slatedb_spark_sharded/reader.py` | `SlateShardedReader` — the class the CLI wraps |
| `slatedb_spark_sharded/storage.py` | `create_s3_client`, `S3ClientConfig` TypedDict |
| `slatedb_spark_sharded/manifest_readers.py` | `DefaultS3ManifestReader`, `ManifestReader` protocol |
| `slatedb_spark_sharded/manifest.py` | `CurrentPointer`, `ParsedManifest` and related models |
| `docs/reader-cli-plan.md` | Full CLI design plan (config schemas, command mapping, etc.) |
