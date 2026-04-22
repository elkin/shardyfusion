---
name: validate-docs
description: Verify shardyfusion documentation matches source-code behavior. Checks public-API references, config dataclass fields, pyproject extras, and CLI surface mentioned anywhere under docs/. USE FOR validating docs sync, docs vs code drift, after any source change to a public API, after editing docs/, before committing docs changes. DO NOT USE FOR running tests, linting code, or general validation (use the `validate` skill).
compatibility: opencode
metadata:
  audience: contributors
  workflow: pre-commit
---

# Validate Docs

Verifies that everything `docs/` claims about the project still matches the source code. **Read-only**: it never edits docs or source files. The single output is a structured Markdown report you act on.

The implementation is shared with the Claude-compatible skill at `.claude/skills/validate-docs/`. Both load the same script.

## When to invoke

Trigger this skill whenever any of the following changed:

- A public symbol exported from `shardyfusion/__init__.py` (rename, add, remove, signature change).
- A field in any config dataclass (`WriteConfig`, `ShardingSpec`, `OutputOptions`, `ManifestOptions`, `VectorSpec`, `RetryConfig`, `S3ConnectionOptions`).
- An entry under `[project.optional-dependencies]` in `pyproject.toml`.
- A subcommand or option of the `shardy` CLI.
- Any file under `docs/`.

## What it checks

1. **Public-API references.** Every symbol referenced in `docs/**/*.md` either:
   - via mkdocstrings `::: shardyfusion.<symbol>`, or
   - via `from shardyfusion import …` / `from shardyfusion.<sub> import …` inside fenced Python blocks, or
   - via inline backticked qualified names like `` `shardyfusion.reader.AsyncShardedReader` `` —
   resolves to a real attribute reachable from the current source tree.
2. **Config dataclass fields.** Every field name appearing in a docs configuration table for `WriteConfig`, `ShardingSpec`, `OutputOptions`, `ManifestOptions`, `VectorSpec`, `RetryConfig`, or `S3ConnectionOptions` exists on that dataclass with the documented default (when a default is shown).
3. **Pyproject extras.** Every `uv sync --extra <name>` and `pip install shardyfusion[<name>]` line in `docs/**/*.md` references an extra defined in `[project.optional-dependencies]`.
4. **CLI surface.** Every `shardy <subcommand>` invocation and every long-form `--flag` referenced in CLI pages exists in `shardyfusion/cli/app.py` (Click command tree).

The skill does **not** check prose, examples that are intentionally illustrative, or external links.

## How to run

The skill ships with `.claude/skills/validate-docs/scripts/check_docs.py` (single source of truth). From the repo root:

```bash
uv run python .claude/skills/validate-docs/scripts/check_docs.py
```

Exit code is `0` when everything passes, `1` when any check fails. The output is a structured Markdown report grouped by check.

## Workflow

1. **Run the script.** Invoke the command above.
2. **Read the report.** It groups findings by check (`api`, `config`, `extras`, `cli`), then by source file, then lists each mismatch with the exact location (`docs/...:line`) and the expected vs actual value.
3. **Fix the docs first.** Documentation must match behavior. If a docs claim is wrong, update the docs.
4. **Only fix source if the source is wrong.** If the discrepancy is a real bug (e.g., a config field was renamed by accident and breaks user code), fix the source — do not paper over by editing the docs.
5. **Re-run until clean.** Loop until the script exits `0`.

## Limitations & escape hatches

- The script intentionally does not parse natural-language prose. Claims about behavior that are not symbol/field/extra/CLI references must be reviewed manually.
- Symbols that are dynamically constructed (e.g., re-exported via `__getattr__`) are resolved by attempting an `importlib.import_module` followed by `getattr`. If a symbol cannot be imported at validation time because the relevant extra is not installed, it is reported as `unverified` (warning), not `failed`.
- Per-file allow-listing: a docs file may include a top-level HTML comment `<!-- validate-docs: ignore -->` to skip the file (use only for hand-curated examples that intentionally diverge).
- A small `example_allowlist` (`{"foodb", "vector-foo"}`) lets `docs/contributing/` reference fictional extras for tutorials without false failures.

## Reporting format (example)

```
# Docs validation report

## api (2 findings)
- docs/use-cases/build-spark-slatedb.md:42  symbol `shardyfusion.writer.spark.write_sharded` resolves OK
- docs/architecture/routing.md:88  symbol `shardyfusion.routing.SnapshotRouter.route_keys` NOT FOUND (did you mean `route_key`?)

## config (1 finding)
- docs/use-cases/build-python-slatedb.md:130  WriteConfig.batch_sizes  NO SUCH FIELD (did you mean `batch_size`?)

## extras (0 findings)

## cli (1 finding)
- docs/reference/cli.md:55  `shardy lookup` not in CLI command tree
```

Treat any `NOT FOUND`, `NO SUCH FIELD`, `not found in`, or mismatch as a hard failure to be resolved before declaring docs ready.
