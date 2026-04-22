# Operate manifest history and rollback

Use the **manifest history** to inspect past publishes and **rollback** to revert `_CURRENT` to a previous manifest.

## When to use

- A bad publish needs to be reverted.
- You need to audit publish history.
- Pre-publish validation: compare the staged manifest against the current one.

## When NOT to use

- Cleaning up old data — see `cleanup` subcommand (referenced from [`operate-cli.md`](operate-cli.md)).
- Forking the snapshot lineage — manifests are append-only, not branchable.

## Install

```bash
uv add 'shardyfusion[cli,read]'
```

## Minimal example

Snapshot location comes from `reader.toml` (or `--current-url`/`SHARDY_CURRENT`); see [`operate-cli.md`](operate-cli.md) for configuration.

```bash
# List recent publishes
shardy history --limit 20

# Roll back by manifest ref
shardy rollback --ref manifests/2026-04-19T08:30:00.000000Z_run_id=abc/manifest

# Or by run id
shardy rollback --run-id abc

# Or by N-back from latest (0 = latest, 1 = previous)
shardy rollback --offset 1
```

Exactly **one** of `--ref`, `--run-id`, or `--offset` must be supplied to `rollback`.

## Configuration

- `history` lists manifests under `manifests/` ordered by timestamp (newest first), capped by `--limit` (default 10).
- `rollback` calls `manifest_store.set_current(ref)` — the same atomic swap a normal publish uses.
- Manifest path format: `manifests/<timestamp>_run_id=<run_id>/manifest`. Timestamp format: `%Y-%m-%dT%H:%M:%S.%fZ`.
- For read-only inspection at a previous manifest **without** rolling back, use the global `--ref` / `--offset` flags (e.g. `shardy --offset 1 info`). These pin the reader without touching `_CURRENT`.

## Functional / Non-functional properties

- Manifests are immutable once written; `_CURRENT` is the only mutable pointer.
- Rollback flips `_CURRENT.format_version=1` to point at the chosen manifest.
- `RunRecord` lifecycle (`runs/<timestamp>_run_id=<run_id>_<uuidhex>/run.yaml`) is independent — rollback does not delete run records.

## Guarantees

- Rollback is atomic at the `_CURRENT` swap.
- All readers picking up the new pointer (via `refresh()` or fresh open) observe the rolled-back manifest.
- Old manifest data remains in S3 until `cleanup` is run.

## Weaknesses

- No automatic rollback on failed publishes — failure leaves `_CURRENT` untouched (that's the design), but if you publish a logically bad manifest, you must rollback manually.
- `cleanup` after rollback can delete the rolled-from manifest's data if it's unreferenced — be careful with order of operations.

## Failure modes & recovery

| Failure | Surface | Recovery |
|---|---|---|
| Missing `_CURRENT` | `click.ClickException("No current manifest found")` | Verify writer published; check `current_url`. |
| `rollback` selector missing or duplicated | `click.UsageError("Exactly one of --ref, --run-id, or --offset is required")` | Pass exactly one selector. |
| `--run-id` not found in history | `click.ClickException("No manifest found with run_id=…")` | Confirm via `history`; widen `--limit` if needed. |
| Manifest body malformed | `ManifestParseError` | Reader falls back to previous (up to `max_fallback_attempts`); investigate writer. |
| Concurrent rollback | Last writer wins on `_CURRENT` | Coordinate operationally. |
| Reader still on old pointer | Stale reads | `reader.refresh()`. |

## See also

- [`operate-cli.md`](operate-cli.md).
- [`architecture/manifest-and-current.md`](../architecture/manifest-and-current.md).
- [`architecture/run-registry.md`](../architecture/run-registry.md).
