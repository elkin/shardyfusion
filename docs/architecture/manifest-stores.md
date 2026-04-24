# Manifest stores

A `ManifestStore` (`shardyfusion/manifest_store.py:61`) is a thin abstraction over the storage of two things:

1. The manifest object itself — `manifests/<run_id>/<timestamp>.sqlite`.
2. The `_CURRENT` pointer.

The store guarantees that pointer updates are observable atomically (at least to the same reader). Implementations differ in what backend they target and how they achieve atomicity.

## Implementations

| Store | Module | Backend | Async | Atomic pointer |
|---|---|---|---|---|
| `S3ManifestStore` | `manifest_store.py:126` | S3 (boto3) | No (sync) | Object PUT (S3 single-key write is atomic). |
| `AsyncS3ManifestStore` | `async_manifest_store.py:32` | S3 (aiobotocore) | Yes | Read-only (`AsyncManifestStore` Protocol at `async_manifest_store.py:22` exposes `load_current` / `load_manifest` / `list_manifests` only). |
| `InMemoryManifestStore` | `manifest_store.py:267` | RAM (test fixture) | No | Dict assignment. |
| `PostgresManifestStore` | `db_manifest_store.py:56` | Postgres (psycopg) | No | Single transaction; pointer is an append-only table. |

The async store is read-only by design: writers run synchronously, so async publish has never been needed. Selection is made by writers via factory wiring; readers receive a store instance directly.

## S3 layout

```
<root>/
  _CURRENT                                                # JSON pointer
  manifests/
    2026-04-20T15:23:01.123456Z_run_id=<run_id>/
      manifest                                            # SQLite manifest (no extension)
    2026-04-20T16:05:12.842901Z_run_id=<run_id>/          # rebuild of same run (rare)
      manifest
    2026-04-20T17:11:09.001234Z_run_id=<other_run_id>/
      manifest
  runs/
    2026-04-20T15:23:01.123456Z_run_id=<run_id>_<uuidhex>/
      run.yaml                                            # run record
  <db_path_template per shard, e.g. db=00000/...>         # data files (template configurable)
```

- Manifest filenames use `_format_manifest_timestamp` (`manifest_store.py:92`) — UTC microsecond-precision ISO-8601 with trailing `Z`, sortable lexicographically.
- The default `manifest_name` is `"manifest"` with no file extension; the SQLite magic header (`SQLite format 3\x00`) at `manifest_store.py:322` identifies the format.
- The shard data path is fully configurable via `WriteConfig.output.db_path_template` (default `"db={db_id:05d}"`) and `output.shard_prefix` (default `"shards"`); the layout above is illustrative.

## Postgres store

`PostgresManifestStore` (`db_manifest_store.py:56`) uses **three tables** (DDL at `db_manifest_store.py:87-134`):

- `<table_name>_builds` (default `shardyfusion_manifests_builds`) — one row per published build.
- `<table_name>_shards` (default `shardyfusion_manifests_shards`) — one row per shard within a build.
- `<pointer_table_name>` (default `shardyfusion_pointer`) — append-only; the latest row by `updated_at DESC` is `_CURRENT`. An index on `updated_at DESC` (`db_manifest_store.py:130`) makes pointer reads cheap.

Pointer updates and manifest writes happen in **one transaction** (`db_manifest_store.py:152-246`, committed at `:231`), so readers cannot observe a half-published state. `_make_ref` (`db_manifest_store.py:46`) builds the pointer payload — for Postgres the `ref` field is just the `run_id` string.

Use Postgres when:

- Multiple writer processes need stronger contention semantics than S3's last-write-wins.
- You want SQL-driven manifest history and audits.
- You want to participate in an existing transactional system.

## In-memory store

`InMemoryManifestStore` (`manifest_store.py:267`) is the test fixture used by unit tests. It is **not** safe for production use — there is no durability and no cross-process visibility.

## What stores do *not* provide

- **Garbage collection of data files.** Stores manage manifests and `_CURRENT` only. Data file cleanup is the writer's job (`cleanup_losers`, `cleanup_stale_attempts`, `cleanup_old_runs` — see [`writer-core.md`](writer-core.md)).
- **Cross-store replication.** If you want manifests in two regions, write to both stores and treat the consistency mismatch as your problem.
- **Schema migration.** Manifest format versions are negotiated by writers; stores treat the payload as opaque bytes.

## Errors

- `PublishManifestError` (`errors.py:84`) — the manifest object write failed.
- `PublishCurrentError` (`errors.py:94`) — the `_CURRENT` pointer write failed (the manifest object is durable but invisible — a future build will overwrite `_CURRENT` and recover).
- `ManifestStoreError` (`errors.py:139`) — generic store failure during read.
- `ManifestParseError` (`errors.py:109`) — the payload was readable but malformed.

See [`error-model.md`](error-model.md) for the full taxonomy.

## See also

- [`manifest-and-current.md`](manifest-and-current.md) — manifest format and two-phase publish.
- The Postgres-backed manifest store is used by setting `WriteConfig.manifest.store` to a `PostgresManifestStore` instance. There is no dedicated use-case page yet — see the source at `shardyfusion/db_manifest_store.py`.
- [`history/design-decisions/adr-001-two-phase-publish.md`](../history/design-decisions/adr-001-two-phase-publish.md).
