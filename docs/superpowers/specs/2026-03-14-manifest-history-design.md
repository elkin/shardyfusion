# Manifest History & Rollback

## Problem

The S3 manifest store overwrites the `_CURRENT` pointer on every publish. While old manifest artifacts persist at their S3 paths, they become undiscoverable — readers have no way to list previous manifests or roll back to one. If a malformed manifest is published, readers fail with no recovery path.

The DB manifest store stores every manifest as a row, but its "current" selection (`ORDER BY created_at DESC LIMIT 1`) is implicit and doesn't support explicit rollback either.

## Goals

1. **Discoverability**: Readers and CLI can list recent manifests in chronological order.
2. **Programmatic rollback**: Operators can point the current pointer at an older manifest, transparently switching all readers on their next refresh.
3. **Auto-fallback on cold start**: When a reader starts and the latest manifest is malformed, it automatically tries previous manifests before failing.
4. **Resilient refresh**: A running reader that encounters a malformed manifest on refresh stays on its current valid state instead of crashing.
5. **CLI history access**: All CLI read commands can target any historical manifest by ref or offset.
6. **Backend symmetry**: The protocol works naturally in both S3 and database backends without forcing S3-specific concepts onto the protocol.

## Non-Goals

- Built-in retention/cleanup of old manifests (use S3 lifecycle rules or external cleanup for the DB pointer table).
- Concurrent writer conflict resolution (last-write-wins on the pointer is acceptable).
- Backward compatibility with old-format manifest paths (library is not yet in production).

## Design

### S3 Path Format

New manifest path with timestamp prefix:

```
s3_prefix/manifests/{ISO-timestamp}_run_id={run_id}/manifest
```

Example:
```
s3://bucket/my-prefix/manifests/2026-03-14T10:30:00.000000Z_run_id=abc123/manifest
```

**Timestamp rules:**
- Always UTC (`datetime.now(timezone.utc)`)
- ISO 8601 format with microsecond precision and `Z` suffix (not `+00:00`)
- Lexicographic ordering of S3 keys = chronological ordering

The `_CURRENT` pointer remains at `s3_prefix/_CURRENT` and is written on every publish. It serves as the fast path for readers (single `GetObject` vs. `ListObjectsV2`).

### New Type: ManifestRef

`CurrentPointer` is demoted from the protocol to an internal implementation detail of `S3ManifestStore`. The protocol uses `ManifestRef` instead — a backend-agnostic pointer:

```python
@dataclass(slots=True)
class ManifestRef:
    ref: str            # full S3 URL or DB run_id — passable to load_manifest()
    run_id: str
    published_at: datetime  # UTC
```

### Protocol Changes

**`ManifestStore`** — `load_current` return type changes, two new methods added:

```python
class ManifestStore(Protocol):
    def publish(
        self, *, run_id: str, required_build: RequiredBuildMeta,
        shards: list[RequiredShardMeta], custom: dict[str, Any],
    ) -> str: ...

    def load_current(self) -> ManifestRef | None: ...  # changed: was CurrentPointer

    def load_manifest(self, ref: str) -> ParsedManifest: ...

    # new
    def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]: ...

    def set_current(self, ref: str) -> None: ...
```

- **`load_current()`**: Returns `ManifestRef | None` instead of `CurrentPointer | None`.
- **`list_manifests(limit)`**: Returns up to `limit` manifests in reverse chronological order (newest first).
- **`set_current(ref)`**: Updates the current pointer to the given manifest ref. Used for explicit rollback.

**`AsyncManifestStore`** — read-only, gains `list_manifests` only:

```python
class AsyncManifestStore(Protocol):
    async def load_current(self) -> ManifestRef | None: ...  # changed: was CurrentPointer
    async def load_manifest(self, ref: str) -> ParsedManifest: ...

    # new
    async def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]: ...
```

`set_current` is omitted from the async protocol — rollback is an operator/CLI action, not a hot-path reader operation.

### S3ManifestStore Implementation

**`publish()`:**
1. Generate timestamp: `datetime.now(timezone.utc)` formatted as `%Y-%m-%dT%H:%M:%S.%fZ`
2. Build manifest path: `s3_prefix/manifests/{timestamp}_run_id={run_id}/manifest`
3. Write manifest artifact via `put_bytes()` (with retry)
4. Write `_CURRENT` pointer (JSON with `manifest_ref`, `run_id`, `updated_at`, etc.) via `put_bytes()` (with retry)
5. Return the manifest URL

**`load_current()`:**
1. Read `_CURRENT` JSON via `try_get_bytes()`
2. Parse into internal `CurrentPointer` (implementation detail)
3. Return `ManifestRef(ref=current.manifest_ref, run_id=current.run_id, published_at=current.updated_at)`

**`list_manifests(limit)`:**
1. `ListObjectsV2(Prefix="s3_prefix/manifests/", Delimiter="/")` — returns `CommonPrefixes`
2. Parse each prefix to extract timestamp and `run_id`
3. S3 returns lexicographic ascending; reverse to get newest-first
4. Return up to `limit` entries as `ManifestRef` objects
5. Construct `ref` for each by appending the manifest filename to the prefix

**`set_current(ref)`:**
1. Extract `run_id` from the ref path
2. Build `_CURRENT` JSON with the given ref
3. Write to `s3_prefix/_CURRENT` via `put_bytes()` (with retry)

### DB ManifestStore Implementation

**Schema — two tables:**

```sql
CREATE TABLE shardyfusion_manifests (
    run_id       TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    payload      JSONB NOT NULL,
    content_type TEXT NOT NULL DEFAULT 'application/json',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE shardyfusion_pointer (
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    manifest_ref  TEXT NOT NULL  -- references shardyfusion_manifests.run_id
);
```

The `pointer` table is an append-only log of what was current at each point in time — from both normal publishes and explicit rollbacks.

**`publish()`:**
In a single transaction:
1. INSERT manifest into `shardyfusion_manifests`
2. INSERT into `shardyfusion_pointer (manifest_ref)` with the new `run_id`

**`load_current()`:**
1. `SELECT manifest_ref, updated_at FROM shardyfusion_pointer ORDER BY updated_at DESC LIMIT 1`
2. If pointer table is empty (fresh deployment): fall back to `SELECT run_id, created_at FROM shardyfusion_manifests ORDER BY created_at DESC LIMIT 1`
3. Return `ManifestRef(ref=run_id, run_id=run_id, published_at=...)`

**`list_manifests(limit)`:**
```sql
SELECT run_id, created_at
FROM shardyfusion_manifests
ORDER BY created_at DESC
LIMIT %(limit)s
```

Returns `ManifestRef(ref=run_id, run_id=run_id, published_at=created_at)`.

**`set_current(ref)`:**
```sql
INSERT INTO shardyfusion_pointer (manifest_ref) VALUES (%(ref)s)
```

Append-only — no UPDATE, no swap, works on any database.

### Reader Fallback Behavior

**Configuration:**
```python
max_fallback_attempts: int = 3  # 0 to disable fallback
```

**Fallback triggers ONLY on `ManifestParseError`** (malformed content). Transport failures (S3 errors, DB connection errors, timeouts) use existing retry logic and propagate up if exhausted — they never trigger fallback to an older manifest.

**Cold-start (reader opening for the first time):**
1. `load_current()` → get `ManifestRef`
2. `load_manifest(ref.ref)` → attempt to parse
3. On `ManifestParseError`:
   a. If `max_fallback_attempts == 0`: raise immediately
   b. `list_manifests(limit=max_fallback_attempts + 1)`
   c. Skip the already-failed ref
   d. Try each remaining ref in order (newest to oldest)
   e. First successful parse → use that manifest, log warning about skipped malformed manifest(s)
   f. All fail → raise `ReaderStateError("No valid manifest found")`

**Running reader (refresh):**
1. `load_current()` → get new `ManifestRef`
2. If same ref as current state → no-op
3. `load_manifest(new_ref.ref)` → attempt to parse
4. On `ManifestParseError`: log warning, keep current state, do not crash
5. On success: swap to new state (existing atomic swap logic)

### AsyncShardedReader Fallback

Same logic as sync reader, using `await` for async manifest store calls. The `max_fallback_attempts` config is shared.

### CLI Integration

**New subcommand — `shardy history`:**
```bash
shardy history              # list recent manifests (default limit=10)
shardy history --limit 20   # list more
```

Output: table of offset index, `published_at`, `run_id`, `ref`.

**New subcommand — `shardy rollback`:**
```bash
shardy rollback --ref <manifest-ref>
shardy rollback --run-id <run_id>
shardy rollback --offset 1        # roll back to previous manifest
```

Calls `set_current()` on the manifest store.

**Existing read commands gain `--ref` and `--offset` options:**
```bash
shardy get 42 --offset 1                         # read from previous manifest
shardy get 42 --ref s3://bucket/.../manifest      # read from exact ref
shardy info --offset 2                            # inspect 2 versions back
shardy multiget 1 2 3 --offset 1
shardy shards --ref s3://bucket/.../manifest
shardy route 42 --offset 1
```

- `--ref` and `--offset` are mutually exclusive
- Both bypass `_CURRENT` discovery
- `--offset 0` is equivalent to no flag (latest manifest)
- `--offset N` resolves via `list_manifests(limit=N+1)`, picks the last entry

**Interactive REPL** gains a `use` command for manifest switching:
```
shardy> use --offset 1
Switched to manifest run_id=abc123 (2026-03-14T09:00:00Z)
shardy> get 42
...
shardy> use --latest
Switched to latest manifest run_id=def456
```

### InMemoryManifestStore (Testing)

The existing `InMemoryManifestStore` gains `list_manifests()` and `set_current()`. It maintains an ordered list of `ManifestRef` entries and tracks a `_current_ref` override for `set_current()`.

## Symmetry Summary

| Concept | S3 | DB |
|---|---|---|
| Manifest storage | Objects at `manifests/{ts}_run_id={id}/` | `shardyfusion_manifests` table rows |
| Primary pointer | `_CURRENT` file (overwritten each publish) | `shardyfusion_pointer` table (append-only) |
| Rollback | Rewrite `_CURRENT` file | INSERT into `shardyfusion_pointer` |
| List history | `ListObjectsV2` on `manifests/` prefix | `SELECT ... ORDER BY created_at DESC` |
| Pointer fallback | N/A (always has `_CURRENT` after first publish) | Falls back to `shardyfusion_manifests` if pointer table is empty |

## Affected Files

| File | Changes |
|---|---|
| `manifest.py` | Add `ManifestRef` dataclass; `CurrentPointer` stays but is no longer part of the protocol |
| `manifest_store.py` | `ManifestStore` protocol: change `load_current` return type to `ManifestRef`, add `list_manifests()` and `set_current()`; `S3ManifestStore`: new path format, implement new methods; `InMemoryManifestStore`: implement new methods |
| `async_manifest_store.py` | `AsyncManifestStore` protocol: change `load_current` return type, add `list_manifests()`; `AsyncS3ManifestStore` and `_SyncManifestStoreAdapter`: implement new method |
| `db_manifest_store.py` | Add `shardyfusion_pointer` table; implement `list_manifests()` and `set_current()` in both PostgreSQL and Comdb2 implementations; update `publish()` to INSERT into pointer table |
| `reader/reader.py` | Add `max_fallback_attempts` config; cold-start fallback logic; resilient refresh on `ManifestParseError` |
| `reader/async_reader.py` | Same fallback logic for async reader |
| `cli/app.py` | Add `history` and `rollback` subcommands; add `--ref`/`--offset` options to read commands |
| `cli/interactive.py` | Add `use` REPL command for manifest switching |
| `cli/config.py` | Parse `--ref`/`--offset` options |
| `_writer_core.py` | Update `publish_to_store()` to generate timestamp-prefixed manifest paths |
| `config.py` | Add `max_fallback_attempts` to reader config |
| `__init__.py` | Export `ManifestRef` |
