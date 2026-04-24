# Manifest and `_CURRENT`

A **manifest** describes one published snapshot: which shards exist, where their database files are, what the sharding spec was, and per-shard statistics (row count, byte size, min/max key). It is a single SQLite database serialized with `con.serialize()`, written as one S3 object.

`_CURRENT` is a tiny JSON object pointing at the latest manifest. Readers read `_CURRENT`, then read the manifest it points at, then route requests.

## Manifest object format

- **Magic header**: `SQLite format 3\x00` (the standard SQLite file magic — readers sniff it before parsing).
- **Builder**: `SqliteManifestBuilder` (`shardyfusion/manifest.py:202`).
- **Parser**: `parse_sqlite_manifest` (`manifest_store.py:339`) → `ParsedManifest` (`manifest.py:107`).
- **Validator**: `_validate_manifest` (`manifest_store.py:414`) — checks shard coverage, format version compatibility, and required metadata.

A manifest carries `BuildResult` (`manifest.py:160`) data: `WriterInfo` (`manifest.py:85`), `RequiredBuildMeta` (`manifest.py:67`), per-shard `RequiredShardMeta` (`manifest.py:94`), `BuildDurations` (`manifest.py:140`), and `BuildStats` (`manifest.py:150`). The MIME type constant is `SQLITE_MANIFEST_CONTENT_TYPE = "application/x-sqlite3"` (`manifest.py:21`).

## Format versions

`_SUPPORTED_MANIFEST_FORMAT_VERSIONS = frozenset({1, 2, 3})` (`manifest_store.py:48`).

| Version | Capability |
|---|---|
| 1 | Legacy. Readers accept it; writers do not produce it. |
| 2 | HASH or CEL **without** `routing_values`, with required `sharding.hash_algorithm`. Default for `RequiredBuildMeta.format_version` (`manifest.py:80`). |
| 3 | Required when `routing_values` is set (categorical CEL). Validation at `manifest_store.py:447`. |

`_manifest_format_version_for_sharding` (`_writer_core.py:796`) returns `3` if `routing_values` is set, else `2`. Readers reject unsupported versions with `ManifestParseError` (`errors.py:109`) at `manifest_store.py:417`.

`required.sharding.hash_algorithm` is required for every manifest. The only supported value today is `xxh3_64`. A manifest that omits the field or names an unsupported algorithm is invalid; readers reject it instead of falling back to a local default.

## `_CURRENT` object format

A `CurrentPointer` (`manifest.py:129`, `format_version: int = 1` at `:136`) serialized as JSON, holding a `ManifestRef` (`manifest.py:25`):

```python
@dataclass(frozen=True, slots=True)
class ManifestRef:
    ref: str            # storage-specific manifest reference
    run_id: str
    published_at: datetime
```

For S3, `ref` is the S3 object URL of the manifest. For Postgres, `ref` is the `run_id` itself. Parsing: `parse_current_pointer_to_ref` (`manifest_store.py:97`).

## Two-phase publish

`publish_to_store` (`_writer_core.py:456`) writes in this strict order:

1. **Manifest object**: `manifests/<timestamp>_run_id=<run_id>/<manifest_name>` — immutable. Default `manifest_name = "manifest"` (no file extension; the SQLite magic header identifies the format). Timestamp uses `MANIFEST_TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"` (`manifest_store.py:51`).
2. **`_CURRENT` pointer**: overwritten in place to point at the manifest from step 1.

A reader observing S3 will see one of three states:
- Old manifest + old `_CURRENT` → old snapshot, fully consistent.
- New manifest written + old `_CURRENT` → reader still sees the old snapshot (the new manifest is invisible).
- New manifest + new `_CURRENT` → reader atomically sees the new snapshot.

There is no "torn" state where `_CURRENT` points at a partially-written manifest, because manifests are written as a single object.

See [`history/design-decisions/adr-001-two-phase-publish.md`](../history/design-decisions/adr-001-two-phase-publish.md) for the rationale.

## Manifest history

Old manifest objects are retained in `manifests/`. They are deleted only by `cleanup_old_runs` (`_writer_core.py:725`), gated by `keep_runs`. Until then, they form the rollback log: pointing `_CURRENT` at a previous manifest object atomically rolls the snapshot back. See [`operate/history-rollback.md`](../operate/history-rollback.md).

## Manifest stores

The manifest object and `_CURRENT` pointer are stored together in a `ManifestStore` (`manifest_store.py:61` Protocol). Implementations are described in [`manifest-stores.md`](manifest-stores.md).

## Loading shard metadata

`load_sqlite_build_meta` (`manifest_store.py:464`) and `SqliteShardLookup` (`manifest_store.py:507`) extract shard URLs and metadata from a parsed manifest. Readers use these to construct adapters per shard.

## See also

- [`manifest-stores.md`](manifest-stores.md) — S3, in-memory, and Postgres implementations.
- [`writer-core.md`](writer-core.md) — `publish_to_store`.
- [`history/design-decisions/adr-001-two-phase-publish.md`](../history/design-decisions/adr-001-two-phase-publish.md).
