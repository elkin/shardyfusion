# 2026-05-24 SQLite overflow chains & configurable page_size

- Status: `implemented`
- Date: `2026-05-24`
- Commit: `aad7800` on branch `claude/sidecar-producer-index-pages-hdqZl`

## Summary

This engineering note documents two related writer-side mitigations for a
single object-storage latency problem: under the range-read SQLite VFS,
every overflow page in a kv chain becomes a separate S3 GET, and a
single 1 MiB value at the default 4 KiB page size can balloon a point
lookup into 200+ round trips.

The changes:

1. **Sidecar v3 → v4** — every kv overflow chain is now enumerated and
   appended to the btree-metadata sidecar, so a future range-mode
   reader can prefetch entire chains in one parallel multi-range
   request rather than walking the next-pointer chain page by page.
2. **Configurable `page_size`** — both SQLite factories
   (`SqliteFactory`, `SqliteVecFactory`) accept
   `page_size ∈ {4096, 8192, 16384, 32768, 65536, "auto"}`. Larger
   pages mean a higher inline-payload threshold, which keeps more
   values out of the overflow chain entirely.
3. **`"auto"` post-write VACUUM** — when `page_size="auto"` the
   adapter measures the on-disk value-size distribution at seal time
   and rewrites the file via `PRAGMA page_size = N; VACUUM;` at the
   size recommended by `sqlite_page_size.recommend_page_size`.
4. **Engine-percentile mode** — distributed writers (Spark, Dask, Ray)
   honour `KeyValueWriteConfig.profile_value_sizes_for_page_size`: they
   sample the input via engine-native ops, compute the p95 in the
   driver, and rebuild the factory before partition dispatch. The
   Python writer rejects the flag and points users at `"auto"`.
5. **VFS prerequisite** — `S3ReadOnlyFile` now parses the SQLite file
   header on open and caches whole pages at the file's *real* page
   size; previously hard-coded to 4 KiB, which would alias multiple
   SQLite pages into a single cache slot for shards written at
   larger sizes.

## 1. What problem is being solved by the changes?

A shard's kv table holds opaque caller-supplied bytes. SQLite stores
rows whose key+value exceeds an inline threshold (~1.0 KiB at 4 KiB
pages, ~16.3 KiB at 64 KiB pages) by stashing the leaf cell with a
pointer to an *overflow page*, which then chains forward via a 4-byte
next-pointer at the head of each subsequent page. For a 1 MiB value at
4 KiB pages, that is **256 overflow pages** behind one row.

The range-mode reader's VFS (`S3ReadOnlyFile`) translates page reads
into HTTP byte-range GETs against S3. The btree-metadata sidecar shipped
in earlier work prefetches every interior page, so point lookups
already cost ~1 GET per row in the *common* case where the value fits
inline. But for large values:

- The leaf cell points to overflow page 1.
- The reader fetches page 1, parses its 4-byte next-pointer, fetches
  page 2, parses its next-pointer, … until the chain ends.
- The chain is purely serial — each fetch's *target* is only known
  after the previous fetch completes — so the request latency is
  `N × RTT`, not `RTT + N × bandwidth`.

A 1 MiB value at 4 KiB pages produces 256 sequential S3 GETs. At a 30 ms
S3 RTT that is 7.7 s for a single point lookup. The same value at 64
KiB pages is 16 pages — or, much better, fits inline at zero overflow
pages.

Two independent levers reduce this cost, and we ship both because they
help different deployments:

- **Wider pages** eliminate overflow entirely for the vast majority of
  values. This is the cheap, durable fix and the right default
  recommendation, but it requires the operator to know their value-size
  distribution at writer-construction time. `"auto"` and the
  engine-percentile flag remove that requirement.
- **A sidecar chain map** turns the serial walk into one parallel
  multi-range fetch even when overflow is unavoidable (values larger
  than the 64 KiB inline ceiling), and it's lossless for callers who
  cannot tune `page_size` (e.g. callers that already pinned a specific
  size for sector-alignment reasons).

## 2. What design decisions were considered, with trade-offs?

### Decision 1: Where to enumerate the overflow chains

#### Option A: Record chains at write time, page-by-page, in the adapter's `flush` path

Pros:
- Zero extra work at seal time; chains are known as rows are inserted.

Cons:
- The adapter does not own SQLite's allocator. The page that backs a
  row's first overflow can move under a VACUUM, an autovacuum, or
  the `"auto"` post-write rewrite. Recording at insert time means
  invalidating and rebuilding the map on every rewrite.
- Couples the adapter's hot path to a sidecar concern that only
  range-mode readers care about.

#### Option B: Enumerate chains once at seal time from `dbstat` + raw page reads (chosen)

Pros:
- Pulls the full set of overflow pages from `dbstat`
  (`pagetype = 'overflow'`), then reconstructs chain structure by
  reading the 4-byte next-pointer at the head of each page from the
  finalized DB bytes. The DB is read-only by then, so the map is
  guaranteed consistent with the bytes a reader will fetch.
- Survives `"auto"` rewrite: the rewrite happens *before* sidecar
  extraction, so the map is built against the final page layout.
- No coupling to insert path; the cost is paid once per shard.

Cons:
- Requires `dbstat` (a SQLite virtual table that is on by default in
  the wheels we use but is technically a build option). The adapter
  raises `BtreeMetaUnavailableError` cleanly when it isn't compiled
  in.

#### Option C: Skip the chain map and rely on page-size widening alone

Pros:
- Smaller change.

Cons:
- Doesn't help workloads where values genuinely exceed the 64 KiB
  inline ceiling. We still want range-mode to be viable for callers
  storing multi-MiB blobs.
- Doesn't help callers who pin `page_size` for other reasons.

We chose B.

### Decision 2: How to pick `page_size` for callers who don't know their distribution

#### Option A: Pure local "auto" — VACUUM with the right page_size after the writer sees the data

Pros:
- Lives entirely in the adapter; no engine-specific code paths.
- Works for every writer (Python, Spark, Dask, Ray) uniformly.
- Uses the exact data the shard contains, not a sample.

Cons:
- Doubles the seal-time disk work (one VACUUM rewrites every page).
  For 1 GiB shards this is bounded — typically a few seconds of
  local SSD I/O — but it's not free.
- Per-shard local cost; engines can't amortize across shards.

#### Option B: Engine-percentile — sample the source DataFrame/Dataset before dispatch

Pros:
- One sample for the entire run; no per-shard VACUUM cost.
- Same `page_size` across all shards in a run, which simplifies
  capacity planning.
- Cheap on every supported engine — `DataFrame.limit(1000).collect()`
  on Spark, `ddf.head(1000, npartitions=-1)` on Dask,
  `ds.take(1000)` on Ray.

Cons:
- Engine-only; no equivalent on the Python writer, which already
  has the full data locally.
- Only as accurate as the sample (1000 rows is enough to pin one of
  five buckets, but pathological skew between sampled and unsampled
  partitions could mis-pick).

#### Option C: Force callers to pick

Pros:
- Simplest semantics.

Cons:
- Callers usually don't know. The whole point of this work is that
  the cliff is invisible until production latency cratters.

We shipped both A and B and made them **mutually exclusive** via
`_validate_page_size_mutual_exclusion` in `config.py`. Combining
`profile_value_sizes_for_page_size=True` with either an explicit
factory `page_size` or `page_size="auto"` raises
`ConfigValidationError` at config construction — never at run time.

### Decision 3: How distributed writers should hand percentile data to the adapter

#### Option A: Per-partition adapter-side accumulation, reduce at finalize

Pros:
- No driver-side sample step.

Cons:
- Page size has to be decided *before* partition dispatch, because
  factories are serialized with the partition. A finalize-time
  decision would mean rewriting every partition's shard.

#### Option B: Driver-side sample, rebuild the factory via `dataclasses.replace`, re-serialize (chosen)

Pros:
- One small `.limit().collect()` (or equivalent) on the driver before
  the writer fans out partitions.
- `dataclasses.replace(factory, page_size=target)` is a typed,
  in-place factory update with no extra protocol surface.
- The new factory is what gets pickled to each partition.

Cons:
- Requires the factory to be a dataclass. Custom non-dataclass
  factories that opt into the flag get a `ConfigValidationError`
  with a clear message instead of a silent no-op.

### Decision 4: How the range-read VFS learns the file's page size

#### Option A: Trust a config setting

Cons:
- A reader instantiated against a shard written at a different
  `page_size` would silently corrupt its own cache (multiple SQLite
  pages aliased into one 4 KiB cache slot).

#### Option B: Parse the SQLite file header on open (chosen)

The SQLite header lives in the first 100 bytes of the database and
encodes `page_size` at offset 16 (big-endian u16, with the value `1`
meaning 65536). `_parse_page_size_from_header` reads exactly that
field; `S3ReadOnlyFile._discover_page_size` calls it once per open.
`xSectorSize` follows the discovered value.

This makes the VFS robust to any `page_size` SQLite supports, with no
operator coordination.

## 3. What is the impact (testability, performance, complexity)?

**Performance.** For an opaque-bytes workload with a typical 8 KiB p95
value, picking `page_size=16384` (the recommendation from
`recommend_page_size`) collapses the overflow chain from ~2 pages per
row to *zero*. A point lookup that previously cost 3 S3 GETs (sidecar
miss + leaf + 1 overflow) drops to 1. The chain-map sidecar is the
fallback for the long-tail rows that overflow anyway; turning a 256-GET
serial walk into a 256-range parallel multi-range fetch flattens
worst-case latency from `N × RTT` to roughly `RTT + bandwidth`.

**Storage.** The chain-map addition to the sidecar is ~12 bytes per
overflow page (chain_id, length, page numbers). For a shard with no
overflow chains the cost is 4 bytes for the chain-count zero. For a
shard heavy in overflow it's still negligible — kilobytes vs the
megabytes of interior-page slabs that already dominate the sidecar.

**Complexity.** The chain enumeration is contained in
`shardyfusion/sqlite_adapter.py` (`_enumerate_overflow_chains`). The
recommender lives in a new pure-Python module
`shardyfusion/sqlite_page_size.py` (~75 lines, no SQLite import). The
engine-percentile picker lives in `shardyfusion/writer/_engine_page_size.py`
and is shared verbatim by all three distributed writers. The Python
writer gains a 5-line refusal of the engine-only flag.

**Testability.** Coverage added:

- `tests/unit/backend/sqlite/test_btreemeta_sidecar.py` — chain map
  round-trips for zero-overflow, single-chain, and multi-chain shards.
- `tests/unit/backend/sqlite/test_auto_page_size.py` — `"auto"`
  picks the recommended size and the recommended size only.
- `tests/unit/backend/sqlite/test_page_size.py` — every supported
  `page_size` produces a readable shard.
- `tests/unit/shared/test_page_size_config.py` — mutual-exclusivity
  matrix: explicit int + flag, `"auto"` + flag, default + flag (the
  last is permitted; the first two raise).
- `tests/unit/shared/test_sqlite_vfs.py` — header parsing for every
  supported size, including the 65536-encoded-as-`1` edge case.
- `tests/unit/vector/test_unified_writer.py` — the vector writer's
  factory plumbing preserves `page_size`.

325 tests pass across the changed code areas; ruff and pyright are
clean on the touched modules.

## 4. API delta

### `SqliteFactory` / `SqliteVecFactory`

New field: `page_size: int | Literal["auto"] = 4096`. Accepted values:
`{4096, 8192, 16384, 32768, 65536, "auto"}`. Anything else raises
`ConfigValidationError` at factory construction.

### `KeyValueWriteConfig`

New field: `profile_value_sizes_for_page_size: bool = False`.

- Honoured by Spark, Dask, Ray writers.
- Rejected by the Python writer with a message pointing at `"auto"`.
- Combining with an explicit factory `page_size` (int *other than* 4096,
  or `"auto"`) raises `ConfigValidationError`.

### Sidecar binary format

`format_version` 3 → 4. The body gains, after the existing
`page_size`/`page_count`/`index`/`page_bytes` section:

```
   ...                  4          chain_count C    (u32)
   ...                  ...        chain_table      (u32 chain_id, u32 length L,
                                                    u32 head_pageno, ...,
                                                    u32 tail_pageno) per chain
```

Readers that only understand v3 should refuse to parse v4. The current
in-tree reader still consumes v3 and ignores the chain map; the chain
map is published in anticipation of the range-mode reader work.

### `S3ReadOnlyFile`

New attribute: `page_size` (int, discovered from the file header at
open). `xSectorSize` returns this value.

## 5. Observed-but-deliberate gotchas

- **`"auto"` doubles seal-time disk I/O.** The post-write VACUUM
  rewrites every page. Operators with extreme shard sizes or tight
  seal-time SLOs should prefer the engine-percentile path.
- **`"auto"` won't downsize.** The recommender returns the *smallest*
  size that fits; it will not shrink a 64 KiB shard that has been
  rewritten by a caller. This is by design — shrinking would
  reintroduce overflow chains.
- **Engine-percentile uses 1000-row samples.** Designed for picking
  one of five discrete buckets, not for accurate percentile
  reporting. A workload with a 99.9th-percentile cliff far above its
  p95 may still want explicit `page_size`.
- **`dbstat` is required for chain enumeration.** Wheels we ship
  against include it; custom SQLite builds without `SQLITE_ENABLE_DBSTAT_VTAB`
  raise `BtreeMetaUnavailableError("dbstat_not_available")` and the
  sidecar is skipped for that shard (existing behaviour).
- **Mutual exclusivity raises at config construction, not at run.**
  This is intentional: invalid configs should fail before partition
  dispatch.

## 6. What was explicitly *not* done, and why

- **Reader-side chain-aware prefetch.** The sidecar now publishes
  enough to do this, but the range-mode reader still walks chains
  serially. Wiring the chain map into the reader's prefetch path is
  the next obvious follow-up; we intentionally separated producer
  and consumer so the sidecar format change can ship and bake.
- **A `page_size`-aware adapter cache eviction policy.** Larger pages
  mean larger cache slots; the LRU's `max_pages` knob is now a
  function of `page_size`. Operators tuning cache size for 64 KiB
  pages should reduce `max_pages` 16× from the 4 KiB default.
  Auto-scaling this was tempting but adds a config interaction we'd
  rather expose explicitly.
- **A migration tool for existing shards.** Pre-existing 4 KiB shards
  keep working under the new reader (header parsing returns 4096 and
  everything proceeds as before). Rewriting them at a larger
  `page_size` is a producer-side operation; we did not write a
  one-shot migrator.
- **A bytes-budget knob on the engine-percentile sampler.** 1000 rows
  is enough for the five-bucket decision. A larger sample buys
  precision we cannot use.

## 7. Post-merge audit

- [ ] Confirm v4 sidecars round-trip through the existing reader (which
      should accept v4 byte-for-byte once the reader is taught the new
      tail, and reject it cleanly until then).
- [ ] Add a microbenchmark covering "1 MiB value at 4 KiB pages" vs
      "1 MiB value at 64 KiB pages" under the range-read VFS to lock
      in the order-of-magnitude latency win.
- [ ] Wire the chain map into the range-mode reader's prefetcher.
- [ ] Revisit cache `max_pages` defaults once 16 KiB+ pages are
      common in production.
