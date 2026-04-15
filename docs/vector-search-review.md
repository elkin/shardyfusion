# Vector Search Integration — Architecture Review

*Reviewed: 2026-04-06*

## Overview

The vector search integration adds approximate nearest-neighbor (ANN) search
to shardyfusion's sharded snapshot model.  It provides:

- A standalone vector write/read pipeline (`vector/writer.py`, `vector/reader.py`)
- Two storage backends: USearch HNSW sidecar and sqlite-vec embedded
- A unified KV+vector mode via `VectorSpec` on `WriteConfig`
- Four sharding strategies: CLUSTER (k-means), LSH, EXPLICIT, CEL

This document captures the current state, weak points, improvement
opportunities, missing cases, and untested scenarios.

---

## 1. Weak Points

### A. No async vector reader path
`AsyncShardedReader` has no vector search counterpart. There is
`ShardedVectorReader` (sync, standalone) and `UnifiedShardedReader` (sync,
extends `ShardedReader`), but no `AsyncUnifiedShardedReader`. Any async
service wanting KV+vector must use `asyncio.to_thread()` as a workaround.

### B. Thread safety during manifest refresh
~~`ShardedVectorReader.refresh()` holds `_refresh_lock` and swaps `_centroids`,
`_hyperplanes`, `_shard_meta`, etc. as individual attribute assignments — not
an atomic state swap. A concurrent `search()` call can see a half-updated
state (new `_shard_meta` but old `_centroids`). The KV reader solved this
with an atomic `_ReaderState` swap + refcount, but the vector reader didn't
adopt that pattern.~~
*Resolved: Recent updates ensure manifest metadata is validated and applied 
consistently, though the vector reader still uses a lock-based refresh rather 
than a full atomic state object swap.*

### C. Shard lock dict grows unboundedly
~~`_shard_locks` entries are never cleaned up when a reader is evicted from
the LRU cache. Over time this dict grows to match every shard ever accessed, not
just `max_cached_shards`.~~
*Resolved: Shard locks are now popped from `_shard_locks` during LRU eviction 
in `_get_or_load_reader`.*

### D. Only the Python writer supports unified KV+vector
The Spark, Dask, and Ray writers have zero `vector_spec` / `vector_fn`
support. Vector search is limited to the Python iterator-based writer, which
is single-process only. This makes it impractical for large-scale production
vector ingestion.

### E. Duplicate merge logic
~~`_merge_top_k()` in `unified_reader.py` and `merge_results()` in
`vector/_merge.py` do the same thing with slightly different interfaces (one
takes a string metric, the other a `DistanceMetric` enum). This is
a consistency risk — a fix to one won't propagate to the other.~~
*Resolved: `UnifiedShardedReader` now uses the common `merge_results` 
function from `shardyfusion.vector._merge`.*

### F. VectorSpec uses strings for metric, VectorIndexConfig uses enums
`VectorSpec.metric` is a plain string (`"cosine"`), while
`VectorIndexConfig.metric` is `DistanceMetric.COSINE`. This mismatch means
implicit conversions happen at several boundary points, and a typo in the
string won't be caught until runtime.


---

## 2. What Can Be Improved

### A. Atomic state swap in ShardedVectorReader

Bundle all manifest-derived state (`_centroids`, `_hyperplanes`,
`_shard_meta`, `_num_dbs`, `_metric`, `_sharding_strategy`, etc.) into a
frozen `_VectorReaderState` dataclass and swap it atomically on refresh —
mirroring the KV reader's `_ReaderState` pattern.  This eliminates the
partial-update race.

### B. Unify merge logic

Delete `_merge_top_k()` from `unified_reader.py` and use `merge_results()`
from `vector/_merge.py` everywhere.  The unified reader would need to convert
its string metric to `DistanceMetric`, but that's a one-liner.

### C. Make VectorSpec.metric an enum (or Literal)

Change `metric: str` to `metric: Literal["cosine", "l2", "dot_product"]` or
use `DistanceMetric` directly, with appropriate lazy import handling to avoid
the numpy dependency in `config.py`.

### D. Evict shard locks alongside readers

When an LRU eviction removes a shard reader, also remove its entry from
`_shard_locks`.  This bounds memory to `max_cached_shards`.

### E. Add search() support to ConcurrentShardedReader

Currently only `ShardedReader` is extended by `UnifiedShardedReader`.  A
`ConcurrentUnifiedShardedReader` variant (or making `UnifiedShardedReader`
work with the concurrent reader) would provide production-grade thread safety
with the pool-mode checkout pattern, refcounted handles, and proper borrow
semantics — all of which are already solved for KV reads.

### F. Batch vector writes in SqliteVecAdapter

Currently vector inserts are row-by-row (`INSERT INTO vec_index ... VALUES (?)`
in a loop).  Batched `executemany` with pre-serialized blobs would
significantly improve write throughput.

---

## 3. Important Cases Not Implemented

### A. Spark/Dask/Ray writer vector support

These frameworks handle the vast majority of production write workloads but
have no vector integration at all.  A production user can't shard-write
millions of vectors through Spark.

### B. Async vector search

No `AsyncUnifiedShardedReader` or async equivalent of `ShardedVectorReader`.
Async services (FastAPI, etc.) must block a thread pool for every search.

### C. Vector index updates / incremental writes

The current model is snapshot-only — a full rebuild each time.  There's no
support for appending vectors to an existing index or doing delta updates.

### D. Filtered/hybrid search

No support for "search vectors WHERE category = X".  The only filtering is
shard-level routing (CEL expression), not per-record metadata filtering within
a shard.

### E. Multi-vector queries with score fusion

No support for querying with multiple vectors and combining scores (e.g.,
reciprocal rank fusion, weighted average).

### F. Quantization-aware distance computation

While quantization is stored in config (`fp16`, `i8`), the distance functions
in `sharding.py` always compute in float32/64.  For CLUSTER/LSH routing, this
doesn't account for quantization-induced drift.

### G. Vector deletion/tombstones

No API to remove individual vectors from an existing index.  The snapshot model
means you rebuild from scratch.

---

## 4. Scenarios Not Covered by Tests

### A. Concurrent refresh + search race condition

No test simulates `refresh()` running while `search()` reads state attributes.
This is the most dangerous untested path.

### B. LRU eviction under concurrent load

No test with `max_cached_shards < num_queried_shards` where multiple threads
hit different cold shards simultaneously.  The evict-just-created guard was
added but never tested with actual concurrency.

### C. Centroids/hyperplanes S3 failure at read time

No test verifies behavior when `get_bytes(centroids_ref)` raises mid-search
after a successful initial load (e.g., transient S3 error during refresh).

### D. USearch adapter with string IDs end-to-end

The writer `id_map` table is tested in isolation (and skipped without usearch),
but there's no integration test proving the full writer→reader round-trip
returns the original string IDs from `search()`.

### E. Corrupted/mismatched manifest custom fields

No test for `manifest.custom["vector"]` with missing keys, wrong types, or
version mismatches.  `_parse_vector_custom()` uses `.get()` with defaults but
some fields like `dim` would crash on `int(None)`.

### F. Large top-k merge correctness

`merge_results` is tested with small inputs.  No test verifies correctness
when `top_k > total_results` across shards, or when all shards return
identical scores (tie-breaking).

### G. SqliteVecAdapter write + search round-trip

No unit test creates a `SqliteVecAdapter`, writes vectors, then reads them
back with `SqliteVecShardReader.search()`.  The integration test exists but
uses moto — no isolated unit-level round-trip.

### H. Unified writer vector_col auto-extraction path

No test exercises the `_auto_vector_fn` closure that builds `vector_fn` from
`columns_fn` + `vector_col`.  The validation is tested, but not the actual
data flow through the auto-generated function.

### I. Rate limiter interaction with vector search

`ShardedVectorReader` supports `rate_limiter` but tests only verify
`acquire()` is called, not behavior under throttling (e.g., that throttled
searches don't deadlock or drop results).

### J. Reader health with stale vector manifest

`health()` reports `staleness_threshold` but no test verifies the `"degraded"`
status when the manifest is older than the threshold with vector metadata
present.
