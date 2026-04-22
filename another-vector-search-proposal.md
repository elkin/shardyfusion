# 2026-03-26 Vector Search Options

- Status: `superseded` — see ADR-005, ADR-006, and `historical-notes/2026-04-19-lancedb-vector-migration.md`
- Date: `2026-03-26`
- Baseline repo commit before this note: `f7b100504383f1c78d9e1ea08f41a595db6c2b72`
- Baseline commit summary: `docs: refresh repository guidance for run registry`
- Implementation status:
  - Implementation completed April 2026.
  - **Engine choice changed**: `usearch` was rejected because it operates in local-only mode and cannot request index data directly from S3. We adopted **LanceDB** as the primary vector backend instead.
  - This note captures early design options explored for adding vector search without changing the existing point-lookup contract.

## Summary

This note outlines ways to add vector search to shardyfusion alongside the existing key-based reader and writer flows.

The main constraints are:

- read-path latency matters most
- the design should scale with snapshot size
- resource usage should stay conservative
- the existing key-lookup API contract should remain valid
- vector support may live in an extra package with native dependencies
- CEL sharding and `routing_context` should compose naturally with vector search, for example `department_id`

The highest-level conclusion is:

- efficient point lookups and efficient vector search can coexist
- they should not be forced through one universal shard layout
- the best default direction is a hybrid design:
  - keep key-routable sharding for point lookups
  - add vector sidecars for ANN + exact rerank
  - use CEL partitioning when filter dimensions such as `department_id` should prune vector search

## Repo Constraints And Existing Extension Points

The current core reader contract is point lookup only.

- `ShardReader` exposes `get(key: bytes) -> bytes | None`
- `ShardReaderFactory` opens one shard reader
- `ShardedReader` and `ConcurrentShardedReader` route keys to one shard and perform `get` and `multi_get`

That contract is visible in:

- `shardyfusion/type_defs.py`
- `shardyfusion/reader/reader.py`
- `shardyfusion/reader/concurrent_reader.py`
- `shardyfusion/routing.py`

There are already several extension seams that make vector support feasible without changing that contract:

- `ManifestOptions.custom_manifest_fields`
  - allows publishing extra metadata without changing required manifest fields
- pluggable `reader_factory`
  - allows alternate read-path backends or sidecar-aware readers
- existing S3 layout with `run_id=...`
  - provides a natural place for vector sidecars
- existing optional dependency model in `pyproject.toml`
  - supports shipping vector functionality as an extra package

## Core Design Tension

Point lookup and vector search want different physical layouts.

### Point Lookup

Point lookup wants deterministic routing:

- `key -> shard`
- one shard hit
- minimal read amplification

That is what the current `HASH` and `CEL` routing model provides.

### Vector Search

Vector search wants one of the following:

- semantic partitions
- filter partitions
- ANN graph or coarse quantization partitions
- a small number of candidate partitions for global search

Those layouts often do not align with key hashing.

This leads to one important design rule:

- do not assume the current point-lookup shard layout is automatically the right vector-search layout

## Option 1: Existing Shards + Shard-Local ANN

### Description

Keep the current shard layout. Add one vector ANN index per shard and run ANN inside the selected shard or shards.

### How It Works

- point lookups continue to use the current sharded reader
- vector search reader opens vector sidecars stored alongside the snapshot
- ANN returns candidate keys
- exact rerank uses exact vectors stored in sidecar files
- final payload fetch uses the existing key reader

### Pros

- minimal disruption to the core design
- cleanest way to preserve the current point-lookup API
- easy to ship as `shardyfusion-vector`
- works naturally when CEL routing selects a specific shard from `routing_context`

### Cons

- hash sharding is a poor vector partitioning strategy
- global search over hash shards tends to require fan-out to many or all shards
- recall suffers if only one hash shard is searched

### Best Fit

- CEL-partitioned workloads
- filtered search such as `department_id`, tenant, region, product line

### Verdict

Strong option when CEL routing is used to prune vector search. Weak option if only hash sharding exists and global semantic search dominates.

## Option 2: CEL Partitioning + Dual Indexes Per Partition

### Description

Use CEL sharding for business partitions such as `department_id`. Within each partition, maintain:

- the normal KV store for point lookups
- a vector ANN sidecar for semantic search

### How It Works

- writes use existing CEL routing
- each partition stores normal payload data and vector sidecars
- `get(key, routing_context=...)` routes to one partition
- `search(vector, routing_context=...)` searches only that partition's vector index
- ANN candidates are exact-reranked inside the partition

### Pros

- elegant composition with existing `routing_context`
- very fast filtered vector search
- point lookup and vector search both stay efficient
- scales well when filter dimensions are meaningful and selective
- no need to alter the core `ShardingStrategy` contract

### Cons

- point lookup is most efficient when the caller knows the routing context
- if callers only know the key, an extra key-directory sidecar may be needed
- global search still needs a strategy beyond "search every CEL partition"

### Best Fit

- the target use case discussed in exploration:
  - search with a vector and `department_id` or similar routing context

### Verdict

This is the strongest default design direction for v1.

## Option 3: Separate Vector Sidecars With A Global Overlay

### Description

Keep the current key-routable shard layout for point lookups. Build vector sidecars in their own directory tree, plus a small global overlay used to choose candidate partitions before ANN fan-out.

Example shape:

```text
s3://bucket/prefix/
  shards/run_id=<run_id>/...
  vectors/run_id=<run_id>/vector-manifest.json
  vectors/run_id=<run_id>/partitions/p=<partition>/index.<engine>
  vectors/run_id=<run_id>/partitions/p=<partition>/keys.npy
  vectors/run_id=<run_id>/partitions/p=<partition>/vectors.f16.npy
  vectors/run_id=<run_id>/overlay/partition-centroids.<engine>
```

### How It Works

- filtered search:
  - use CEL routing or metadata to choose one or a few partitions
  - ANN-search only those partitions
- global search:
  - search a tiny overlay index of partition centroids or coarse clusters
  - shortlist the top partitions
  - run ANN inside those partitions
  - rerank exact vectors
  - fetch final payloads through the existing key reader

### Pros

- preserves current point lookup unchanged
- avoids full fan-out for global search
- fits the existing `run_id`-scoped storage layout
- allows independent evolution of vector internals

### Cons

- more moving parts than pure shard-local ANN
- requires a vector-specific manifest or sidecar metadata contract
- promotion of KV snapshot and vector artifacts must be coordinated

### Best Fit

- mixed workloads:
  - some filtered CEL queries
  - some global semantic search

### Verdict

This is the best general-purpose hybrid design if global search matters in addition to CEL-filtered search.

## Option 4: Vector-Only Partitioning

### Description

Create a vector-native partitioning strategy optimized only for ANN search. Do not promise efficient point lookup for this layout.

### How It Works

- vectors are partitioned by semantic clustering or ANN-specific partitioning
- a vector-only reader performs ANN + exact rerank
- the layout is treated as a separate artifact family, not as a core shardyfusion snapshot contract

### Pros

- simple and honest when point lookup is not required
- can optimize aggressively for ANN quality and throughput
- does not need to preserve deterministic `key -> shard` routing

### Cons

- no efficient point lookup guarantee
- should not be represented as a normal core shardyfusion sharding mode
- introducing this into the current `ShardingStrategy` would blur the existing reader contract

### Best Fit

- vector-only applications
- corpora where payload fetch by key is not a first-class access path

### Verdict

Valid as an optional mode in `shardyfusion-vector`, but not the best primary design for the main package.

## Option 5: Replace Core Sharding With Vector Sharding

### Description

Introduce vector-search sharding into the same core sharding abstraction used today for point lookups.

### Pros

- superficially simple because there is one sharding vocabulary

### Cons

- breaks the implicit contract that a core shard layout supports efficient `get(key)`
- makes manifests look reader-compatible even when they are not
- pushes vector-specific assumptions into the core routing model
- makes the normal reader API harder to reason about

### Verdict

This should be rejected.

## Hash Sharding Versus CEL Sharding For Vector Search

### Hash Sharding

Hash sharding remains good for point lookup.

For vector search it has a major weakness:

- semantic neighbors are distributed randomly
- good recall usually requires searching many shards
- searching one shard is fast but low quality
- searching all shards restores quality but increases read amplification

Hash sharding plus shard-local ANN is therefore acceptable only as a fallback.

### CEL Sharding

CEL sharding is much more promising for vector workloads when it reflects meaningful business partitions.

Examples:

- `department_id`
- tenant
- locale
- region
- product line

Benefits:

- natural search pruning
- lower memory footprint per partition
- lower latency due to fewer indexes opened or searched
- better fit for mixed point-lookup and vector-search workloads

## Point Lookup And Vector Search At The Same Time

It is possible to support both efficiently, but usually not from one universal shard layout alone.

The clean pattern is:

1. store payloads in key-routable shards
2. store vector indexes as sidecars
3. return keys from ANN
4. fetch payloads through the existing point-lookup path

This duplicates indexes, not the whole payload store.

That is a good tradeoff because:

- point lookup keeps its fast path
- vector search gets its own optimized structures
- the existing API contract stays stable

## Point Lookup Without CEL Routing Context

CEL partitioning is attractive for filtered vector search, but it introduces one practical issue:

- what if the caller knows only the key and not the routing context?

The main mitigation is a small key-directory sidecar:

- `key -> partition metadata`

That allows:

- one cheap directory lookup
- then a normal KV shard lookup

This directory should be optional. If callers already know the routing context, it is unnecessary.

## ANN Engine Options

The exploration considered several vector engines.

### USearch

Most attractive default engine for `shardyfusion-vector`.

Why:

- supports ANN and exact search
- supports disk-backed serving and memory-mapped views
- relatively lightweight compared with Faiss
- fits well with partition-local indexes plus mmap exact vectors

Risks:

- filtering should be handled primarily by partition selection rather than relying on Python-level predicates

### Faiss

Strong option for larger-scale deployments.

Why:

- rich family of ANN indexes
- supports memory-performance tradeoffs such as `HNSW`, `IVF`, and `PQ`
- good fit when global search scale becomes large enough to justify heavier tuning

Risks:

- heavier dependency
- more operational and tuning complexity

### hnswlib

Useful fallback but not the preferred default.

Why:

- good HNSW implementation
- simple interface

Risks:

- less attractive disk-view story than USearch
- Python-level filtering is not a strong fit for multi-threaded filtered search

### sqlite-vec

Interesting mainly for SQLite-centric or smaller deployments.

Why:

- natural fit for SQLite-backed data
- appealing for simpler embedded setups

Risks:

- not the strongest option for large-scale ANN serving
- better viewed as a targeted backend than as the default vector engine

## Recommended Direction

### Primary Recommendation

Build vector search as a separate package:

- `shardyfusion-vector`

Ship hybrid vector support first:

- preserve existing key-routable sharding in core shardyfusion
- add vector sidecars under the same `run_id`
- support ANN + exact rerank
- use CEL `routing_context` to prune vector search whenever possible
- add a small global overlay for mixed workloads that need unfiltered semantic search

### Recommended V1 Shape

1. Default to CEL-partitioned vector search when filter dimensions exist.
2. Use shard-local or partition-local ANN indexes.
3. Store exact vectors in mmap-friendly files for reranking.
4. Return keys from ANN and fetch payloads via the existing point-lookup path.
5. Keep vector-only partitioning as an optional separate mode in the extra package.

### What Not To Do

- do not add vector-native sharding to the core `ShardingStrategy`
- do not assume hash shards are good ANN partitions
- do not force the existing `ShardReader.get()` contract to grow vector semantics

## Recommended Phasing

### Phase 1 (Implemented)

- ~~create `shardyfusion-vector[usearch]`~~ → created `shardyfusion[vector-lancedb]` (LanceDB backend; usearch rejected for lack of S3 support)
- implement Python writer and reader first
- support:
  - CEL-filtered vector search
  - ANN + exact rerank
  - payload fetch by key through existing readers

### Phase 2 (Implemented)

- add global overlay partition selection for mixed workloads
- add async vector reader (`AsyncShardedVectorReader`, `AsyncUnifiedShardedReader`)
- add docs and benchmarks

### Phase 3 (Implemented)

- add Spark, Dask, and Ray vector build flows
- ~~optionally add Faiss backend~~ → sqlite-vec added as lightweight alternative; LanceDB retained for production scale
- optionally add vector-only partitioning mode

## Final Recommendation

The best overall design is not "vector sharding instead of point-lookup sharding".

The best overall design is:

- key-routable sharding for payload access
- vector sidecars for ANN
- CEL partitioning when business filters should prune search
- optional global overlay for broader semantic search

This keeps shardyfusion's current reader contract intact while still allowing efficient vector search and efficient point lookups in the same snapshot family.
