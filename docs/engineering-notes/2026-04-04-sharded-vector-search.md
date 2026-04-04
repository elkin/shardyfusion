# 2026-04-04 Sharded Vector Search

- Status: `implemented`
- Date: `2026-04-04`
- Baseline repo commit before this change series: `c90ba94`
- Implementation commits:
  - `9ffdfe9` `feat: add sharded vector search (write HNSW indices, search across shards)`
  - `fcd5288` `fix: add VectorReaderHealth export, lazy adapter imports, remove faiss placeholder`

## Summary

This change adds a `shardyfusion.vector` subpackage for sharded approximate nearest neighbor (ANN) vector search. It provides a Python iterator-based writer that builds per-shard HNSW indices and uploads them to S3, and a reader that lazily downloads shard indices and fans out searches with top-k result merging.

The design reuses the existing shardyfusion manifest and S3 infrastructure but introduces a separate sharding model (cluster, LSH, explicit) suited to vector similarity workloads rather than the existing key-hash model.

## 1. What problem is being solved or functionality being added by the changes?

shardyfusion supports sharded KV lookups but has no support for vector similarity search. Users with embedding workloads need to:

- Build per-shard HNSW indices from vector collections
- Upload indices to S3 alongside a manifest
- Search across shards with multi-probe routing and top-k merge
- Support multiple distance metrics (cosine, L2, dot product)

The existing KV sharding model (xxh3_64 hash or CEL expression) is not suitable for vector workloads because it distributes vectors without regard for their geometric proximity. Vector search benefits from locality-aware sharding (cluster or LSH) that places similar vectors on the same shard, reducing the number of shards that need to be probed at query time.

## 2. What design decisions were considered with their pros and cons and trade offs?

### Decision 1: How to shard vectors across indices

#### Option A: Reuse the existing key-hash sharding

Pros:
- no new sharding code
- consistent with KV writer behavior
- uniform shard sizes

Cons:
- vectors with similar embeddings end up on different shards
- every search must query every shard (no routing optimization)
- num_probes optimization is impossible because assignment has no geometric meaning
- defeats the purpose of sharded ANN search

#### Option B: K-means cluster-based sharding

Pros:
- similar vectors land on the same shard by construction
- at query time, only the nearest centroid shards need to be probed
- well-understood IVF-style partitioning from the ANN literature
- centroid training can be done from a sample of the data

Cons:
- requires a training step or user-provided centroids
- uneven shard sizes if cluster sizes vary
- centroids must be persisted for the reader

#### Option C: Locality-sensitive hashing (LSH)

Pros:
- no training step required
- deterministic assignment from random hyperplanes
- supports multi-probe search by flipping hash bits
- hyperplanes are cheap to generate and persist

Cons:
- less precise locality than k-means
- hash collisions can group dissimilar vectors
- multi-probe quality degrades as num_dbs increases relative to hash bit count

#### Option D: Explicit user-assigned shards

Pros:
- maximum flexibility for users with domain-specific partitioning
- trivial implementation
- no training or random state

Cons:
- puts all routing decisions on the user
- no built-in multi-probe at query time

**Chosen approach**: All three (B, C, D) are implemented as `VectorShardingStrategy.CLUSTER`, `LSH`, and `EXPLICIT`. Option A was rejected because it provides no geometric locality. The three strategies cover the common use cases: CLUSTER for best search quality, LSH for training-free deployment, and EXPLICIT for users who manage their own partitioning.

### Decision 2: How to store vector metadata in the manifest

#### Option A: Extend RequiredBuildMeta with vector-specific fields

Pros:
- vector metadata is first-class in the manifest schema
- type-checked by Pydantic validation

Cons:
- requires modifying the core manifest model, which all readers parse
- introduces vector-specific fields (dim, metric, centroids_ref) into a model shared with KV writers
- risks manifest format compatibility issues with non-vector readers
- increases coupling between vector and KV code paths

#### Option B: Use the existing custom_manifest_fields dict

Pros:
- no changes to core manifest models or manifest format version
- vector metadata is isolated in a `"vector"` key within the custom dict
- non-vector readers ignore the custom fields entirely
- avoids manifest schema migration for all existing deployments

Cons:
- vector metadata is untyped at the manifest level (dict, not Pydantic model)
- reader must manually parse and validate the custom dict
- no schema enforcement at publish time

#### Option C: Create a separate vector manifest alongside the standard manifest

Pros:
- complete isolation from KV manifest
- can evolve independently

Cons:
- duplicates manifest infrastructure (publish, load, list, set_current)
- two manifest stores to configure and manage
- complex coordination if both KV and vector share a prefix

**Chosen approach**: Option B. The custom fields dict already exists for user-defined metadata. Storing vector config there avoids touching the core manifest schema while keeping the publish/load/refresh infrastructure entirely shared. The reader parses the `"vector"` key from `manifest.custom` and reconstructs the index config and sharding metadata. Centroids and hyperplanes are stored as separate `.npy` files in S3 and referenced by URL in the custom dict.

### Decision 3: How to structure the per-shard index adapter

#### Option A: Build HNSW graph in Python, serialize custom format

Pros:
- full control over the graph format
- no external library dependency

Cons:
- implementing HNSW correctly in pure Python is error-prone and slow
- custom format requires custom deserialization
- no established tooling for inspection or debugging

#### Option B: Use usearch as the index engine with a pluggable adapter protocol

Pros:
- usearch is a mature, fast HNSW implementation
- supports multiple distance metrics and quantization (fp16, i8)
- save/load to file is built in
- deferred import means no hard dependency at package level
- adapter protocol allows swapping to FAISS or other engines later

Cons:
- adds an optional C++ dependency (usearch)
- usearch API may change between versions
- payload storage is not built into usearch (requires a sidecar)

#### Option C: Use FAISS as the index engine

Pros:
- widely used, well-tested
- supports IVF, PQ, and other advanced index types

Cons:
- heavier dependency (faiss-cpu or faiss-gpu)
- API is more complex than usearch for simple HNSW
- no clear advantage for the HNSW-only use case

**Chosen approach**: Option B with `VectorIndexWriter` and `VectorShardReader` protocols. The usearch adapter is the default implementation, imported lazily. Payloads are stored in a SQLite sidecar database alongside each shard's index file. The protocol design allows adding a FAISS adapter later without changing the writer or reader. The pre-existing `HnswGraphBuilder` / `HnswGraph` / `HnswNode` types in `types.py` were kept for potential future graph inspection use but are not used by the current adapter — usearch manages the HNSW graph internally.

### Decision 4: How to handle reader shard lifecycle

#### Option A: Preload all shard readers on initialization

Pros:
- first search has no cold-start latency
- simple implementation

Cons:
- downloads all N shard indices from S3 at startup, even if only a few are queried
- high memory usage for large shard counts
- slow startup time proportional to num_dbs

#### Option B: Lazy loading with LRU eviction

Pros:
- only downloads shards that are actually queried
- bounded memory via max_cached_shards
- amortized cold start across queries
- double-checked locking prevents duplicate downloads under concurrency

Cons:
- first query to a shard has download latency
- LRU eviction may cause re-downloads for access patterns that cycle through many shards

#### Option C: Lazy loading with TTL-based eviction

Pros:
- time-bounded cache freshness

Cons:
- adds timer complexity
- TTL does not bound memory usage
- less useful than LRU for shard access patterns

**Chosen approach**: Option B with optional preload. The reader uses an `OrderedDict` as an LRU cache with per-shard locks for concurrent download safety. `preload_shards=True` is supported for users who want eager loading. `max_cached_shards` controls the eviction threshold.

### Decision 5: How to merge results across shards

#### Option A: Sort all results, take top-k

Pros:
- simple implementation

Cons:
- O(N log N) where N is total results across all shards
- wasteful when N >> top_k

#### Option B: Heap-based top-k selection

Pros:
- O(N log k) time complexity
- uses Python's built-in heapq.nsmallest / nlargest
- naturally handles metric-dependent ordering (lower-is-better for L2/cosine, higher-is-better for dot product)

Cons:
- marginal complexity over sort for small result sets

**Chosen approach**: Option B. `heapq.nsmallest` for L2 and cosine (lower score = better), `heapq.nlargest` for dot product (higher score = better). The implementation is in `_merge.py`.

## 3. What implementation was chosen and why?

The implementation creates a `shardyfusion/vector/` subpackage with the following structure:

### Module Layout

- `types.py` — Enums (`DistanceMetric`, `VectorShardingStrategy`), data classes (`VectorRecord`, `SearchResult`, `VectorSearchResponse`, `VectorShardDetail`, `VectorSnapshotInfo`), and protocols (`VectorIndexWriter`, `VectorIndexWriterFactory`, `VectorShardReader`, `VectorShardReaderFactory`).
- `config.py` — `VectorIndexConfig` (dim, metric, HNSW params, quantization), `VectorShardingSpec` (strategy, num_probes, centroids/hyperplanes), `VectorWriteConfig` (top-level config mirroring `WriteConfig` structure).
- `sharding.py` — Pure-numpy sharding implementations: k-means++ training (`train_centroids_kmeans`), cluster assignment and probing, LSH hyperplane generation, LSH hashing and probing, and a unified `route_vector_to_shards` dispatcher.
- `_merge.py` — Heap-based top-k merge across per-shard result lists, metric-aware.
- `adapters/usearch_adapter.py` — `USearchWriter` (builds usearch.Index + SQLite payloads locally, uploads to S3 on close), `USearchShardReader` (downloads from S3, loads index and payloads), and corresponding factories. usearch is imported inside `_import_usearch()` to keep it optional.
- `writer.py` — `write_vector_sharded()` function with single-process implementation. Buffers records by shard, flushes in batches to the adapter, handles centroid training and hyperplane generation, uploads sharding metadata to S3, and publishes a manifest with vector config in custom fields.
- `reader.py` — `ShardedVectorReader` class with lazy shard loading, LRU eviction, thread-pool fan-out for multi-shard search, manifest refresh, health monitoring, and shard inspection methods.
- `__init__.py` — Public API exports (18 names).
- `adapters/__init__.py` — Lazy `__getattr__` exports for USearch adapter types.

### Write Pipeline

1. Validate config (dim > 0, s3_prefix set, sharding spec consistent).
2. Resolve sharding: train centroids if requested, generate hyperplanes for LSH, validate explicit shard IDs.
3. Create S3 client from credential provider.
4. Iterate records: assign each to a shard via `_assign_shard()`, buffer in `_ShardState`.
5. When batch reaches `config.batch_size`, flush to adapter via `add_batch()`.
6. After iteration: flush remaining batches, checkpoint each adapter.
7. Upload centroids/hyperplanes as `.npy` files to `{s3_prefix}/vector_meta/`.
8. Build `RequiredBuildMeta` (using `KeyEncoding.RAW` and `ShardingStrategy.HASH` as placeholders) and `RequiredShardMeta` list from shard states.
9. Publish manifest via `ManifestStore.publish()` with vector metadata in `custom["vector"]`.
10. Return `BuildResult`.

### Read Pipeline

1. Load manifest via `ManifestStore.load_current()` + `load_manifest()`.
2. Parse `manifest.custom["vector"]` to reconstruct `VectorIndexConfig`, sharding strategy, and num_probes.
3. Download centroids/hyperplanes from S3 refs in the custom dict.
4. On `search(query, top_k)`: route query to shard IDs via `route_vector_to_shards()`, fan out to shards (thread pool or sequential), merge results via `_merge.merge_results()`.
5. Shard readers are loaded lazily on first access, cached in an `OrderedDict` LRU, and evicted when `max_cached_shards` is exceeded.
6. `refresh()` reloads the manifest, closes old shard readers, and rebuilds state.

### Manifest Integration

Vector metadata stored in `manifest.custom["vector"]`:

```json
{
  "dim": 128,
  "metric": "cosine",
  "index_type": "hnsw",
  "quantization": null,
  "total_vectors": 100000,
  "sharding_strategy": "cluster",
  "num_probes": 2,
  "centroids_ref": "s3://bucket/prefix/vector_meta/centroids.npy",
  "num_hash_bits": 8,
  "hyperplanes_ref": "s3://bucket/prefix/vector_meta/hyperplanes.npy"
}
```

This avoids any changes to the core manifest schema.

### Adapter Protocol

The `VectorIndexWriter` protocol mirrors the existing `DbAdapter` pattern:

- `add_batch(ids, vectors, payloads)` — add vectors to the index
- `flush()` — persist current state to local files
- `checkpoint() -> str | None` — return a content hash for the manifest
- `close()` — upload local files to S3
- Context manager support via `__enter__` / `__exit__`

The `VectorShardReader` protocol:

- `search(query, top_k, ef) -> list[SearchResult]` — search the shard
- `close()` — release resources

### Payload Storage

The USearch adapter stores payloads in a SQLite sidecar (`payloads.db`) alongside the index file (`index.usearch`). Each shard has two files uploaded to S3:

- `{db_url}/index.usearch` — the usearch HNSW index
- `{db_url}/payloads.db` — SQLite with `(id TEXT PRIMARY KEY, payload TEXT)` table

The reader downloads both files, loads the index, and joins search results with payloads via SQL lookups.

## Known Limitations

### Not Yet Implemented

- **Parallel writer mode**: `write_vector_sharded(..., parallel=True)` raises `ConfigValidationError`. Multi-process writing would require the same spool-file approach as the KV Python parallel writer.
- **Async reader**: No `AsyncShardedVectorReader` yet. The sync reader with thread-pool fan-out covers the initial use case.
- **FAISS adapter**: The adapter protocol supports it, but no implementation exists.
- **Dask/Ray/Spark writers**: Only the Python iterator-based writer is implemented. Framework writers would follow the same pattern as existing KV framework writers.

### Design Trade-offs Accepted

- **Placeholder sharding in manifest**: `RequiredBuildMeta.sharding` is set to `ShardingStrategy.HASH` as a placeholder since the core `ManifestShardingSpec` does not support vector sharding strategies. The actual vector sharding config lives in `custom["vector"]`. This is fine because non-vector readers never interpret the sharding field for vector manifests.
- **Payload sidecar**: Storing payloads in a separate SQLite file adds a second S3 GET per shard load. An alternative would be embedding payloads in the index file format, but usearch does not support arbitrary payload storage, and the SQLite approach keeps payloads queryable.
- **No graph inspection**: The `HnswGraph` / `HnswNode` types exist in `types.py` but are not populated by the usearch adapter. usearch manages the graph internally and does not expose node-level structure. These types remain available for potential future adapters that do expose graph internals.
