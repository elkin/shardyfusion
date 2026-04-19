# Vector Search over Sharded SQLite

## Overview
`shardyfusion` provides serverless vector search directly against sharded datasets. The generic `ShardedVectorReader` uses a two-step flow: it first uses snapshot metadata to identify target shards, then queries those shards through the configured vector backend. Today that generic path is typically backed by `LanceDB`. The unified SQLite path (`UnifiedShardedReader` with `sqlite-vec`) is a separate read path that stores KV and vector data together in SQLite shards.

## Routing Strategies
To handle different workloads, `shardyfusion` supports four distinct vector routing strategies:

1. **LSH (Locality Sensitive Hashing):** 
   - **Mechanism:** Uses random projections (hyperplanes) to map vectors to buckets.
   - **Best for:** High-throughput streaming writes and global searches.
   - **Reader Path:** The reader uses the same hyperplanes to find the primary bucket and then probes neighboring buckets (multi-probe LSH) to increase recall.

2. **CLUSTER (Centroid-based):**
   - **Mechanism:** Uses K-Means clustering to find optimal centroids during the write phase.
   - **Best for:** Datasets where data naturally clusters; generally provides better recall than LSH.
   - **Reader Path:** The reader calculates the distance from the query vector to all centroids and probes the Top-N closest shards.

3. **CEL (Common Expression Language):**
   - **Mechanism:** Uses a CEL expression to route vectors based on metadata (e.g., `tenant_id`) provided in a routing context.
   - **Best for:** Multi-tenant isolation and hard boundaries.
   - **Reader Path:** The reader evaluates the CEL expression with the provided context to determine the target shard.

4. **EXPLICIT:**
   - **Mechanism:** The user provides the shard ID directly.
   - **Best for:** Applications that already know where their data lives.

## Architecture & Implementation

### Manifest & Metadata
Vector snapshots use the normal manifest format. Vector-specific metadata is stored under `custom["vector"]`:
- **`dim`**: Dimensionality of the vectors.
- **`metric`**: Distance metric (`cosine`, `l2`, or `dot_product`).
- **`sharding_strategy`**: The routing strategy used.
- **`centroids_ref` / `hyperplanes_ref`**: S3 pointers to the numpy arrays used for routing.

### The "Double-Dip" Search Process
When a user calls `.search(query_vector, top_k=10)`, the following happens:
1. **Routing:** The `ShardedVectorReader` uses the active routing strategy to identify the target `db_id`s.
2. **Fan-out:** The reader concurrently dispatches the search query to the identified shards.
3. **Local Search:** Each shard performs a local ANN search through its configured vector adapter and returns its local Top-K.
4. **Global Merge:** The reader collects all local results and performs a final global sort/merge to return the true Top-K nearest neighbors.

Routing inputs depend on the strategy:
- **CLUSTER / LSH:** the query vector alone is enough to derive target shards.
- **CEL:** the query must include `routing_context=...` so the CEL expression can resolve the shard.
- **EXPLICIT:** the query must include `shard_ids=[...]` because shard selection is provided by the caller.

Examples:

```python
# CLUSTER / LSH
reader.search(query_vector, top_k=10)

# CEL
reader.search(query_vector, top_k=10, routing_context={"tenant_id": "acme"})

# EXPLICIT
reader.search(query_vector, top_k=10, shard_ids=[3, 7])
```

### Storage Backends
Vector search is implemented via specialized adapters:
- **`LanceDB`**: A high-performance HNSW sidecar that can be composed with any KV backend (SlateDB or SQLite). This is the typical backend for the generic `ShardedVectorReader` path.
- **`sqlite-vec`**: Direct integration with SQLite via the `sqlite-vec` extension. This is used in the unified SQLite KV+vector path.

### Write Pipeline
The `write_vector_sharded` pipeline handles the complexity of routing setup:
- For **CLUSTER**, it can optionally perform a sampling phase to train centroids using K-Means.
- For **LSH**, it generates deterministic hyperplanes based on a seed.
- It ensures that all writers (whether Python or distributed) produce the same shard assignments for the same vector.
