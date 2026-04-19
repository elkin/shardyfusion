# Vector Search

## Overview

`shardyfusion` provides serverless vector search directly against sharded datasets. The search uses a "Double-Dip" process:
1. **Routing:** The reader evaluates the query against metadata (like centroids, hyperplanes, or CEL rules) to identify target shards.
2. **Fan-out:** It concurrently dispatches the query to these shards.
3. **Local Search:** Each shard performs an Approximate Nearest Neighbor (ANN) search via its backend adapter and returns a local Top-K.
4. **Global Merge:** The reader performs a final global sort/merge to return the true Top-K nearest neighbors.

## Modes of Operation

Vector search in `shardyfusion` operates in two primary modes:

1. **Unified KV + Vector:** Configured using `WriteConfig(vector_spec=...)` in distributed writers (Spark, Ray, Dask). This mode embeds both key-value data for point lookups and vector embeddings in the same snapshot. 
2. **Standalone Vector:** Focused entirely on indexing and querying vectors. Configured via `VectorWriteConfig` (often used via the Python writer).

## Storage Backends & Use Cases (Adapters)

Vector indexing and searching are handled via specialized adapters. The backend choice dictates performance characteristics and deployment complexity.

### LanceDB (High-Performance Sidecar)
Uses `.lance` datasets built with HNSW and optional Product Quantization (PQ) / Scalar Quantization (SQ). 

**When to use LanceDB:**
*   **High Dimensionality & Massive Scale:** Ideal for 768d, 1536d, or 3072d embeddings (OpenAI, Cohere). Memory footprint during search is a primary concern, and LanceDB's PQ (`quantization="i8"`) compresses these effectively.
*   **Advanced Index Control:** When you require fine-grained tuning of HNSW parameters (like `M`, `ef_construction`) to balance recall and latency.
*   **Vector-Only Pipelines:** When your application strictly performs similarity searches and does not require complex KV point-lookups.

*Requires the `shardyfusion[vector]` extra.*

### sqlite-vec (Embedded Unified DB)
Direct integration with SQLite via the `sqlite-vec` extension, storing KV and vector data together in a single `shard.db` file.

**When to use sqlite-vec:**
*   **Operational Simplicity:** You want a single `shard.db` file per shard containing both your KV data and your vector index, avoiding sidecar `.lance` files and extra S3 objects.
*   **Small to Medium Datasets:** Great for datasets where vectors easily fit into memory and aggressive quantization isn't necessary.
*   **Multi-modal Queries:** When you expect to need point lookups via standard key fetches alongside nearest-neighbor searches.

*Requires the `shardyfusion[vector-sqlite]` extra.*

## Sharding Strategies (Routing)

To handle different data shapes and workload profiles, `shardyfusion` supports four distinct vector routing strategies:

1. **`cluster` (Locality Sensitive Clustering - K-Means)**
   *   **Mechanism:** Finds optimal centroids during the write phase and routes vectors to their nearest centroid's shard.
   *   **Best for:** **Maximizing recall**. Datasets that naturally group together yield the best retrieval quality when clustered.

2. **`lsh` (Locality Sensitive Hashing)**
   *   **Mechanism:** Uses deterministic random projections (hyperplanes) to map vectors to buckets.
   *   **Best for:** **Massive ingestion throughput**. Hashing is virtually free compared to training centroids. Ideal for streaming data or scenarios where read-time multi-probe searches offset slightly lower precision.

3. **`cel` (Common Expression Language)**
   *   **Mechanism:** Routes vectors based on structured metadata using a CEL expression (e.g., `tenant_id == "acme"`).
   *   **Best for:** **Strict tenant isolation** and hard domain boundaries.

4. **`explicit`**
   *   **Mechanism:** The caller manually dictates the shard ID.
   *   **Best for:** Applications that maintain external directories or have highly customized assignment logic.

## Usage Guide & Code Examples

### 1. Writing Unified Vectors (PySpark)

Here is how you configure a distributed Spark writer to build a unified KV+Vector index using LanceDB and the `cluster` routing strategy.

```python
import numpy as np
from shardyfusion.config import WriteConfig, VectorSpec, OutputOptions, ManifestOptions
from shardyfusion.vector.config import VectorSpecSharding
from shardyfusion.manifest_store import S3ManifestStore
from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy
from shardyfusion.vector.adapters.lancedb_adapter import LanceDbWriterFactory
from shardyfusion.writer.spark.writer import write_vector_sharded

# Define vector dimensions and sharding
vector_spec = VectorSpec(
    dim=128,
    vector_col="embedding",
    sharding=VectorSpecSharding(
        strategy="cluster",
        train_centroids=True,  # Automatically trains k-means centroids on driver
    ),
)

config = WriteConfig(
    num_dbs=8,
    s3_prefix="s3://my-bucket/vectors-run",
    vector_spec=vector_spec,
    sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
    output=OutputOptions(run_id="run-123", local_root="/tmp/writer"),
    adapter_factory=LanceDbWriterFactory(
        # The factory will serialize configuration for distributed Spark executors
        s3_connection_options=opts, 
        credential_provider=creds
    ),
    credential_provider=creds,
    s3_connection_options=opts,
    manifest=ManifestOptions(
        store=S3ManifestStore("s3://my-bucket/vectors-run", ...)
    ),
)

# df is a PySpark DataFrame with 'id' and 'embedding' columns
result = write_vector_sharded(
    df, 
    config, 
    vector_col="embedding", 
    id_col="id"
)
```

### 2. Reading and Searching

Use `UnifiedShardedReader` (if the index contains KV data) or `ShardedVectorReader` for vector-only indices. For high-concurrency environments like `asyncio`-based web services, use the `AsyncShardedVectorReader`.

### Synchronous Reader

```python
from shardyfusion.vector.reader import ShardedVectorReader
import numpy as np

reader = ShardedVectorReader(
    s3_prefix="s3://my-bucket/vectors-run",
    local_root="/tmp/reader",
)

# Perform a Vector Search
query_vector = np.random.randn(128).astype(np.float32)

# The reader uses the routing metadata in the manifest to fan out only to relevant shards.
results = reader.search(query_vector, top_k=10)
for res in results:
    print(res.id, res.score, res.payload)

reader.close()
```

### Asynchronous Reader (FastAPI / aiohttp)

```python
from shardyfusion.vector.async_reader import AsyncShardedVectorReader
import numpy as np

# Use the factory method `open` to load the manifest asynchronously
reader = await AsyncShardedVectorReader.open(
    s3_prefix="s3://my-bucket/vectors-run",
    local_root="/tmp/reader",
    max_concurrency=4,  # limits concurrent shard S3 downloads
)

# Perform an Async Vector Search
query_vector = np.random.randn(128).astype(np.float32)

# Shards are queried concurrently for lower latency
results = await reader.search(query_vector, top_k=10)
for res in results:
    print(res.id, res.score, res.payload)

await reader.close()
```

for res in results:
    print(f"ID: {res.id}, Score: {res.score}, Payload: {res.payload}")
```
