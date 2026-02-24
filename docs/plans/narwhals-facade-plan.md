# Narwhals Facade Plan for Slatefusion

## Executive Summary

This plan evaluates using [narwhals](https://github.com/narwhals-dev/narwhals) as a
DataFrame abstraction layer so that `slatefusion` can accept PySpark, Dask, Polars,
pandas, cuDF, and other narwhals-supported backends as input — rather than being
locked to PySpark for the writer path.

**Key finding:** Narwhals can replace the *DataFrame transformation* layer (column
operations, filtering, type checking) but **cannot** replace the *distributed
execution* layer (RDD partitioning, `mapPartitionsWithIndex`, `TaskContext`, Spark ML
Bucketizer, `approxQuantile`). A hybrid approach is required.

---

## 1. Current Architecture Analysis

### PySpark Usage Map

PySpark is used in exactly **2 modules** (798 lines, ~20% of the codebase):

| Module | PySpark APIs Used | Purpose |
|---|---|---|
| `writer/spark/sharding.py` | `F.pmod`, `F.xxhash64`, `F.col`, `F.lit`, `F.expr`, `F.cast`, `Bucketizer`, `df.approxQuantile`, `df.withColumn`, `df.where`, `df.limit`, `df.count`, `df.schema`, `df.sortWithinPartitions`, `df.rdd`, `RDD.map`, `RDD.partitionBy` | Shard ID assignment, partitioning |
| `writer/spark/writer.py` | `df.sparkSession`, `df.persist/unpersist`, `df.select`, `df.limit`, `df.collect`, `RDD.mapPartitionsWithIndex`, `RDD.collect`, `TaskContext`, `StorageLevel`, `SparkSession` | Orchestration, distributed execution |

### The 80% That's Already Backend-Agnostic

The rest of the codebase (reader, routing, config, manifest, publishing, CLI, serde)
is **pure Python** with zero PySpark dependency. The Python writer
(`writer/python/writer.py`) already proves the core write pipeline works without
Spark, using `multiprocessing` for parallelism.

### PySpark Operations Categorized

**Replaceable by Narwhals** (DataFrame transformations):
- `df.withColumn(name, expr)` → `nw.LazyFrame.with_columns()`
- `df.select(cols)` → `nw.LazyFrame.select()`
- `df.where(condition)` / `df.filter()` → `nw.LazyFrame.filter()`
- `df.limit(n)` → `nw.LazyFrame.head(n)`
- `df.count()` → `nw.LazyFrame.collect()` then len, or backend-specific
- Column type checking (`df.schema[col].dataType`) → `nw.LazyFrame.schema`
- Column casting (`.cast("int")`) → `nw.Expr.cast(nw.Int32)`
- `df.sortWithinPartitions(col)` → no direct equivalent, but sort is available

**NOT replaceable by Narwhals** (distributed execution / backend-specific):
- `F.pmod(F.xxhash64(...), F.lit(n))` — no hash function in narwhals
- `F.expr(sql_string)` — narwhals has no SQL expression passthrough
- `Bucketizer` (Spark ML) — not a DataFrame op
- `df.approxQuantile()` — statistical method, not in narwhals
- `df.rdd` → `RDD.map()` → `RDD.partitionBy()` — RDD-level partitioning
- `RDD.mapPartitionsWithIndex()` — distributed execution primitive
- `TaskContext.get()` — Spark worker context
- `df.persist()` / `df.unpersist()` — caching control
- `SparkSession` / `df.sparkSession` — session management

---

## 2. Narwhals Capabilities Summary

| Feature | Available? | Notes |
|---|---|---|
| `select`, `with_columns` | Yes | Core Polars-like API |
| `filter` | Yes | Boolean expressions |
| `sort` | Yes | `sort(by, descending)` |
| `group_by` + `agg` | Yes | Standard aggregations |
| `join` | Yes | Various join types |
| `cast` | Yes | Type coercion |
| Schema inspection | Yes | `df.schema`, `df.collect_schema()` |
| `head(n)` | Yes | Equivalent to `limit` |
| `collect()` | Yes | Materialize lazy frame |
| `to_native()` | Yes | Return original backend object |
| `get_native_namespace()` | Yes | Get backend module (escape hatch) |
| Hash functions (xxhash64) | **No** | Must use escape hatch |
| SQL expressions | **No** | Must use escape hatch |
| RDD operations | **No** | Spark-specific, no abstraction |
| Approximate quantiles | **No** | Backend-specific statistical method |
| Partitioning control | **No** | Backend-specific execution detail |
| UDFs | **No** | Not in narwhals scope |

### Narwhals Backend Tiers

| Tier | Backends | Capabilities |
|---|---|---|
| **Full** (eager + lazy) | pandas, Polars, Modin, cuDF, PyArrow | All narwhals operations |
| **Lazy-only** | PySpark, Dask, DuckDB, Ibis, Daft, SQLFrame | Lazy operations only (no Series) |

---

## 3. Proposed Architecture

### 3.1 Design Principle: Layered Abstraction with Escape Hatches

```
                    User code
                        │
                        ▼
              ┌─────────────────────┐
              │   write_sharded()   │  ← unified entry point
              │   (narwhals facade) │
              └────────┬────────────┘
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
   ┌─────────────────┐  ┌──────────────────┐
   │  DataFrame ops   │  │  Execution engine │
   │  (narwhals)      │  │  (backend-native) │
   │                  │  │                   │
   │  • schema check  │  │  • partitioning   │
   │  • add columns   │  │  • distributed    │
   │  • filter/sort   │  │    map/collect    │
   │  • type coercion │  │  • hash sharding  │
   └─────────────────┘  │  • approxQuantile │
                        └──────────────────┘
```

### 3.2 Module Structure

```
slatedb_spark_sharded/
├── writer/
│   ├── _frame_ops.py          # NEW: narwhals-based DataFrame operations
│   ├── _execution.py          # NEW: execution engine protocol + registry
│   ├── _engines/              # NEW: backend-specific execution engines
│   │   ├── __init__.py
│   │   ├── _spark_engine.py   #   PySpark: RDD partitioning + mapPartitionsWithIndex
│   │   ├── _dask_engine.py    #   Dask: partition map + delayed execution
│   │   ├── _polars_engine.py  #   Polars: rayon-parallel partition writes
│   │   └── _python_engine.py  #   Fallback: collect-to-iterator + multiprocessing
│   ├── spark/
│   │   ├── writer.py          # EXISTING: kept as thin wrapper calling unified path
│   │   └── sharding.py        # EXISTING: Spark-specific hash/range sharding preserved
│   └── python/
│       └── writer.py          # EXISTING: kept as thin wrapper
```

### 3.3 New Abstractions

#### 3.3.1 `FrameInput` — Narwhals-Wrapped Input

```python
import narwhals as nw
from narwhals.typing import IntoFrameT

def write_sharded_df(
    df: IntoFrameT,
    config: WriteConfig,
    *,
    key_col: str,
    value_spec: ValueSpec,
    **kwargs,
) -> BuildResult:
    """Backend-agnostic sharded writer entry point."""
    nw_frame = nw.from_native(df, eager_or_interchange_only=False)
    # ... dispatch to appropriate engine
```

#### 3.3.2 `ExecutionEngine` Protocol

```python
from typing import Protocol

class ExecutionEngine(Protocol):
    """Backend-specific execution strategy for distributed shard writing."""

    def add_shard_column(
        self,
        native_df: Any,
        key_col: str,
        num_dbs: int,
        sharding: ShardingSpec,
    ) -> tuple[Any, ShardingSpec]:
        """Add _slatedb_db_id column using backend-native operations."""
        ...

    def execute_partition_writes(
        self,
        native_df_with_shard_id: Any,
        num_dbs: int,
        write_fn: Callable[[int, Iterable[tuple[int, Any]]], Iterator[ShardAttemptResult]],
    ) -> list[ShardAttemptResult]:
        """Partition data by shard ID and execute writes per partition."""
        ...

    def compute_approx_quantiles(
        self,
        native_df: Any,
        col: str,
        probabilities: list[float],
        rel_error: float,
    ) -> list[float]:
        """Compute approximate quantiles for range boundary detection."""
        ...
```

#### 3.3.3 `BackendDetector` — Engine Selection

```python
def detect_engine(nw_frame: nw.LazyFrame) -> ExecutionEngine:
    """Select execution engine based on the native backend."""
    impl = nw_frame.implementation
    if impl.is_pyspark():
        return SparkExecutionEngine()
    if impl.is_dask():
        return DaskExecutionEngine()
    if impl.is_polars():
        return PolarsExecutionEngine()
    # Fallback: collect to Python and use multiprocessing
    return PythonExecutionEngine()
```

---

## 4. Engine Implementations

### 4.1 Spark Engine (`_spark_engine.py`)

This is essentially the current `writer/spark/` code, restructured:

```python
class SparkExecutionEngine:
    def add_shard_column(self, native_df, key_col, num_dbs, sharding):
        # Existing sharding.add_db_id_column() — unchanged
        return add_db_id_column(native_df, key_col=key_col, ...)

    def execute_partition_writes(self, native_df, num_dbs, write_fn):
        # Existing RDD partitioning + mapPartitionsWithIndex — unchanged
        pair_rdd = native_df.rdd.map(lambda r: (int(r[DB_ID_COL]), r))
        partitioned = pair_rdd.partitionBy(num_dbs, lambda k: int(k))
        return partitioned.mapPartitionsWithIndex(write_fn).collect()

    def compute_approx_quantiles(self, native_df, col, probabilities, rel_error):
        return native_df.approxQuantile(col, probabilities, rel_error)
```

### 4.2 Dask Engine (`_dask_engine.py`)

```python
class DaskExecutionEngine:
    def add_shard_column(self, native_df, key_col, num_dbs, sharding):
        import dask.dataframe as dd
        # Use map_partitions with Python xxhash routing
        def assign_shard(partition, key_col, num_dbs, key_encoding):
            import xxhash
            partition["_slatedb_db_id"] = partition[key_col].apply(
                lambda k: _xxhash64_db_id(k, num_dbs, key_encoding)
            )
            return partition
        return native_df.map_partitions(assign_shard, key_col, num_dbs, ...), resolved

    def execute_partition_writes(self, native_df, num_dbs, write_fn):
        # Shuffle by shard ID, then process each partition
        shuffled = native_df.set_index("_slatedb_db_id", sorted=True)
        # Use groupby + apply or partition-level iteration
        ...

    def compute_approx_quantiles(self, native_df, col, probabilities, rel_error):
        # Dask supports .quantile() on Series
        return native_df[col].quantile(probabilities).compute().tolist()
```

### 4.3 Polars Engine (`_polars_engine.py`)

```python
class PolarsExecutionEngine:
    def add_shard_column(self, native_df, key_col, num_dbs, sharding):
        import polars as pl
        # Polars has no xxhash64 — use Python routing via map_elements
        df = native_df.with_columns(
            pl.col(key_col).map_elements(
                lambda k: _xxhash64_db_id(k, num_dbs, key_encoding),
                return_dtype=pl.Int32,
            ).alias("_slatedb_db_id")
        )
        return df, resolved

    def execute_partition_writes(self, native_df, num_dbs, write_fn):
        # Group by shard ID, iterate groups — Polars is single-node but fast
        for db_id in range(num_dbs):
            shard_df = native_df.filter(pl.col("_slatedb_db_id") == db_id)
            # Write shard using iterator
            ...

    def compute_approx_quantiles(self, native_df, col, probabilities, rel_error):
        return [native_df[col].quantile(p) for p in probabilities]
```

### 4.4 Python Fallback Engine (`_python_engine.py`)

```python
class PythonExecutionEngine:
    def add_shard_column(self, native_df, key_col, num_dbs, sharding):
        # Collect to Python, route using existing _route_key
        # Return as list of (db_id, record) or similar
        ...

    def execute_partition_writes(self, native_df, num_dbs, write_fn):
        # Existing writer/python/writer.py logic
        # multiprocessing with queues per shard
        ...

    def compute_approx_quantiles(self, native_df, col, probabilities, rel_error):
        # numpy-based quantile on collected data
        import numpy as np
        values = [row[col] for row in native_df]
        return [float(np.quantile(values, p)) for p in probabilities]
```

---

## 5. What Narwhals Handles (the Common Layer)

These operations use narwhals and work identically across all backends:

### 5.1 Schema Validation

```python
def validate_key_column(nw_frame: nw.LazyFrame, key_col: str, strategy: ShardingStrategy):
    schema = nw_frame.collect_schema()
    if key_col not in schema:
        raise ShardAssignmentError(f"Key column `{key_col}` not found")

    dtype = schema[key_col]
    if strategy == ShardingStrategy.HASH:
        if dtype not in (nw.Int32, nw.Int64, nw.UInt32, nw.UInt64):
            raise ShardAssignmentError(f"Hash sharding requires integer key; got {dtype}")
```

### 5.2 Shard ID Validation

```python
def validate_shard_ids(nw_frame: nw.LazyFrame, num_dbs: int):
    """Validate all shard IDs are in [0, num_dbs)."""
    invalid = nw_frame.filter(
        (nw.col(DB_ID_COL).is_null())
        | (nw.col(DB_ID_COL) < 0)
        | (nw.col(DB_ID_COL) >= num_dbs)
    ).head(1).collect()
    if len(invalid) > 0:
        raise ShardAssignmentError("Computed db_id out of range [0, num_dbs-1].")
```

### 5.3 Sampling for Routing Verification

```python
def sample_for_verification(nw_frame: nw.LazyFrame, key_col: str, n: int = 20):
    """Sample rows for routing verification."""
    return nw_frame.select(key_col, DB_ID_COL).head(n).collect()
```

### 5.4 Row Count / Basic Stats

```python
# Narwhals-based count
count = nw_frame.select(nw.col(key_col).count()).collect().item()
```

---

## 6. What Requires Escape Hatches

These operations **must** drop to the native backend via `nw.to_native()`:

| Operation | Why | Escape Hatch Pattern |
|---|---|---|
| Hash sharding (`xxhash64`) | Narwhals has no hash functions | `to_native()` → engine-specific |
| SQL expression passthrough | Narwhals has no `expr()` | Spark: `F.expr()`, others: N/A |
| Bucketizer (Spark ML) | ML library, not DataFrame op | Spark-only, others use Python |
| `approxQuantile` | Statistical method, not in narwhals | `to_native()` → engine-specific |
| RDD partitioning | Execution primitive, not data transform | Engine-specific execution |
| `mapPartitionsWithIndex` | Distributed execution | Engine-specific execution |
| `TaskContext` | Worker runtime context | Spark-only |
| `persist/unpersist` | Caching control | Engine-specific, optional |

---

## 7. Migration Plan

### Phase 1: Add Narwhals + Engine Protocol (non-breaking)

**Goal:** Introduce narwhals dependency, create the `ExecutionEngine` protocol, and
wrap existing Spark code as the first engine implementation.

1. Add `narwhals>=1.20` to core dependencies in `pyproject.toml`
2. Create `writer/_execution.py` with the `ExecutionEngine` protocol
3. Create `writer/_engines/_spark_engine.py` wrapping existing `writer/spark/` code
4. Create `writer/_frame_ops.py` with narwhals-based schema validation, shard ID
   validation, and sampling helpers
5. **No behavior changes** — existing `write_sharded_spark()` continues working

### Phase 2: Unified Entry Point

**Goal:** Create `write_sharded_df()` that accepts any narwhals-compatible DataFrame.

1. Create `writer/writer.py` with `write_sharded_df(df, config, ...)` that:
   - Wraps input with `nw.from_native()`
   - Detects backend via `nw_frame.implementation`
   - Dispatches to the appropriate `ExecutionEngine`
   - Uses narwhals for validation, native backend for execution
2. Update `write_sharded_spark()` to delegate to `write_sharded_df()` internally
3. Update `write_sharded()` (Python writer) to be the fallback engine

### Phase 3: Polars Engine

**Goal:** Support Polars DataFrames natively without collecting to Python.

1. Implement `PolarsExecutionEngine`:
   - Hash sharding via `map_elements` with Python `xxhash`
   - Range sharding via `cut()` / manual bucketing
   - Partition writes via filter-per-shard (Polars is single-node, fast in-process)
2. Add tests using Polars DataFrames

### Phase 4: Dask Engine

**Goal:** Support Dask DataFrames for distributed writes without Spark.

1. Implement `DaskExecutionEngine`:
   - Hash sharding via `map_partitions` with Python `xxhash`
   - Distributed execution via Dask's `map_partitions` or shuffle + groupby
   - Approximate quantiles via Dask's `.quantile()`
2. Add Dask to optional dependencies
3. Add tests using Dask DataFrames

### Phase 5: pandas / cuDF / PyArrow Engines (via Python fallback)

**Goal:** Any eager backend works via the Python fallback path.

1. The `PythonExecutionEngine` handles any backend narwhals can `.collect()` to rows
2. For cuDF specifically, consider a GPU-aware engine that keeps data on-device
3. Ensure `to_native()` → iterate rows → write shards works for all eager backends

---

## 8. Dependency Changes

```toml
# pyproject.toml changes
[project]
dependencies = [
  "xxhash>=3.4",
  "pydantic>=2.0",
  "narwhals>=1.20",        # NEW: DataFrame abstraction layer
]

[project.optional-dependencies]
read = [
  "slatedb",
  "boto3>=1.28",
]
writer-spark = [             # RENAMED from "writer"
  "pyspark>=3.3",
  "slatedb",
  "boto3>=1.28",
]
writer-polars = [            # NEW
  "polars>=0.20",
  "slatedb",
  "boto3>=1.28",
]
writer-dask = [              # NEW
  "dask[dataframe]>=2024.1",
  "slatedb",
  "boto3>=1.28",
]
writer = [                   # Meta-extra: all writer backends
  "pyspark>=3.3",
  "polars>=0.20",
  "dask[dataframe]>=2024.1",
  "slatedb",
  "boto3>=1.28",
]
```

---

## 9. Public API Surface

### Current API

```python
# Spark writer (requires PySpark)
from slatedb_spark_sharded import write_sharded_spark
result = write_sharded_spark(spark_df, config, key_col="id", value_spec=spec)

# Python writer (no PySpark)
from slatedb_spark_sharded import write_sharded
result = write_sharded(records, config, key_fn=..., value_fn=...)
```

### Proposed API (additive, non-breaking)

```python
# NEW: Backend-agnostic writer (accepts any narwhals-supported DataFrame)
from slatedb_spark_sharded import write_sharded_df
result = write_sharded_df(polars_df, config, key_col="id", value_spec=spec)
result = write_sharded_df(spark_df, config, key_col="id", value_spec=spec)
result = write_sharded_df(pandas_df, config, key_col="id", value_spec=spec)
result = write_sharded_df(dask_df, config, key_col="id", value_spec=spec)

# PRESERVED: Existing APIs continue working unchanged
from slatedb_spark_sharded import write_sharded_spark  # still works
from slatedb_spark_sharded import write_sharded         # still works
```

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Narwhals API subset too limited | Some validations can't be expressed | Use escape hatches; narwhals covers 90% of validation needs |
| Hash sharding inconsistency across backends | Different backends produce different shard assignments for same key | All backends use the same Python `xxhash` routing; Spark uses SQL `xxhash64` verified against Python |
| Performance regression from narwhals wrapper | Extra indirection on hot path | Narwhals overhead is negligible for wrapping; execution uses native backend directly |
| Dask shuffle/repartition complexity | Distributed shard writes require careful partitioning | Dask engine can use `set_index` shuffle or fall back to collect-per-shard |
| Narwhals breaking changes | New versions may alter API | Pin `narwhals>=1.20,<3`; narwhals has strong backwards-compat policy |
| `approxQuantile` not available on all backends | Range sharding with auto-boundaries won't work | Require explicit boundaries for non-Spark backends, or provide backend-specific quantile implementations |

---

## 11. Testing Strategy

### Unit Tests

- **Narwhals frame ops** (`test_frame_ops.py`): Test schema validation, shard ID
  validation, sampling — parameterized across pandas, Polars, and PySpark inputs
- **Engine protocol** (`test_execution.py`): Test `detect_engine()` returns correct
  engine for each backend type
- **Per-engine tests**: Each engine tested with its native DataFrame type

### Integration Tests

- **Cross-backend consistency** (`test_cross_backend.py`): Same dataset, same config
  → same shard assignments across Spark, Polars, pandas engines
- **Routing contract**: Existing `test_routing_contract.py` extended to verify all
  engines produce shard IDs consistent with Python routing

### Compatibility Matrix

| Backend | Hash Sharding | Range (explicit) | Range (auto) | Custom Expr |
|---|---|---|---|---|
| PySpark | Yes | Yes | Yes | Yes |
| Polars | Yes | Yes | No* | No |
| Dask | Yes | Yes | Yes** | No |
| pandas | Yes | Yes | No* | No |

\* Auto-boundaries require a quantile implementation; could add numpy-based fallback.
\** Dask has `.quantile()` built-in.

---

## 12. Decision Points Requiring Input

1. **Package rename?** The package is currently `slatedb_spark_sharded`. If it
   supports Polars/Dask/pandas, should it be renamed to `slatefusion` (matching the
   repo name) or keep the current name?

2. **Narwhals as core vs optional dependency?** Adding narwhals to core `dependencies`
   means all users get it. Alternative: make it optional and only use when available.
   Recommendation: core dependency — it's lightweight (zero transitive deps) and
   enables the unified API.

3. **Custom expression sharding?** Currently `CUSTOM_EXPR` allows raw Spark SQL.
   Should we define a backend-agnostic custom sharding protocol (e.g., a Python
   callable), or keep it Spark-only? Recommendation: add a `custom_callable` option
   that works on any backend, keep `custom_expr` as Spark-only.

4. **Minimum narwhals version?** PySpark lazy support was added in narwhals ~1.20.
   Pin to `>=1.20`.

---

## 13. Summary: What Changes, What Doesn't

### Changes

- New `write_sharded_df()` unified entry point accepting any DataFrame backend
- New `ExecutionEngine` protocol with per-backend implementations
- Narwhals used for schema validation, shard ID validation, and basic DataFrame ops
- New optional dependency groups for Polars, Dask writers
- `narwhals>=1.20` added to core dependencies

### Doesn't Change

- `write_sharded_spark()` API — preserved, delegates internally
- `write_sharded()` (Python iterator API) — preserved as fallback engine
- Reader path — no changes (already pure Python)
- CLI — no changes
- Manifest format — no changes
- Sharding algorithms — identical across all backends (same xxhash64 routing)
- S3 publishing — no changes
