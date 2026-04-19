# 2026-04-19 LanceDB Vector Migration and Testing Expansion

- Status: `implemented`
- Date: `2026-04-19`

## Summary

This engineering note documents the migration of our sharded approximate nearest neighbor (ANN) vector search backend from `usearch` to `LanceDB`, alongside the maturation of our unified KV+vector path using `sqlite-vec`. It covers the core design changes for the index engine switch, the resolution of data invariant bugs (specifically ID type coercion), and a significant expansion of integration and end-to-end (E2E) testing against real and mocked S3 environments.

## 1. What problem is being solved or functionality being added by the changes?

The initial vector search implementation proved difficult to scale and test using the `usearch` engine and mock test adapters. We needed an engine that offered better performance, native on-disk/S3 integration, and robust Python tooling. We also needed to address data integrity bugs and testing blind spots:

1. **Engine Limitations**: `usearch` was not meeting our needs for mature indexing and query tooling.
2. **String ID Coercion Bug**: When using `LanceDB`, string IDs containing only digits (e.g., `"123"`) were blindly coerced into integers by PyArrow during the `search()` operation, causing lookups to fail or return mismatched types.
3. **Storage Configuration**: LanceDB requires explicit `storage_options` to connect to local mock S3 instances (Moto/Garage) used in our CI environments, which was hindering robust testing.
4. **Table Creation Lifecycle**: LanceDB's `create_table` method threw errors when called repeatedly on existing tables during batch insertions.
5. **Testing Fidelity**: Integration and E2E tests relied on mock adapters (`FakeVectorWriter`, `FakeUnifiedAdapter`), masking real-world bugs in S3 connectivity and engine behavior.

## 2. What design decisions were considered with their pros and cons and trade offs?

### Decision 1: Choosing the Primary Vector Index Engine

#### Option A: Stick with `usearch` or pure Python
Pros:
- No migration cost.
Cons:
- Lack of mature features (quantization, built-in disk persistence formats) compared to other engines.

#### Option B: FAISS
Pros:
- Industry standard, highly optimized.
Cons:
- Complex API, heavy dependencies, no built-in payload/metadata storage.

#### Option C: LanceDB (Lance columnar format)
Pros:
- Mature, fast HNSW implementation.
- Deep PyArrow integration, supporting multiple distance metrics and quantization (fp16, i8).
- Pluggable adapter protocol allowed a clean swap.
Cons:
- Adds an optional dependency.
- Requires sidecar for arbitrary payloads (which we solve with a SQLite payload DB alongside the LanceDB index).

**Chosen approach**: Option C. We migrated to `LanceDB` as the default dedicated vector search engine, while maintaining `sqlite-vec` for the unified KV+vector path.

### Decision 2: Connecting LanceDB to local mock S3 (Moto/Garage) for testing

#### Option A: Hardcode storage options in test fixtures
Pros:
- Simple.
Cons:
- Leaks test environment details into the adapter.

#### Option B: Dynamic extraction from Boto3 client
Pros:
- Transparent to the application; works seamlessly across real S3 and test S3.
Cons:
- Requires custom logic to parse endpoint URLs.

**Chosen approach**: Option B. We added `_extract_storage_options(s3_client)` in the LanceDB adapter to automatically infer `aws_endpoint`, `aws_region`, and `aws_allow_http` from the boto3 client's configuration.

### Decision 3: PyArrow ID Type Coercion

**Chosen approach**: Rather than force all IDs to be strings at the application boundary (which would break backwards compatibility), we explicitly defined PyArrow schema fields (`pa.string()`, `pa.int64()`) when adding data and querying. This fixed the `LanceDbShardReader.search()` bug where numeric string IDs were natively preserved.

### Decision 4: Table Creation Flow in LanceDB

**Chosen approach**: We introduced a state check in the writer (`if self._table is None: create_table(...) else: self._table.add(...)`). This avoids catching and ignoring exceptions (which could hide actual storage errors) while supporting multi-batch operations cleanly.

## 3. What implementation was chosen and why?

### Adapter Implementation (`vector/adapters/lancedb_adapter.py`)
- Implemented `LanceDBWriter` and `LanceDBShardReader` conforming to the `VectorIndexWriter` and `VectorShardReader` protocols.
- Fixed PyArrow schema definitions to preserve string vs. integer IDs.
- Implemented dynamic `storage_options` extraction for seamless S3/Moto compatibility.
- Implemented robust batch table addition logic.

### Test Coverage Expansion
- **Unit Tests**: Added `tests/unit/vector/test_lancedb_reader.py` covering reader instantiation, search functionality, and type preservation. Utilized `pytest.importorskip("lancedb")` to ensure test collection doesn't fail in tox environments where the optional `lancedb` dependency isn't installed.
- **Integration Tests**: Replaced mock adapters with real `LanceDbWriterFactory`, `LanceDbReaderFactory`, `SqliteVecFactory`, and `SqliteVecReaderFactory` in `test_vector_writer_reader_local_s3.py` and `test_unified_write_read_local_s3.py` against Moto S3. Updated `tox.ini` to explicitly include the `backend-vector-lancedb` dependency group for the `vector-integration` environment so the real adapters can be tested.
- **E2E Tests**: Augmented Spark E2E tests (`test_spark_vector_e2e.py`). Added reader-side verifications to `sqlite-vec` paths. Added `test_spark_vector_lancedb_write_and_read` to verify the end-to-end Spark distributed writer-to-LanceDB reader pipeline against Garage S3.

## Known Limitations

- **Payload Storage**: The LanceDB adapter stores arbitrary payloads in a separate `payloads.db` SQLite sidecar. This requires fetching two files from S3 per shard.
- **Storage Options Coverage**: The dynamic extraction handles standard AWS, Moto, and Garage S3 configurations. Non-standard S3-compatible endpoints might require manual overrides in the future.