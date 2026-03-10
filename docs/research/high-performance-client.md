# High-Performance Shardyfusion Client — Research & Recommendations

## Context

Shardyfusion writes sharded SlateDB snapshots to S3. The current Python reader (`ConcurrentShardedReader`) works but is constrained by Python's GIL and per-call overhead. For online serving (low-latency point lookups), a Rust-core client with Python bindings would eliminate these bottlenecks while keeping the Python ecosystem accessible.

The goal is a **separate repository** containing a Rust crate + Python package (via PyO3/maturin) optimized for low-latency `get`/`multi_get` lookups against shardyfusion snapshots.

---

## Recommended Architecture: Rust Core + PyO3 Python Bindings

### Why This Approach

1. **SlateDB is already Rust** — the `slatedb` crate provides the native reader API. Using it directly from Rust eliminates the Python→PyO3→Rust roundtrip that the current `slatedb` PyPI package incurs per call.
2. **Routing is CPU-bound** — xxhash64 computation, key encoding, and shard selection are pure computation. Rust eliminates Python interpreter overhead entirely.
3. **`multi_get` parallelism without GIL** — Rust can fan out across shards using `tokio` tasks, completely bypassing Python's GIL. The current Python reader uses `ThreadPoolExecutor` but still contends on the GIL for routing/result assembly.
4. **`object_store` crate** — SlateDB already uses this for S3 access. Reusing it avoids boto3 overhead and gives async I/O natively.
5. **Single process** — no network hop (vs. a gRPC sidecar), no serialization overhead.

### Architecture Diagram

```
Python Service
  │
  ▼
┌─────────────────────────────────┐
│  shardyfusion-client (Python)   │  ← PyO3 bindings (maturin)
│  ShardedClient class            │
├─────────────────────────────────┤
│  shardyfusion-client-core (Rust)│  ← Pure Rust crate
│  ┌───────────┐ ┌──────────────┐ │
│  │ Router    │ │ ShardManager │ │
│  │ (xxhash64)│ │ (SlateDB     │ │
│  │           │ │  readers)    │ │
│  └───────────┘ └──────────────┘ │
│  ┌───────────┐ ┌──────────────┐ │
│  │ Manifest  │ │ object_store │ │
│  │ Loader    │ │ (S3 async)   │ │
│  └───────────┘ └──────────────┘ │
└─────────────────────────────────┘
```

### Alternatives Considered

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **Pure Rust + PyO3** (recommended) | Max performance, true parallelism, zero-copy | Rust expertise needed, complex build | Best for online serving |
| **Python + existing slatedb PyPI** | Simple, fast to build | GIL limits concurrency, Python overhead | Current reader already does this |
| **Rust gRPC service** | Language-agnostic, independent scaling | Network hop latency, deployment complexity | Overkill for single-process serving |
| **C++ with pybind11** | Good performance | No SlateDB C++ bindings exist, more work | Not practical |

---

## Repo Structure (new repo: `shardyfusion-client`)

```
shardyfusion-client/
├── Cargo.toml                    # Workspace root
├── pyproject.toml                # Python package (maturin build)
├── crates/
│   ├── core/                     # Pure Rust library crate
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── router.rs         # xxhash64 routing (port from shardyfusion/routing.py)
│   │       ├── manifest.rs       # Manifest + CURRENT pointer parsing (serde_json)
│   │       ├── shard_manager.rs  # Opens/caches SlateDB readers per shard
│   │       ├── client.rs         # ShardedClient: get/multi_get/refresh
│   │       └── error.rs
│   └── python/                   # PyO3 bindings crate
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs            # #[pymodule] exposing ShardedClient to Python
├── tests/
│   ├── rust/                     # Rust integration tests
│   └── python/                   # Python tests (pytest)
├── benches/                      # Criterion benchmarks
└── examples/
    └── lookup.py                 # Python usage example
```

---

## Key Components

### 1. Router (`crates/core/src/router.rs`)

Port of `shardyfusion/routing.py`. Must replicate the **sharding invariant exactly**:

```rust
use xxhash_rust::xxh64::xxh64;

const XXHASH64_SEED: u64 = 42;

fn xxhash64_db_id(key: u64, num_dbs: u32, key_encoding: KeyEncoding) -> u32 {
    let payload = key.to_le_bytes(); // 8-byte little-endian
    let digest = xxh64(&payload, XXHASH64_SEED);
    let signed = digest as i64; // reinterpret as signed (matches JVM)
    (signed.rem_euclid(num_dbs as i64)) as u32
}
```

- Use `xxhash-rust` crate (pure Rust, no C dependency)
- Support HASH, RANGE (binary search on boundaries), and CUSTOM_EXPR (fallback to range)
- Must pass cross-validation against Python `routing.py` with the same test vectors from `test_routing_contract.py`

### 2. Manifest Loader (`crates/core/src/manifest.rs`)

- Parse `_CURRENT` pointer JSON → extract `manifest_ref`
- Fetch and parse manifest JSON → `RequiredBuildMeta` + `Vec<RequiredShardMeta>`
- Use `serde` + `serde_json` for deserialization
- Use `object_store` crate for S3 fetches (async, with retry)
- Match the JSON schema from `shardyfusion/schemas/manifest.schema.json`

### 3. Shard Manager (`crates/core/src/shard_manager.rs`)

- Open one `slatedb::db::Db` (or read-only reader) per shard at init
- Cache open readers — no per-request open/close
- Support `refresh()` to atomically swap to new manifest/readers
- Use `Arc<RwLock<ShardState>>` for concurrent access
- Support checkpoint-based reads (`checkpoint_id` from manifest)

### 4. Client (`crates/core/src/client.rs`)

```rust
pub struct ShardedClient {
    router: Router,
    shards: Arc<RwLock<ShardState>>,
    // ...
}

impl ShardedClient {
    pub async fn new(s3_prefix: &str, local_root: &Path, config: ClientConfig) -> Result<Self>;
    pub async fn get(&self, key: u64) -> Result<Option<Vec<u8>>>;
    pub async fn multi_get(&self, keys: &[u64]) -> Result<HashMap<u64, Option<Vec<u8>>>>;
    pub async fn refresh(&self) -> Result<bool>;
    pub fn route_key(&self, key: u64) -> u32;
}
```

- `multi_get`: group keys by shard, fan out reads via `tokio::spawn` (true parallelism, no GIL)
- Connection pool per shard for concurrent readers

### 5. Python Bindings (`crates/python/src/lib.rs`)

```python
# Python API (exposed via PyO3)
from shardyfusion_client import ShardedClient

client = ShardedClient(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/cache",
    max_workers=4,
)

# Single lookup
value: bytes | None = client.get(42)

# Batch lookup
results: dict[int, bytes | None] = client.multi_get([1, 2, 3, 42])

# Routing only (no DB access)
shard_id: int = client.route_key(42)

# Reload manifest
changed: bool = client.refresh()

# Cleanup
client.close()
```

Key PyO3 patterns:
- Release the GIL during Rust I/O (`py.allow_threads(|| ...)`)
- Return `PyBytes` for zero-copy where possible
- Use `pyo3-asyncio` if the Python service uses asyncio (optional)
- Accept `int` keys directly (PyO3 handles Python int → Rust u64)

---

## Performance Considerations

| Aspect | Current Python Reader | Proposed Rust Client |
|--------|----------------------|---------------------|
| Routing overhead | Python interpreter per key | Native xxhash64, ~10ns/key |
| GIL contention | ThreadPoolExecutor still needs GIL for routing | GIL released during all Rust work |
| S3 I/O | boto3 (synchronous) | object_store (async, tokio) |
| SlateDB access | PyO3 round-trip per get() | Direct Rust API call |
| multi_get parallelism | Thread pool (GIL-limited) | tokio tasks (true parallelism) |
| Memory | Python objects per key/value | Zero-copy bytes where possible |

Expected improvements:
- **p50 latency**: 2-5x improvement (eliminates Python overhead)
- **p99 latency**: 5-10x improvement (eliminates GIL contention spikes)
- **multi_get throughput**: 3-10x improvement (true parallelism across shards)

---

## Critical Correctness Requirements

1. **Sharding invariant**: The Rust router MUST produce identical shard IDs as `shardyfusion/routing.py` for every key. Test with:
   - Same edge-case keys from `tests/unit/writer/test_routing_contract.py` (~200 keys)
   - Property-based tests (proptest crate) mirroring hypothesis tests
   - Cross-validation script that runs both Python and Rust routers on random keys

2. **Manifest compatibility**: Must parse manifests produced by all shardyfusion writer paths (Spark, Dask, Ray, Python). Validate against `shardyfusion/schemas/manifest.schema.json`.

3. **Key encoding**: Must support both `u64be` (8-byte) and `u32be` (4-byte) encodings, matching the encode/decode logic in `shardyfusion/serde.py` and `shardyfusion/routing.py`.

---

## Rust Dependencies

```toml
[dependencies]
slatedb = "0.10"           # SlateDB Rust crate (same version as PyPI)
object-store = "0.11"      # S3/GCS/ABS access (used by SlateDB)
xxhash-rust = { version = "0.8", features = ["xxh64"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["full"] }
pyo3 = { version = "0.23", features = ["extension-module"] }
pyo3-asyncio = { version = "0.23", features = ["tokio-runtime"] }  # optional
thiserror = "2"
tracing = "0.1"
```

---

## Build & Packaging

- **Build tool**: [maturin](https://github.com/PyO3/maturin) (standard for PyO3 projects)
- **Python package name**: `shardyfusion-client`
- **Wheel distribution**: maturin builds platform-specific wheels (manylinux, macOS, Windows)
- **CI**: GitHub Actions with `maturin-action` for cross-platform wheel builds
- **Rust edition**: 2021
- **MSRV**: Match SlateDB's MSRV

---

## Testing Strategy

1. **Routing contract tests** (Rust unit tests):
   - Port all edge-case keys from `test_routing_contract.py`
   - Property tests via `proptest` crate
   - Cross-validate against Python router via a test script that compares outputs

2. **Manifest parsing tests** (Rust unit tests):
   - Test with sample manifests from different writer paths
   - Validate against JSON schema

3. **Integration tests** (Rust):
   - Write data with shardyfusion Python writer → read with Rust client
   - Use moto or localstack for S3

4. **Python binding tests** (pytest):
   - Test `ShardedClient` Python API end-to-end
   - Compare results with existing `ConcurrentShardedReader`

5. **Benchmarks** (Criterion):
   - `get` latency (p50, p99)
   - `multi_get` throughput (keys/sec)
   - Routing throughput (keys/sec)

---

## Implementation Phases

### Phase 1: Rust Core (routing + manifest)
- Implement `Router` with xxhash64 hash and range strategies
- Implement manifest/CURRENT JSON parsing
- Port and pass all routing contract tests

### Phase 2: SlateDB Integration
- Integrate `slatedb` crate for shard reads
- Implement `ShardManager` with reader caching
- Implement `ShardedClient` with `get`/`multi_get`

### Phase 3: PyO3 Bindings
- Expose `ShardedClient` to Python via PyO3
- GIL release during I/O
- maturin packaging + wheel builds

### Phase 4: Testing & Validation
- Cross-validation with Python reader on real snapshots
- Performance benchmarks
- CI pipeline

---

## References

- [SlateDB](https://github.com/slatedb/slatedb) — Rust LSM-tree engine on object storage
- [PyO3](https://github.com/PyO3/pyo3) — Rust bindings for Python
- [maturin](https://github.com/PyO3/maturin) — Build and publish PyO3 crates as Python packages
- [Nine Rules for Writing Python Extensions in Rust](https://towardsdatascience.com/nine-rules-for-writing-python-extensions-in-rust-d35ea3a4ec29/) — Best practices
- [xxhash-rust](https://crates.io/crates/xxhash-rust) — Pure Rust xxhash implementation
- [slatedb PyPI](https://pypi.org/project/slatedb/) — Current Python bindings (v0.10.0)
