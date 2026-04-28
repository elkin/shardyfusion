# End-to-End Tests

E2E tests validate the complete write → S3 → read pipeline against a real
S3-compatible service ([Garage](https://garagehq.deuxfleurs.fr/)).  They are
the only tests that exercise actual object-store I/O — unit and integration
tests use moto mocks.

## Running

```bash
just d-e2e            # build container, start Garage, run tests
just d-e2e -p 4       # override parallelism (default: 2 tox envs)
```

Requires a container engine (Podman or Docker).  Set `CONTAINER_ENGINE=docker`
if not using Podman.

## Infrastructure

```
┌──────────────────────────────────────────────────┐
│  compose-e2e.yaml                                │
│                                                  │
│  ┌────────────┐       ┌────────────────────────┐ │
│  │   garage    │◄──────│        tests           │ │
│  │  (S3 API)   │ HTTP  │  tox -m e2e -p 2      │ │
│  │  port 3900  │       │                        │ │
│  │  admin 3903 │       │  uv sync + tox         │ │
│  └────────────┘       └────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

**Garage** is a lightweight S3-compatible store.  The entrypoint script
(`docker/garage-entrypoint.sh`) starts the server, assigns a cluster layout,
and creates a readiness marker (`/tmp/garage-ready`).  The test container waits
for Garage's health check before starting.

**Session fixture** (`tests/e2e/conftest.py:garage_s3_service`) creates a
random bucket and API key via the Garage admin API at the start of the session.
All tests in a session share this bucket.  Garage requires **path-style
addressing** (virtual-hosted-style is not supported).

## What the tests validate

### 1. Smoke: sharding correctness (per writer framework)

Each writer framework (Spark, Python, Dask, Ray) runs 7 smoke tests against
a 10-row dataset (`SMOKE_DATA`: keys 0–9, byte values, group column "a"/"b"):

| Test | Sharding | Config | Expected shards |
|------|----------|--------|-----------------|
| `test_smoke_hash` | HASH | `num_dbs=3` | 3 |
| `test_smoke_hash_num_dbs_2` | HASH | `num_dbs=2` | 2 |
| `test_smoke_hash_max_keys_per_shard` | HASH | `max_keys_per_shard=5` | 2 (auto) |
| `test_smoke_cel_key_modulo` | CEL | `key % 3` | 3 |
| `test_smoke_cel_shard_hash` | CEL | `shard_hash(key) % 3u` | 3 |
| `test_smoke_cel_key_identity` | CEL | `uint(key)` | 10 |
| `test_smoke_cel_routing_context` | CEL | `group` column, `routing_values=["a","b"]` | 2 |

**Assertions per smoke test:**

1. **Winner count**: `len(result.winners) == expected_num_shards`
2. **Row conservation**: `sum(row_count) == 10` (no rows lost or duplicated)
3. **Shard placement**: each key exists in exactly the shard predicted by the
   routing function (`xxh3_db_id` for HASH, compiled CEL expression for CEL) —
   verified by opening raw shard readers per shard
4. **Roundtrip read**: `ShardedReader` and `ConcurrentShardedReader` both
   return correct values for all keys via `multi_get`

These tests are the primary defense of the **sharding invariant**: writer and
reader compute identical shard IDs for every key.

### 2. Manifest publishing (per writer framework)

One test per framework writes 24 rows to 4 shards and verifies:

- Manifest YAML is published to S3 at the expected prefix
- `_CURRENT` JSON pointer is published and points to the manifest
- Manifest contains correct `run_id`, `num_dbs`, shard count
- Total `row_count` across shards equals 24

Python writer has two variants: `parallel=False` (sequential) and
`parallel=True` (multiprocessing with spawn).

### 3. Reader refresh (per writer framework)

One test per framework validates the reader's hot-reload path:

1. Writer v1 publishes 32 rows (values `"old-{i}"`)
2. Reader opens and reads — gets old values
3. Writer v2 publishes 32 rows (values `"new-{i}"`) to a new manifest
4. `reader.refresh()` returns `True` — new manifest loaded
5. Reader reads — gets new values
6. `reader.refresh()` returns `False` — no further change

### 4. Reader manifest loading

A standalone reader test (`test_reader_e2e.py`) constructs a manifest and
`_CURRENT` pointer by hand (no writer involved), uploads them to S3, and
verifies that `ShardedReader` can load and serve `get`/`multi_get` requests.

### 5. SQLite range-read VFS (dedicated)

Two dedicated tests in `test_sqlite_range_reader_e2e.py` exercise the APSW
VFS path that serves point lookups via S3 Range requests (the shard DB is
never downloaded in full):

- **Smoke hash roundtrip** (3 shards) — writes via `SqliteFactory`, reads via
  `SqliteRangeReaderFactory`
- **Reader loads manifest** — verifies range-read reader can serve
  `get`/`multi_get` against a hand-built manifest

These are NOT parameterized — they use `SqliteRangeReaderFactory` directly and
require `apsw` (skipped via `pytest.importorskip` when absent).

## Storage backends

All smoke, publish, refresh, and reader-manifest tests are **parameterized
over storage backends** via the `backend` pytest fixture
(`tests/e2e/conftest.py`).  Each test runs twice — once with SlateDB and once
with SQLite — producing test IDs like `test_smoke_hash[slatedb]` /
`test_smoke_hash[sqlite]`.

| Backend | Adapter | Reader | Notes |
|---------|---------|--------|-------|
| SlateDB | `real_file_adapter_factory` | `slatedb.SlateDBReader` (file-backed) | Shard data on local filesystem; only manifests go to Garage |
| SQLite  | `SqliteFactory` | `SqliteReaderFactory` | Full download reader; shard data uploaded/downloaded from Garage |

The `BackendFixture` dataclass bundles `adapter_factory` and `reader_factory`
so scenario helpers are backend-agnostic.  Both backends satisfy the
`DbAdapterFactory` and `ShardReaderFactory` protocols.

## Test matrix

| Framework | Smoke (7×2) | Property (4×2) | Publish (×2) | Refresh (×2) | Total |
|-----------|-------------|----------------|--------------|--------------|-------|
| Spark     | 14 | 8 | 2 | 2 | 26 |
| Python    | 14 | 8 | 4 (seq + parallel) | 2 | 28 |
| Dask      | 14 | 8 | 2 | 2 | 26 |
| Ray       | 14 | 8 | 2 | 2 | 26 |
| Reader    | — | — | — | — | 2 |
| Range-read | 1 | — | — | — | 2 (smoke + manifest) |
| **Total** | 56 | 32 | 10 | 8 | **110** |

CEL smoke tests require `cel-expr-python` (installed via `mod-cel` dependency
group).  On Python 3.14, where `cel-expr-python` has no wheels, these tests
skip via `pytest.importorskip`.  SQLite range-read tests require `apsw`
(installed via `backend-sqlite-range` dependency group).

## Shared helpers

Test logic is factored into two helper modules to avoid duplication across
writer frameworks:

- **`tests/helpers/smoke_scenarios.py`** — `SMOKE_DATA`, `EXPECTED`,
  `run_smoke_write_then_read_scenario`, `run_smoke_cel_scenario`.  These
  functions accept a `write_fn` callback and `adapter_factory`/`reader_factory`
  parameters so each framework provides its own write implementation while
  sharing assertion logic across backends.

- **`tests/helpers/property_scenarios.py`** — generated KV datasets plus
  `run_property_kv_e2e_scenario`.  The property tests share the same assertions
  across writer frameworks and backends for HASH, HASH with
  `max_keys_per_shard`, CEL routing-context, and CEL categorical sharding.

- **`tests/helpers/s3_test_scenarios.py`** — `run_writer_publishes_manifest_scenario`,
  `run_writer_reader_refresh_scenario`, and framework-specific variants.  Accept
  `adapter_factory` and `reader_factory` parameters for backend-agnostic
  execution.  Used by both moto integration tests and Garage e2e tests.

## Tox environments

```
py{311,312,313,314}-read-e2e
py311-sparkwriter-spark35-e2e
py{311,312,313,314}-sparkwriter-spark4-e2e
py{311,312,313,314}-pythonwriter-e2e
py{311,312,313}-daskwriter-e2e          # Dask not on py3.14
py{311,312,313}-raywriter-e2e           # Ray not on py3.14
```

The compose file runs `tox p -m e2e -p 2 -o --parallel-no-spinner` — two
tox environments execute in parallel with combined output.
