# Testing

The test suite is layered into **unit**, **integration**, and **e2e**, mirrored across many capability/backend combinations through tox.

## Layout

```
tests/
├── unit/
│   ├── shared/              # cross-cutting: config, manifest, ordering, types
│   ├── read/                # sync reader
│   ├── read_async/          # async reader
│   ├── writer/              # writer-core, sharding, retry, parallel
│   ├── backend/{slatedb,sqlite}/
│   ├── cel/                 # CEL routing
│   ├── metrics/             # Prometheus + OTel collectors
│   ├── cli/                 # Click CLI
│   └── vector/              # vector adapters and search
├── integration/
│   ├── read/                # moto S3 + real readers
│   ├── writer/              # moto S3 + real writers (incl. Spark)
│   ├── backend/sqlite/
│   └── vector/
├── e2e/                     # Garage S3 via docker/compose-e2e.yaml
│   ├── read/
│   └── writer/
└── helpers/                 # shared fixtures and scenarios
    ├── s3_test_scenarios.py
    ├── smoke_scenarios.py
    └── run_record_assertions.py
```

## Tox labels

| Label | Runs |
|---|---|
| `quality` | lint, format, type checks, package build, docs check |
| `unit` | All `py{311,312,313}-*-unit` envs |
| `integration` | All `py{311,312,313}-*-integration` envs |
| `e2e` | All `py{311,312,313}-*-e2e` envs (requires container engine) |
| `smoke` | Cross-capability smoke envs (`all-spark35-unit`, `all-spark35-integration`) |

```bash
uv run tox -m quality
uv run tox -m unit
uv run tox -m integration
uv run tox -m e2e
```

Or via `just`: `just quality`, `just unit`, `just integration`, `just d-e2e`.

## Tox env naming

Envs are composed from three axes:

```
py<version>-<capability>-<backend>-<level>
```

Examples:

- `py312-pythonwriter-slatedb-unit` — Python writer, SlateDB backend, unit tests, Python 3.12.
- `py311-sparkwriter-spark35-slatedb-integration` — Spark 3.5 writer with SlateDB, integration tests.
- `py313-vector-lancedb-unit` — LanceDB vector adapter, unit tests, Python 3.13.

Capability/backend factors map to dependency groups in `pyproject.toml` via `[testenv].dependency_groups` in `tox.ini`. The full list is in [`operate/tox-matrix.md`](../operate/tox-matrix.md).

## Targeted runs while developing

Tox is comprehensive but slow. For tight loops:

```bash
uv run pytest -q tests/unit/writer
uv run pytest -q tests/unit/cli
uv run pytest -q tests/integration/read
uv run pytest -q -k "test_routing_contract"     # property test
```

Unit tests use `pytest-xdist` for parallelism: `uv run pytest -n 4 -q tests/unit`.

## Test adapters

`shardyfusion/testing.py` provides three SlateDB-shaped fakes for use in tests:

| Class | Purpose |
|---|---|
| `FakeSlateDbAdapter` | In-memory; fast for unit tests of `_writer_core`. |
| `FileBackedSlateDbAdapter` | On-disk; verifies serialization without network. |
| `RealSlateDbFileAdapter` | Real SlateDB pointed at local disk. |

These are the only test doubles in shipped code; anything else lives under `tests/`.

## Shared scenarios

`tests/helpers/s3_test_scenarios.py` defines parameterized scenarios reused across moto (integration) and Garage (e2e). When you add behavior that touches S3 publishing, add the scenario there once and both backends pick it up.

`tests/helpers/run_record_assertions.py` centralizes run-record YAML assertions so the format stays consistent.

## Property tests

`tests/unit/writer/core/test_routing_contract.py` uses [`hypothesis`](https://hypothesis.readthedocs.io) to verify the writer-reader sharding invariant: a key the writer assigned to shard *N* must be routed to shard *N* by `SnapshotRouter`. Any change to `_writer_core` routing logic, `routing.py`, or `sharding_types.py` must keep this passing.

## Adding tests

| Change | Required tests |
|---|---|
| New behavior in `_writer_core.py` / `routing.py` | Unit + the routing-contract property test must still pass. |
| New adapter | Unit (with `FakeSlateDbAdapter`-style double) + integration (moto S3). |
| New writer flavor | Unit + integration; e2e if the writer publishes manifests. |
| New CLI command | Unit (Click runner) — see `tests/unit/cli`. |
| New metric event | Unit in `tests/unit/metrics`. |
| New use-case page | Validate-docs check; an integration test exercising the documented snippet is encouraged. |

## Before requesting review

```bash
just ci d-e2e
```

Quality + unit + integration on the host, then end-to-end inside containers against Garage. PRs must not turn this red.

## See also

- [`operate/tox-matrix.md`](../operate/tox-matrix.md) — full env list.
- [`operate/cloud-testing.md`](../operate/cloud-testing.md) — running e2e against real S3.
- [`extras-and-dependencies.md`](extras-and-dependencies.md) — keeping `pyproject.toml` and `tox.ini` in sync.
