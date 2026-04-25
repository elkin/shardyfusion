---
hide:
  - navigation
---

# shardyfusion

A library for **sharded SlateDB/SQLite snapshots** and **sharded vector search** — use either independently or together.

The docs are organized by **what you want to do**. Pick a branch below.

## Use-case map

Click any node to jump to its page. (See the [full index](use-cases/index.md) for a navigational tree.)

<div style="overflow:auto; max-width:100%;">
<div style="min-width:360px;">

```mermaid
%%{init: {'flowchart': {'ranksep': 30, 'nodesep': 4}}}%%
flowchart TD
  classDef root fill:#ff6b6b,stroke:#c92a2a,color:#fff,stroke-width:2px
  classDef category fill:#339af0,stroke:#1864ab,color:#fff,stroke-width:2px
  classDef group fill:#e9ecef,stroke:#495057,color:#212529,stroke-width:1px
  classDef leaf fill:#fff,stroke:#868e96,color:#212529,stroke-width:1px
  classDef shared fill:#845ef7,stroke:#5f3dc4,color:#fff,stroke-width:2px
  classDef op fill:#51cf66,stroke:#2b8a3e,color:#fff,stroke-width:2px

  subgraph kv["KV storage"]
    direction TB
    KV_PY[Python]:::leaf
    KV_SP[Spark]:::leaf
    KV_DK[Dask]:::leaf
    KV_RY[Ray]:::leaf
    KV_BUILD[Build]:::group
    KV_RD_SY[Sync]:::group
    KV_RD_AS[Async]:::group
    KV_RD_SY_SL[SlateDB]:::leaf
    KV_RD_SY_SQ[SQLite]:::leaf
    KV_RD_AS_SL[SlateDB]:::leaf
    KV_RD_AS_SQ[SQLite]:::leaf
    KV_OV[Overview]:::leaf

    KV_PY --> KV_BUILD
    KV_SP --> KV_BUILD
    KV_DK --> KV_BUILD
    KV_RY --> KV_BUILD
    KV_RD_SY_SL --> KV_RD_SY
    KV_RD_SY_SQ --> KV_RD_SY
    KV_RD_AS_SL --> KV_RD_AS
    KV_RD_AS_SQ --> KV_RD_AS
    KV_BUILD --> KV_OV
    KV_RD_SY --> KV_OV
    KV_RD_AS --> KV_OV
  end

  subgraph kvv["KV + Vector"]
    direction TB
    KVV_CO[Composite]:::leaf
    KVV_UN[Unified]:::leaf
    KVV_BUILD[Build]:::group
    KVV_RD_SY[Sync]:::leaf
    KVV_RD_AS[Async]:::leaf
    KVV_OV[Overview]:::leaf

    KVV_CO --> KVV_BUILD
    KVV_UN --> KVV_BUILD
    KVV_BUILD --> KVV_OV
    KVV_RD_SY --> KVV_OV
    KVV_RD_AS --> KVV_OV
  end

  subgraph vec["Vector search"]
    direction TB
    VEC_LN[LanceDB]:::leaf
    VEC_SV[sqlite-vec]:::leaf
    VEC_BUILD[Build]:::group
    VEC_RD_SY[Sync]:::leaf
    VEC_RD_AS[Async]:::leaf
    VEC_OV[Overview]:::leaf

    VEC_LN --> VEC_BUILD
    VEC_SV --> VEC_BUILD
    VEC_BUILD --> VEC_OV
    VEC_RD_SY --> VEC_OV
    VEC_RD_AS --> VEC_OV
  end

  SHARED[Shared snapshot workflow]:::shared --> ROOT
  kv --> ROOT
  kvv --> ROOT
  vec --> ROOT

  ROOT([What do you want to do?]):::root

  ROOT --> OP[Operations]:::op

  subgraph ops[""]
    direction TB
    OP_CLI[CLI]:::leaf
    OP_HIST[History &amp; rollback]:::leaf
    OP_OBS[Observability]:::group
    OP_GUIDES[Guides]:::group
    OP_PROM[Prometheus]:::leaf
    OP_OTEL[OTel]:::leaf
    OP_PROD[Production]:::leaf
    OP_CLOUD[Cloud testing]:::leaf
    OP_TOX[Tox matrix]:::leaf

    OP --> OP_CLI
    OP --> OP_HIST
    OP --> OP_OBS
    OP --> OP_GUIDES
    OP_OBS --> OP_PROM
    OP_OBS --> OP_OTEL
    OP_GUIDES --> OP_PROD
    OP_GUIDES --> OP_CLOUD
    OP_GUIDES --> OP_TOX
  end

  click KV_OV href "use-cases/kv-storage/overview/"
  click SHARED href "use-cases/shared-snapshot-workflow/"
  click KV_PY href "use-cases/kv-storage/build/python/"
  click KV_SP href "use-cases/kv-storage/build/spark/"
  click KV_DK href "use-cases/kv-storage/build/dask/"
  click KV_RY href "use-cases/kv-storage/build/ray/"
  click KV_RD_SY_SL href "use-cases/kv-storage/read/sync/slatedb/"
  click KV_RD_SY_SQ href "use-cases/kv-storage/read/sync/sqlite/"
  click KV_RD_AS_SL href "use-cases/kv-storage/read/async/slatedb/"
  click KV_RD_AS_SQ href "use-cases/kv-storage/read/async/sqlite/"

  click KVV_OV href "use-cases/kv-vector/overview/"
  click KVV_CO href "use-cases/kv-vector/build/composite/"
  click KVV_UN href "use-cases/kv-vector/build/unified/"
  click KVV_RD_SY href "use-cases/kv-vector/read/sync/"
  click KVV_RD_AS href "use-cases/kv-vector/read/async/"

  click VEC_OV href "use-cases/vector/overview/"
  click VEC_LN href "use-cases/vector/build/lancedb/"
  click VEC_SV href "use-cases/vector/build/sqlite-vec/"
  click VEC_RD_SY href "use-cases/vector/read/sync/"
  click VEC_RD_AS href "use-cases/vector/read/async/"

  click OP_CLI href "operate/cli/"
  click OP_HIST href "operate/history-rollback/"
  click OP_PROM href "operate/prometheus-metrics/"
  click OP_OTEL href "operate/otel-metrics/"
  click OP_PROD href "operate/production/"
  click OP_CLOUD href "operate/cloud-testing/"
  click OP_TOX href "operate/tox-matrix/"
```

</div>
</div>

## Sections

- **[Use cases](use-cases/index.md)** — task-oriented guides organized by use-case type (KV, KV+Vector, Vector), plus the shared snapshot workflow behind all of them.
- **[Operations](operate/index.md)** — CLI, history & rollback, metrics, production checks, cloud testing, and tox matrix.
- **[Architecture](architecture/writer-core.md)** — internal design: writer core, sharding, routing, manifest, run registry, adapters, observability, error model.
- **[Reference](reference/api.md)** — public API, configuration objects, CLI, glossary.
- **[Contributing](contributing/index.md)** — local development, testing, adding adapters/writers/use-cases, documentation policy.
- **[History](history/index.md)** — ADRs, open plans, original engineering notes.

## Quick orientation

- Default storage backends: **SlateDB** (KV), **LanceDB** (vector search).
- Snapshot layout: per-shard databases under `s3_prefix/`, plus an immutable manifest under `manifests/<timestamp>_run_id=<run_id>/manifest`, plus a single mutable pointer `_CURRENT`.
- Publish is **two-phase**: write manifest, swap `_CURRENT`. See [ADR-001](history/design-decisions/adr-001-two-phase-publish.md).
- Python `>=3.11,<3.14`.
