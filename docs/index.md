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
  classDef default font-size:18px

  ROOT([What do you want to do?])
  style ROOT font-size:24px,font-weight:bold

  KV_HDR[KV storage]
  style KV_HDR font-size:20px,font-weight:bold
  KVV_HDR[KV + Vector]
  style KVV_HDR font-size:20px,font-weight:bold
  VEC_HDR[Vector search]
  style VEC_HDR font-size:20px,font-weight:bold
  OP_HDR[Operations]
  style OP_HDR font-size:20px,font-weight:bold

  ROOT --> KV_HDR
  ROOT --> KVV_HDR
  ROOT --> VEC_HDR
  ROOT --> OP_HDR

  KV_HDR --> kv
  KVV_HDR --> kvv
  VEC_HDR --> vec
  OP_HDR --> op

  subgraph kv[" "]
    direction LR
    KV_OV[Overview]
    KV_BUILD[Build]
    KV_RD_SY[Sync]
    KV_RD_AS[Async]
    KV_PY[Python]
    KV_SP[Spark]
    KV_DK[Dask]
    KV_RY[Ray]
    KV_RD_SY_SL[SlateDB]
    KV_RD_SY_SQ[SQLite]
    KV_RD_AS_SL[SlateDB]
    KV_RD_AS_SQ[SQLite]

    KV_OV --> KV_BUILD
    KV_OV --> KV_RD_SY
    KV_OV --> KV_RD_AS
    KV_BUILD --> KV_PY
    KV_BUILD --> KV_SP
    KV_BUILD --> KV_DK
    KV_BUILD --> KV_RY
    KV_RD_SY --> KV_RD_SY_SL
    KV_RD_SY --> KV_RD_SY_SQ
    KV_RD_AS --> KV_RD_AS_SL
    KV_RD_AS --> KV_RD_AS_SQ
  end

  subgraph kvv[" "]
    direction LR
    KVV_OV[Overview]
    KVV_BUILD[Build]
    KVV_RD_SY[Sync]
    KVV_RD_AS[Async]
    KVV_CO[Composite]
    KVV_UN[Unified]

    KVV_OV --> KVV_BUILD
    KVV_OV --> KVV_RD_SY
    KVV_OV --> KVV_RD_AS
    KVV_BUILD --> KVV_CO
    KVV_BUILD --> KVV_UN
  end

  subgraph vec[" "]
    direction LR
    VEC_OV[Overview]
    VEC_BUILD[Build]
    VEC_RD_SY[Sync]
    VEC_RD_AS[Async]
    VEC_LN[LanceDB]
    VEC_SV[sqlite-vec]

    VEC_OV --> VEC_BUILD
    VEC_OV --> VEC_RD_SY
    VEC_OV --> VEC_RD_AS
    VEC_BUILD --> VEC_LN
    VEC_BUILD --> VEC_SV
  end

  subgraph op[" "]
    direction LR
    OP_OV[Overview]
    OP_CLI[CLI]
    OP_HIST[History &amp; rollback]
    OP_OBS[Observability]
    OP_GUIDES[Guides]
    OP_PROM[Prometheus]
    OP_OTEL[OTel]
    OP_PROD[Production]
    OP_CLOUD[Cloud testing]
    OP_TOX[Tox matrix]

    OP_OV --> OP_CLI
    OP_OV --> OP_HIST
    OP_OV --> OP_OBS
    OP_OV --> OP_GUIDES
    OP_OBS --> OP_PROM
    OP_OBS --> OP_OTEL
    OP_GUIDES --> OP_PROD
    OP_GUIDES --> OP_CLOUD
    OP_GUIDES --> OP_TOX
  end

  click KV_OV href "use-cases/kv-storage/overview/"
  click KV_BUILD href "use-cases/kv-storage/build/"
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

  click OP_OV href "operate/"
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
