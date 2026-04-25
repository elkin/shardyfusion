# Shared Snapshot Workflow

Every shardyfusion use case follows the same high-level data workflow:

1. a writer receives records, vectors, or records with vectors
2. sharding logic assigns each item to one shard
3. the writer uploads immutable shard objects to an S3 bucket/prefix
4. the writer publishes a manifest, then updates `_CURRENT`
5. readers load the manifest and fetch only the shard data needed for each request

Use this page when you want the project-wide model. If you already know whether you need KV, vector-only, or KV+vector, you can jump straight to that use-case overview; each one still stands on its own.

---

## End-to-end flow

```mermaid
flowchart LR
    subgraph Caller
        IN[Input data<br/>records / vectors / both]
        Q[Read request<br/>keys / query vector / routing context]
    end

    subgraph Writer
        W[Writer]
        R{Sharding logic}
        BM[Build manifest]
        W --> R
    end

    IN --> W

    subgraph S3["S3 bucket / prefix"]
        S0[Shard 0 object]
        S1[Shard 1 object]
        S2[Shard 2 object]
        M[Immutable manifest]
        C[_CURRENT pointer]
    end

    subgraph Reader
        LR[Load _CURRENT]
        LM[Load manifest]
        RR{Route request}
        OUT[Return results]
    end

    R --> S0
    R --> S1
    R --> S2
    S0 --> BM
    S1 --> BM
    S2 --> BM
    BM --> M
    M --> C

    Q --> LR
    LR --> C
    LR --> LM
    LM --> M
    LM --> RR
    RR -->|needed shard only| S0
    RR -->|needed shard only| S1
    RR -->|needed shard only| S2
    RR --> OUT
```

The concrete routing decision differs by use case:

| Use case | What gets routed | Reader behavior |
|---|---|---|
| KV | lookup keys | route each key to exactly one shard |
| Vector-only | query vector and optional routing context | search all shards or a routed subset, then merge top-k results |
| KV+vector | lookup keys and vectors | dispatch point lookups to KV data and searches to vector data from the same snapshot |

The shared contract is the manifest. It records shard locations, routing metadata, build metadata, and backend-specific custom fields such as vector index configuration.

---

## Two-phase publish

Publishing has two visible phases:

1. **Write manifest object** — immutable and timestamped under `manifests/`.
2. **Update `_CURRENT` pointer** — small mutable object pointing at the manifest.

```mermaid
sequenceDiagram
    participant W as Writer
    participant S3 as S3 / Object Store
    participant R as Reader

    W->>S3: PUT shard objects
    W->>S3: PUT manifests/<timestamp>_run_id=<run_id>/manifest
    Note over S3: Manifest is durable but not current yet
    W->>S3: PUT _CURRENT -> manifest ref
    Note over S3: New snapshot is now current

    R->>S3: GET _CURRENT
    S3-->>R: ManifestRef
    R->>S3: GET manifest by ref
    S3-->>R: shard URLs + routing metadata
    R->>S3: GET only required shard data
```

Readers observe one of three states:

| State | Manifest | `_CURRENT` | What readers see |
|---|---|---|---|
| Before publish | old | old | Old snapshot |
| Mid-publish | new written | old | Old snapshot; new manifest is invisible |
| After publish | new | new | New snapshot atomically |

There is no mixed snapshot where some shards come from the old manifest and some from the new one. Readers pin their state to one manifest at a time.

---

## Failure tolerance

| Failure point | Result | Recovery |
|---|---|---|
| Shard write fails | No manifest is published for that attempt. Readers keep using the old snapshot. | Rerun the writer. Stale attempts can be cleaned up later. |
| Manifest write fails | `_CURRENT` is unchanged. Readers keep using the old snapshot. | Rerun the writer. |
| `_CURRENT` update fails after manifest write | The manifest exists but is invisible to normal readers. | Rerun the writer or later clean up the orphaned manifest. |
| Current manifest is malformed at reader startup | Reader can try previous manifests, up to its fallback limit. | Fix or roll back `_CURRENT`; see [History & rollback](../operate/history-rollback.md). |

The important boundary is that `_CURRENT` is updated only after the manifest object exists. Readers never get a pointer to a manifest that was only half-written.

---

## Snapshot history and reader migration

Each publish creates a new immutable snapshot. Old snapshots stay in the bucket until cleanup removes them, which gives you rollback history and lets existing readers migrate on their own schedule.

```mermaid
flowchart LR
    subgraph S3["S3 / Object store"]
        direction TB

        subgraph snapN1_new["Snapshot N+1 (run_id=ghi) — current"]
            MAN3["manifest"]
            SH3_0["shard 0 object"]
            SH3_1["shard 1 object"]
            MAN3 --> SH3_0
            MAN3 --> SH3_1
        end

        subgraph snapN["Snapshot N (run_id=def)"]
            MAN2["manifest"]
            SH2_0["shard 0 object"]
            SH2_1["shard 1 object"]
            MAN2 --> SH2_0
            MAN2 --> SH2_1
        end

        subgraph snapN1["Snapshot N-1 (run_id=abc)"]
            MAN1["manifest"]
            SH1_0["shard 0 object"]
            SH1_1["shard 1 object"]
            MAN1 --> SH1_0
            MAN1 --> SH1_1
        end

        CUR["_CURRENT pointer"] -->|points to| MAN3
    end

    subgraph "Reader A (new)"
        RA["open()"] -->|reads _CURRENT| MAN3
        RA --> SH3_0
        RA --> SH3_1
    end

    subgraph "Reader B (already open)"
        RB["open() earlier"] -->|pinned to| MAN2
        RB --> SH2_0
        RB --> SH2_1
        RB -.->|"refresh() when ready"| MAN3
    end
```

- New readers load `_CURRENT` and use the newest manifest.
- Already-open readers stay pinned to the manifest they loaded.
- `refresh()` moves a reader to the current manifest when the application is ready.
- Rollback is the same mechanism in reverse: point `_CURRENT` at an older manifest.

---

## Concrete use cases

- [Sharded KV storage](kv-storage/overview.md) adds key encoding, HASH/CEL routing, and KV reader choices.
- [Sharded KV storage with vector search](kv-vector/overview.md) adds vector metadata and a unified reader surface for point lookup plus ANN search.
- [Sharded vector search](vector/overview.md) adds vector sharding strategies and scatter-gather result merging.

For implementation details, see [Manifest & `_CURRENT`](../architecture/manifest-and-current.md), [Manifest stores](../architecture/manifest-stores.md), and [History & rollback](../operate/history-rollback.md).
