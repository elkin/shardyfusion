# Adding a use-case page

Use-case pages follow the branching structure shown in the [home page use-case map](../index.md#use-case-map):

1. **Conceptual overview** (e.g. `kv-storage/overview.md`) — explains shared concepts: sharding, manifests, two-phase publish, safety properties. These are the primary entry points for users.
2. **Leaf pages** (e.g. `kv-storage/build/python.md`, `kv-storage/read/sync/slatedb.md`) — backend-specific or writer-specific details. These can be thinner because shared concepts live in the parent overview.

## When to add a page

Add a page when you add:

- A new writer flavor (add under `kv-storage/build/` or `kv-vector/build/` with writer tabs).
- A new read-side combination (add under the appropriate `read/` directory).
- A new operational scenario (add under `operate/`).
- A new top-level use-case type (e.g. a fourth use-case family alongside KV, KV+Vector, Vector).

If the new feature is a *variation* of an existing leaf (different config knob), extend the existing leaf page; don't add a new one.

## Page types

### Overview pages

Overview pages explain **shared concepts** for a use-case family. They should:

- Be self-contained for users (deep-link to `architecture/` for implementation details).
- Include visual diagrams (mermaid) to explain data flow, state machines, or comparisons.
- Link to all child leaf pages at the bottom.

Required sections (flexible order):

```markdown
# <Use-case family name>

<One-sentence summary.>

## <Concept 1>

## <Concept 2>

## Child pages

- [Build → ...](build/...)
- [Read → ...](read/...)
```

### Leaf pages

Leaf pages cover **one specific backend or writer**. They can drop sections that are fully covered by the parent overview, but must **link to the overview** for those details.

Required sections (exact order):

```markdown
# <Verb + backend / reader type>

<One-sentence framing.>

## When to use

## When NOT to use

## Install

## Minimal example

## Configuration

## Functional properties   # only backend/writer-specific

## Non-functional properties  # only backend/writer-specific

## Guarantees              # only if different from overview

## Weaknesses

## Failure modes & recovery

## See also
```

Leaf pages **must** include a link to the parent overview in their "See also" section. Example:

```markdown
## See also

- [KV Storage Overview](../overview.md) — sharding, manifests, two-phase publish, safety
```

### Writer tabs

When a build page supports multiple writer flavors (e.g. KV+Vector composite), use **content tabs** to show Python / Spark / Dask / Ray examples:

```markdown
=== "Python"

    ```python
    ...
    ```

=== "Spark"

    ```python
    ...
    ```
```

This requires the `pymdownx.tabbed` extension (already enabled in `mkdocs.yml`).

## Style rules

- **No marketing.** Describe what the code does, including its limits.
- **No comparisons that don't exist in code.** "Faster than X" needs a benchmark in `tests/` or it doesn't go in.
- **Use real symbol names with file:line refs**: `WriteConfig.batch_size` (`shardyfusion/config.py:201`).
- **Prefer absolute correctness to brevity.** If the failure-mode table has 10 rows, list 10 rows.
- **Cross-link generously.** Every "Configuration" field that has its own section in `architecture/` should link there. Every "When NOT to use" alternative should link.
- **Be visual.** Include mermaid diagrams for data flow, state machines, and comparisons. Diagrams significantly improve comprehension.

## What goes in `Configuration`

Only the **fields a user touches for this use case**. The full surface lives in `reference/config.md`. Each row:

```markdown
| Field | Default | Purpose |
|---|---|---|
| `num_dbs` | `None` | Number of shards. Required (>0) for HASH sharding without `max_keys_per_shard`. |
```

Include defaults verbatim from source. If the default is computed (e.g. `field(default_factory=ShardingSpec)`), say so.

## What goes in `Failure modes & recovery`

One row per *user-observable* failure. Surface column = the exception class or signal. Recovery = what the user does (not what the system does internally).

Don't list S3 transient retries — they're handled internally and aren't user-observable.

## Validate-docs requirements

When you add a use-case page, `validate-docs` will check:

1. The `Install` extras name exists in `pyproject.toml`.
2. Every code symbol with a `file:line` reference exists at that line.
3. Every cross-link target file exists.
4. Every `MetricEvent` name mentioned exists in `_events.py`.
5. Required section headings exist (adapted for page type).

Run before pushing:

```bash
just docs-check
```

## Index updates

Adding a use-case page also requires:

1. **`docs/index.md`** — add the page as a clickable node in the mermaid use-case map.
2. **`docs/use-cases/index.md`** — add the page to the navigational tree.
3. **`mkdocs.yml`** — add the page under the correct `nav` branch.

All are checked by `mkdocs build --strict`.

## Don't add

- A page describing a single config knob without an end-to-end scenario. Document the knob in the relevant existing page or in `reference/config.md`.
- A page that duplicates 80%+ of an existing page. Extend the existing page with a new "Variant" section instead.
- A page about an internal refactor. Those go in `history/historical-notes/` (engineering notes) or `history/design-decisions/` (ADRs).

## See also

- [`documentation-policy.md`](documentation-policy.md) — overall docs structure.
- [`architecture/index.md`](../architecture/index.md) — what the architecture pages cover (so you don't duplicate).
- The `validate` skill in `.opencode/skills/validate/`.
