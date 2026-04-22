# Adding a use-case page

Use-case pages are the primary entry point for users. There are 18 of them, each covering one (writer × backend) or one operational scenario. The structure is **locked** — every page follows the same 11-section template.

This makes the docs predictable, comparable across backends, and verifiable by the `validate-docs` skill (`.claude/skills/validate-docs/SKILL.md`) skill.

## When to add a page

Add a use-case page when you add:

- A new writer flavor + backend combination (e.g. `build-myframework-slatedb`).
- A new operational scenario (e.g. `operate-postgres-manifest-store`).
- A new read-side combination (e.g. `read-async-foodb`).

If the new feature is a *variation* of an existing use case (different config knob), extend the existing page; don't add a new one.

## The locked template

Every use-case page has these sections in this order. Section names are exact.

```markdown
# <Verb the user does, including writer + backend if relevant>

<One-sentence framing: who this is for and what they get.>

## When to use

- Bulleted list of conditions.

## When NOT to use

- Bulleted list with explicit pointers to the right alternative.

## Install

\```bash
uv add 'shardyfusion[<extra>]'
\```

## Minimal example

A complete, runnable snippet. Output expectations included as comments.

## Configuration

A table of the relevant `WriteConfig` / `ShardedReader` / etc. fields with defaults and purpose. Only fields used by THIS use case.

## Functional properties

- Streaming-safe? Deterministic? Atomic publish?

## Non-functional properties

- Memory profile, parallelism model, backpressure.

## Guarantees

What a successful return means. What persists.

## Weaknesses

What this approach does badly.

## Failure modes & recovery

| Failure | Surface | Recovery |
|---|---|---|
...

## See also

- Links to architecture pages.
- Sibling use-case pages.
- Related ADRs.
```

The prototype is `docs/use-cases/build-python-slatedb.md`. Read it before writing a new page.

## Style rules

- **No marketing.** Describe what the code does, including its limits.
- **No comparisons that don't exist in code.** "Faster than X" needs a benchmark in `tests/` or it doesn't go in.
- **Use real symbol names with file:line refs**: `WriteConfig.batch_size` (`shardyfusion/config.py:201`).
- **Prefer absolute correctness to brevity.** If the failure-mode table has 10 rows, list 10 rows.
- **Cross-link generously.** Every "Configuration" field that has its own section in `architecture/` should link there. Every "When NOT to use" alternative should link.

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
5. The 11 section headings exist in the right order.

Run before pushing:

```bash
uv run python .claude/skills/validate-docs/scripts/check_docs.py
```

## Index updates

Adding a use-case page also requires:

1. **`docs/index.md`** — add the page as a clickable node in the mermaid use-case map.
2. **`mkdocs.yml`** — add the page under `nav.Use cases:`.

Both are checked by `mkdocs build --strict`.

## Don't add

- A page describing a single config knob without an end-to-end scenario. Document the knob in the relevant existing page or in `reference/config.md`.
- A page that duplicates 80%+ of an existing page. Extend the existing page with a new "Variant" section instead — but only if it's a small variant. Otherwise it's two use cases.
- A page about an internal refactor. Those go in `history/historical-notes/` (engineering notes) or `history/design-decisions/` (ADRs).

## See also

- [`documentation-policy.md`](documentation-policy.md) — overall docs structure.
- [`architecture/index.md`](../architecture/index.md) — what the architecture pages cover (so you don't duplicate).
- The validate-docs skill in `.claude/skills/validate-docs/`.
