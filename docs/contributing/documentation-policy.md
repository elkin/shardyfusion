# Documentation policy

Docs are first-class. They are part of the deliverable, not an afterthought, and they must match source-code behavior. This page explains the structure, what goes where, and the rules that keep it true.

## Layout

```
docs/
├── index.md                          # landing page with mermaid use-case map
├── use-cases/                        # task-oriented pages plus the shared snapshot workflow
├── operate/                          # CLI, rollback, metrics, cloud testing, tox matrix
├── architecture/                     # 11 conceptual pages (shared internals)
├── contributing/                     # 8 operational pages for contributors
├── reference/                        # API surface, config schema, CLI, glossary
└── history/
    ├── design-decisions/             # ADRs (Architecture Decision Records)
    ├── implementation-plans/         # Open or recently-completed plans
    └── historical-notes/             # Verbatim engineering notes (point-in-time)
```

This is **not** strict Diátaxis — there's no "tutorial" / "how-to" split because every use-case page already includes a runnable example *and* the conditions for choosing it.

## What goes where

| Question the reader is asking | Page type |
|---|---|
| "How do I do X?" | `use-cases/` |
| "How do I run or verify shardyfusion?" | `operate/` |
| "Why does X work the way it does internally?" | `architecture/` |
| "How do I change shardyfusion to do Y?" | `contributing/` |
| "What is the exact signature of `Z()`?" | `reference/` |
| "Why was decision D made?" | `history/design-decisions/` |
| "What was the plan for feature F?" | `history/implementation-plans/` |
| "What did we learn during incident I?" | `history/historical-notes/` |

If a topic could plausibly go in two places, prefer the more user-facing one. Architecture pages should not duplicate use-case content.

## Source is the truth

Every claim in docs must match source behavior. When source changes, docs change in the same PR. When docs say something happens at `foo.py:123`, line 123 must contain the thing.

The `validate` skill (`.opencode/skills/validate/SKILL.md`) checks:

- Symbol references (`file:line`) point at real code.
- Documented extras exist in `pyproject.toml`.
- Documented `MetricEvent` names exist in `_events.py`.
- Section structure of use-case pages follows the locked template.
- Cross-links resolve.

`mkdocs build --strict` catches link rot and rendering errors. Run both before requesting review.

## No legacy / no redirects

The project has no users yet. When superseded, docs are **deleted** rather than redirected. This keeps the docs tree a faithful snapshot of current code, not an archaeology dig.

When that policy changes (first external user), this page changes with it.

## ADR process

Architecture Decision Records live in `docs/history/design-decisions/`. Format:

```markdown
# <Decision title>

**Status**: Accepted | Superseded by <ADR>
**Date**: YYYY-MM-DD

## Context

What forced the decision? What were the constraints?

## Decision

What we decided.

## Consequences

What this enables; what it costs; what we now can't easily change.

## See also

- Related code (`file:line`).
- Use-case pages affected.
- Superseding ADR if any.
```

ADRs are **append-only**: superseded ones stay in the tree with their status updated, so newcomers can reconstruct the reasoning.

When to write an ADR:

- A choice between two viable approaches with non-trivial tradeoffs.
- A change to a public surface (manifest format version, error hierarchy, default backend).
- A reversal of a prior ADR.

When *not* to write one:

- Routine refactors with no user-visible effect.
- Bug fixes.
- Adding an adapter / writer / use-case (the contributing pages cover the process).

## Implementation plans

`docs/history/implementation-plans/` contains plans for non-trivial changes that span more than one PR. A plan is a working document during implementation and a historical artifact afterward.

A plan is "closed" when its checklist is complete; closed plans stay in the tree (renamed to start with `closed-`) so the history is preserved.

## Engineering notes

`docs/history/historical-notes/` contains dated notes (`YYYY-MM-DD-topic.md`) capturing point-in-time observations: bug investigations, performance findings, design alternatives considered and discarded.

Notes are **never edited after the dated day** except to add a forward-pointer at the top:

```markdown
> **Update YYYY-MM-DD**: superseded by [ADR-NNN](../design-decisions/...).
```

This preserves the temporal record.

## Diagrams

Use Mermaid (`\`\`\`mermaid` blocks) — rendered natively by Material for MkDocs. Avoid PNG/SVG except for screenshots, because they can't be diffed.

The landing-page use-case map uses Mermaid `flowchart` with native `click NodeId "url"` syntax — no JS needed.

## Code examples

- Must be runnable as written (after `pip install` of the documented extra).
- Must use real symbol names from `shardyfusion/`.
- Output expectations as Python comments (`# → b"row-42"`).
- Prefer minimal examples; add a second example only if it materially changes the picture.

## Cross-links

Use relative links (`../architecture/sharding.md`), not absolute URLs. `mkdocs build --strict` catches breakage.

When a use-case page mentions a concept that has its own architecture page, link to it on first mention.

## See also

- [`adding-a-use-case.md`](adding-a-use-case.md) — the locked template.
- [`testing.md`](testing.md) — how validate-docs fits into CI.
- The `validate` skill in `.opencode/skills/validate/SKILL.md`.
