# Contributing

shardyfusion is a small library with a strong policy bias: documentation must match source-code behavior, every optional dependency must be gated, and every public surface should be reachable from a use-case page.

These pages are the operational counterpart to [`architecture/`](../architecture/index.md). Architecture explains *what is* and *why*; contributing explains *what to do* when you're changing it.

## Pages

- [`local-development.md`](local-development.md) — bootstrap a clone, run `just doctor`, container vs. host workflows.
- [`testing.md`](testing.md) — the test pyramid (unit / integration / e2e), tox env naming, shared helpers.
- [`extras-and-dependencies.md`](extras-and-dependencies.md) — how to add a new optional dependency without breaking the base install.
- [`adding-an-adapter.md`](adding-an-adapter.md) — worked example: implement a new KV backend.
- [`adding-a-writer.md`](adding-a-writer.md) — worked example: add a new distributed-writer flavor.
- [`adding-a-use-case.md`](adding-a-use-case.md) — the locked use-case page template and validate-docs contract.
- [`documentation-policy.md`](documentation-policy.md) — the docs/ layout, ADR process, history.

## Non-negotiables

1. **Source is the truth.** Docs must match what `shardyfusion/` actually does. The `validate` skill (`.opencode/skills/validate/SKILL.md`) is run on every PR.
2. **Optional imports.** New backend or framework deps go behind an extra. Never import them at the top of `shardyfusion/__init__.py`. See [`architecture/optional-imports.md`](../architecture/optional-imports.md).
3. **Tests first for behavior changes.** Add a unit test (and integration where routing or publishing is affected) before changing `_writer_core.py`, `routing.py`, `manifest_store.py`, or any adapter.
4. **Run `just ci d-e2e` before requesting review.** Quality + unit + integration + end-to-end against Garage.
5. **No legacy/redirect docs.** The project has no users; we delete superseded docs rather than keeping shims.

## Style

- Python 3.11+ with full type hints.
- Ruff for lint and format. `just fix` auto-applies both.
- `snake_case` / `PascalCase` / `UPPER_SNAKE_CASE`.
- `@dataclass(slots=True)` for performance-sensitive models.
- Keep Spark logic in `writer/spark/`; do not introduce Python UDFs when Spark built-ins suffice.

## See also

- [`architecture/index.md`](../architecture/index.md) — the conceptual map.
- [`history/design-decisions/`](../history/design-decisions/index.md) — ADRs explaining why a thing is the way it is.
