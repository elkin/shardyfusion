# Extras and dependencies

shardyfusion has many optional features (writers, backends, vector engines, metrics backends, CLI). The base install must stay minimal; everything else hides behind an extra.

This page is the operational guide. The conceptual reasoning is in [`architecture/optional-imports.md`](../architecture/optional-imports.md).

## Two parallel surfaces

`pyproject.toml` declares dependencies in **two** places, and they must stay aligned:

| Surface | Used by | Defined under |
|---|---|---|
| `[project.optional-dependencies]` | End users (`pip install 'shardyfusion[<extra>]'`) | `pyproject.toml` |
| `[dependency-groups]` | Tox / CI envs | `pyproject.toml` |

The user-facing extras compose into bundles (e.g. `read-slatedb-async = ["shardyfusion[read-slatedb]", "aiobotocore>=2.12"]`).

The dependency groups (`backend-slatedb`, `cap-writer-spark`, `mod-cel`, …) are the atomic units used by `tox.ini` to assemble per-env dependency sets.

## Base dependencies

Always installed (`[project.dependencies]`):

```toml
xxhash>=3.4
pydantic>=2.0
pyyaml>=6.0
```

Anything else must be gated.

## Adding a new optional dependency

Worked example: adding a new vector backend, `vector-foo`, that depends on `foo-vector>=1.0`.

### 1. Add the dependency group

```toml
# pyproject.toml — [dependency-groups]
backend-vector-foo = [
  "foo-vector>=1.0",
  "numpy>=1.24",
]
```

This is the atomic unit tox composes from.

### 2. Add the user-facing extra

```toml
# pyproject.toml — [project.optional-dependencies]
vector-foo = ["foo-vector>=1.0", "numpy>=1.24", "boto3>=1.28"]
```

Convention: the extra mirrors the dependency-group contents, with `boto3` added if the backend reads from S3.

### 3. Add the lazy import wrapper

```python
# shardyfusion/vector/adapters/foo_adapter.py
def _import_foo() -> Any:
    try:
        import foo_vector
    except ImportError as exc:
        raise ImportError(
            "foo vector backend requires `pip install shardyfusion[vector-foo]`"
        ) from exc
    return foo_vector
```

Never import `foo_vector` at module top-level if anything in `shardyfusion/__init__.py`'s import graph would pull it.

### 4. Wire into tox

```ini
# tox.ini — [testenv].dependency_groups
vector-foo: backend-vector-foo
```

Then add envs to `env_list` and the `unit` / `integration` labels:

```
py{311,312,313}-vector-foo-unit
py{311,312,313}-vector-foo-integration
```

### 5. Document the extra

Add a row to:

- `scripts/generate_extras_matrix.py` (`EXTRA_META`) — regenerates the [Extras matrix](../use-cases/extras-matrix.md).
- [`architecture/optional-imports.md`](../architecture/optional-imports.md) — the canonical extras index.
- A new use-case page (e.g. `docs/use-cases/vector/build/foo.md`) — see [`adding-a-use-case.md`](adding-a-use-case.md).

Then regenerate the matrix page:

```bash
uv run python scripts/generate_extras_matrix.py
```

### 6. Verify with `docs-check`

```bash
just docs-check
```

The skill cross-checks:

- Every extra documented in docs is in `pyproject.toml`.
- Every extra in `pyproject.toml` is documented.
- No top-level `from foo_vector import …` exists in modules reachable from `shardyfusion/__init__.py`.

### 7. Update CI matrix

```bash
just ci-matrix
```

This regenerates `.github/ci-matrix.json` from the tox env list.

## When to add an alias

When two extras would have identical contents, the older / more-discoverable name becomes a thin alias:

```toml
read-sqlite-adaptive = ["shardyfusion[sqlite-adaptive]"]
```

Aliases are convenient for users but should not multiply — keep one canonical name per backend. Avoid backend-implicit aliases (e.g. a bare `vector` that silently means LanceDB); prefer explicit names like `vector-lancedb` so every extra advertises its backend in its name.

## When something belongs in `[project.dependencies]` instead

Promote to base only if **all** of these hold:

- Used by the reader **and** at least three writer flavors.
- Pure Python with broad version support (3.11–3.13).
- No transitive conflicts with any current extra.
- Footprint <1 MiB compressed.

`pyyaml` qualifies (used by run records and run registry); `boto3` does not (huge transitive surface, optional for some backends).

## When to bump a pinned upper bound

Upper bounds (e.g. `slatedb<0.12`) exist when an upstream had a known-breaking change. To bump:

1. Bump the pin in `pyproject.toml` and the matching `[dependency-groups]` entry.
2. Run `just ci d-e2e`.
3. If green, commit. If red, the pin stays where it is and a follow-up issue captures the breakage.

Never silently widen a pin without running the full matrix.

## Removing an extra

Removing an extra is a breaking change. Steps:

1. Confirm no use-case page documents it.
2. Remove from `pyproject.toml` (both surfaces).
3. Remove tox envs and label entries.
4. Remove the lazy-import wrapper.
5. Run `validate-docs` — it will surface any stale references.
6. Note the removal in the next ADR or release notes.

## See also

- [`architecture/optional-imports.md`](../architecture/optional-imports.md) — the pattern's design.
- [`operate/tox-matrix.md`](../operate/tox-matrix.md) — full env list.
- [`adding-an-adapter.md`](adding-an-adapter.md) — the canonical worked example.
