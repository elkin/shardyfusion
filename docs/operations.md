# Operations

## CI workflows

- `CI` workflow:
  - quality gates (`ruff`, `ty`, `pyright`)
  - package build smoke checks
  - unit and integration test jobs
- `Docs` workflow:
  - docs build on pull requests
  - docs publish to GitHub Pages on `main`
- `Release` workflow:
  - tag-triggered release validation + PyPI publish

## Running checks locally

```bash
# Quality
tox -e lint,format,type

# Package check
tox -e package

# Unit matrix (parallel tox envs)
tox p -p 2

# Integration subsets
tox -e py311-read-integration,py311-writer-spark4-integration
```

## Memory tuning for tests

To avoid OOM when parallelizing:

- per-unit-env pytest is capped to `-n 2`
- tox environment-level parallelism should be capped (`tox p -p 2`)
