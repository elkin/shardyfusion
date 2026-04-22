# Local development

shardyfusion uses [`uv`](https://docs.astral.sh/uv/) for dependency management and [`just`](https://github.com/casey/just) as the task runner. Everything is reproducible from a fresh clone in three commands.

## Prerequisites

| Tool | Required for |
|---|---|
| `uv` | Dependency management; everything. |
| `python3` | 3.11–3.13 (matches `requires-python = ">=3.11,<3.14"`). |
| `just` | All recipe shortcuts. |
| `java` (17+) | Spark writer tests only. Optional for everything else. |
| `podman` or `docker` | Containerized CI workflows and end-to-end tests. |

`just doctor` reports what is and isn't available without changing anything.

## Bootstrap

```bash
git clone https://github.com/elkin/shardyfusion
cd shardyfusion
just setup            # uv sync --all-extras --dev, then verify import
```

`just setup` is idempotent — re-run it whenever `pyproject.toml` changes.

## Daily workflow

| Recipe | What it does |
|---|---|
| `just sync` | `uv sync --all-extras --dev` (faster than full setup). |
| `just fix` | `ruff check --fix` + `ruff format`. Run before commits. |
| `just docs` | `mkdocs build --strict`. |
| `just docs-serve` | Live-reload docs at <http://localhost:8000>. |
| `just quality` | Lint, format check, type check, package, docs check. |
| `just unit` | All unit tests across py311–py313 and Spark 3.5/4. |
| `just integration` | Integration tests (moto S3, real Spark sessions). |
| `just ci` | `quality + unit + integration` in sequence. |

## Containerized workflows

The container path uses an isolated uv project env at `/opt/shardyfusion-venv`, so it does not collide with your host `.venv`.

| Recipe | What it does |
|---|---|
| `just d-build` | Build the CI image from `docker/ci.Dockerfile`. |
| `just d-shell` | Shell inside the container with deps pre-synced. |
| `just d <command>` | Run any command in the container. |
| `just d-quality` / `d-unit` / `d-integration` / `d-e2e` / `d-ci` | Containerized counterparts. |
| `just d-clean` | Remove cached uv/tox volumes. |

End-to-end tests (`just d-e2e`) require a container engine because they spin up a real Garage S3 cluster via `docker/compose-e2e.yaml`.

Switch engines: `CONTAINER_ENGINE=docker just d-build`.

## Devcontainer

`.devcontainer/devcontainer.json` provides a VS Code / GitHub Codespaces config. Open the repo in a devcontainer and `just setup` is the only manual step.

## Cleanup

| Recipe | Removes |
|---|---|
| `just clean` | Caches (`.ruff_cache`, `.pytest_cache`, `.tox` outputs, `dist`, `htmlcov`, `__pycache__`). |
| `just clean-all` | Above + `.venv` + `.tox`. |

## Common pitfalls

- **`shardyfusion not importable` after a pull**: dependencies drifted. Run `just setup`.
- **Spark tests skipped silently**: Java not on `PATH` or version <17. Check with `just doctor`.
- **`mkdocs build --strict` fails after a doc edit**: a link broke or `validate-docs` flagged drift. Run the `validate-docs` skill (`.claude/skills/validate-docs/SKILL.md`).

## See also

- [`testing.md`](testing.md) — what the test labels mean and how to scope a run.
- [`extras-and-dependencies.md`](extras-and-dependencies.md) — when to add a dependency vs. an extra.
- [`operations/tox-matrix.md`](../operations/tox-matrix.md) — the full tox env list.
