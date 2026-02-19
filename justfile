set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

engine := env_var_or_default("CONTAINER_ENGINE", "podman")
image := env_var_or_default("CONTAINER_IMAGE", "slatedb-spark-sharded-ci")
workspace := "/workspace"
uv_cache_volume := "slatedb-spark-sharded-uv-cache"
uv_venv_volume := "slatedb-spark-sharded-uv-venv"
uv_project_env := "/opt/slatedb-venv"

_default:
    @just --list

# ── Local ────────────────────────────────────────────────────────────────────

# Install all dependencies into the local venv
sync:
    uv sync --all-extras --dev

# Auto-fix ruff lint and format issues
fix:
    uv run ruff check --fix .
    uv run ruff format .

# Lint, format, type checks, package build, docs check
quality:
    uv run tox -m quality

# Lint, format, type checks, package build, docs check (parallel)
quality-p:
    uv run tox p -m quality -p 4

# Unit tests
unit:
    uv run tox -m unit

# Unit tests (parallel)
unit-p:
    uv run tox p -m unit -p 2

# Integration tests
integration:
    uv run tox -m integration

# Integration tests (parallel)
integration-p:
    uv run tox p -m integration -p 2

# Quality + unit + integration in sequence
ci: quality unit integration

# ── Docker ───────────────────────────────────────────────────────────────────

docker-build:
    {{engine}} build -f docker/ci.Dockerfile -t {{image}} .

docker-shell:
    {{engine}} run --rm -it \
      -v "{{invocation_directory()}}:{{workspace}}" \
      -v "{{uv_cache_volume}}:/root/.cache/uv" \
      -v "{{uv_venv_volume}}:{{uv_project_env}}" \
      -w {{workspace}} \
      -e UV_PROJECT_ENVIRONMENT={{uv_project_env}} \
      {{image}} \
      /bin/bash -lc "uv sync --all-extras --dev --quiet && exec /bin/bash"

# Run an arbitrary command inside the container
d +cmd:
    {{engine}} run --rm -it \
      -v "{{invocation_directory()}}:{{workspace}}" \
      -v "{{uv_cache_volume}}:/root/.cache/uv" \
      -v "{{uv_venv_volume}}:{{uv_project_env}}" \
      -w {{workspace}} \
      -e UV_PROJECT_ENVIRONMENT={{uv_project_env}} \
      {{image}} \
      /bin/bash -lc "uv sync --all-extras --dev --quiet && {{cmd}}"

# Lint, format, type checks, package build, docs check (in container)
d-quality:
    just d uv run tox -m quality

# Lint, format, type checks, package build, docs check (in container, parallel)
d-quality-p:
    just d uv run tox p -m quality -p 4

# Unit tests (in container)
d-unit:
    just d uv run tox -m unit

# Unit tests (in container, parallel)
d-unit-p:
    just d uv run tox p -m unit -p 2

# Integration tests (in container)
d-integration:
    just d uv run tox -m integration

# Integration tests (in container, parallel)
d-integration-p:
    just d uv run tox p -m integration -p 2

# Quality + unit + integration in sequence (in container)
d-ci:
    just d-quality && just d-unit && just d-integration
