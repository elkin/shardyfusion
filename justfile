set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

engine := env_var_or_default("CONTAINER_ENGINE", "podman")
image := env_var_or_default("CONTAINER_IMAGE", "shardyfusion-ci")
workspace := "/workspace"
uv_cache_volume := "shardyfusion-uv-cache"
uv_venv_volume := "shardyfusion-uv-venv"
uv_project_env := "/opt/shardyfusion-venv"

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
[arg('p', short='p', help='tox parallel envs')]
quality p="2":
    uv run tox p -m quality -p {{p}}

# Unit tests
[arg('n', short='n', help='pytest-xdist workers')]
[arg('p', short='p', help='tox parallel envs')]
unit n="4" p="2":
    PYTEST_WORKERS={{n}} uv run tox p -m unit -p {{p}}

# Integration tests
[arg('p', short='p', help='tox parallel envs')]
integration p="2":
    uv run tox p -m integration -p {{p}}

# Quality + unit + integration in sequence
[arg('n', short='n', help='pytest-xdist workers')]
[arg('p', short='p', help='tox parallel envs')]
ci n="4" p="2":
    just quality -p {{p}} && just unit -n {{n}} -p {{p}} && just integration -p {{p}}

# ── Docker ───────────────────────────────────────────────────────────────────

d-build:
    {{engine}} build -f docker/ci.Dockerfile -t {{image}} .

d-shell:
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
[arg('p', short='p', help='tox parallel envs')]
d-quality p="2":
    just d "uv run tox p -m quality -p {{p}}"

# Unit tests (in container)
[arg('n', short='n', help='pytest-xdist workers')]
[arg('p', short='p', help='tox parallel envs')]
d-unit n="4" p="2":
    just d "PYTEST_WORKERS={{n}} uv run tox p -m unit -p {{p}}"

# Integration tests (in container)
[arg('p', short='p', help='tox parallel envs')]
d-integration p="2":
    just d "uv run tox p -m integration -p {{p}}"

# End-to-end tests against Garage (in container via compose)
d-e2e:
    docker/run-e2e.sh {{engine}}

# Quality + unit + integration in sequence (in container)
[arg('n', short='n', help='pytest-xdist workers')]
[arg('p', short='p', help='tox parallel envs')]
d-ci n="4" p="2":
    just d-quality -p {{p}} && just d-unit -n {{n}} -p {{p}} && just d-integration -p {{p}}
