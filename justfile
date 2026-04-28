set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

engine := env_var_or_default("CONTAINER_ENGINE", "podman")
image := env_var_or_default("CONTAINER_IMAGE", "shardyfusion-ci")
workspace := "/workspace"
uv_cache_volume := "shardyfusion-uv-cache"
uv_venv_volume := "shardyfusion-uv-venv"
tox_cache_volume := "shardyfusion-tox-cache"
uv_project_env := "/opt/shardyfusion-venv"

_default:
    @just --list

_check-venv:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -d .venv ]; then
        echo "error: .venv/ not found — run 'just setup' first" >&2
        exit 1
    fi
    if ! .venv/bin/python -c "import shardyfusion" 2>/dev/null; then
        echo "error: dependencies out of sync — run 'just setup' to fix" >&2
        exit 1
    fi

_check-java:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v java &>/dev/null; then
        echo "error: java not found — install Java 17+" >&2
        exit 1
    fi
    java_ver=$(java -version 2>&1 | head -1 | sed -E 's/.*"([0-9]+).*/\1/')
    if [ "$java_ver" -lt 17 ] 2>/dev/null; then
        echo "error: Java 17+ required (found Java $java_ver)" >&2
        exit 1
    fi

# ── Setup ────────────────────────────────────────────────────────────────────

# Bootstrap a fresh clone (idempotent)
[group('setup')]
setup:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "── checking prerequisites ──"
    for cmd in uv python3; do
        if ! command -v "$cmd" &>/dev/null; then
            echo "FAIL  $cmd not found — install it first" >&2
            exit 1
        fi
        printf "  ok  %s (%s)\n" "$cmd" "$(command -v "$cmd")"
    done
    echo "── syncing dependencies ──"
    uv sync --all-extras --dev
    echo "── verifying install ──"
    uv run python -c "import shardyfusion; from importlib.metadata import version; print(f'  ok  shardyfusion {version(\"shardyfusion\")}')"
    if command -v java &>/dev/null; then
        printf "  ok  java (%s)\n" "$(java -version 2>&1 | head -1)"
    else
        echo "  warn  java not found — Spark writer tests will be skipped"
    fi
    echo "── setup complete ──"

# Read-only environment health check
[group('setup')]
doctor:
    #!/usr/bin/env bash
    set -euo pipefail
    ok=0 warn=0 fail=0

    check_required() {
        if command -v "$1" &>/dev/null; then
            printf "  ok    %s (%s)\n" "$1" "$(command -v "$1")"
            ok=$((ok + 1))
        else
            printf "  FAIL  %s not found\n" "$1"
            fail=$((fail + 1))
        fi
    }

    check_optional() {
        if command -v "$1" &>/dev/null; then
            printf "  ok    %s (%s)\n" "$1" "$(command -v "$1")"
            ok=$((ok + 1))
        else
            printf "  warn  %s not found — %s\n" "$1" "$2"
            warn=$((warn + 1))
        fi
    }

    echo "── required tools ──"
    check_required uv
    check_required python3
    check_required just

    echo "── optional tools ──"
    check_optional java "needed for Spark writer tests only"

    echo "── project state ──"
    if [ -d .venv ]; then
        printf "  ok    .venv/ exists\n"
        ok=$((ok + 1))
    else
        printf "  FAIL  .venv/ missing — run 'just setup'\n"
        fail=$((fail + 1))
    fi

    if uv run python -c "import shardyfusion" 2>/dev/null; then
        printf "  ok    shardyfusion is importable\n"
        ok=$((ok + 1))
    else
        printf "  FAIL  shardyfusion not importable — run 'just setup'\n"
        fail=$((fail + 1))
    fi

    echo "── summary: $ok ok, $warn warn, $fail fail ──"
    [ "$fail" -eq 0 ]

# Remove caches and build artifacts
[group('setup')]
clean:
    rm -rf .ruff_cache .pytest_cache .mypy_cache .hypothesis
    rm -rf dist site build
    rm -rf shardyfusion.egg-info
    rm -rf htmlcov .coverage .coverage.* coverage.xml
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Remove caches, .venv, and .tox (full reset)
[group('setup')]
clean-all: clean
    rm -rf .venv .tox

# ── Dev ──────────────────────────────────────────────────────────────────────

# Install all dependencies into the local venv
[group('dev')]
sync:
    uv sync --all-extras --dev

# Auto-fix ruff lint and format issues
[group('dev')]
fix: _check-venv
    uv run ruff check --fix .
    uv run ruff format .

# Build documentation
[group('dev')]
docs: _check-venv
    uv run mkdocs build --strict

# Serve documentation locally with live reload
[group('dev')]
docs-serve: _check-venv
    uv run mkdocs serve --livereload --watch shardyfusion/

# Regenerate .github/ci-matrix.json from tox env lists
[group('dev')]
ci-matrix: _check-venv
    uv run python scripts/generate_ci_matrix.py

# ── Test ─────────────────────────────────────────────────────────────────────

# Lint, format, type checks, package build, docs check
[group('test')]
[arg('p', short='p', help='tox parallel envs')]
quality p="2": _check-venv _check-java
    uv run tox p -m quality -p {{p}}

# Unit tests (pass path to run a specific test directory/module)
[group('test')]
[arg('n', short='n', help='pytest-xdist workers')]
[arg('p', short='p', help='tox parallel envs')]
unit n="4" p="2" path="": _check-venv _check-java
    PYTEST_WORKERS={{n}} uv run tox p -m unit -p {{p}} {{if path != "" { "-- " + path } else { "" }}}

# Integration tests (pass path to run a specific test directory/module)
[group('test')]
[arg('n', short='n', help='pytest-xdist workers')]
[arg('p', short='p', help='tox parallel envs')]
integration n="4" p="2" path="": _check-venv _check-java
    PYTEST_WORKERS={{n}} uv run tox p -m integration -p {{p}} {{if path != "" { "-- " + path } else { "" }}}

# Quality + unit + integration in sequence
[group('test')]
[arg('n', short='n', help='pytest-xdist workers')]
[arg('p', short='p', help='tox parallel envs')]
ci n="4" p="2":
    PYTEST_WORKERS={{n}} uv run tox p -m quality -m unit -m integration -p {{p}}

# Run unit and integration tests with coverage and produce a report
[group('test')]
[arg('n', short='n', help='pytest-xdist workers')]
coverage n="4": _check-venv _check-java
    RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 SPARK_LOCAL_IP=127.0.0.1 uv run pytest --cov -n {{n}} -q tests/unit tests/integration
    uv run coverage html
    @echo "HTML report: htmlcov/index.html"

# ── Container ────────────────────────────────────────────────────────────────

# Build the CI container image
[group('container')]
d-build:
    {{engine}} build -f docker/ci.Dockerfile -t {{image}} .

# Open a shell in the CI container
[group('container')]
d-shell:
    {{engine}} run --rm -it \
      -v "{{invocation_directory()}}:{{workspace}}" \
      -v "{{uv_cache_volume}}:/root/.cache/uv" \
      -v "{{uv_venv_volume}}:{{uv_project_env}}" \
      -v "{{tox_cache_volume}}:{{workspace}}/.tox" \
      -w {{workspace}} \
      -e UV_PROJECT_ENVIRONMENT={{uv_project_env}} \
      {{image}} \
      /bin/bash -lc "uv sync --all-extras --dev --quiet && exec /bin/bash"

# Run an arbitrary command inside the container
[group('container')]
d +cmd:
    {{engine}} run --rm -it \
      -v "{{invocation_directory()}}:{{workspace}}" \
      -v "{{uv_cache_volume}}:/root/.cache/uv" \
      -v "{{uv_venv_volume}}:{{uv_project_env}}" \
      -v "{{tox_cache_volume}}:{{workspace}}/.tox" \
      -w {{workspace}} \
      -e UV_PROJECT_ENVIRONMENT={{uv_project_env}} \
      {{image}} \
      /bin/bash -lc "uv sync --all-extras --dev --quiet && {{cmd}}"

# Lint, format, type checks, package build, docs check (in container)
[group('container')]
[arg('p', short='p', help='tox parallel envs')]
d-quality p="2":
    just d "uv run tox p -m quality -p {{p}}"

# Unit tests (in container; pass path to run a specific test directory/module)
[group('container')]
[arg('n', short='n', help='pytest-xdist workers')]
[arg('p', short='p', help='tox parallel envs')]
d-unit n="4" p="2" path="":
    just d "PYTEST_WORKERS={{n}} uv run tox p -m unit -p {{p}} {{if path != "" { "-- " + path } else { "" }}}"

# Integration tests (in container; pass path to run a specific test directory/module)
[group('container')]
[arg('n', short='n', help='pytest-xdist workers')]
[arg('p', short='p', help='tox parallel envs')]
d-integration n="4" p="2" path="":
    just d "PYTEST_WORKERS={{n}} uv run tox p -m integration -p {{p}} {{if path != "" { "-- " + path } else { "" }}}"

# End-to-end tests against Garage (in container via compose)
[group('container')]
[arg('p', short='p', help='tox parallel envs')]
d-e2e p="2": d-build
    E2E_PARALLEL={{p}} docker/run-e2e.sh {{engine}}

# Quality + unit + integration in sequence (in container)
[group('container')]
[arg('n', short='n', help='pytest-xdist workers')]
[arg('p', short='p', help='tox parallel envs')]
d-ci n="4" p="2":
    PYTEST_WORKERS={{n}} just d "uv run tox p -m quality -m unit -m integration -p {{p}}"

# Remove container cache volumes
[group('container')]
d-clean:
    {{engine}} volume rm -f {{uv_cache_volume}} {{uv_venv_volume}} {{tox_cache_volume}}
