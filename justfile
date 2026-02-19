set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

engine := env_var_or_default("CONTAINER_ENGINE", "podman")
image := env_var_or_default("CONTAINER_IMAGE", "slatedb-spark-sharded-ci")
workspace := "/workspace"
uv_cache_volume := "slatedb-spark-sharded-uv-cache"
uv_venv_volume := "slatedb-spark-sharded-uv-venv"
uv_project_env := "/opt/slatedb-venv"

default:
    @just --list

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

d +cmd:
    {{engine}} run --rm -it \
      -v "{{invocation_directory()}}:{{workspace}}" \
      -v "{{uv_cache_volume}}:/root/.cache/uv" \
      -v "{{uv_venv_volume}}:{{uv_project_env}}" \
      -w {{workspace}} \
      -e UV_PROJECT_ENVIRONMENT={{uv_project_env}} \
      {{image}} \
      /bin/bash -lc "uv sync --all-extras --dev --quiet && {{cmd}}"
