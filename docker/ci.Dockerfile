FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_PROJECT_ENVIRONMENT=/opt/slatedb-venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        git \
        libatomic1 \
        openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Install uv for package/env/build/publish workflows.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Install Python 3.10, 3.12, 3.13, 3.14 so tox environments run (and are not skipped).
RUN uv python install 3.10 3.12 3.13 3.14 \
    && ln -sf "$(uv python find 3.10)" /usr/local/bin/python3.10 \
    && ln -sf "$(uv python find 3.12)" /usr/local/bin/python3.12 \
    && ln -sf "$(uv python find 3.13)" /usr/local/bin/python3.13 \
    && ln -sf "$(uv python find 3.14)" /usr/local/bin/python3.14

RUN mkdir -p /opt/slatedb-venv

# Pre-install project dependencies into the container-local uv environment.
# This ensures runtime tools (e.g. slatedb, pyspark, tox deps) are available
# even before mounting the local workspace.
WORKDIR /tmp/slatedb-spark-sharded-deps
COPY pyproject.toml uv.lock ./
RUN uv sync --all-extras --dev --no-install-project --quiet

WORKDIR /workspace

# Example usage:
#   podman build -f docker/ci.Dockerfile -t slatedb-spark-sharded-ci .
#   podman run --rm -it -v "$PWD:/workspace" -w /workspace slatedb-spark-sharded-ci uv sync --all-extras --dev
