FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_SYSTEM_PYTHON=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        git \
        openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Install uv for package/env/build/publish workflows.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Install Python 3.10 so tox py310 environments run (and are not skipped).
RUN uv python install 3.10 \
    && ln -sf "$(uv python find 3.10)" /usr/local/bin/python3.10

WORKDIR /workspace

# Example usage:
#   podman build -f docker/ci.Dockerfile -t slatedb-spark-sharded-ci .
#   podman run --rm -it -v "$PWD:/workspace" -w /workspace slatedb-spark-sharded-ci uv sync --all-extras --dev
