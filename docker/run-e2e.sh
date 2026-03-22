#!/usr/bin/env bash
# Run e2e tests via compose, returning the test exit code.
#
# Usage: docker/run-e2e.sh [ENGINE]
#   ENGINE defaults to "podman"
#
# Works around podman rootless teardown errors that pollute the exit
# code from "compose up --exit-code-from".
set -uo pipefail

engine="${1:-podman}"
project_name="shardyfusion-e2e"

# Resolve paths relative to the repo root (where the justfile lives).
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
compose_file="$repo_root/docker/compose-e2e.yaml"

compose=("${engine}" compose -p "$project_name" -f "$compose_file")

"${compose[@]}" build

# Clean up any stale stack first. Under podman-compose this avoids noisy
# "no such container" messages during force-recreate.
"${compose[@]}" down >/dev/null 2>&1 || true

# Start stack in background, then follow the actual test container logs
# directly. This avoids podman-compose service-name lookups like
# "docker_tests_1", which can be flaky when the compose file lives under
# docker/.
"${compose[@]}" up -d

# Retrieve the real test container ID via compose labels. Compose ps
# output varies between implementations, so query the engine directly.
cid=""
for _ in {1..20}; do
    cid=$(
        "${engine}" ps -aq \
            --filter "label=com.docker.compose.project=${project_name}" \
            --filter "label=com.docker.compose.service=tests" \
            | head -1
    )
    if [ -n "$cid" ]; then
        break
    fi
    sleep 1
done

if [ -n "$cid" ]; then
    "${engine}" logs -f "$cid" 2>&1 || true
    rc=$("${engine}" wait "$cid" 2>/dev/null | tail -1 || echo 1)
else
    echo "ERROR: test container did not start within 20s" >&2
    rc=1
fi

# Suppress stderr: podman rootless emits "kill network process: permission
# denied" during teardown which is harmless but noisy.
"${compose[@]}" down >/dev/null 2>&1 || true
exit "$rc"
