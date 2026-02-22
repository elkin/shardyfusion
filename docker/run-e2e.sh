#!/usr/bin/env bash
# Run e2e tests via compose, returning the test exit code.
#
# Usage: docker/run-e2e.sh [ENGINE] [TOX_ARGS...]
#   ENGINE defaults to "podman"
#   TOX_ARGS defaults to "-m e2e"
#
# Works around podman rootless teardown errors that pollute the exit
# code from "compose up --exit-code-from".
set -uo pipefail

engine="${1:-podman}"
shift 2>/dev/null || true
export E2E_TOX_ARGS="${*:--m e2e}"

# Resolve paths relative to the repo root (where the justfile lives).
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
compose_file="$repo_root/docker/compose-e2e.yaml"

compose=("${engine}" compose -f "$compose_file")

"${compose[@]}" build

# Start stack in background, follow test logs, then inspect the actual
# test container exit code (immune to podman cleanup errors).
"${compose[@]}" up -d

# Stream test output to the terminal.  "logs -f" exits once the
# container stops, so we don't need a separate wait.
"${compose[@]}" logs -f tests 2>&1 || true

# Retrieve the real exit code of the tests container.  Query the engine
# directly (compose ps output varies between implementations).
cid=$("${engine}" ps -aq --filter "label=com.docker.compose.service=tests" | head -1)
if [ -n "$cid" ]; then
    rc=$("${engine}" inspect --format '{{.State.ExitCode}}' "$cid" 2>/dev/null || echo 1)
else
    rc=1
fi

# Suppress stderr: podman rootless emits "kill network process: permission
# denied" during teardown which is harmless but noisy.
"${compose[@]}" down 2>/dev/null || true
exit "$rc"
