#!/bin/bash
# PostToolUse hook: auto-format Python files after Edit/Write.
# Reads tool_input from stdin JSON, runs ruff format + ruff check --fix.
set -euo pipefail

FILE=$(cat | jq -r '.tool_input.file_path // empty')

# Only act on Python files
if [[ "$FILE" == *.py && -f "$FILE" ]]; then
    cd "$CLAUDE_PROJECT_DIR"
    uv run ruff format "$FILE" 2>/dev/null || true
    uv run ruff check --fix "$FILE" 2>/dev/null || true
fi

exit 0
