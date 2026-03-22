#!/usr/bin/env python3
"""Generate .github/ci-matrix.json from tox env lists.

Usage:
    python scripts/generate_ci_matrix.py          # write matrix file
    python scripts/generate_ci_matrix.py --check   # verify committed file is up to date
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = REPO_ROOT / ".github" / "ci-matrix.json"

LABELS = ("unit", "integration", "smoke")


def _tox_envs(label: str) -> list[str]:
    result = subprocess.run(
        ["uv", "run", "tox", "-m", label, "-l"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    return [e.strip() for e in result.stdout.splitlines() if e.strip()]


def _build_entry(env_name: str) -> dict[str, object]:
    match = re.match(r"py(\d)(\d{2})-", env_name)
    if match is None:
        raise SystemExit(f"Cannot infer Python version from: {env_name}")
    python_version = f"{match.group(1)}.{int(match.group(2))}"
    return {
        "name": env_name.replace("-", " / "),
        "tox-env": env_name,
        "python-version": python_version,
        "needs-java": "sparkwriter" in env_name or "-all-" in env_name,
    }


def generate() -> dict[str, object]:
    matrices: dict[str, object] = {}
    for label in LABELS:
        envs = _tox_envs(label)
        matrices[label] = {"include": [_build_entry(e) for e in envs]}
    return matrices


def main() -> None:
    check_mode = "--check" in sys.argv

    matrices = generate()
    generated = json.dumps(matrices, indent=2) + "\n"

    if check_mode:
        if not MATRIX_PATH.exists():
            print(f"FAIL: {MATRIX_PATH} does not exist", file=sys.stderr)
            print("Run: just ci-matrix", file=sys.stderr)
            raise SystemExit(1)
        committed = MATRIX_PATH.read_text()
        if committed != generated:
            print(
                f"FAIL: {MATRIX_PATH} is stale — regenerate with: just ci-matrix",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print(f"OK: {MATRIX_PATH} is up to date")
    else:
        MATRIX_PATH.write_text(generated)
        print(f"Wrote {MATRIX_PATH}")


if __name__ == "__main__":
    main()
