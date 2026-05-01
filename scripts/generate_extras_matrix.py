#!/usr/bin/env python3
"""Generate docs/use-cases/extras-matrix.md from pyproject.toml extras.

Run manually whenever extras change:
    uv run python scripts/generate_extras_matrix.py
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
OUTPUT = REPO_ROOT / "docs" / "use-cases" / "extras-matrix.md"

# Extras that are NOT user-facing runtime extras.
DEV_EXTRAS = {"test", "quality", "docs"}

_RE_EXTRA_REF = re.compile(r"shardyfusion\[([A-Za-z0-9_\-]+)\]")


class _Meta(TypedDict, total=False):
    section: str
    task: str
    notes: str


# fmt: off
EXTRA_META: dict[str, _Meta] = {
    # ---- Backend building blocks ----
    "slatedb": {
        "section": "Backend building blocks",
        "task": "SlateDB storage engine",
        "notes": "Base dep for SlateDB-backed readers and writers.",
    },
    "sqlite": {
        "section": "Backend building blocks",
        "task": "SQLite storage engine",
        "notes": "Base dep for SQLite-backed readers and writers.",
    },
    "sqlite-range": {
        "section": "Backend building blocks",
        "task": "SQLite range-read VFS",
        "notes": "APSW + obstore for S3 range-reads without full download.",
    },

    # ---- KV Storage — Read ----
    "read-slatedb": {
        "section": "KV Storage — Read",
        "task": "Sync reader (SlateDB)",
        "notes": "Sync SlateDB reader. Pulls in `slatedb`.",
    },
    "read-slatedb-async": {
        "section": "KV Storage — Read",
        "task": "Async reader (SlateDB)",
        "notes": "Async S3 manifest store + SlateDB shards via aiobotocore.",
    },
    "read-sqlite": {
        "section": "KV Storage — Read",
        "task": "Sync SQLite reader (download)",
        "notes": "Downloads full DB locally before opening.",
    },
    "read-sqlite-range": {
        "section": "KV Storage — Read",
        "task": "Sync SQLite reader (range-read)",
        "notes": "Uses APSW VFS to read S3 pages on demand.",
    },
    "sqlite-adaptive": {
        "section": "KV Storage — Read",
        "task": "Adaptive SQLite reader deps",
        "notes": "Composes `sqlite` + `sqlite-range` so AdaptiveSqliteReaderFactory can pick per snapshot.",
    },
    "read-sqlite-adaptive": {
        "section": "KV Storage — Read",
        "task": "Sync adaptive SQLite reader",
        "notes": "Alias for `sqlite-adaptive`. Auto-picks download vs range per snapshot.",
    },
    "sqlite-async": {
        "section": "KV Storage — Read",
        "task": "Async SQLite readers",
        "notes": "Async wrappers for both download and range-read SQLite.",
    },
    "sqlite-adaptive-async": {
        "section": "KV Storage — Read",
        "task": "Async adaptive SQLite reader",
        "notes": "Async adaptive policy + aiobotocore.",
    },

    # ---- KV Storage — Write ----
    "writer-python-slatedb": {
        "section": "KV Storage — Write",
        "task": "Python writer (SlateDB)",
        "notes": "Pure Python, single-process or multiprocessing.",
    },
    "writer-python-sqlite": {
        "section": "KV Storage — Write",
        "task": "Python writer (SQLite)",
        "notes": "Pure Python writing SQLite shards.",
    },
    "writer-spark": {
        "section": "KV Storage — Write",
        "task": "Spark framework (no backend)",
        "notes": "PySpark + pandas + pyarrow. Combine with `slatedb`, `sqlite`, `vector-lancedb`, or `vector-sqlite`.",
    },
    "writer-spark-slatedb": {
        "section": "KV Storage — Write",
        "task": "Spark writer (SlateDB)",
        "notes": "Requires Java 17. PySpark ≥3.3.",
    },
    "writer-spark-sqlite": {
        "section": "KV Storage — Write",
        "task": "Spark writer (SQLite)",
        "notes": "Requires Java 17. PySpark ≥3.3.",
    },
    "writer-dask": {
        "section": "KV Storage — Write",
        "task": "Dask framework (no backend)",
        "notes": "dask[dataframe]. Combine with `slatedb`, `sqlite`, `vector-lancedb`, or `vector-sqlite`.",
    },
    "writer-dask-slatedb": {
        "section": "KV Storage — Write",
        "task": "Dask writer (SlateDB)",
        "notes": "Dask DataFrame input.",
    },
    "writer-dask-sqlite": {
        "section": "KV Storage — Write",
        "task": "Dask writer (SQLite)",
        "notes": "Dask DataFrame input writing SQLite shards.",
    },
    "writer-ray": {
        "section": "KV Storage — Write",
        "task": "Ray framework (no backend)",
        "notes": "ray[data]. Combine with `slatedb`, `sqlite`, `vector-lancedb`, or `vector-sqlite`.",
    },
    "writer-ray-slatedb": {
        "section": "KV Storage — Write",
        "task": "Ray writer (SlateDB)",
        "notes": "Ray Dataset input.",
    },
    "writer-ray-sqlite": {
        "section": "KV Storage — Write",
        "task": "Ray writer (SQLite)",
        "notes": "Ray Dataset input writing SQLite shards.",
    },

    # ---- Vector Search — Distributed Write ----
    "writer-spark-vector-lancedb": {
        "section": "Vector Search — Distributed Write",
        "task": "Spark vector writer (LanceDB)",
        "notes": "PySpark DataFrame → sharded LanceDB index. Requires Java 17.",
    },
    "writer-spark-vector-sqlite": {
        "section": "Vector Search — Distributed Write",
        "task": "Spark vector writer (sqlite-vec)",
        "notes": "PySpark DataFrame → sharded sqlite-vec index. Requires Java 17.",
    },
    "writer-dask-vector-lancedb": {
        "section": "Vector Search — Distributed Write",
        "task": "Dask vector writer (LanceDB)",
        "notes": "Dask DataFrame → sharded LanceDB index.",
    },
    "writer-dask-vector-sqlite": {
        "section": "Vector Search — Distributed Write",
        "task": "Dask vector writer (sqlite-vec)",
        "notes": "Dask DataFrame → sharded sqlite-vec index.",
    },
    "writer-ray-vector-lancedb": {
        "section": "Vector Search — Distributed Write",
        "task": "Ray vector writer (LanceDB)",
        "notes": "Ray Dataset → sharded LanceDB index.",
    },
    "writer-ray-vector-sqlite": {
        "section": "Vector Search — Distributed Write",
        "task": "Ray vector writer (sqlite-vec)",
        "notes": "Ray Dataset → sharded sqlite-vec index.",
    },

    # ---- Vector Search ----
    "vector-lancedb": {
        "section": "Vector Search",
        "task": "Vector search (LanceDB)",
        "notes": "HNSW index via LanceDB.",
    },
    "vector-sqlite": {
        "section": "Vector Search",
        "task": "Vector search (sqlite-vec)",
        "notes": "sqlite-vec unified KV+vector in single DB.",
    },

    # ---- KV + Vector ----
    "unified-slatedb-lancedb": {
        "section": "KV + Vector",
        "task": "Unified KV+vector (SlateDB + LanceDB)",
        "notes": "Composite SlateDB + LanceDB sidecar. Enables UnifiedShardedReader.",
    },
    "unified-sqlite-vec": {
        "section": "KV + Vector",
        "task": "Unified KV+vector (sqlite-vec)",
        "notes": "Single-file sqlite-vec backend. Enables UnifiedShardedReader.",
    },

    # ---- Operations & Observability ----
    "cli-minimal": {
        "section": "Operations & Observability",
        "task": "CLI binary (no backend)",
        "notes": "`shardy` command with click only. Combine with a reader extra.",
    },
    "cli": {
        "section": "Operations & Observability",
        "task": "Kitchen-sink CLI",
        "notes": "All reader backends + vector + CEL bundled.",
    },
    "metrics-prometheus": {
        "section": "Operations & Observability",
        "task": "Prometheus metrics",
        "notes": "PrometheusCollector for writer/reader events.",
    },
    "metrics-otel": {
        "section": "Operations & Observability",
        "task": "OpenTelemetry metrics",
        "notes": "OtelCollector for writer/reader events.",
    },

    # ---- Routing ----
    "cel": {
        "section": "Routing",
        "task": "CEL expression routing",
        "notes": "Custom sharding rules via cel-expr-python.",
    },

    # ---- Development ----
    "all": {
        "section": "Development",
        "task": "Everything (runtime)",
        "notes": "Convenience bundle of all runtime extras. Excludes dev/test/quality/docs.",
    },
    "test": {
        "section": "Development",
        "task": "Test dependencies",
        "notes": "pytest, hypothesis, moto, etc.",
    },
    "quality": {
        "section": "Development",
        "task": "Lint & type-check dependencies",
        "notes": "ruff, pyright.",
    },
    "docs": {
        "section": "Development",
        "task": "Documentation dependencies",
        "notes": "MkDocs + plugins.",
    },
}
# fmt: on


SECTION_ORDER = [
    "Backend building blocks",
    "KV Storage — Read",
    "KV Storage — Write",
    "Vector Search — Distributed Write",
    "Vector Search",
    "KV + Vector",
    "Operations & Observability",
    "Routing",
    "Development",
]


def load_extras() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return set(data.get("project", {}).get("optional-dependencies", {}).keys())


def load_extra_dependencies() -> dict[str, set[str]]:
    """Return {extra_name: {other_extra_names it depends on}}."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    deps: dict[str, set[str]] = {}
    for name, items in data.get("project", {}).get("optional-dependencies", {}).items():
        refs: set[str] = set()
        for item in items:
            for m in _RE_EXTRA_REF.finditer(item):
                refs.add(m.group(1))
        deps[name] = refs
    return deps


def _node_id(name: str) -> str:
    return f"n_{name.replace('-', '_')}"


def _safe_subgraph_id(section: str) -> str:
    return (
        section.lower()
        .replace(" — ", "_")
        .replace(" ", "_")
        .replace("&", "and")
        .replace("+", "plus")
    )


def generate() -> str:
    extras = load_extras()
    extra_deps = load_extra_dependencies()

    # Validate completeness
    missing = extras - set(EXTRA_META.keys())
    if missing:
        print(
            f"ERROR: Extras in pyproject.toml but missing from EXTRA_META: {sorted(missing)}",
            file=sys.stderr,
        )
        print("Update EXTRA_META in scripts/generate_extras_matrix.py", file=sys.stderr)
        sys.exit(1)

    # Group by section
    by_section: dict[str, list[tuple[str, _Meta]]] = {s: [] for s in SECTION_ORDER}
    for name in sorted(
        extras, key=lambda n: (EXTRA_META[n]["section"], EXTRA_META[n]["task"])
    ):
        meta = EXTRA_META[name]
        by_section.setdefault(meta["section"], []).append((name, meta))

    # Build runtime list (exclude dev extras)
    runtime_names = {n for n in sorted(extras) if n not in DEV_EXTRAS}

    lines: list[str] = []
    lines.append("# Extras Matrix")
    lines.append("")
    lines.append(
        "This page maps every user-facing use case to the `pip install` / `uv sync --extra` "
        "target you need.  It is auto-generated by `scripts/generate_extras_matrix.py`; "
        "run that script after adding or renaming an extra."
    )
    lines.append("")
    lines.append("## Common install commands")
    lines.append("")
    lines.append("```bash")
    lines.append("# Reader only (SlateDB backend)")
    lines.append("uv sync --extra read-slatedb")
    lines.append("")
    lines.append("# Async reader (SlateDB backend)")
    lines.append("uv sync --extra read-slatedb-async")
    lines.append("")
    lines.append("# Spark writer (SlateDB backend; requires Java 17)")
    lines.append("uv sync --extra writer-spark-slatedb")
    lines.append("")
    lines.append("# Kitchen-sink CLI (all read backends + vector)")
    lines.append("uv sync --extra cli")
    lines.append("")
    lines.append("# Everything (all runtime extras)")
    lines.append("uv sync --extra all")
    lines.append("```")
    lines.append("")

    # ---- Mermaid diagram (runtime extras only) ----
    lines.append("## Visual map")
    lines.append("")
    lines.append(
        "Arrows point from a base extra to the extras that build on top of it."
    )
    lines.append("")
    lines.append("```mermaid")
    lines.append("%%{init: {'flowchart': {'ranksep': 30, 'nodesep': 4}}}%%")
    lines.append("flowchart LR")
    lines.append("  classDef default font-size:16px")
    lines.append("")

    # Subgraphs per section
    for section in SECTION_ORDER:
        items = by_section.get(section, [])
        runtime_items = [(n, m) for n, m in items if n not in DEV_EXTRAS]
        if not runtime_items:
            continue
        safe_id = _safe_subgraph_id(section)
        lines.append(f"  subgraph {safe_id}[{section}]")
        lines.append("    direction TB")
        for name, meta in runtime_items:
            nid = _node_id(name)
            label = f"{name}\\n{meta['task']}"
            lines.append(f'    {nid}("{label}")')
        lines.append("  end")
        lines.append("")

    # Dependency edges (runtime only, skip dev extras)
    # Skip edges from `all` because it is a convenience bundle that depends on
    # everything — drawing them creates unreadable clutter.
    drawn: set[tuple[str, str]] = set()
    for src in sorted(runtime_names):
        if src == "all":
            continue
        for dst in sorted(extra_deps.get(src, set())):
            if dst not in runtime_names or dst == "all":
                continue
            # Draw edge from dependency -> dependent (base -> derived)
            edge = (src, dst)
            if edge not in drawn:
                drawn.add(edge)
                lines.append(f"  {_node_id(src)} --> {_node_id(dst)}")

    lines.append("```")
    lines.append("")

    # ---- Tables (all extras) ----
    lines.append("## Full table")
    lines.append("")

    for section in SECTION_ORDER:
        items = by_section.get(section, [])
        if not items:
            continue
        lines.append(f"### {section}")
        lines.append("")
        lines.append("| Extra | Task | Notes |")
        lines.append("|---|---|---|")
        for name, meta in items:
            notes = meta.get("notes", "")
            lines.append(f"| `{name}` | {meta['task']} | {notes} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "*Generated by `scripts/generate_extras_matrix.py`.  "
        "Do not edit this file by hand — it will be overwritten on the next run.*"
    )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    if not PYPROJECT.exists():
        print(f"pyproject.toml not found at {PYPROJECT}", file=sys.stderr)
        return 1

    content = generate()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
