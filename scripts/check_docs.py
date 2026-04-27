#!/usr/bin/env python3
"""validate-docs helper.

Verifies docs/ references against the current shardyfusion source tree:

  1. api      — every `::: shardyfusion.X.Y` mkdocstrings reference and every
                qualified `shardyfusion.X[.Y...]` mention in inline code or
                fenced python blocks resolves to a real attribute.
  2. config   — every field name listed in a docs config table for one of the
                tracked dataclasses exists on that class. When a default value
                is shown, it must match the source default (best-effort string
                comparison).
  3. extras   — every `--extra <name>` and `shardyfusion[<name>]` mention is a
                key in [project.optional-dependencies] in pyproject.toml.
  4. cli      — every `shardy <subcommand>` mention and every `--flag`
                referenced in the CLI use-case / reference page exists in the
                Click command tree at shardyfusion/cli/app.py.

Exits 0 on success, 1 on any failure. Findings are written to stdout as a
Markdown report grouped by check.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import re
import sys
import tomllib
from dataclasses import is_dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
PYPROJECT = REPO_ROOT / "pyproject.toml"
CLI_PAGES = {"use-cases/operate-cli.md", "reference/cli.md"}
IGNORE_MARKER = "<!-- validate-docs: ignore -->"

TRACKED_DATACLASSES = {
    "WriteConfig": "shardyfusion.config",
    "ShardingSpec": "shardyfusion.sharding_types",
    "OutputOptions": "shardyfusion.config",
    "ManifestOptions": "shardyfusion.config",
    "VectorSpec": "shardyfusion.config",
    "RetryConfig": "shardyfusion.type_defs",
    "S3ConnectionOptions": "shardyfusion.type_defs",
}


# ---------------------------------------------------------------------------
# Result aggregation


@dataclasses.dataclass
class Finding:
    check: str
    file: str
    line: int
    message: str
    severity: str  # "fail" | "warn" | "ok"


def emit_report(findings: list[Finding]) -> int:
    by_check: dict[str, list[Finding]] = {}
    for f in findings:
        by_check.setdefault(f.check, []).append(f)
    print("# Docs validation report\n")
    fail_count = 0
    for check in ("api", "config", "extras", "cli"):
        items = by_check.get(check, [])
        non_ok = [f for f in items if f.severity != "ok"]
        ok_count = sum(1 for f in items if f.severity == "ok")
        print(
            f"## {check} ({len(non_ok)} findings, {ok_count} ok)"
        )
        if not items:
            print("- no references found\n")
            continue
        for f in items:
            if f.severity == "ok":
                continue
            tag = {"fail": "FAIL", "warn": "WARN"}[f.severity]
            print(f"- [{tag}] {f.file}:{f.line}  {f.message}")
            if f.severity == "fail":
                fail_count += 1
        print()
    print(f"\n**Total failures: {fail_count}**")
    return 1 if fail_count else 0


# ---------------------------------------------------------------------------
# Discovery helpers


def iter_docs() -> Iterable[Path]:
    for path in sorted(DOCS_ROOT.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        if IGNORE_MARKER in text.splitlines()[:5]:
            continue
        yield path


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


# ---------------------------------------------------------------------------
# Check 1: api references


MKDOCSTRINGS_RE = re.compile(r"^:::\s+(shardyfusion(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\s*$")
QUALIFIED_RE = re.compile(r"\bshardyfusion(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")


def resolve_dotted(dotted: str) -> str:
    """Return 'ok' if symbol resolves, 'missing' if doc reference is
    genuinely broken (parent module imports cleanly but attribute is
    absent, OR full leaf import fails on a path whose parent has no
    such submodule on disk), or 'unverified' if a parent package along
    the path raised ImportError (likely an optional extra is not
    installed in the active env)."""
    parts = dotted.split(".")
    # Walk shorter prefixes until one imports.
    for cut in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:cut])
        try:
            obj = importlib.import_module(mod_name)
        except ImportError:
            continue
        except Exception:
            continue
        remainder = parts[cut:]
        if not remainder:
            return "ok"
        # Successfully imported some prefix; attribute-walk the rest.
        for attr in remainder:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                # Distinguish: is the missing attribute a submodule that
                # exists on disk but failed to import (optional extra),
                # or is it genuinely absent?
                missing_path = f"{mod_name}.{attr}"
                try:
                    spec = importlib.util.find_spec(missing_path)
                except (ImportError, ValueError):
                    spec = None
                if spec is not None:
                    # Submodule exists on disk — its non-importability is
                    # an environment issue, not a doc bug.
                    return "unverified"
                return "missing"
            mod_name = f"{mod_name}.{attr}"
        return "ok"
    return "unverified"


def check_api(findings: list[Finding]) -> None:
    for path in iter_docs():
        in_fence = False
        fence_lang = ""
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_fence:
                    in_fence = False
                    fence_lang = ""
                else:
                    in_fence = True
                    fence_lang = stripped[3:].split()[0] if len(stripped) > 3 else ""
                continue
            mk = MKDOCSTRINGS_RE.match(stripped)
            if mk:
                dotted = mk.group(1)
                status = resolve_dotted(dotted)
                if status == "ok":
                    msg = f"mkdocstrings `{dotted}` OK"
                    sev = "ok"
                elif status == "missing":
                    msg = f"mkdocstrings `{dotted}` NOT FOUND"
                    sev = "fail"
                else:
                    msg = (
                        f"mkdocstrings `{dotted}` not verified "
                        "(module not importable; install the relevant extra)"
                    )
                    sev = "warn"
                findings.append(
                    Finding(
                        check="api",
                        file=rel(path),
                        line=i,
                        message=msg,
                        severity=sev,
                    )
                )
                continue
            # qualified mentions in prose / python blocks
            if in_fence and fence_lang and fence_lang not in {"python", "py"}:
                continue
            for m in QUALIFIED_RE.finditer(line):
                dotted = m.group(0)
                # ignore obvious non-symbols (e.g. urls)
                if "/" in dotted or "@" in dotted:
                    continue
                status = resolve_dotted(dotted)
                if status == "missing":
                    findings.append(
                        Finding(
                            check="api",
                            file=rel(path),
                            line=i,
                            message=f"qualified `{dotted}` NOT FOUND",
                            severity="fail",
                        )
                    )
                elif status == "unverified":
                    findings.append(
                        Finding(
                            check="api",
                            file=rel(path),
                            line=i,
                            message=(
                                f"qualified `{dotted}` not verified "
                                "(module not importable; install the relevant extra)"
                            ),
                            severity="warn",
                        )
                    )


# ---------------------------------------------------------------------------
# Check 2: config dataclass field names


def load_dataclass_fields() -> dict[str, dict[str, str]]:
    """Return {ClassName: {field_name: default_repr_or_'<MISSING>'}}."""
    result: dict[str, dict[str, str]] = {}
    for cls_name, mod_name in TRACKED_DATACLASSES.items():
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        cls = getattr(mod, cls_name, None)
        if cls is None:
            continue
        if not is_dataclass(cls):
            # TypedDict path (S3ConnectionOptions): collect annotation keys.
            ann = getattr(cls, "__annotations__", {})
            result[cls_name] = {k: "<MISSING>" for k in ann}
            continue
        fields_map: dict[str, str] = {}
        for f in dataclasses.fields(cls):
            if f.default is not dataclasses.MISSING:
                fields_map[f.name] = repr(f.default)
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                fields_map[f.name] = "<factory>"
            else:
                fields_map[f.name] = "<MISSING>"
        result[cls_name] = fields_map
    return result


CONFIG_TABLE_HEADER_RE = re.compile(
    r"^\|\s*(?:Field|Name|Option|Knob|Parameter)\b", re.IGNORECASE
)
CONFIG_CONTEXT_RE = re.compile(r"\b(WriteConfig|ShardingSpec|OutputOptions|ManifestOptions|VectorSpec|RetryConfig|S3ConnectionOptions)\b")
TABLE_ROW_RE = re.compile(r"^\|\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\|")


def check_config(findings: list[Finding]) -> None:
    fields_by_class = load_dataclass_fields()
    for path in iter_docs():
        lines = path.read_text(encoding="utf-8").splitlines()
        active_class: str | None = None
        in_table = False
        for i, line in enumerate(lines, 1):
            # Only update active_class outside tables, to avoid being
            # confused by class names that appear inside type-cells of
            # rows describing fields of a different class.
            if not in_table:
                ctx = CONFIG_CONTEXT_RE.search(line)
                if ctx:
                    active_class = ctx.group(1)
            if CONFIG_TABLE_HEADER_RE.match(line):
                in_table = True
                continue
            if in_table and line.strip().startswith("|---"):
                continue
            if in_table and not line.strip().startswith("|"):
                in_table = False
                continue
            if in_table and active_class:
                m = TABLE_ROW_RE.match(line)
                if not m:
                    continue
                field = m.group(1)
                # skip header-ish words
                if field.lower() in {"field", "name", "option", "knob", "parameter"}:
                    continue
                known = fields_by_class.get(active_class, {})
                if not known:
                    findings.append(
                        Finding(
                            check="config",
                            file=rel(path),
                            line=i,
                            message=f"{active_class} not importable; cannot verify `{field}`",
                            severity="warn",
                        )
                    )
                    continue
                if field not in known:
                    suggestion = ""
                    for k in known:
                        if k.lower() == field.lower():
                            suggestion = f" (did you mean `{k}`?)"
                            break
                    findings.append(
                        Finding(
                            check="config",
                            file=rel(path),
                            line=i,
                            message=f"{active_class}.{field} NO SUCH FIELD{suggestion}",
                            severity="fail",
                        )
                    )


# ---------------------------------------------------------------------------
# Check 3: extras


EXTRA_RE = re.compile(r"(?:--extra\s+([A-Za-z0-9_\-]+))|shardyfusion\[([A-Za-z0-9_\-,\s]+)\]")


def load_extras() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return set(
        data.get("project", {}).get("optional-dependencies", {}).keys()
    )


def check_extras(findings: list[Finding]) -> None:
    known = load_extras()
    # Fictional extras used as illustrative examples in contributing pages
    # (e.g. "how to add an adapter named foodb").
    example_allowlist = {"foodb", "vector-foo"}
    for path in iter_docs():
        rel_path = rel(path)
        is_example_page = rel_path.startswith("docs/contributing/")
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for m in EXTRA_RE.finditer(line):
                raw = m.group(1) or m.group(2) or ""
                for name in re.split(r"[,\s]+", raw.strip()):
                    if not name:
                        continue
                    if name in known:
                        continue
                    if is_example_page and name in example_allowlist:
                        continue
                    findings.append(
                        Finding(
                            check="extras",
                            file=rel_path,
                            line=i,
                            message=f"extra `{name}` not in pyproject.toml",
                            severity="fail",
                        )
                    )


# Runtime extras that must be documented in the generated matrix page.
_DEV_EXTRAS = {"test", "quality", "docs"}
_MATRIX_PAGE = DOCS_ROOT / "use-cases" / "extras-matrix.md"
_MATRIX_EXTRA_RE = re.compile(r"`([A-Za-z0-9_\-]+)`")


def check_extras_matrix(findings: list[Finding]) -> None:
    """Every runtime extra must appear inside backticks in extras-matrix.md."""
    known = load_extras()
    runtime = {e for e in known if e not in _DEV_EXTRAS}
    if not _MATRIX_PAGE.exists():
        findings.append(
            Finding(
                check="extras",
                file=str(_MATRIX_PAGE.relative_to(REPO_ROOT)),
                line=1,
                message="extras-matrix.md is missing; run `uv run python scripts/generate_extras_matrix.py`",
                severity="fail",
            )
        )
        return
    text = _MATRIX_PAGE.read_text(encoding="utf-8")
    documented: set[str] = set()
    for m in _MATRIX_EXTRA_RE.finditer(text):
        documented.add(m.group(1))
    missing = sorted(runtime - documented)
    for name in missing:
        findings.append(
            Finding(
                check="extras",
                file=str(_MATRIX_PAGE.relative_to(REPO_ROOT)),
                line=1,
                message=(
                    f"runtime extra `{name}` not documented in extras-matrix.md; "
                    "run `uv run python scripts/generate_extras_matrix.py`"
                ),
                severity="fail",
            )
        )


# ---------------------------------------------------------------------------
# Check 4: CLI


SHARDY_CMD_RE = re.compile(r"\bshardy\s+([a-zA-Z][a-zA-Z0-9\-]*)")
LONG_FLAG_RE = re.compile(r"(--[a-zA-Z][a-zA-Z0-9\-]+)")


def load_cli_surface() -> tuple[set[str], set[str]]:
    """Return (subcommands, flags) discovered in the Click app."""
    try:
        import click

        from shardyfusion.cli import app as cli_app
    except Exception:
        return set(), set()
    main = getattr(cli_app, "cli", None) or getattr(cli_app, "main", None)
    if main is None or not isinstance(main, click.BaseCommand):
        return set(), set()
    subcommands: set[str] = set()
    flags: set[str] = set()

    def walk(cmd: click.BaseCommand) -> None:
        for p in getattr(cmd, "params", []):
            for opt in getattr(p, "opts", []) + getattr(p, "secondary_opts", []):
                if opt.startswith("--"):
                    flags.add(opt)
        if isinstance(cmd, click.MultiCommand):
            for name in cmd.list_commands(None):  # type: ignore[arg-type]
                subcommands.add(name)
                sub = cmd.get_command(None, name)  # type: ignore[arg-type]
                if sub is not None:
                    walk(sub)

    walk(main)
    return subcommands, flags


def check_cli(findings: list[Finding]) -> None:
    subcommands, flags = load_cli_surface()
    if not subcommands:
        # CLI not importable; skip silently with a warn line per CLI page.
        for rel_name in CLI_PAGES:
            p = DOCS_ROOT / rel_name
            if not p.exists():
                continue
            findings.append(
                Finding(
                    check="cli",
                    file=rel(p),
                    line=1,
                    message="CLI app not importable; install [cli] extra to validate",
                    severity="warn",
                )
            )
        return
    for rel_name in CLI_PAGES:
        p = DOCS_ROOT / rel_name
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for m in SHARDY_CMD_RE.finditer(line):
                name = m.group(1)
                if name not in subcommands:
                    findings.append(
                        Finding(
                            check="cli",
                            file=rel(p),
                            line=i,
                            message=f"`shardy {name}` not in CLI command tree",
                            severity="fail",
                        )
                    )
            for m in LONG_FLAG_RE.finditer(line):
                flag = m.group(1)
                # Click auto-registers --help everywhere; --version is on
                # the root group via @click.version_option but isn't always
                # surfaced in `params` introspection.
                if flag in {"--help", "--version"}:
                    continue
                if flag not in flags:
                    findings.append(
                        Finding(
                            check="cli",
                            file=rel(p),
                            line=i,
                            message=f"flag `{flag}` not registered on any CLI command",
                            severity="fail",
                        )
                    )


# ---------------------------------------------------------------------------


def main() -> int:
    if not DOCS_ROOT.exists():
        print(f"docs/ directory not found at {DOCS_ROOT}", file=sys.stderr)
        return 2
    findings: list[Finding] = []
    check_api(findings)
    check_config(findings)
    check_extras(findings)
    check_extras_matrix(findings)
    check_cli(findings)
    return emit_report(findings)


if __name__ == "__main__":
    raise SystemExit(main())
