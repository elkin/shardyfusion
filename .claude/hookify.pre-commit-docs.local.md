---
name: pre-commit-docs
enabled: true
event: bash
action: warn
pattern: git\s+commit
---

**Documentation sync check required before committing.**

Before proceeding with this commit, verify that documentation is consistent with the code changes being committed:

1. **README.md** — Verify it reflects the current public API, install instructions, usage examples, and development workflow. Check that any new features, changed commands, or modified APIs are documented.

2. **CLAUDE.md** — Verify it reflects the current module structure, commands, architecture, testing notes, and error hierarchy. Check that:
   - The "Architecture" section matches actual module dependencies
   - The "Commands" section lists current build/test/lint commands
   - The "Error Hierarchy" table includes all error classes
   - The "Public API Summary" lists all exports from `__init__.py`
   - The "Testing Notes" section covers any new test directories or markers

3. **AGENTS.md** — Verify it reflects the current project structure, build commands, test commands, and coding style.

4. **docs/ directory** — Check that MkDocs source files are consistent with code changes (API reference, guides, configuration docs).

If any documentation is stale or missing coverage for the changes being committed, update it before proceeding with the commit.
