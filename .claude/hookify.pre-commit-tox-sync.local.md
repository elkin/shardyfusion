---
name: pre-commit-tox-sync
enabled: true
event: bash
action: warn
pattern: git\s+commit
---

**Tox configuration sync check required before committing.**

Before proceeding with this commit, verify that `tox.ini` is consistent with the test suite:

1. **Test directory coverage** — Verify all `tests/` subdirectories (unit, integration, e2e, and their children like `tests/unit/cli/`, `tests/unit/writer/spark/`, `tests/unit/writer/dask/`, `tests/unit/writer/python/`, etc.) have corresponding tox environments.

2. **Stale environments** — Check for tox environments that reference removed or renamed test paths. Remove any stale environments.

3. **Label completeness** — Verify tox labels (`quality`, `unit`, `integration`, `e2e`) include all relevant environments. New test directories or writer paths should be covered by appropriate labels.

4. **New environments** — If new test directories or writer paths were added in this commit, add corresponding tox environments and include them in the appropriate labels.

If any inconsistencies are found, fix `tox.ini` before proceeding with the commit.
