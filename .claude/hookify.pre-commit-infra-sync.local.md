---
name: pre-commit-infra-sync
enabled: true
event: bash
action: block
pattern: git\s+commit
---

**Infrastructure version sync check required before committing.**

Before proceeding with this commit, verify that Docker, GitHub Actions, and tox all use consistent Python and Java versions:

1. **Python version consistency** — Verify that the same Python versions are used across:
   - `docker/ci.Dockerfile` (installed interpreters)
   - `tox.ini` (test matrix basepython entries)
   - `.github/workflows/ci.yml` (matrix strategy)
   - Currently expected: Python 3.11, 3.12, 3.13, 3.14

2. **GitHub Actions workflow consistency** — Verify that:
   - `.github/workflows/ci.yml` matrix matches Docker and tox Python versions
   - `.github/workflows/release.yml` uses a consistent base Python version
   - `.github/workflows/docs.yml` uses a consistent base Python version

3. **Java version consistency** — Verify that Java 17 (temurin) is used consistently across all Spark-requiring configurations:
   - `.github/workflows/ci.yml` (setup-java steps)
   - `docker/ci.Dockerfile` (Java installation)
   - `tox.ini` (any Java-related settings)

4. **Version drift** — Flag and fix any interpreter or runtime version drift between these files.

If any inconsistencies are found, fix them before proceeding with the commit.
