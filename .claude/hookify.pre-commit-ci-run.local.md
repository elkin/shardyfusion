---
name: pre-commit-ci-run
enabled: true
event: bash
action: block
pattern: git\s+commit
---

**CI and E2E tests must pass before committing.**

Before proceeding with this commit, run the full test suite and fix any failures:

1. **Run `just ci`** — This executes quality checks (lint, format, type-check), unit tests, and integration tests in sequence. All must pass.

2. **Run `just d-e2e`** — This executes end-to-end tests against Garage S3 via compose. All must pass.

If any tests fail, diagnose and fix the failures before proceeding with the commit. Do not skip or ignore failing tests.
