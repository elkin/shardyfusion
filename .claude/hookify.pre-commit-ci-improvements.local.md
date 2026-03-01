---
name: pre-commit-ci-improvements
enabled: true
event: bash
action: warn
pattern: git\s+commit
---

**Consider CI/validation improvements for these changes.**

Review the changes being committed and suggest improvements only with concrete justification:

- **New dependency category** — If a new optional extra or dependency group was added, suggest adding an import smoke-test in CI.
- **New file type** — If a new file type was introduced (e.g., YAML configs, SQL migrations), suggest adding an appropriate linter or validator.
- **New writer path** — If a new writer implementation was added, suggest adding corresponding tox environments, routing contract tests, and integration test scenarios.
- **New test markers** — If new pytest markers were introduced, suggest adding them to CI matrix configurations.
- **Type-checking gaps** — If new modules were added, suggest adding pyright project configs if they aren't covered by existing ones.

This is advisory only. Proceed with the commit if no improvements are warranted.
