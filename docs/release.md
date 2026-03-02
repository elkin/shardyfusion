# Release Process

## One-time setup

Configure PyPI Trusted Publishing for this repository/workflow.

The release workflow uses GitHub OIDC (`id-token`) and does not require storing a long-lived PyPI token.

## Cut a release

1. Bump version in `pyproject.toml`:

```bash
uv version 0.1.1
```

2. Commit and merge to `main`.
3. Create and push a version tag:

```bash
git tag v0.1.1
git push origin v0.1.1
```

4. `Release` workflow runs:
   - lint/style/type checks
   - unit + integration checks
   - `uv build`
   - `uv publish --trusted-publishing always`

## Verify

```bash
uv pip install shardyfusion==0.1.1
uv run python -c "import shardyfusion; print('ok')"
```

## If release fails

- fix the issue and push a new tag version (recommended)
- avoid reusing the same version number after partial publish
