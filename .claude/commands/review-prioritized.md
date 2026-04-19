# Prioritized File-by-File Branch Review

Review all changes in the current branch compared to main.

## Step 1 — Triage

Run `git diff main...HEAD --stat` and classify every changed file into one of three risk tiers:

**CRITICAL** — assign this tier if the file:
- Handles data storage, serialization, or persistence
- Implements core algorithms or business logic
- Manages concurrency, locking, or coordination
- Touches networking, RPC, or protocol handling
- Modifies authentication, authorization, or security boundaries
- Changes configuration parsing or feature flags
- Is on a hot path (high-throughput, latency-sensitive)

**NORMAL** — assign this tier if the file:
- Adds or modifies tests
- Updates non-critical application logic
- Changes internal utilities used by critical code
- Modifies build or CI configuration

**LOW** — assign this tier if the file:
- Updates documentation only
- Changes comments, formatting, or style
- Modifies dev tooling, linting configs, or scripts
- Renames without logic changes

Print the triage table:

| File | Lines Changed | Risk Tier | Reason |
|------|--------------|-----------|--------|

Then proceed to review in order: all CRITICAL files first, then NORMAL, then LOW.

## Step 2 — Deep Review (per file)

For each file, run `git diff main...HEAD -- <filepath>` and analyze across ALL of the following dimensions. Skip dimensions that genuinely don't apply to the file, but err on the side of including rather than skipping.

### Correctness
- **Bugs**: Logic errors, off-by-one, race conditions, nil/null derefs, resource leaks, deadlocks, incorrect error handling, use-after-close, integer overflow
- **Contracts**: Pre/post-condition violations, broken type invariants, API contract changes (breaking or subtle), implicit assumptions not enforced

### Testing
- **Test quality**: Are assertions meaningful? Are tests brittle or robust? Do they fail when the code is wrong? Is mocking appropriate?
- **Test coverage**: Untested new/changed code paths, untested edge cases (empty, boundary, error, concurrent), modified logic whose tests don't cover new behavior

### Reliability & Operations
- **Failure modes**: Behavior under dependency failure, retry storms, cascading failures, missing timeouts, partial failure handling, graceful degradation
- **Observability**: Missing metrics/logs/traces for new paths, silent failures, insufficient error context for production diagnosis

### Performance
- **Hot paths**: Unnecessary allocations, hidden quadratic complexity, lock contention, unbounded growth, excessive copying, missing caching on critical paths

### Documentation
- **Accuracy**: Stale comments or docs that no longer match the code
- **Gaps**: Undocumented public APIs, config options, behaviors, or complex logic lacking explanation

### Risk
- **Production risk**: Rollback safety, feature flag needs, data migration concerns, backward/forward compatibility, blast radius
- **Quick wins**: Low-effort improvements worth making now — naming, dead code, deduplication, simplification

### Per-file output format

For each file, print:

```
=== path/to/file.ext [CRITICAL|NORMAL|LOW] ===
```

Then list findings. Each finding must have:
- **Severity**: `CRITICAL` / `WARNING` / `SUGGESTION`
- **Line**: exact line number or range
- **Category**: which dimension (e.g., Bugs, Test coverage, Failure modes)
- **Finding**: what the problem is and why it matters
- **Recommendation**: concrete fix or direction

If a file has no findings, print: `No issues found.`

Spend the most effort and token budget on CRITICAL files. For LOW-risk files, a quick scan is sufficient.

## Step 3 — Summary

After all files are reviewed, produce:

### Findings by severity

| Severity | Count |
|----------|-------|
| CRITICAL | N |
| WARNING  | N |
| SUGGESTION | N |

### All CRITICAL findings (repeated for quick scanning)

List every CRITICAL finding with its file:line for easy reference.

### Deployment checklist

Concrete items to verify before and after shipping, derived from the findings.

### Overall risk rating

Rate the changeset: **LOW / MEDIUM / HIGH / CRITICAL**

Provide a one-paragraph justification covering the most significant risks and whether the test coverage is adequate for the changes made.
