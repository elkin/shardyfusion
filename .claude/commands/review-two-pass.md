# Two-Pass Branch Review

Review all changes in the current branch compared to main using `git diff main...HEAD`.

## Pass Selection

This command runs in two passes. Check which pass is requested via the $ARGUMENTS variable:
- If `$ARGUMENTS` is "1" or "correctness" or empty → run **Pass 1**
- If `$ARGUMENTS` is "2" or "ops" → run **Pass 2**
- If `$ARGUMENTS` is "full" → run both passes sequentially

---

## Pass 1 — Correctness & Quality

Go file by file through the diff. For each file, analyze:

1. **Bugs**: Logic errors, off-by-one errors, race conditions, nil/null dereferences, resource leaks (unclosed handles, missing defers/finally), integer overflow, incorrect error handling (swallowed errors, wrong error propagated), deadlocks, use-after-free or use-after-close patterns.

2. **Contracts**: Function pre/post-condition violations, type invariant breakage, API contract changes (breaking or subtly incompatible), implicit assumptions that aren't enforced, violated LSP or interface contracts, changed semantics without updated callers.

3. **Test quality**: Do new/modified tests assert meaningful behavior or just exercise code? Are tests brittle (tied to implementation details, ordering, timing)? Are test names descriptive? Is test setup/teardown correct? Are mocks appropriate or over-used? Do tests actually fail when the code is wrong?

4. **Test coverage gaps**: New/changed code paths with no test coverage. Untested edge cases (empty input, boundary values, error paths, concurrent access). Untested combinations of conditions. Modified logic whose existing tests don't cover the new behavior.

5. **Quick wins**: Obvious improvements that are low-effort and high-value — naming improvements, dead code removal, duplicated logic that should be extracted, overly complex expressions that can be simplified, missing early returns, unnecessary nesting.

### Output format for Pass 1

For every finding, provide:
- **Severity**: `CRITICAL` / `WARNING` / `SUGGESTION`
- **File and line**: exact `file:line` reference
- **Category**: which of the 5 dimensions above
- **Description**: what the problem is and why it matters
- **Recommendation**: concrete fix or direction

End with a summary table:

| Severity | Count |
|----------|-------|
| CRITICAL | N |
| WARNING | N |
| SUGGESTION | N |

Followed by a list of all CRITICAL findings repeated for quick scanning.

---

## Pass 2 — Reliability, Operations & Performance

Go file by file through the same diff. For each file, analyze:

1. **Reliability & failure modes**: How does this code behave when dependencies fail? Are there retry storms, cascading failure risks, or missing circuit breakers? Is there proper timeout propagation? Are partial failures handled (half-written state, torn reads)? Is there proper graceful degradation?

2. **Observability**: Are new code paths observable? Missing metrics, logs, or trace spans for important operations. Are errors logged with sufficient context? Can you diagnose a production issue from the telemetry this code emits? Are there silent failures?

3. **Hot path performance**: Unnecessary heap allocations on critical paths, O(n²) or worse hiding in loops, lock contention under concurrency, missing caching or memoization opportunities, unbounded growth (maps/slices that grow without limits), excessive copying of large structures, inefficient serialization/deserialization.

4. **Documentation state**: Are existing comments, docstrings, READMEs, and inline docs still accurate after these changes? Are there stale comments that now describe the wrong behavior? Do architectural docs (if any) need updating?

5. **Documentation gaps**: New public APIs, exported functions, configuration options, environment variables, CLI flags, or behaviors that lack documentation. Complex logic that needs explanatory comments. Missing or incomplete changelog entries.

6. **Production risk assessment**: What could go wrong when this ships? Is this change rollback-safe? Does it need a feature flag? Are there data migration risks? Could it cause a thundering herd on deploy? Are there backward/forward compatibility concerns for data formats or APIs? What is the blast radius if this change has a bug?

### Output format for Pass 2

Same format as Pass 1 — severity, file:line, category, description, recommendation.

End with:
1. The same summary table by severity
2. A **deployment checklist** — concrete items to verify before and after shipping these changes
3. An **overall risk rating** for the changeset: LOW / MEDIUM / HIGH / CRITICAL, with a one-paragraph justification
