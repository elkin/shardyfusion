# 2026-03-15 Reliable Loser Cleanup Plan

- Status: `plan`
- Date: `2026-03-15`
- Original location: `docs/superpowers/plans/2026-03-15-reliable-loser-cleanup.md`
- Commit metadata: unavailable; this note existed only in the working tree before it was moved under `Engineering Notes`

**Goal:** Make post-write cleanup of losing shard attempts reliable by adding retries to `delete_prefix`, persisting failed cleanup URLs to S3, and providing a `shardy cleanup` CLI command for deferred retry.

**Architecture:** Three layers — (1) `delete_prefix` gains retry support via the existing `_retry_s3_operation` helper, (2) `cleanup_losers` persists any remaining failures to a `_cleanup/` S3 prefix as small JSON records, (3) a new `shardy cleanup` CLI command processes those records. This lets the write pipeline succeed without blocking on cleanup, while providing an operator-accessible deferred retry path. A short lifecycle rule on `_cleanup/` auto-expires stale records.

**Tech Stack:** Python 3.11+, boto3 (S3), click (CLI), pytest + unittest.mock (tests)

---

## File Structure

| File | Responsibility |
|---|---|
| `shardyfusion/storage.py` | Add retry support to `delete_prefix` |
| `shardyfusion/_writer_core.py` | Extend `cleanup_losers` to persist failures; add `CleanupResult` |
| `shardyfusion/metrics/_events.py` | Add `CLEANUP_COMPLETED` and `CLEANUP_RECORD_WRITTEN` events |
| `shardyfusion/cli/app.py` | New `shardy cleanup` subcommand |
| `tests/unit/shared/test_storage.py` | Tests for `delete_prefix` retry behavior |
| `tests/unit/writer/test_writer_core_resilience.py` | Tests for `cleanup_losers` persistence |
| `tests/unit/cli/test_app.py` | Tests for `shardy cleanup` CLI command |

---

## Chunk 1: `delete_prefix` with retries

### Task 1: Add retry support to `delete_prefix`

`delete_prefix` currently has zero retries — a single transient S3 error abandons the entire deletion. The existing `_retry_s3_operation` helper handles exponential backoff for transient errors and is already used by `put_bytes` and `get_bytes`. We'll use it to wrap each S3 API call inside `delete_prefix`.

**Files:**
- Modify: `shardyfusion/storage.py:306-357` (`delete_prefix`)
- Test: `tests/unit/shared/test_storage.py`

- [ ] **Step 1: Write failing test — transient error retried during delete**

In `tests/unit/shared/test_storage.py`, add:

```python
class TestDeletePrefixRetry:
    """delete_prefix retries transient S3 errors."""

    def test_retries_transient_list_error(self, monkeypatch):
        """Transient error on list_objects_v2 is retried."""
        monkeypatch.setattr("shardyfusion.storage.time.sleep", lambda _: None)
        calls = 0

        class FakeClient:
            def list_objects_v2(self, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise _FakeTransientS3Error()
                return {"Contents": [{"Key": "prefix/obj1"}], "IsTruncated": False}

            def delete_objects(self, **kwargs):
                pass

        from shardyfusion.storage import delete_prefix

        deleted = delete_prefix("s3://bucket/prefix/", s3_client=FakeClient())
        assert deleted == 1
        assert calls == 2  # first call failed, second succeeded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/shared/test_storage.py::TestDeletePrefixRetry::test_retries_transient_list_error -v`
Expected: FAIL — `delete_prefix` catches the exception and returns 0 instead of retrying.

- [ ] **Step 3: Write failing test — transient error retried during delete_objects**

```python
    def test_retries_transient_delete_error(self, monkeypatch):
        """Transient error on delete_objects is retried."""
        monkeypatch.setattr("shardyfusion.storage.time.sleep", lambda _: None)
        delete_calls = 0

        class FakeClient:
            def list_objects_v2(self, **kwargs):
                return {"Contents": [{"Key": "prefix/obj1"}], "IsTruncated": False}

            def delete_objects(self, **kwargs):
                nonlocal delete_calls
                delete_calls += 1
                if delete_calls == 1:
                    raise _FakeTransientS3Error()

        from shardyfusion.storage import delete_prefix

        deleted = delete_prefix("s3://bucket/prefix/", s3_client=FakeClient())
        assert deleted == 1
        assert delete_calls == 2
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/unit/shared/test_storage.py::TestDeletePrefixRetry::test_retries_transient_delete_error -v`
Expected: FAIL

- [ ] **Step 5: Write failing test — non-transient error not retried**

```python
    def test_non_transient_error_not_retried(self, monkeypatch):
        """Non-transient error (e.g. AccessDenied) is not retried."""
        monkeypatch.setattr("shardyfusion.storage.time.sleep", lambda _: None)

        class FakeClient:
            def list_objects_v2(self, **kwargs):
                exc = Exception("AccessDenied")
                exc.response = {"Error": {"Code": "AccessDenied"}}
                raise exc

        from shardyfusion.storage import delete_prefix

        # Non-transient: returns 0, does not retry
        deleted = delete_prefix("s3://bucket/prefix/", s3_client=FakeClient())
        assert deleted == 0
```

- [ ] **Step 6: Run test to verify it fails or passes**

Run: `uv run pytest tests/unit/shared/test_storage.py::TestDeletePrefixRetry::test_non_transient_error_not_retried -v`
Expected: May pass since current code also returns 0 — that's fine, it confirms the contract.

- [ ] **Step 7: Write failing test — retry_config is respected**

```python
    def test_retry_config_respected(self, monkeypatch):
        """Custom RetryConfig controls retry count."""
        monkeypatch.setattr("shardyfusion.storage.time.sleep", lambda _: None)
        calls = 0

        class FakeClient:
            def list_objects_v2(self, **kwargs):
                nonlocal calls
                calls += 1
                raise _FakeTransientS3Error()

        from shardyfusion.storage import delete_prefix
        from shardyfusion.type_defs import RetryConfig

        deleted = delete_prefix(
            "s3://bucket/prefix/",
            s3_client=FakeClient(),
            retry_config=RetryConfig(max_retries=1),
        )
        assert deleted == 0
        assert calls == 2  # initial + 1 retry
```

- [ ] **Step 8: Run test to verify it fails**

Run: `uv run pytest tests/unit/shared/test_storage.py::TestDeletePrefixRetry::test_retry_config_respected -v`
Expected: FAIL — `delete_prefix` doesn't accept `retry_config` yet.

- [ ] **Step 9: Implement `delete_prefix` with retries**

Refactor `delete_prefix` in `shardyfusion/storage.py:306-357`. The key change: wrap each page of list+delete in `_retry_s3_operation`. Add `retry_config` parameter.

```python
def delete_prefix(
    prefix_url: str,
    *,
    s3_client: Any = None,
    metrics_collector: MetricsCollector | None = None,
    retry_config: RetryConfig | None = None,
) -> int:
    """Delete all objects under an S3 prefix. Returns the number of objects deleted.

    Best-effort: logs errors but does not raise on transient failures.
    Uses exponential backoff for transient S3 errors (default: 3 retries).
    """
    client = s3_client or create_s3_client()
    bucket, key_prefix = parse_s3_url(prefix_url)

    deleted = 0
    continuation_token: str | None = None

    try:
        while True:
            list_kwargs: dict[str, Any] = {
                "Bucket": bucket,
                "Prefix": key_prefix,
                "MaxKeys": 1000,
            }
            if continuation_token is not None:
                list_kwargs["ContinuationToken"] = continuation_token

            response = _retry_s3_operation(
                lambda: client.list_objects_v2(**list_kwargs),
                operation_name="list_objects_v2",
                url=prefix_url,
                metrics_collector=metrics_collector,
                retry_config=retry_config,
            )
            contents = response.get("Contents", [])
            if not contents:
                break

            objects = [{"Key": obj["Key"]} for obj in contents]
            _retry_s3_operation(
                lambda: client.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": objects, "Quiet": True},
                ),
                operation_name="delete_objects",
                url=prefix_url,
                metrics_collector=metrics_collector,
                retry_config=retry_config,
            )
            deleted += len(objects)

            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
    except Exception as exc:
        log_failure(
            "s3_delete_prefix_failed",
            severity=FailureSeverity.ERROR,
            logger=_logger,
            error=exc,
            prefix_url=prefix_url,
            deleted_so_far=deleted,
        )

    return deleted
```

**Important:** The lambdas in `_retry_s3_operation` must capture their arguments by value at call time to avoid the late-binding closure pitfall. In the loop body, `list_kwargs`, `objects`, `bucket` are all defined before the lambda, so this is safe. However, the `objects` lambda must capture the current `objects` list — since we reassign `objects` each iteration, the lambda binds to the local variable which is correct per iteration.

- [ ] **Step 10: Run all retry tests**

Run: `uv run pytest tests/unit/shared/test_storage.py -v`
Expected: All pass

- [ ] **Step 11: Run full unit suite for regressions**

Run: `uv run pytest tests/unit/ -x -q`
Expected: All pass, no regressions

- [ ] **Step 12: Commit**

```bash
git add shardyfusion/storage.py tests/unit/shared/test_storage.py
git commit -m "feat: add retry support to delete_prefix for transient S3 errors"
```

---

## Chunk 2: `cleanup_losers` persists failures + `CleanupResult`

### Task 2: Add `CleanupResult` dataclass and persist failures

When `cleanup_losers` successfully deletes some losers but fails on others, it should persist the failed URLs to `_cleanup/run_id=.../pending.json` so a separate job can retry.

**Files:**
- Modify: `shardyfusion/_writer_core.py:285-332` (`cleanup_losers`)
- Modify: `shardyfusion/metrics/_events.py` (add events)
- Test: `tests/unit/writer/test_writer_core_resilience.py`

- [ ] **Step 1: Add metric events**

In `shardyfusion/metrics/_events.py`, add after `BATCH_WRITTEN`:

```python
    CLEANUP_COMPLETED = "write.cleanup_completed"
    CLEANUP_RECORD_WRITTEN = "write.cleanup_record_written"
```

- [ ] **Step 2: Run lint to verify**

Run: `uv run ruff check shardyfusion/metrics/_events.py`
Expected: Clean

- [ ] **Step 3: Write failing test — `CleanupResult` returned with deleted/failed counts**

In `tests/unit/writer/test_writer_core_resilience.py`, add:

```python
from shardyfusion._writer_core import CleanupResult


class TestCleanupLosersResult:
    """cleanup_losers returns a CleanupResult with deleted/failed breakdown."""

    @patch("shardyfusion.storage.delete_prefix")
    @patch("shardyfusion.storage.create_s3_client")
    def test_returns_cleanup_result(
        self, mock_create_client: MagicMock, mock_delete: MagicMock
    ) -> None:
        mock_delete.return_value = 5
        mock_create_client.return_value = MagicMock()

        winners = [_winner(0)]
        all_urls = [
            "s3://b/p/db=00000/attempt=00",
            "s3://b/p/db=00000/attempt=01",
        ]

        result = cleanup_losers(all_urls, winners)
        assert isinstance(result, CleanupResult)
        assert result.deleted == 5
        assert result.failed_urls == []
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/unit/writer/test_writer_core_resilience.py::TestCleanupLosersResult::test_returns_cleanup_result -v`
Expected: FAIL — `cleanup_losers` returns `int`, not `CleanupResult`

- [ ] **Step 5: Write failing test — failed URLs tracked in result**

```python
    @patch("shardyfusion.storage.delete_prefix")
    @patch("shardyfusion.storage.create_s3_client")
    def test_failed_urls_tracked(
        self, mock_create_client: MagicMock, mock_delete: MagicMock
    ) -> None:
        """URLs where delete_prefix returns 0 are tracked as failed."""
        mock_delete.side_effect = [5, 0]  # first loser cleaned, second failed
        mock_create_client.return_value = MagicMock()

        winners = [_winner(0)]
        all_urls = [
            "s3://b/p/db=00000/attempt=00",  # winner
            "s3://b/p/db=00000/attempt=01",  # loser — cleaned (5 objects)
            "s3://b/p/db=00000/attempt=02",  # loser — failed (0 objects)
        ]

        result = cleanup_losers(all_urls, winners)
        assert result.deleted == 5
        assert result.failed_urls == ["s3://b/p/db=00000/attempt=02"]
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/unit/writer/test_writer_core_resilience.py::TestCleanupLosersResult::test_failed_urls_tracked -v`
Expected: FAIL

- [ ] **Step 7: Write failing test — pending record written to S3**

```python
    @patch("shardyfusion.storage.put_bytes")
    @patch("shardyfusion.storage.delete_prefix")
    @patch("shardyfusion.storage.create_s3_client")
    def test_pending_record_written_on_failure(
        self, mock_create_client: MagicMock, mock_delete: MagicMock, mock_put: MagicMock
    ) -> None:
        """When losers fail to delete, a pending cleanup record is written to S3."""
        mock_delete.return_value = 0  # all deletes fail
        mock_create_client.return_value = MagicMock()

        winners = [_winner(0)]
        all_urls = [
            "s3://b/p/db=00000/attempt=00",
            "s3://b/p/db=00000/attempt=01",
        ]

        cleanup_losers(
            all_urls,
            winners,
            s3_prefix="s3://b/p",
            run_id="test-run-123",
        )
        mock_put.assert_called_once()
        call_args = mock_put.call_args
        assert "_cleanup/" in call_args.args[0]
        assert "test-run-123" in call_args.args[0]
```

- [ ] **Step 8: Run test to verify it fails**

Run: `uv run pytest tests/unit/writer/test_writer_core_resilience.py::TestCleanupLosersResult::test_pending_record_written_on_failure -v`
Expected: FAIL

- [ ] **Step 9: Write failing test — no pending record when all deletes succeed**

```python
    @patch("shardyfusion.storage.put_bytes")
    @patch("shardyfusion.storage.delete_prefix")
    @patch("shardyfusion.storage.create_s3_client")
    def test_no_pending_record_when_all_succeed(
        self, mock_create_client: MagicMock, mock_delete: MagicMock, mock_put: MagicMock
    ) -> None:
        """No pending record written when all losers are successfully cleaned."""
        mock_delete.return_value = 5
        mock_create_client.return_value = MagicMock()

        winners = [_winner(0)]
        all_urls = [
            "s3://b/p/db=00000/attempt=00",
            "s3://b/p/db=00000/attempt=01",
        ]

        cleanup_losers(
            all_urls,
            winners,
            s3_prefix="s3://b/p",
            run_id="test-run-123",
        )
        mock_put.assert_not_called()
```

- [ ] **Step 10: Run test to verify it fails**

Run: `uv run pytest tests/unit/writer/test_writer_core_resilience.py::TestCleanupLosersResult -v`
Expected: All FAIL

- [ ] **Step 11: Implement `CleanupResult` and updated `cleanup_losers`**

In `shardyfusion/_writer_core.py`, add `CleanupResult` dataclass and update `cleanup_losers`:

```python
@dataclass(slots=True)
class CleanupResult:
    """Result of a cleanup_losers operation."""

    deleted: int
    failed_urls: list[str]


def cleanup_losers(
    all_attempt_urls: Sequence[str],
    winners: list[RequiredShardMeta],
    *,
    s3_client: Any | None = None,
    metrics_collector: MetricsCollector | None = None,
    s3_prefix: str | None = None,
    run_id: str | None = None,
) -> CleanupResult:
    """Delete temp databases for non-winning attempts.

    Best-effort: logs errors but does not raise.

    When s3_prefix and run_id are provided, any URLs that fail to delete
    are persisted to ``s3_prefix/_cleanup/run_id=.../pending.json`` for
    deferred retry (e.g. via ``shardy cleanup``).

    Returns:
        CleanupResult with counts of deleted objects and list of failed URLs.
    """
    from .storage import create_s3_client, delete_prefix, put_bytes

    winner_urls = {w.db_url for w in winners}
    total_deleted = 0
    num_losers = 0
    failed_urls: list[str] = []
    client = s3_client or create_s3_client()

    for url in all_attempt_urls:
        if url not in winner_urls:
            num_losers += 1
            deleted = delete_prefix(
                url,
                s3_client=client,
                metrics_collector=metrics_collector,
            )
            total_deleted += deleted
            if deleted > 0:
                log_event(
                    "loser_attempt_cleaned",
                    level=logging.DEBUG,
                    logger=_logger,
                    db_url=url,
                    objects_deleted=deleted,
                )
            else:
                failed_urls.append(url)

    if total_deleted > 0:
        log_event(
            "losers_cleanup_completed",
            logger=_logger,
            total_objects_deleted=total_deleted,
            num_losers=num_losers,
        )

    if metrics_collector is not None:
        metrics_collector.emit(
            MetricEvent.CLEANUP_COMPLETED,
            {
                "total_deleted": total_deleted,
                "num_losers": num_losers,
                "num_failed": len(failed_urls),
            },
        )

    # Persist failed URLs for deferred retry
    if failed_urls and s3_prefix is not None and run_id is not None:
        _write_cleanup_record(
            s3_prefix=s3_prefix,
            run_id=run_id,
            failed_urls=failed_urls,
            s3_client=client,
            metrics_collector=metrics_collector,
        )

    return CleanupResult(deleted=total_deleted, failed_urls=failed_urls)


def _write_cleanup_record(
    *,
    s3_prefix: str,
    run_id: str,
    failed_urls: list[str],
    s3_client: Any,
    metrics_collector: MetricsCollector | None = None,
) -> None:
    """Persist failed cleanup URLs to S3 for deferred retry. Best-effort."""
    import json

    from .storage import join_s3, put_bytes

    record = {
        "run_id": run_id,
        "failed_urls": failed_urls,
        "created_at": _utc_now().isoformat(),
    }
    record_url = join_s3(s3_prefix, "_cleanup", f"run_id={run_id}", "pending.json")

    try:
        put_bytes(
            record_url,
            json.dumps(record).encode(),
            "application/json",
            s3_client=s3_client,
        )
        log_event(
            "cleanup_record_written",
            logger=_logger,
            record_url=record_url,
            num_failed=len(failed_urls),
        )
        if metrics_collector is not None:
            metrics_collector.emit(
                MetricEvent.CLEANUP_RECORD_WRITTEN,
                {"num_failed": len(failed_urls)},
            )
    except Exception:
        # Best-effort: log but don't raise
        log_failure(
            "cleanup_record_write_failed",
            severity=FailureSeverity.ERROR,
            logger=_logger,
            record_url=record_url,
            num_failed=len(failed_urls),
        )
```

Add `CleanupResult` to imports in `__init__.py` if it's part of the public API, or keep it internal.

- [ ] **Step 12: Update callers of `cleanup_losers` in all 7 writer files**

Each writer currently calls `cleanup_losers(...)` without capturing or using the return value. Two changes per writer:

1. Pass `s3_prefix=config.s3_prefix` and `run_id=run_id` to `cleanup_losers` so it can persist failures.
2. No need to change the return value handling — `CleanupResult` replaces the `int` return, but callers don't use it.

Files to update (the `cleanup_losers(...)` call in each):
- `shardyfusion/writer/spark/writer.py` — add `s3_prefix=config.s3_prefix, run_id=run_id`
- `shardyfusion/writer/dask/writer.py` — same
- `shardyfusion/writer/ray/writer.py` — same
- `shardyfusion/writer/python/writer.py` — same
- `shardyfusion/writer/spark/single_db_writer.py` — same
- `shardyfusion/writer/dask/single_db_writer.py` — same
- `shardyfusion/writer/ray/single_db_writer.py` — same

Example (Spark sharded writer):
```python
        cleanup_losers(
            write_outcome.all_attempt_urls,
            write_outcome.winners,
            metrics_collector=mc,
            s3_prefix=config.s3_prefix,
            run_id=run_id,
        )
```

- [ ] **Step 13: Update existing `TestCleanupLosers` tests**

The existing tests in `TestCleanupLosers` assert `cleanup_losers` returns `int`. Update them to expect `CleanupResult`:

- `assert deleted == 10` → `assert result.deleted == 10` (rename `deleted` → `result`)
- `assert deleted == 0` → `assert result.deleted == 0`

- [ ] **Step 14: Run all tests**

Run: `uv run pytest tests/unit/writer/test_writer_core_resilience.py -v`
Expected: All pass

- [ ] **Step 15: Run lint and type check**

Run: `uv run ruff check shardyfusion/_writer_core.py shardyfusion/metrics/_events.py && uv run pyright shardyfusion/_writer_core.py`
Expected: Clean

- [ ] **Step 16: Run full unit suite for regressions**

Run: `uv run pytest tests/unit/ -x -q`
Expected: All pass

- [ ] **Step 17: Commit**

```bash
git add shardyfusion/_writer_core.py shardyfusion/metrics/_events.py shardyfusion/writer/ tests/unit/writer/test_writer_core_resilience.py
git commit -m "feat: cleanup_losers returns CleanupResult and persists failures to S3"
```

---

## Chunk 3: `shardy cleanup` CLI command

### Task 3: Add `shardy cleanup` subcommand

A new CLI command that lists pending cleanup records under `_cleanup/` and retries the deletions.

**Files:**
- Modify: `shardyfusion/cli/app.py` (add `cleanup` command)
- Test: `tests/unit/cli/test_app.py`

- [ ] **Step 1: Write failing test — `shardy cleanup` lists and processes pending records**

In `tests/unit/cli/test_app.py`, add a test class. First check how existing CLI tests are structured.

Read `tests/unit/cli/test_app.py` to understand the test patterns (click `CliRunner`, fixture setup, etc.). Match the existing style.

The test should:
1. Set up a fake S3 with a `_cleanup/run_id=.../pending.json` record
2. Invoke `shardy cleanup --s3-prefix s3://bucket/prefix`
3. Assert the pending record's URLs were deleted
4. Assert the pending record itself was removed after successful cleanup

```python
class TestCleanupCommand:
    """Tests for the `shardy cleanup` CLI command."""

    def test_processes_pending_records(self, ...):
        """cleanup command retries pending deletions."""
        # Test body depends on existing CLI test patterns — match them.
        ...

    def test_no_pending_records(self, ...):
        """cleanup command with no pending records exits cleanly."""
        ...

    def test_dry_run(self, ...):
        """cleanup --dry-run lists pending records without deleting."""
        ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/cli/test_app.py::TestCleanupCommand -v`
Expected: FAIL — command doesn't exist yet.

- [ ] **Step 3: Add `"cleanup"` to the early-return guard in `cli()` group**

In `shardyfusion/cli/app.py`, the `cli()` group function (line 229) has an early return for `"schema"` to skip reader/S3 config loading. Add `"cleanup"` to this guard so the command works standalone without a reader config:

```python
    # schema and cleanup subcommands need no reader/S3 setup
    if ctx.invoked_subcommand in {"schema", "cleanup"}:
        return
```

- [ ] **Step 4: Implement `shardy cleanup` command**

In `shardyfusion/cli/app.py`, add the `cleanup` subcommand. Key design decisions:
- Uses its own `--s3-prefix` option (not the group's reader config) since it operates standalone.
- Uses `try_get_bytes` (not `get_bytes`) to gracefully handle records deleted by lifecycle rules between listing and reading.
- Accepts `--s3-option` for endpoint/credentials (same pattern as the group options).

```python
@cli.command("cleanup")
@click.option(
    "--s3-prefix",
    required=True,
    help="S3 prefix to scan for pending cleanup records.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="List pending records without deleting.",
)
@click.option(
    "--endpoint-url",
    default=None,
    help="S3 endpoint URL (for non-AWS providers).",
)
def cleanup_cmd(
    s3_prefix: str,
    dry_run: bool,
    endpoint_url: str | None,
) -> None:
    """Retry failed loser cleanup from previous writes.

    Scans _cleanup/ under the given S3 prefix for pending.json records
    and retries the deletions. Successfully cleaned records are removed.
    """
    import json

    from ..storage import create_s3_client, delete_prefix, parse_s3_url, try_get_bytes

    connection_options = {"endpoint_url": endpoint_url} if endpoint_url else None
    client = create_s3_client(connection_options=connection_options)

    cleanup_prefix = s3_prefix.rstrip("/") + "/_cleanup/"
    bucket, key_prefix = parse_s3_url(cleanup_prefix)

    # List all pending.json files under _cleanup/
    paginator_token = None
    pending_records: list[str] = []
    while True:
        list_kwargs: dict[str, Any] = {
            "Bucket": bucket, "Prefix": key_prefix, "MaxKeys": 1000,
        }
        if paginator_token:
            list_kwargs["ContinuationToken"] = paginator_token
        response = client.list_objects_v2(**list_kwargs)
        for obj in response.get("Contents", []):
            if obj["Key"].endswith("pending.json"):
                pending_records.append(obj["Key"])
        if not response.get("IsTruncated"):
            break
        paginator_token = response.get("NextContinuationToken")

    if not pending_records:
        click.echo("No pending cleanup records found.")
        return

    click.echo(f"Found {len(pending_records)} pending cleanup record(s).")

    for record_key in pending_records:
        record_url = f"s3://{bucket}/{record_key}"
        # try_get_bytes returns None if the record was deleted (e.g. by lifecycle rule)
        raw = try_get_bytes(record_url, s3_client=client)
        if raw is None:
            click.echo(f"  Skipping (already removed): {record_url}")
            continue
        record = json.loads(raw)
        failed_urls = record.get("failed_urls", [])
        run_id = record.get("run_id", "unknown")

        click.echo(f"  run_id={run_id}: {len(failed_urls)} URL(s) to clean")

        if dry_run:
            for url in failed_urls:
                click.echo(f"    {url}")
            continue

        still_failed: list[str] = []
        for url in failed_urls:
            deleted = delete_prefix(url, s3_client=client)
            if deleted > 0:
                click.echo(f"    Cleaned: {url} ({deleted} objects)")
            else:
                still_failed.append(url)
                click.echo(f"    Still failed: {url}")

        if not still_failed:
            # All cleaned — remove the pending record
            client.delete_object(Bucket=bucket, Key=record_key)
            click.echo(f"  Record removed: {record_url}")
        else:
            click.echo(f"  {len(still_failed)} URL(s) still pending")
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/cli/test_app.py::TestCleanupCommand -v`
Expected: All pass

- [ ] **Step 6: Run lint**

Run: `uv run ruff check shardyfusion/cli/app.py`
Expected: Clean

- [ ] **Step 7: Manual smoke test**

Run: `uv run shardy cleanup --help`
Expected: Shows help text for the cleanup command with `--s3-prefix` and `--dry-run` options.

- [ ] **Step 8: Commit**

```bash
git add shardyfusion/cli/app.py tests/unit/cli/test_app.py
git commit -m "feat: add 'shardy cleanup' command for deferred loser cleanup retry"
```

---

## Chunk 4: Documentation and CLAUDE.md sync

### Task 4: Update docs

**Files:**
- Modify: `docs/how-it-works.md` — update cleanup note to mention deferred retry
- Modify: `docs/cli.md` — add `cleanup` subcommand documentation
- Modify: `CLAUDE.md` — add `cleanup` to CLI subcommand list, update cleanup behavior description

- [ ] **Step 1: Update `docs/how-it-works.md`**

In the S3 layout notes section, update the cleanup line (currently says "automatically cleaned up after publish via `cleanup_losers()` (best-effort)") to:

```
- Non-winning attempt paths are cleaned up after publish via `cleanup_losers()` (best-effort with retries). If cleanup fails, pending records are written to `_cleanup/run_id=.../pending.json` for deferred retry via `shardy cleanup`.
```

- [ ] **Step 2: Update `docs/cli.md`**

Add `cleanup` to the subcommand list. Document `--s3-prefix` (required) and `--dry-run` options.

- [ ] **Step 3: Update `CLAUDE.md`**

In the CLI section, add `cleanup --s3-prefix PREFIX [--dry-run]` to the subcommand list.

In the "Gotchas & Non-obvious Behavior" section, add:
```
- **`cleanup_losers()` returns `CleanupResult`**: The return type changed from `int` to `CleanupResult(deleted=int, failed_urls=list[str])`. Failed cleanup URLs are persisted to `_cleanup/run_id=.../pending.json` when `s3_prefix` and `run_id` are provided.
```

- [ ] **Step 4: Run verification**

```bash
grep -r "cleanup" docs/cli.md docs/how-it-works.md CLAUDE.md | head -20
```

- [ ] **Step 5: Commit**

```bash
git add docs/how-it-works.md docs/cli.md CLAUDE.md
git commit -m "docs: document cleanup_losers deferred retry and shardy cleanup command"
```

---

## Verification Checklist

After all tasks are complete:

- [ ] `uv run pytest tests/unit/writer/test_writer_core_resilience.py -v` — all cleanup tests pass
- [ ] `uv run pytest tests/unit/shared/test_storage.py -v` — delete_prefix retry tests pass
- [ ] `uv run pytest tests/unit/cli/test_app.py -v` — cleanup CLI tests pass
- [ ] `uv run pytest tests/unit/ -x -q` — full unit suite, no regressions
- [ ] `uv run ruff check shardyfusion/ tests/` — lint clean
- [ ] `uv run pyright shardyfusion/_writer_core.py shardyfusion/storage.py shardyfusion/cli/app.py` — type check clean
- [ ] `uv run shardy cleanup --help` — CLI help works

## Operational Note

After deploying this change, set up an S3 lifecycle rule for the `_cleanup/` prefix:

```json
{
  "Rules": [{
    "ID": "expire-cleanup-records",
    "Filter": {"Prefix": "_cleanup/"},
    "Status": "Enabled",
    "Expiration": {"Days": 30}
  }]
}
```

This auto-expires stale cleanup records that are no longer actionable.
