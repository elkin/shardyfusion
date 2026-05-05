"""Tests for credential provider types and env file materialization."""

from __future__ import annotations

import os
import pickle
import threading
from pathlib import Path

import pytest

from shardyfusion.credentials import (
    EnvCredentialProvider,
    S3Credentials,
    StaticCredentialProvider,
    _AppliedEnvContext,
    _parse_dotenv,
    apply_env_file,
    materialize_env_file,
    remove_env_file,
)


class TestS3Credentials:
    def test_defaults_to_none(self):
        creds = S3Credentials()
        assert creds.access_key_id is None
        assert creds.secret_access_key is None
        assert creds.session_token is None

    def test_frozen(self):
        creds = S3Credentials(access_key_id="AKIA")
        with pytest.raises(AttributeError):
            creds.access_key_id = "other"  # type: ignore[misc]

    def test_picklable(self):
        """Picklability is required for Spark/Dask/Ray/multiprocessing serialization."""
        creds = S3Credentials(access_key_id="AKIA", secret_access_key="secret")
        restored = pickle.loads(pickle.dumps(creds))
        assert restored == creds


class TestStaticCredentialProvider:
    def test_resolve_returns_credentials(self):
        provider = StaticCredentialProvider(
            access_key_id="AKIA",
            secret_access_key="secret",
            session_token="token",
        )
        creds = provider.resolve()
        assert creds.access_key_id == "AKIA"
        assert creds.secret_access_key == "secret"
        assert creds.session_token == "token"

    def test_resolve_defaults(self):
        provider = StaticCredentialProvider()
        creds = provider.resolve()
        assert creds.access_key_id is None
        assert creds.secret_access_key is None

    def test_picklable(self):
        """Picklability is required for Spark/Dask/Ray/multiprocessing serialization."""
        provider = StaticCredentialProvider(access_key_id="AKIA")
        restored = pickle.loads(pickle.dumps(provider))
        assert restored.resolve() == provider.resolve()

    def test_frozen(self):
        provider = StaticCredentialProvider(access_key_id="AKIA")
        with pytest.raises(AttributeError):
            provider.access_key_id = "other"  # type: ignore[misc]


class TestEnvCredentialProvider:
    def test_reads_s3_env_vars(self, monkeypatch):
        monkeypatch.setenv("S3_ACCESS_KEY_ID", "s3key")
        monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "s3secret")
        monkeypatch.setenv("S3_SESSION_TOKEN", "s3token")
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

        creds = EnvCredentialProvider().resolve()
        assert creds.access_key_id == "s3key"
        assert creds.secret_access_key == "s3secret"
        assert creds.session_token == "s3token"

    def test_falls_back_to_aws_env_vars(self, monkeypatch):
        monkeypatch.delenv("S3_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("S3_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.delenv("S3_SESSION_TOKEN", raising=False)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "awskey")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "awssecret")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "awstoken")

        creds = EnvCredentialProvider().resolve()
        assert creds.access_key_id == "awskey"
        assert creds.secret_access_key == "awssecret"
        assert creds.session_token == "awstoken"

    def test_handles_missing_vars(self, monkeypatch):
        for var in [
            "S3_ACCESS_KEY_ID",
            "AWS_ACCESS_KEY_ID",
            "S3_SECRET_ACCESS_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "S3_SESSION_TOKEN",
            "AWS_SESSION_TOKEN",
        ]:
            monkeypatch.delenv(var, raising=False)

        creds = EnvCredentialProvider().resolve()
        assert creds.access_key_id is None
        assert creds.secret_access_key is None
        assert creds.session_token is None

    def test_picklable(self):
        """Picklability is required for Spark/Dask/Ray/multiprocessing serialization."""
        provider = EnvCredentialProvider()
        restored = pickle.loads(pickle.dumps(provider))
        assert isinstance(restored, EnvCredentialProvider)


class TestMaterializeEnvFile:
    def test_creates_file_with_correct_content(self):
        creds = S3Credentials(
            access_key_id="AKIA",
            secret_access_key="secret",
            session_token="token",
        )
        path = materialize_env_file(creds)
        try:
            content = open(path).read()
            assert "AWS_ACCESS_KEY_ID=AKIA" in content
            assert "AWS_SECRET_ACCESS_KEY=secret" in content
            assert "AWS_SESSION_TOKEN=token" in content
        finally:
            remove_env_file(path)

    def test_file_permissions(self):
        creds = S3Credentials(access_key_id="AKIA", secret_access_key="secret")
        path = materialize_env_file(creds)
        try:
            mode = os.stat(path).st_mode & 0o777
            assert mode == 0o600, f"Expected 0600, got {oct(mode)}"
        finally:
            remove_env_file(path)

    def test_omits_none_fields(self):
        creds = S3Credentials(access_key_id="AKIA")
        path = materialize_env_file(creds)
        try:
            content = open(path).read()
            assert "AWS_ACCESS_KEY_ID=AKIA" in content
            assert "AWS_SECRET_ACCESS_KEY" not in content
            assert "AWS_SESSION_TOKEN" not in content
        finally:
            remove_env_file(path)

    def test_empty_credentials_creates_empty_file(self):
        creds = S3Credentials()
        path = materialize_env_file(creds)
        try:
            content = open(path).read()
            assert content == ""
        finally:
            remove_env_file(path)

    def test_cleanup(self):
        creds = S3Credentials(access_key_id="AKIA")
        path = materialize_env_file(creds)
        assert os.path.exists(path)
        remove_env_file(path)
        assert not os.path.exists(path)

    def test_remove_nonexistent_is_safe(self):
        remove_env_file("/tmp/nonexistent_shardyfusion_test_file")


# ---------------------------------------------------------------------------
# _parse_dotenv (S8)
# ---------------------------------------------------------------------------


class TestParseDotenv:
    def test_simple_pairs(self) -> None:
        assert _parse_dotenv("A=1\nB=two\n") == {"A": "1", "B": "two"}

    def test_skips_blank_and_comment_lines(self) -> None:
        text = "# top comment\nA=1\n\n# another comment\nB=2\n"
        assert _parse_dotenv(text) == {"A": "1", "B": "2"}

    def test_strips_export_prefix(self) -> None:
        assert _parse_dotenv("export AWS_ACCESS_KEY_ID=AKIA\n") == {
            "AWS_ACCESS_KEY_ID": "AKIA"
        }
        # Leading whitespace before key after ``export`` is tolerated.
        assert _parse_dotenv("export   FOO=bar\n") == {"FOO": "bar"}

    def test_strips_matching_quotes(self) -> None:
        assert _parse_dotenv('A="quoted"\n') == {"A": "quoted"}
        assert _parse_dotenv("B='single'\n") == {"B": "single"}

    def test_preserves_mismatched_quotes(self) -> None:
        # Mismatched quotes are preserved literally — not unquoted.
        assert _parse_dotenv("C=\"mismatch'\n") == {"C": "\"mismatch'"}

    def test_empty_value_allowed(self) -> None:
        assert _parse_dotenv("EMPTY=\n") == {"EMPTY": ""}

    def test_value_with_equals_signs(self) -> None:
        # ``str.partition('=')`` splits on the first ``=`` only.
        assert _parse_dotenv("URL=postgres://u:p@h:5432/db?x=1&y=2\n") == {
            "URL": "postgres://u:p@h:5432/db?x=1&y=2"
        }

    def test_skips_lines_without_equals(self) -> None:
        assert _parse_dotenv("invalid line no equals\nA=1\n") == {"A": "1"}

    def test_skips_empty_keys(self) -> None:
        # ``=value`` with no key is silently dropped.
        assert _parse_dotenv("=orphan\nA=1\n") == {"A": "1"}


# ---------------------------------------------------------------------------
# _AppliedEnvContext / apply_env_file (S8 + W5)
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the test keys are absent at start; ``monkeypatch`` records
    their pre-test values and restores on teardown."""
    for k in ("_SF_TEST_KEY_A", "_SF_TEST_KEY_B", "_SF_TEST_KEY_C"):
        monkeypatch.delenv(k, raising=False)


class TestAppliedEnvFile:
    def test_none_is_no_op(self, clean_env: None) -> None:
        before = dict(os.environ)
        with apply_env_file(None):
            assert os.environ == before
        assert os.environ == before

    def test_sets_then_pops_when_key_was_absent(
        self, clean_env: None, tmp_path: Path
    ) -> None:
        p = tmp_path / "creds.env"
        p.write_text("_SF_TEST_KEY_A=alpha\n")

        assert "_SF_TEST_KEY_A" not in os.environ
        with apply_env_file(str(p)):
            assert os.environ["_SF_TEST_KEY_A"] == "alpha"
        assert "_SF_TEST_KEY_A" not in os.environ

    def test_restores_pre_existing_value(
        self, clean_env: None, tmp_path: Path
    ) -> None:
        os.environ["_SF_TEST_KEY_A"] = "original"
        p = tmp_path / "creds.env"
        p.write_text("_SF_TEST_KEY_A=overridden\n")

        with apply_env_file(str(p)):
            assert os.environ["_SF_TEST_KEY_A"] == "overridden"
        assert os.environ["_SF_TEST_KEY_A"] == "original"

    def test_missing_file_logs_warning_and_no_op(
        self,
        clean_env: None,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        missing = tmp_path / "does-not-exist.env"
        before = dict(os.environ)
        with caplog.at_level("WARNING"):
            with apply_env_file(str(missing)):
                assert os.environ == before
        assert os.environ == before
        assert any(
            "Failed to read env file" in rec.getMessage() for rec in caplog.records
        )

    def test_empty_pairs_does_not_acquire_lock(
        self, clean_env: None, tmp_path: Path
    ) -> None:
        # A file with only comments produces no pairs; no lock is held.
        p = tmp_path / "empty.env"
        p.write_text("# only a comment\n\n")
        ctx = _AppliedEnvContext(str(p))
        with ctx:
            assert ctx._locked is False
        assert ctx._locked is False

    def test_lock_released_after_exit(
        self, clean_env: None, tmp_path: Path
    ) -> None:
        """A subsequent ``apply_env_file`` works after the previous one
        exits — guards against a regression where ``__exit__`` forgets
        to release the lock and the next caller deadlocks."""
        p = tmp_path / "creds.env"
        p.write_text("_SF_TEST_KEY_A=v1\n")

        with apply_env_file(str(p)):
            pass
        finished = threading.Event()

        def _retry() -> None:
            with apply_env_file(str(p)):
                assert os.environ["_SF_TEST_KEY_A"] == "v1"
            finished.set()

        t = threading.Thread(target=_retry)
        t.start()
        t.join(timeout=2.0)
        assert finished.is_set(), "second apply_env_file deadlocked → lock leaked"


class TestAppliedEnvFileConcurrency:
    """W5: serialise the entire context body across threads so concurrent
    enters with overlapping keys do not corrupt the snapshot/restore."""

    def test_concurrent_overlapping_keys_restore_to_absent(
        self, clean_env: None, tmp_path: Path
    ) -> None:
        files = []
        for i in range(8):
            p = tmp_path / f"creds_{i}.env"
            p.write_text(f"_SF_TEST_KEY_A=v{i}\n")
            files.append(str(p))

        barrier = threading.Barrier(len(files))
        errors: list[BaseException] = []
        err_lock = threading.Lock()

        def _worker(env_file: str) -> None:
            barrier.wait()
            try:
                with apply_env_file(env_file):
                    # Body sees *some* value; the W5 invariant is about
                    # the post-exit state, not the body view.
                    assert os.environ.get("_SF_TEST_KEY_A") is not None
            except BaseException as exc:
                with err_lock:
                    errors.append(exc)
                raise

        threads = [threading.Thread(target=_worker, args=(f,)) for f in files]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive(), "thread deadlocked on env mutation lock"

        assert errors == []
        # Every thread saw the key as absent on entry → all threads
        # popped on exit. End state = absent, despite N concurrent
        # mutations to the same key.
        assert "_SF_TEST_KEY_A" not in os.environ

    def test_concurrent_overlapping_keys_restore_pre_existing(
        self, clean_env: None, tmp_path: Path
    ) -> None:
        os.environ["_SF_TEST_KEY_A"] = "original"

        files = []
        for i in range(4):
            p = tmp_path / f"creds_{i}.env"
            p.write_text(f"_SF_TEST_KEY_A=v{i}\n")
            files.append(str(p))

        def _worker(env_file: str) -> None:
            with apply_env_file(env_file):
                pass

        threads = [threading.Thread(target=_worker, args=(f,)) for f in files]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert os.environ.get("_SF_TEST_KEY_A") == "original"
