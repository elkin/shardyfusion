"""Tests for credential provider types and env file materialization."""

import os
import pickle

import pytest

from shardyfusion.credentials import (
    EnvCredentialProvider,
    S3Credentials,
    StaticCredentialProvider,
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
