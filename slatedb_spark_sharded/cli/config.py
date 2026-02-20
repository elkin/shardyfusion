"""Configuration loading, resolution, and S3 client factory for the reader CLI."""

from __future__ import annotations

import os
import stat
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..type_defs import S3ClientConfig

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReaderConfig:
    """Non-sensitive reader settings from [reader] in reader.toml."""

    current_url: str | None = None
    local_root: str = "/tmp/slatefusion"
    thread_safety: str = "lock"
    max_workers: int = 4
    slate_env_file: str | None = None
    credentials_profile: str = "default"


@dataclass
class OutputConfig:
    """Output formatting settings from [output] in reader.toml."""

    format: str = "jsonl"
    value_encoding: str = "base64"
    null_repr: str = "null"


@dataclass
class CredentialsProfile:
    """One named profile from credentials.toml (credentials + S3 options)."""

    endpoint_url: str | None = None
    region: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None
    # botocore / connection options
    addressing_style: str | None = None
    signature_version: str | None = None
    verify_ssl: bool | str = True
    connect_timeout: int = 10
    read_timeout: int = 30
    max_attempts: int = 3


# ---------------------------------------------------------------------------
# TOML loaders
# ---------------------------------------------------------------------------

_READER_SEARCH_PATHS = [
    Path("reader.toml"),
    Path.home() / ".config" / "slatefusion" / "reader.toml",
]

_CREDS_SEARCH_PATHS = [
    Path("credentials.toml"),
    Path.home() / ".config" / "slatefusion" / "credentials.toml",
]


def _find_file(
    explicit: str | None, env_var: str, search_paths: list[Path]
) -> Path | None:
    """Return first resolved path using: explicit flag → env var → search paths."""
    if explicit:
        return Path(explicit)
    env = os.getenv(env_var)
    if env:
        return Path(env)
    for path in search_paths:
        if path.exists():
            return path
    return None


def load_reader_config(
    config_path: str | None = None,
) -> tuple[ReaderConfig, OutputConfig]:
    """Load and return (ReaderConfig, OutputConfig) from the resolved reader.toml.

    Falls back to defaults when no file is found.
    """
    path = _find_file(config_path, "SLATE_READER_CONFIG", _READER_SEARCH_PATHS)
    if path is None:
        return ReaderConfig(), OutputConfig()

    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    reader_raw = data.get("reader", {})
    output_raw = data.get("output", {})

    reader_cfg = ReaderConfig(
        current_url=reader_raw.get("current_url") or None,
        local_root=reader_raw.get("local_root", "/tmp/slatefusion"),
        thread_safety=reader_raw.get("thread_safety", "lock"),
        max_workers=int(reader_raw.get("max_workers", 4)),
        slate_env_file=reader_raw.get("slate_env_file") or None,
        credentials_profile=reader_raw.get("credentials_profile", "default"),
    )

    output_cfg = OutputConfig(
        format=output_raw.get("format", "jsonl"),
        value_encoding=output_raw.get("value_encoding", "base64"),
        null_repr=output_raw.get("null_repr", "null"),
    )

    return reader_cfg, output_cfg


def load_credentials_profile(
    profile_name: str = "default",
    credentials_path: str | None = None,
) -> CredentialsProfile | None:
    """Load a named profile from the resolved credentials.toml.

    Returns None when no credentials file is found (boto3 chain is used as fallback).
    Warns when the file has permissions wider than 0600 on UNIX.
    """
    path = _find_file(credentials_path, "SLATE_READER_CREDENTIALS", _CREDS_SEARCH_PATHS)
    if path is None:
        return None

    # Warn on overly-permissive credentials file
    if sys.platform != "win32":
        mode = path.stat().st_mode & 0o777
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            print(
                f"Warning: credentials file {path} has permissions {oct(mode)} "
                "— consider restricting to 0600.",
                file=sys.stderr,
            )

    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    raw = data.get(profile_name)
    if raw is None:
        raise KeyError(f"Credentials profile '{profile_name}' not found in {path}")

    verify_ssl_raw = raw.get("verify_ssl", True)
    verify_ssl: bool | str
    if isinstance(verify_ssl_raw, bool):
        verify_ssl = verify_ssl_raw
    elif isinstance(verify_ssl_raw, str):
        verify_ssl = verify_ssl_raw
    else:
        verify_ssl = bool(verify_ssl_raw)

    return CredentialsProfile(
        endpoint_url=raw.get("endpoint_url") or None,
        region=raw.get("region") or None,
        access_key_id=raw.get("access_key_id") or None,
        secret_access_key=raw.get("secret_access_key") or None,
        session_token=raw.get("session_token") or None,
        addressing_style=raw.get("addressing_style") or None,
        signature_version=raw.get("signature_version") or None,
        verify_ssl=verify_ssl,
        connect_timeout=int(raw.get("connect_timeout", 10)),
        read_timeout=int(raw.get("read_timeout", 30)),
        max_attempts=int(raw.get("max_attempts", 3)),
    )


# ---------------------------------------------------------------------------
# CURRENT URL resolution
# ---------------------------------------------------------------------------

_S3_OPTION_TYPES: dict[str, type] = {
    "addressing_style": str,
    "signature_version": str,
    "verify_ssl": bool,
    "connect_timeout": int,
    "read_timeout": int,
    "max_attempts": int,
}


def coerce_s3_option(key: str, value: str) -> bool | int | str:
    """Coerce a --s3-option value string to the expected Python type."""
    target = _S3_OPTION_TYPES.get(key)
    if target is bool:
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"Cannot coerce '{value}' to bool for option '{key}'")
    if target is int:
        return int(value)
    return value  # str or unknown key → leave as string


def resolve_current_url(
    positional: str | None,
    reader_cfg: ReaderConfig,
) -> str:
    """Resolve CURRENT URL from positional arg → env var → config file."""
    if positional:
        return positional
    env = os.getenv("SLATE_READER_CURRENT")
    if env:
        return env
    if reader_cfg.current_url:
        return reader_cfg.current_url
    raise SystemExit(
        "Error: CURRENT URL is required. Provide it as a positional argument, "
        "set SLATE_READER_CURRENT env var, or set current_url in reader.toml."
    )


def split_current_url(current_url: str) -> tuple[str, str]:
    """Return (s3_prefix, current_name) by splitting the last path segment."""
    current_url = current_url.rstrip("/")
    idx = current_url.rfind("/")
    if idx < 0:
        raise ValueError(
            f"Cannot split CURRENT URL into prefix and name: {current_url!r}"
        )
    return current_url[:idx], current_url[idx + 1 :]


# ---------------------------------------------------------------------------
# S3ClientConfig factory
# ---------------------------------------------------------------------------


def build_s3_client_config(
    profile: CredentialsProfile | None,
    s3_option_overrides: dict[str, bool | int | str] | None = None,
) -> S3ClientConfig:
    """Assemble S3ClientConfig from credentials profile + per-invocation overrides."""
    cfg: S3ClientConfig = {}

    if profile is not None:
        if profile.endpoint_url:
            cfg["endpoint_url"] = profile.endpoint_url
        if profile.region:
            cfg["region_name"] = profile.region
        if profile.access_key_id:
            cfg["access_key_id"] = profile.access_key_id
        if profile.secret_access_key:
            cfg["secret_access_key"] = profile.secret_access_key
        if profile.session_token:
            cfg["session_token"] = profile.session_token
        if profile.addressing_style:
            cfg["addressing_style"] = profile.addressing_style
        if profile.signature_version:
            cfg["signature_version"] = profile.signature_version
        cfg["verify_ssl"] = profile.verify_ssl
        cfg["connect_timeout"] = profile.connect_timeout
        cfg["read_timeout"] = profile.read_timeout
        cfg["max_attempts"] = profile.max_attempts

    # Apply per-invocation overrides from --s3-option
    if s3_option_overrides:
        for key, val in s3_option_overrides.items():
            cfg[key] = val  # type: ignore[literal-required]

    return cfg
