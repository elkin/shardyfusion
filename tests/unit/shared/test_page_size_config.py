"""Mutual-exclusivity tests for the page_size selection mechanisms.

A writer config must specify at most one page_size strategy:

* Explicit ``page_size`` on the SQLite factory (any supported int).
* ``page_size="auto"`` on the factory (post-write VACUUM picks per shard).
* ``kv.profile_value_sizes_for_page_size=True`` on the writer config
  (distributed engines compute the percentile upstream).

The default — ``page_size=4096`` with the flag unset — is the
"no choice made" baseline.  Any other combination is a configuration
error raised at config construction.
"""

from __future__ import annotations

import pytest

from shardyfusion.config import KeyValueWriteConfig
from shardyfusion.errors import ConfigValidationError
from shardyfusion.sqlite_adapter import SqliteFactory


class TestPageSizeMutualExclusion:
    def test_defaults_pass(self) -> None:
        KeyValueWriteConfig()

    def test_explicit_int_only(self) -> None:
        KeyValueWriteConfig(adapter_factory=SqliteFactory(page_size=8192))

    def test_auto_only(self) -> None:
        KeyValueWriteConfig(adapter_factory=SqliteFactory(page_size="auto"))

    def test_flag_with_default_factory(self) -> None:
        # Default factory has page_size=4096 which is the "no choice"
        # baseline, so adding the flag is allowed.
        KeyValueWriteConfig(
            adapter_factory=SqliteFactory(),
            profile_value_sizes_for_page_size=True,
        )

    def test_flag_with_explicit_int_rejected(self) -> None:
        with pytest.raises(ConfigValidationError) as excinfo:
            KeyValueWriteConfig(
                adapter_factory=SqliteFactory(page_size=16384),
                profile_value_sizes_for_page_size=True,
            )
        msg = str(excinfo.value)
        assert "profile_value_sizes_for_page_size" in msg
        assert "page_size=16384" in msg

    def test_flag_with_auto_rejected(self) -> None:
        with pytest.raises(ConfigValidationError) as excinfo:
            KeyValueWriteConfig(
                adapter_factory=SqliteFactory(page_size="auto"),
                profile_value_sizes_for_page_size=True,
            )
        msg = str(excinfo.value)
        assert "profile_value_sizes_for_page_size" in msg
        assert "page_size='auto'" in msg

    def test_factory_rejects_unsupported_int(self) -> None:
        with pytest.raises(ConfigValidationError):
            SqliteFactory(page_size=3000)

    def test_factory_rejects_unknown_string(self) -> None:
        with pytest.raises(ConfigValidationError):
            SqliteFactory(page_size="adaptive")  # type: ignore[arg-type]


class TestPythonWriterRejectsProfileFlag:
    """The Python writer cannot consume the distributed-engine flag."""

    def test_python_writer_rejects_flag(self) -> None:
        # Imports gated because the writer module pulls in a lot of
        # transitive deps; the rejection lives near the entry point.
        from shardyfusion.writer.python.writer import _reject_engine_profile_flag

        cfg = type(
            "_Stub",
            (),
            {"kv": KeyValueWriteConfig(profile_value_sizes_for_page_size=True)},
        )()
        with pytest.raises(ConfigValidationError) as excinfo:
            _reject_engine_profile_flag(cfg)
        msg = str(excinfo.value)
        assert "Python writer" in msg
        assert "page_size='auto'" in msg

    def test_python_writer_accepts_when_flag_unset(self) -> None:
        from shardyfusion.writer.python.writer import _reject_engine_profile_flag

        cfg = type("_Stub", (), {"kv": KeyValueWriteConfig()})()
        _reject_engine_profile_flag(cfg)  # no raise
