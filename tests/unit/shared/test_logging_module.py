"""Tests for shardyfusion.logging helpers."""

import logging

from shardyfusion.logging import (
    FailureSeverity,
    get_logger,
    log_event,
    log_failure,
)


class TestGetLogger:
    def test_strips_package_prefix(self):
        logger = get_logger("shardyfusion.writer.spark.writer")
        assert logger.name == "shardyfusion.writer.spark.writer"

    def test_adds_package_prefix_for_bare_name(self):
        logger = get_logger("my_module")
        assert logger.name == "shardyfusion.my_module"

    def test_returns_child_of_root_logger(self):
        root = logging.getLogger("shardyfusion")
        child = get_logger("shardyfusion.storage")
        assert child.parent is root

    def test_hierarchy_across_modules(self):
        """Loggers from different modules share the same root."""
        writer = get_logger("shardyfusion.writer.spark.writer")
        reader = get_logger("shardyfusion.reader.reader")
        root = logging.getLogger("shardyfusion")
        # Both must be descendants
        assert _is_descendant(writer, root)
        assert _is_descendant(reader, root)


class TestLogEvent:
    def test_emits_info_by_default(self, caplog):
        with caplog.at_level(logging.INFO, logger="shardyfusion"):
            log_event("test_event", foo="bar", count=42)
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.message == "test_event"
        assert record.levelno == logging.INFO
        assert record.slatedb == {"foo": "bar", "count": 42}  # type: ignore[attr-defined]

    def test_emits_debug_level(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="shardyfusion"):
            log_event("debug_event", level=logging.DEBUG, x=1)
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.DEBUG

    def test_uses_explicit_logger(self, caplog):
        custom = get_logger("shardyfusion.test_custom")
        with caplog.at_level(logging.INFO, logger="shardyfusion.test_custom"):
            log_event("custom_event", logger=custom, key="val")
        assert len(caplog.records) == 1
        assert caplog.records[0].name == "shardyfusion.test_custom"
        assert caplog.records[0].slatedb == {"key": "val"}  # type: ignore[attr-defined]

    def test_falls_back_to_root_logger_when_no_logger(self, caplog):
        with caplog.at_level(logging.INFO, logger="shardyfusion"):
            log_event("root_event")
        assert len(caplog.records) == 1
        assert caplog.records[0].name == "shardyfusion"

    def test_skips_when_level_disabled(self, caplog):
        with caplog.at_level(logging.WARNING, logger="shardyfusion"):
            log_event("should_not_appear", level=logging.INFO)
        assert len(caplog.records) == 0


class TestLogFailure:
    def test_transient_logs_at_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="shardyfusion"):
            log_failure(
                "s3_transient",
                severity=FailureSeverity.TRANSIENT,
                url="s3://bucket/key",
            )
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.WARNING
        assert record.slatedb["severity"] == "transient"  # type: ignore[attr-defined]
        assert record.slatedb["url"] == "s3://bucket/key"  # type: ignore[attr-defined]

    def test_error_logs_at_error(self, caplog):
        with caplog.at_level(logging.ERROR, logger="shardyfusion"):
            log_failure("write_fail", severity=FailureSeverity.ERROR)
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.ERROR

    def test_critical_logs_at_critical(self, caplog):
        with caplog.at_level(logging.CRITICAL, logger="shardyfusion"):
            log_failure("data_loss", severity=FailureSeverity.CRITICAL)
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.CRITICAL

    def test_includes_error_repr(self, caplog):
        exc = ValueError("bad input")
        with caplog.at_level(logging.ERROR, logger="shardyfusion"):
            log_failure("fail", severity=FailureSeverity.ERROR, error=exc)
        record = caplog.records[0]
        assert "ValueError" in record.slatedb["error"]  # type: ignore[attr-defined]
        assert "bad input" in record.slatedb["error"]  # type: ignore[attr-defined]

    def test_includes_traceback_when_requested(self, caplog):
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            with caplog.at_level(logging.ERROR, logger="shardyfusion"):
                log_failure(
                    "fail",
                    severity=FailureSeverity.ERROR,
                    error=exc,
                    include_traceback=True,
                )
        record = caplog.records[0]
        tb = record.slatedb["traceback"]  # type: ignore[attr-defined]
        assert isinstance(tb, list)
        assert any("RuntimeError" in line for line in tb)

    def test_uses_explicit_logger(self, caplog):
        custom = get_logger("shardyfusion.test_failures")
        with caplog.at_level(logging.ERROR, logger="shardyfusion.test_failures"):
            log_failure("custom_fail", severity=FailureSeverity.ERROR, logger=custom)
        assert len(caplog.records) == 1
        assert caplog.records[0].name == "shardyfusion.test_failures"

    def test_skips_when_level_disabled(self, caplog):
        with caplog.at_level(logging.CRITICAL, logger="shardyfusion"):
            log_failure("should_skip", severity=FailureSeverity.ERROR)
        assert len(caplog.records) == 0


def _is_descendant(child: logging.Logger, ancestor: logging.Logger) -> bool:
    """Walk parent chain to check if child descends from ancestor."""
    current = child
    while current:
        if current is ancestor:
            return True
        current = current.parent  # type: ignore[assignment]
    return False
