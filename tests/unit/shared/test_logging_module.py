"""Tests for shardyfusion.logging helpers."""

import logging

import pytest

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
        assert record.shardyfusion == {"foo": "bar", "count": 42}  # type: ignore[attr-defined]

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
        assert caplog.records[0].shardyfusion == {"key": "val"}  # type: ignore[attr-defined]

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
        assert record.shardyfusion["severity"] == "transient"  # type: ignore[attr-defined]
        assert record.shardyfusion["url"] == "s3://bucket/key"  # type: ignore[attr-defined]

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
        assert "ValueError" in record.shardyfusion["error"]  # type: ignore[attr-defined]
        assert "bad input" in record.shardyfusion["error"]  # type: ignore[attr-defined]

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
        tb = record.shardyfusion["traceback"]  # type: ignore[attr-defined]
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


class TestLogContext:
    def test_fields_merged_into_log_event(self, caplog):
        from shardyfusion.logging import LogContext

        with caplog.at_level(logging.INFO, logger="shardyfusion"):
            with LogContext(run_id="abc", writer_type="spark"):
                log_event("test_event", shard=0)
        record = caplog.records[0]
        assert record.shardyfusion == {
            "run_id": "abc",
            "writer_type": "spark",
            "shard": 0,
        }  # type: ignore[attr-defined]

    def test_nesting_inner_overrides_outer(self, caplog):
        from shardyfusion.logging import LogContext

        with caplog.at_level(logging.INFO, logger="shardyfusion"):
            with LogContext(run_id="outer", mode="batch"):
                with LogContext(run_id="inner"):
                    log_event("nested")
        record = caplog.records[0]
        # Inner overrides 'run_id', outer 'mode' is preserved
        assert record.shardyfusion["run_id"] == "inner"  # type: ignore[attr-defined]
        assert record.shardyfusion["mode"] == "batch"  # type: ignore[attr-defined]

    def test_outer_restored_after_exit(self, caplog):
        from shardyfusion.logging import LogContext

        with caplog.at_level(logging.INFO, logger="shardyfusion"):
            with LogContext(run_id="outer"):
                with LogContext(run_id="inner"):
                    pass
                log_event("after_inner")
        record = caplog.records[0]
        assert record.shardyfusion["run_id"] == "outer"  # type: ignore[attr-defined]


class TestJsonFormatter:
    def test_produces_valid_json(self):
        import json

        from shardyfusion.logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="shardyfusion.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test_event",
            args=(),
            exc_info=None,
        )
        record.shardyfusion = {"run_id": "abc", "shard": 0}  # type: ignore[attr-defined]
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "shardyfusion.test"
        assert parsed["event"] == "test_event"
        assert parsed["run_id"] == "abc"
        assert parsed["shard"] == 0
        assert "timestamp" in parsed

    def test_no_shardyfusion_fields(self):
        import json

        from shardyfusion.logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="shardyfusion.test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="bare_event",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["event"] == "bare_event"
        assert parsed["level"] == "WARNING"


class TestConfigureLogging:
    @pytest.fixture(autouse=True)
    def _preserve_logger(self):
        """Save and restore shardyfusion logger state around each test."""
        logger = logging.getLogger("shardyfusion")
        original_level = logger.level
        original_handlers = list(logger.handlers)
        yield
        logger.handlers[:] = original_handlers
        logger.setLevel(original_level)

    def test_sets_handler_on_shardyfusion_logger(self):
        import io

        from shardyfusion.logging import configure_logging

        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=stream)

        logger = logging.getLogger("shardyfusion")
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 1
        assert logger.handlers[0].stream is stream  # type: ignore[attr-defined]

    def test_json_format_option(self):
        import io

        from shardyfusion.logging import JsonFormatter, configure_logging

        stream = io.StringIO()
        configure_logging(level=logging.INFO, json_format=True, stream=stream)

        logger = logging.getLogger("shardyfusion")
        assert isinstance(logger.handlers[0].formatter, JsonFormatter)


def _is_descendant(child: logging.Logger, ancestor: logging.Logger) -> bool:
    """Walk parent chain to check if child descends from ancestor."""
    current = child
    while current:
        if current is ancestor:
            return True
        current = current.parent  # type: ignore[assignment]
    return False
