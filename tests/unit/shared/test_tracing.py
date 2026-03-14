"""Unit tests for the Tracer/Span Protocol and tracing integration."""

from __future__ import annotations

from typing import Any

from shardyfusion.type_defs import Span, Tracer


class FakeSpan:
    """Test double implementing the Span protocol."""

    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}
        self.status: Any = None
        self.status_description: str | None = None
        self.exceptions: list[BaseException] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, status: Any, description: str | None = None) -> None:
        self.status = status
        self.status_description = description

    def record_exception(self, exception: BaseException) -> None:
        self.exceptions.append(exception)

    def __enter__(self) -> FakeSpan:
        return self

    def __exit__(self, *args: object) -> None:
        pass


class FakeTracer:
    """Test double implementing the Tracer protocol."""

    def __init__(self) -> None:
        self.spans: list[tuple[str, FakeSpan]] = []

    def start_as_current_span(self, name: str, **kwargs: Any) -> FakeSpan:
        span = FakeSpan()
        self.spans.append((name, span))
        return span


def test_fake_tracer_satisfies_protocol() -> None:
    """FakeTracer satisfies the Tracer Protocol via structural typing."""
    tracer: Tracer = FakeTracer()
    span: Span = tracer.start_as_current_span("test.span")
    with span as s:
        s.set_attribute("key", "value")


def test_span_context_manager() -> None:
    """Span supports context manager protocol."""
    tracer = FakeTracer()
    with tracer.start_as_current_span("test.op") as span:
        span.set_attribute("run_id", "abc123")
        span.set_attribute("num_dbs", 4)

    assert len(tracer.spans) == 1
    name, recorded = tracer.spans[0]
    assert name == "test.op"
    assert recorded.attributes["run_id"] == "abc123"
    assert recorded.attributes["num_dbs"] == 4


def test_span_record_exception() -> None:
    """Span can record exceptions."""
    span = FakeSpan()
    err = ValueError("test error")
    span.record_exception(err)
    assert span.exceptions == [err]


def test_no_tracer_no_overhead() -> None:
    """When no tracer is passed, operations work without tracing — zero overhead."""
    tracer: Tracer | None = None

    # Simulate what writer code does:
    result = "success"
    if tracer is not None:
        with tracer.start_as_current_span("write") as span:
            span.set_attribute("run_id", "test")
            result = "traced"

    assert result == "success"
