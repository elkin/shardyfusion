"""MetricsCollector protocol definition."""

from typing import Any, Protocol

from ._events import MetricEvent


class MetricsCollector(Protocol):
    """Protocol for receiving metric events.

    Implementations must be thread-safe.  The ``emit()`` method is called
    synchronously; buffer internally if blocking is a concern.
    """

    def emit(self, event: MetricEvent, payload: dict[str, Any]) -> None: ...
