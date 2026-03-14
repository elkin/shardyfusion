"""Configurable metrics collection for observability.

Backward-compatible: ``from shardyfusion.metrics import MetricEvent`` works
exactly as before the module-to-package promotion.
"""

from ._events import MetricEvent
from ._protocol import MetricsCollector

__all__ = ["MetricEvent", "MetricsCollector"]
