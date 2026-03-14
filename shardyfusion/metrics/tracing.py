"""OpenTelemetry tracing convenience helpers.

Requires the ``metrics-otel`` extra: ``pip install shardyfusion[metrics-otel]``
"""


def get_tracer(name: str = "shardyfusion"):
    """Return an OpenTelemetry Tracer for shardyfusion.

    This is a thin convenience wrapper around ``opentelemetry.trace.get_tracer()``.
    Users who don't install the ``metrics-otel`` extra simply don't call this function.
    """
    from opentelemetry import trace

    return trace.get_tracer(name)
