import os

# Tests never read a developer's .env or print span trees.
os.environ["TRACING_EXPORTER"] = "none"

import pytest  # noqa: E402
from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)


@pytest.fixture
def spans() -> InMemorySpanExporter:
    """Finished spans from the code under test."""
    _exporter.clear()
    return _exporter


def by_name(exporter: InMemorySpanExporter, prefix: str):
    return [s for s in exporter.get_finished_spans() if s.name.startswith(prefix)]
