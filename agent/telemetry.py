"""OpenTelemetry setup.

Spans follow the OpenTelemetry GenAI semantic conventions (gen_ai.*) so
backends that understand them, Langfuse and LangSmith included, show model,
token and tool details without extra mapping.

TRACING_EXPORTER picks where spans go; several can be combined with commas:
  console       an indented span tree per run on stdout (default)
  console_json  the stock OpenTelemetry JSON-per-span console output
  otlp          any OTLP/HTTP backend, configured by the standard
                OTEL_EXPORTER_OTLP_* environment variables
  langfuse      Langfuse's OTLP endpoint (LANGFUSE_PUBLIC_KEY / SECRET_KEY / HOST)
  langsmith     LangSmith's OTLP endpoint (LANGSMITH_API_KEY / PROJECT / ENDPOINT)
  none          tracing off
"""

import base64
import sys
from collections import defaultdict
from collections.abc import Sequence
from typing import TextIO

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from config import settings

TRACER_NAME = "ai-research-agent"

_provider: TracerProvider | None = None


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


# ─── Console tree exporter ───────────────────────────────────────────────────

# Attributes worth showing inline, in display order, with short labels.
_SUMMARY_ATTRS = [
    ("agent.iteration", "#"),
    ("agent.step.kind", "kind"),
    ("gen_ai.response.model", "model"),
    ("gen_ai.tool.name", "tool"),
    ("agent.status", "status"),
    ("gen_ai.response.finish_reasons", "stop"),
    ("gen_ai.usage.input_tokens", "in"),
    ("gen_ai.usage.output_tokens", "out"),
    ("agent.cost_usd", "cost$"),
    ("llm.cost_usd", "cost$"),
    ("llm.attempts", "attempts"),
    ("tool.attempts", "attempts"),
    ("tool.fallback", "fallback"),
    ("tool.error", "error"),
    ("agent.brief_valid", "valid"),
    ("error.type", "error"),
]


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if value < 1 else f"{value:.2f}"
    if isinstance(value, tuple | list):
        return ",".join(str(v) for v in value)
    return str(value)


class TreeConsoleExporter(SpanExporter):
    """Buffers spans per trace and prints the whole tree when the root span
    ends. Children end before their parents, so a flat per-span printout
    reads backwards; this reads like the run did."""

    def __init__(self, out: TextIO = sys.stdout):
        self._out = out
        self._pending: dict[int, list[ReadableSpan]] = defaultdict(list)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            trace_id = span.context.trace_id
            self._pending[trace_id].append(span)
            if span.parent is None:
                self._print_tree(trace_id, self._pending.pop(trace_id))
        return SpanExportResult.SUCCESS

    def _print_tree(self, trace_id: int, spans: list[ReadableSpan]) -> None:
        children: dict[int | None, list[ReadableSpan]] = defaultdict(list)
        for s in spans:
            children[s.parent.span_id if s.parent else None].append(s)
        for group in children.values():
            group.sort(key=lambda s: s.start_time or 0)

        lines = [f"trace {trace.format_trace_id(trace_id)}"]

        def walk(span: ReadableSpan, depth: int) -> None:
            duration_ms = ((span.end_time or 0) - (span.start_time or 0)) / 1e6
            attrs = span.attributes or {}
            shown, seen = [], set()
            for key, label in _SUMMARY_ATTRS:
                if key in attrs and label not in seen:
                    shown.append(f"{label}={_format_value(attrs[key])}")
                    seen.add(label)
            events = [e.name for e in span.events]
            if events:
                shown.append(f"events={','.join(events)}")
            lines.append(
                f"{'  ' * depth}{span.name}  {duration_ms:.0f}ms  {' '.join(shown)}".rstrip()
            )
            for child in children.get(span.context.span_id, []):
                walk(child, depth + 1)

        for root in children.get(None, []):
            walk(root, 1)
        self._out.write("\n".join(lines) + "\n")
        self._out.flush()

    def shutdown(self) -> None:
        self._pending.clear()


# ─── Exporter selection ──────────────────────────────────────────────────────


def _otlp_exporter(endpoint: str | None = None, headers: dict[str, str] | None = None):
    # Imported lazily so the console path has no protobuf/requests dependency.
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    if endpoint is None:
        return OTLPSpanExporter()  # reads OTEL_EXPORTER_OTLP_* env vars
    return OTLPSpanExporter(endpoint=endpoint, headers=headers or {})


def _langfuse_exporter():
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        raise ValueError("langfuse exporter needs LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
    token = base64.b64encode(
        f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
    ).decode()
    return _otlp_exporter(
        f"{settings.langfuse_host.rstrip('/')}/api/public/otel/v1/traces",
        {"Authorization": f"Basic {token}"},
    )


def _langsmith_exporter():
    if not settings.langsmith_api_key:
        raise ValueError("langsmith exporter needs LANGSMITH_API_KEY")
    return _otlp_exporter(
        f"{settings.langsmith_endpoint.rstrip('/')}/otel/v1/traces",
        {"x-api-key": settings.langsmith_api_key, "Langsmith-Project": settings.langsmith_project},
    )


def build_processors(exporters: str) -> list:
    processors = []
    for name in (n.strip().lower() for n in exporters.split(",")):
        if not name or name == "none":
            continue
        if name == "console":
            processors.append(SimpleSpanProcessor(TreeConsoleExporter()))
        elif name == "console_json":
            processors.append(SimpleSpanProcessor(ConsoleSpanExporter()))
        elif name == "otlp":
            processors.append(BatchSpanProcessor(_otlp_exporter()))
        elif name == "langfuse":
            processors.append(BatchSpanProcessor(_langfuse_exporter()))
        elif name == "langsmith":
            processors.append(BatchSpanProcessor(_langsmith_exporter()))
        else:
            raise ValueError(f"Unknown TRACING_EXPORTER {name!r}")
    return processors


def setup_tracing(exporters: str | None = None) -> TracerProvider | None:
    """Register a global tracer provider. Safe to call more than once; only
    the first call takes effect, matching OpenTelemetry's global provider."""
    global _provider
    if _provider is not None:
        return _provider
    processors = build_processors(exporters if exporters is not None else settings.tracing_exporter)
    if not processors:
        return None
    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.tracing_service_name})
    )
    for processor in processors:
        provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    _provider = provider
    return provider


def shutdown_tracing() -> None:
    """Flush batched spans; call on process exit."""
    global _provider
    if _provider is not None:
        _provider.shutdown()
        _provider = None
