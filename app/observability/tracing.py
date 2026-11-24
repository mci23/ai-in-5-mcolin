# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenTelemetry distributed tracing configuration and span helpers."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from app.observability.pii_redactor import redact_pii

_TRACING_CONFIGURED = False


def configure_tracing(
    service_name: str = "weather-comparison-agent",
) -> trace.TracerProvider:
    """Configures an OpenTelemetry TracerProvider once if none is registered."""
    global _TRACING_CONFIGURED
    current_provider = trace.get_tracer_provider()
    if _TRACING_CONFIGURED or isinstance(current_provider, TracerProvider):
        _TRACING_CONFIGURED = True
        return current_provider

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if os.getenv("OTEL_EXPORT_TO_GCP", "false").lower() == "true" and os.getenv(
        "GOOGLE_CLOUD_PROJECT"
    ):
        try:
            from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

            provider.add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter()))
        except Exception:
            pass
    elif os.getenv("OTEL_CONSOLE_EXPORT", "false").lower() == "true":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _TRACING_CONFIGURED = True
    return provider


def get_tracer(name: str = "weather_comparison_agent") -> trace.Tracer:
    """Returns a configured OpenTelemetry tracer."""
    configure_tracing()
    return trace.get_tracer(name)


@contextlib.contextmanager
def traced_span(
    span_name: str,
    *,
    tracer_name: str = "weather_comparison_agent",
    attributes: dict[str, Any] | None = None,
) -> Iterator[trace.Span]:
    """Starts an OpenTelemetry span with PII-scrubbed primitive attributes."""
    tracer = get_tracer(tracer_name)
    with tracer.start_as_current_span(span_name) as span:
        if attributes:
            sanitized = redact_pii(attributes)
            for key, value in sanitized.items():
                if isinstance(value, (str, bool, int, float)):
                    span.set_attribute(key, value)
                elif value is not None:
                    span.set_attribute(key, str(value))
        yield span
