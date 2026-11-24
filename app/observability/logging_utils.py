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

"""Structured JSON logging with automatic trace correlation and PII scrubbing."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace

from app.observability.pii_redactor import redact_pii


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as Cloud Logging-compatible structured JSON."""

    def format(self, record: logging.LogRecord) -> str:
        span = trace.get_current_span()
        span_ctx = span.get_span_context() if span else None
        trace_id = (
            format(span_ctx.trace_id, "032x")
            if span_ctx and span_ctx.is_valid
            else None
        )
        span_id = (
            format(span_ctx.span_id, "016x") if span_ctx and span_ctx.is_valid else None
        )

        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": redact_pii(record.getMessage()),
        }
        if trace_id:
            payload["trace_id"] = trace_id
            payload["span_id"] = span_id

        extra_fields = getattr(record, "structured_fields", None)
        if isinstance(extra_fields, dict):
            payload.update(redact_pii(extra_fields))

        if record.exc_info:
            payload["exception"] = redact_pii(self.formatException(record.exc_info))

        return json.dumps(payload, default=str, ensure_ascii=False)


def get_structured_logger(name: str) -> logging.Logger:
    """Returns a logger configured with StructuredJsonFormatter."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    return logger


def log_structured_event(
    logger: logging.Logger,
    event_type: str,
    message: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emits a structured JSON event with an explicit event_type and metadata."""
    logger.log(
        level,
        message,
        extra={"structured_fields": {"event_type": event_type, **fields}},
    )
