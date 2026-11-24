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

"""PII redaction utilities for logging, tracing, and persistent memory."""

from __future__ import annotations

import os
import re
from typing import Any

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("API_KEY", re.compile(r"\b(?:AIza[0-9A-Za-z_-]{20,}|sk-[A-Za-z0-9]{20,})\b")),
    ("BEARER_TOKEN", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (
        "PHONE",
        re.compile(
            r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{3})[\s.-]?\d{3,4}[\s.-]?\d{4}(?!\d)"
        ),
    ),
    (
        "IPV4",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
    ),
)

_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "api_key",
        "authorization",
        "bearer",
        "credential",
        "email",
        "password",
        "phone",
        "secret",
        "ssn",
        "token",
    }
)


class PIIRedactor:
    """Redacts PII and secrets before telemetry or long-term memory persistence.

    Uses deterministic regex scrubbing in all environments, and optionally
    augments scrubbing with Google Cloud Sensitive Data Protection (Cloud DLP)
    when ``ENABLE_CLOUD_DLP=true`` and a GCP project ID is configured.
    """

    def __init__(
        self,
        *,
        enable_cloud_dlp: bool | None = None,
        project_id: str | None = None,
        dlp_client: Any | None = None,
    ) -> None:
        if enable_cloud_dlp is None:
            enable_cloud_dlp = os.getenv("ENABLE_CLOUD_DLP", "false").lower() == "true"
        self.enable_cloud_dlp = enable_cloud_dlp
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self._dlp_client = dlp_client

    def redact_text(self, text: str) -> str:
        """Scrubs sensitive patterns from a text string."""
        if not text:
            return text

        scrubbed = text
        if self.enable_cloud_dlp and self.project_id:
            scrubbed = self._redact_with_cloud_dlp(scrubbed)

        for label, pattern in _PATTERNS:
            scrubbed = pattern.sub(f"[REDACTED_{label}]", scrubbed)
        return scrubbed

    def _redact_with_cloud_dlp(self, text: str) -> str:
        """Calls Cloud DLP deidentify_content when enabled, with safe fallback."""
        try:
            if self._dlp_client is None:
                from google.cloud import dlp_v2

                self._dlp_client = dlp_v2.DlpServiceClient()

            parent = f"projects/{self.project_id}/locations/global"
            inspect_config = {
                "info_types": [
                    {"name": "EMAIL_ADDRESS"},
                    {"name": "PHONE_NUMBER"},
                    {"name": "CREDIT_CARD_NUMBER"},
                    {"name": "US_SOCIAL_SECURITY_NUMBER"},
                ]
            }
            deidentify_config = {
                "info_type_transformations": {
                    "transformations": [
                        {
                            "primitive_transformation": {
                                "replace_with_info_type_config": {}
                            }
                        }
                    ]
                }
            }
            response = self._dlp_client.deidentify_content(
                request={
                    "parent": parent,
                    "deidentify_config": deidentify_config,
                    "inspect_config": inspect_config,
                    "item": {"value": text},
                }
            )
            return response.item.value
        except Exception:
            # Never block execution or leak raw text if DLP RPC fails; regex runs next.
            return text

    def redact_value(self, value: Any, *, field_name: str | None = None) -> Any:
        """Recursively redacts strings, dicts, lists, and sensitive-named fields."""
        if field_name and any(s in field_name.lower() for s in _SENSITIVE_FIELD_NAMES):
            return "[REDACTED_SECRET]"
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, dict):
            return {
                str(k): self.redact_value(v, field_name=str(k))
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [self.redact_value(item) for item in value]
        if isinstance(value, tuple):
            return [self.redact_value(item) for item in value]
        return value


_default_redactor = PIIRedactor()


def redact_pii(value: Any) -> Any:
    """Convenience helper using the process-wide PIIRedactor."""
    return _default_redactor.redact_value(value)
