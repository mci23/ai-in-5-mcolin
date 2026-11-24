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

"""Runner-wide ADK BasePlugin for guardrails, policy checks, and intent/outcome telemetry."""

from __future__ import annotations

import logging
import re
import time
from datetime import date, timedelta
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools import BaseTool, ToolContext
from google.genai import types

from app.observability.logging_utils import get_structured_logger, log_structured_event
from app.observability.pii_redactor import redact_pii

logger = get_structured_logger("weather_comparison.plugin")

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b"),
    re.compile(r"(?i)\breveal\s+(?:your\s+)?(?:system\s+prompt|api\s+key|secrets?)\b"),
    re.compile(r"(?i)\boverride\s+(?:safety|guardrails|policy)\b"),
)


def validate_historical_date_window(
    target_date_str: str,
    *,
    today: date | None = None,
) -> tuple[bool, str | None]:
    """Checks that target_date_str is an ISO date strictly within the past 365 days."""
    ref_today = today or date.today()
    min_date = ref_today - timedelta(days=365)
    max_date = ref_today - timedelta(days=1)

    try:
        parsed = date.fromisoformat(target_date_str.strip())
    except ValueError:
        return (
            False,
            f"Date '{target_date_str}' is not valid YYYY-MM-DD. Choose between {min_date.isoformat()} and {max_date.isoformat()}.",
        )

    if parsed >= ref_today:
        return (
            False,
            f"Historical date '{target_date_str}' must be in the past (on or before {max_date.isoformat()}).",
        )
    if parsed < min_date:
        return (
            False,
            f"Historical date '{target_date_str}' is older than 1 year (earliest allowed: {min_date.isoformat()}).",
        )
    return True, None


class AgentOpsGuardrailsAndObservabilityPlugin(BasePlugin):
    """Enforces safety/policy guardrails and logs intent-vs-outcome telemetry."""

    def __init__(self) -> None:
        super().__init__(name="agentops_guardrails_and_observability")
        self._tool_start_times: dict[str, float] = {}

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        """Blocks obvious prompt-injection/secret-exfiltration attempts before LLM execution."""
        latest_user_text = ""
        if llm_request.contents:
            for content in reversed(llm_request.contents):
                if content.role == "user" and content.parts:
                    latest_user_text = " ".join(
                        part.text
                        for part in content.parts
                        if isinstance(getattr(part, "text", None), str) and part.text
                    )
                    if latest_user_text:
                        break

        for pattern in _INJECTION_PATTERNS:
            if pattern.search(latest_user_text):
                log_structured_event(
                    logger,
                    "guardrail_blocked_input",
                    "Blocked unsafe prompt injection attempt",
                    level=logging.WARNING,
                    agent_name=callback_context.agent_name,
                    invocation_id=callback_context.invocation_id,
                    matched_rule=pattern.pattern,
                )
                return LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part.from_text(
                                text=(
                                    "I cannot bypass safety policies or reveal system instructions/secrets. "
                                    "I can help you compare today's weather in Mexico City with historical "
                                    "weather in São Paulo and New York within the past year."
                                )
                            )
                        ],
                    )
                )

        log_structured_event(
            logger,
            "model_intent",
            "Dispatching LLM request",
            agent_name=callback_context.agent_name,
            invocation_id=callback_context.invocation_id,
            content_count=len(llm_request.contents or []),
        )
        return None

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        """Performs lightweight self-evaluation / grounding audit on LLM outputs."""
        text_parts: list[str] = []
        if llm_response.content and llm_response.content.parts:
            text_parts = [
                p.text
                for p in llm_response.content.parts
                if isinstance(getattr(p, "text", None), str) and p.text
            ]
        combined_text = "\n".join(text_parts)

        # Redact any accidental secret patterns from model text if ever echoed.
        scrubbed_text = redact_pii(combined_text)
        log_structured_event(
            logger,
            "model_outcome",
            "Received LLM response",
            agent_name=callback_context.agent_name,
            invocation_id=callback_context.invocation_id,
            response_chars=len(combined_text),
            has_function_calls=bool(
                llm_response.content
                and llm_response.content.parts
                and any(
                    getattr(p, "function_call", None)
                    for p in llm_response.content.parts
                )
            ),
        )
        if scrubbed_text != combined_text and llm_response.content and text_parts:
            return LlmResponse(
                content=types.Content(
                    role=llm_response.content.role or "model",
                    parts=[types.Part.from_text(text=scrubbed_text)],
                )
            )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        """Logs tool intent and enforces domain policy constraints before execution."""
        call_key = (
            getattr(tool_context, "function_call_id", None)
            or f"{tool_context.invocation_id}:{tool.name}:{id(tool_args)}"
        )
        self._tool_start_times[call_key] = time.perf_counter()

        log_structured_event(
            logger,
            "tool_intent",
            f"Agent intends to call tool '{tool.name}'",
            agent_name=tool_context.agent_name,
            invocation_id=tool_context.invocation_id,
            tool_name=tool.name,
            intended_args=redact_pii(tool_args),
        )

        # Policy guardrail: validate historical date window before invoking network APIs.
        if tool.name == "fetch_historical_weather_for_city_and_date":
            target_date = str(tool_args.get("target_date", ""))
            is_valid, reason = validate_historical_date_window(target_date)
            if not is_valid:
                log_structured_event(
                    logger,
                    "guardrail_blocked_tool",
                    f"Policy blocked '{tool.name}' due to out-of-range date",
                    level=logging.WARNING,
                    tool_name=tool.name,
                    reason=reason,
                )
                return {
                    "status": "error",
                    "error_code": "POLICY_DATE_OUT_OF_RANGE",
                    "message": reason,
                    "recovery_hint": (
                        "Inform the user that historical comparisons are restricted to the past "
                        "365 days (excluding today and future dates) and ask for or propose a valid date."
                    ),
                }
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Logs actual tool outcome and latency alongside original intent."""
        call_key = (
            getattr(tool_context, "function_call_id", None)
            or f"{tool_context.invocation_id}:{tool.name}:{id(tool_args)}"
        )
        started = self._tool_start_times.pop(call_key, None)
        duration_ms = (
            round((time.perf_counter() - started) * 1000, 2) if started else None
        )

        outcome_status = (
            result.get("status", "completed")
            if isinstance(result, dict)
            else "completed"
        )
        log_structured_event(
            logger,
            "tool_outcome",
            f"Tool '{tool.name}' finished with status '{outcome_status}'",
            agent_name=tool_context.agent_name,
            invocation_id=tool_context.invocation_id,
            tool_name=tool.name,
            intended_args=redact_pii(tool_args),
            outcome_status=outcome_status,
            duration_ms=duration_ms,
        )
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        """Logs tool exceptions and returns a guided recovery payload instead of crashing."""
        call_key = (
            getattr(tool_context, "function_call_id", None)
            or f"{tool_context.invocation_id}:{tool.name}:{id(tool_args)}"
        )
        started = self._tool_start_times.pop(call_key, None)
        duration_ms = (
            round((time.perf_counter() - started) * 1000, 2) if started else None
        )

        log_structured_event(
            logger,
            "tool_error",
            f"Tool '{tool.name}' raised an unhandled exception",
            level=logging.ERROR,
            agent_name=tool_context.agent_name,
            invocation_id=tool_context.invocation_id,
            tool_name=tool.name,
            intended_args=redact_pii(tool_args),
            duration_ms=duration_ms,
            error_type=type(error).__name__,
            error_message=str(error),
        )
        return {
            "status": "error",
            "error_code": "UNEXPECTED_TOOL_FAILURE",
            "message": f"Tool '{tool.name}' failed: {type(error).__name__}: {error}",
            "recovery_hint": (
                "Explain the transient failure clearly to the user and retry once or suggest "
                "re-checking the requested parameters."
            ),
        }
