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

"""Non-blocking background memory consolidation tasks."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from app.observability.logging_utils import get_structured_logger, log_structured_event
from app.observability.tracing import traced_span

logger = get_structured_logger("weather_comparison.async_memory")

_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


async def _consolidate_session_memory(callback_context: CallbackContext) -> None:
    """Persists session events to the configured MemoryService in the background."""
    with traced_span(
        "memory.async_consolidation",
        attributes={
            "agent_name": callback_context.agent_name,
            "invocation_id": callback_context.invocation_id,
        },
    ):
        try:
            await callback_context.add_session_to_memory()
            log_structured_event(
                logger,
                "async_memory_consolidated",
                "Background memory consolidation completed",
                agent_name=callback_context.agent_name,
                invocation_id=callback_context.invocation_id,
            )
        except ValueError as exc:
            # Occurs when Runner is instantiated without a memory_service (e.g. minimal tests).
            log_structured_event(
                logger,
                "async_memory_skipped",
                "Skipped async memory consolidation: no memory service attached",
                level=logging.DEBUG,
                reason=str(exc),
            )
        except Exception as exc:
            log_structured_event(
                logger,
                "async_memory_failed",
                "Background memory consolidation failed",
                level=logging.WARNING,
                error=str(exc),
            )


async def schedule_async_memory_callback(
    callback_context: CallbackContext,
) -> types.Content | None:
    """ADK after_agent_callback that schedules non-blocking memory persistence.

    Returns ``None`` immediately so the user response is never delayed by memory
    redaction, indexing, or storage I/O.
    """
    try:
        inv_ctx = callback_context.get_invocation_context()
        if getattr(inv_ctx, "memory_service", None) is None:
            return None
    except Exception:
        return None

    task = asyncio.create_task(_consolidate_session_memory(callback_context))
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    log_structured_event(
        logger,
        "async_memory_scheduled",
        "Scheduled background memory consolidation task",
        agent_name=callback_context.agent_name,
        pending_tasks=len(_BACKGROUND_TASKS),
    )
    return None


async def flush_pending_memory_tasks() -> None:
    """Awaits all in-flight background memory tasks (useful for shutdown and tests)."""
    if not _BACKGROUND_TASKS:
        return
    tasks = list(_BACKGROUND_TASKS)
    await asyncio.gather(*tasks, return_exceptions=True)
