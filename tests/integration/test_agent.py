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

from __future__ import annotations

import os
from pathlib import Path

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import app as adk_app
from app.app_utils import services


def test_app_services_and_persistent_stores_wire_cleanly(
    tmp_path: Path, monkeypatch
) -> None:
    """Verifies persistent session and memory service initialization without LLM calls."""
    monkeypatch.setenv("WEATHER_AGENT_DATA_DIR", str(tmp_path))
    services.get_session_service.cache_clear()
    services.get_memory_service.cache_clear()

    session_service = services.get_session_service()
    memory_service = services.get_memory_service()

    assert session_service is not None
    assert memory_service is not None
    assert adk_app.root_agent is not None
    assert adk_app.root_agent.name == "weather_comparison_agent"


def test_agent_stream() -> None:
    """Integration test for end-to-end ADK runner streaming (requires Gemini credentials)."""
    if not (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("INTEGRATION_TEST")
    ):
        return

    session_service = InMemorySessionService()
    memory_service = services.get_memory_service()

    session = session_service.create_session_sync(
        user_id="test_user", app_name=adk_app.name
    )
    runner = Runner(
        app=adk_app,
        session_service=session_service,
        memory_service=memory_service,
    )

    message = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text="Compare today's weather in Mexico City with 2026-06-15 in São Paulo and New York."
            )
        ],
    )

    events = list(
        runner.run(
            new_message=message,
            user_id="test_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )
    assert len(events) > 0, "Expected at least one message"

    has_text_content = any(
        event.content
        and event.content.parts
        and any(part.text for part in event.content.parts)
        for event in events
    )
    assert has_text_content, "Expected at least one message with text content"
