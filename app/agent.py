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

"""Multi-agent ADK orchestration for the Weather Comparison Agent."""

from __future__ import annotations

from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models import Gemini
from google.adk.tools import AgentTool
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.genai import types

from app.config import get_settings
from app.memory.async_memory import schedule_async_memory_callback
from app.observability.tracing import configure_tracing
from app.plugins.agentops_plugin import AgentOpsGuardrailsAndObservabilityPlugin
from app.prompts import (
    COMPARISON_ANALYST_INSTRUCTION,
    COORDINATOR_INSTRUCTION,
    CURRENT_MEXICO_CITY_INSTRUCTION,
    HISTORICAL_NEW_YORK_INSTRUCTION,
    HISTORICAL_SAO_PAULO_INSTRUCTION,
)
from app.tools.weather_tools import (
    compute_weather_comparison_metrics,
    confirm_historical_comparison_date,
    fetch_current_weather_for_city,
    fetch_historical_weather_for_city_and_date,
    resolve_city_coordinates,
    save_weather_preference_or_summary,
)

configure_tracing()
_settings = get_settings()
_RETRY_OPTIONS = types.HttpRetryOptions(attempts=3)


def _create_mexico_city_current_agent() -> Agent:
    return Agent(
        name="mexico_city_current_agent",
        model=Gemini(model=_settings.fast_model, retry_options=_RETRY_OPTIONS),
        description="Fetches today's live and daily weather for Mexico City.",
        instruction=CURRENT_MEXICO_CITY_INSTRUCTION,
        tools=[resolve_city_coordinates, fetch_current_weather_for_city],
        output_key="mexico_city_current_weather",
    )


def _create_sao_paulo_historical_agent() -> Agent:
    return Agent(
        name="sao_paulo_historical_agent",
        model=Gemini(model=_settings.fast_model, retry_options=_RETRY_OPTIONS),
        description="Fetches historical weather for São Paulo on a specific past-year date.",
        instruction=HISTORICAL_SAO_PAULO_INSTRUCTION,
        tools=[resolve_city_coordinates, fetch_historical_weather_for_city_and_date],
        output_key="sao_paulo_historical_weather",
    )


def _create_new_york_historical_agent() -> Agent:
    return Agent(
        name="new_york_historical_agent",
        model=Gemini(model=_settings.fast_model, retry_options=_RETRY_OPTIONS),
        description="Fetches historical weather for New York on a specific past-year date.",
        instruction=HISTORICAL_NEW_YORK_INSTRUCTION,
        tools=[resolve_city_coordinates, fetch_historical_weather_for_city_and_date],
        output_key="new_york_historical_weather",
    )


def _create_comparison_analyst_agent() -> Agent:
    return Agent(
        name="comparison_analyst_agent",
        model=Gemini(model=_settings.pro_model, retry_options=_RETRY_OPTIONS),
        description="Computes deterministic deltas and writes the final weather comparison.",
        instruction=COMPARISON_ANALYST_INSTRUCTION,
        tools=[compute_weather_comparison_metrics],
        output_key="comparison_analysis",
    )


def create_weather_comparison_pipeline() -> SequentialAgent:
    """Builds the Parallel -> Sequential multi-agent data gathering and analysis pipeline."""
    parallel_gatherer = ParallelAgent(
        name="parallel_weather_gatherer",
        description=(
            "Concurrently fetches Mexico City current weather and historical weather "
            "for São Paulo and New York."
        ),
        sub_agents=[
            _create_mexico_city_current_agent(),
            _create_sao_paulo_historical_agent(),
            _create_new_york_historical_agent(),
        ],
    )
    return SequentialAgent(
        name="weather_comparison_pipeline",
        description=(
            "Executes parallel city weather retrieval (Mexico City today, São Paulo and "
            "New York on a specific historical date) followed by deterministic comparative analysis."
        ),
        sub_agents=[
            parallel_gatherer,
            _create_comparison_analyst_agent(),
        ],
    )


def create_root_agent() -> Agent:
    """Builds the top-level Weather Comparison Coordinator agent."""
    pipeline = create_weather_comparison_pipeline()
    return Agent(
        # Keep in sync with create_params.root_agent_name in agents-cli-manifest.yaml.
        name=_settings.root_agent_name,
        model=Gemini(model=_settings.pro_model, retry_options=_RETRY_OPTIONS),
        description=(
            "Compares today's weather in Mexico City against a specific day in the past "
            "year in São Paulo and New York."
        ),
        instruction=COORDINATOR_INSTRUCTION,
        generate_content_config=types.GenerateContentConfig(temperature=0.1),
        tools=[
            PreloadMemoryTool(),
            confirm_historical_comparison_date,
            save_weather_preference_or_summary,
            AgentTool(agent=pipeline),
        ],
        after_agent_callback=schedule_async_memory_callback,
    )


root_agent = create_root_agent()

app = App(
    name=_settings.app_name,
    root_agent=root_agent,
    plugins=[AgentOpsGuardrailsAndObservabilityPlugin()],
    resumability_config=ResumabilityConfig(is_resumable=True),
    events_compaction_config=EventsCompactionConfig(
        token_threshold=_settings.compaction_token_threshold,
        event_retention_size=_settings.compaction_event_retention,
        summarizer=LlmEventSummarizer(
            llm=Gemini(model=_settings.fast_model, retry_options=_RETRY_OPTIONS)
        ),
    ),
)
