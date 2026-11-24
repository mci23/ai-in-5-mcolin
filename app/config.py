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

"""Central application configuration and GCP Secret Manager integration."""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(
    os.getenv("WEATHER_AGENT_DATA_DIR", PROJECT_ROOT / ".weather_agent_data")
)


class SecretResolutionError(RuntimeError):
    """Raised when a required secret cannot be resolved securely."""


def fetch_secret_from_gcp_secret_manager(
    secret_name: str,
    project_id: str,
    version: str = "latest",
    client: Any | None = None,
) -> str:
    """Fetches a secret payload from Google Cloud Secret Manager.

    Args:
        secret_name: Short name of the secret in Secret Manager.
        project_id: GCP project ID hosting the secret.
        version: Secret version alias or number (defaults to 'latest').
        client: Optional pre-configured SecretManagerServiceClient for testing.

    Returns:
        The decoded UTF-8 secret value.
    """
    if not project_id or not secret_name:
        raise SecretResolutionError(
            "Both project_id and secret_name are required for Secret Manager lookup."
        )
    if client is None:
        from google.cloud import secretmanager

        client = secretmanager.SecretManagerServiceClient()

    resource_name = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"
    response = client.access_secret_version(request={"name": resource_name})
    return response.payload.data.decode("utf-8").strip()


def resolve_secret_or_env(
    env_var: str,
    secret_name: str | None = None,
    *,
    default: str | None = None,
    project_id: str | None = None,
    use_secret_manager: bool | None = None,
    secret_client: Any | None = None,
) -> str | None:
    """Resolves configuration secrets via Secret Manager first, then env vars.

    In Cloud Run or GCP deployments (`USE_SECRET_MANAGER=true`), secrets are
    retrieved from Secret Manager rather than hardcoded in code or config.
    In local prototype mode, environment variables (e.g. from `.env`) are used
    directly without requiring GCP Secret Manager access.
    """
    if use_secret_manager is None:
        use_secret_manager = os.getenv("USE_SECRET_MANAGER", "false").lower() == "true"
    resolved_project = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")

    if use_secret_manager and secret_name and resolved_project:
        try:
            return fetch_secret_from_gcp_secret_manager(
                secret_name=secret_name,
                project_id=resolved_project,
                client=secret_client,
            )
        except Exception as exc:
            # Fall back to env var if explicitly provided locally, otherwise fail loudly.
            env_fallback = os.getenv(env_var)
            if env_fallback:
                return env_fallback
            raise SecretResolutionError(
                f"Failed to load secret '{secret_name}' from GCP Secret Manager "
                f"and '{env_var}' is not set: {exc}"
            ) from exc

    return os.getenv(env_var, default)


@dataclass(frozen=True)
class AppSettings:
    """Runtime settings for the Weather Comparison Agent."""

    app_name: str
    root_agent_name: str
    pro_model: str
    fast_model: str
    open_meteo_forecast_url: str
    open_meteo_archive_url: str
    open_meteo_geocoding_url: str
    http_timeout_seconds: float
    session_db_url: str
    memory_store_path: Path
    enable_cloud_dlp: bool
    compaction_token_threshold: int
    compaction_event_retention: int
    gemini_api_key: str | None
    weather_api_key: str | None


@functools.lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Loads validated application settings from Secret Manager / environment."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    default_session_db = (
        f"sqlite+aiosqlite:///{(DATA_DIR / 'adk_sessions.db').as_posix()}"
    )
    default_memory_path = DATA_DIR / "long_term_memory.json"

    return AppSettings(
        app_name="app",
        root_agent_name="weather_comparison_agent",
        # Strategic Model Routing: Pro for planning & synthesis, Flash for fast tool sub-agents.
        pro_model=os.getenv("PRO_MODEL", "gemini-3.1-pro-preview"),
        fast_model=os.getenv("FAST_MODEL", "gemini-3.1-flash-lite"),
        open_meteo_forecast_url=os.getenv(
            "OPEN_METEO_FORECAST_URL", "https://api.open-meteo.com/v1/forecast"
        ),
        open_meteo_archive_url=os.getenv(
            "OPEN_METEO_ARCHIVE_URL", "https://archive-api.open-meteo.com/v1/archive"
        ),
        open_meteo_geocoding_url=os.getenv(
            "OPEN_METEO_GEOCODING_URL", "https://geocoding-api.open-meteo.com/v1/search"
        ),
        http_timeout_seconds=float(os.getenv("WEATHER_HTTP_TIMEOUT_SECONDS", "15.0")),
        session_db_url=os.getenv("SESSION_DB_URL", default_session_db),
        memory_store_path=Path(os.getenv("MEMORY_STORE_PATH", default_memory_path)),
        enable_cloud_dlp=os.getenv("ENABLE_CLOUD_DLP", "false").lower() == "true",
        compaction_token_threshold=int(
            os.getenv("COMPACTION_TOKEN_THRESHOLD", "16000")
        ),
        compaction_event_retention=int(os.getenv("COMPACTION_EVENT_RETENTION", "6")),
        gemini_api_key=resolve_secret_or_env(
            "GEMINI_API_KEY",
            secret_name=os.getenv("GEMINI_API_KEY_SECRET_NAME", "gemini-api-key"),
        ),
        weather_api_key=resolve_secret_or_env(
            "WEATHER_API_KEY",
            secret_name=os.getenv("WEATHER_API_KEY_SECRET_NAME", "weather-api-key"),
        ),
    )
