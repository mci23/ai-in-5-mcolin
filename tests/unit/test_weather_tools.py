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

"""Unit tests for Open-Meteo weather tools, schemas, comparison math, and HITL."""

from __future__ import annotations

import json
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools.weather_tools import (
    compute_weather_comparison_metrics,
    confirm_historical_comparison_date,
    fetch_current_weather_for_city,
    fetch_historical_weather_for_city_and_date,
    resolve_city_coordinates,
    save_weather_preference_or_summary,
)


def _mock_httpx_response(payload: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_resolve_canonical_cities() -> None:
    cdmx = await resolve_city_coordinates("Mexico City", "Mexico")
    sp = await resolve_city_coordinates("Sao Paulo", "")
    ny = await resolve_city_coordinates("New York", "US")

    assert cdmx["status"] == "success"
    assert cdmx["timezone_name"] == "America/Mexico_City"
    assert sp["city_name"] == "São Paulo"
    assert ny["timezone_name"] == "America/New_York"


@pytest.mark.asyncio
async def test_resolve_city_invalid_input_returns_guided_error() -> None:
    res = await resolve_city_coordinates("A", "")
    assert res["status"] == "error"
    assert res["error_code"] == "INVALID_CITY_INPUT"
    assert "recovery_hint" in res


@pytest.mark.asyncio
async def test_fetch_current_weather_for_mexico_city() -> None:
    mock_payload = {
        "current": {
            "temperature_2m": 20.5,
            "relative_humidity_2m": 51,
            "weather_code": 2,
            "wind_speed_10m": 4.2,
        },
        "daily": {
            "time": [date.today().isoformat()],
            "weather_code": [2],
            "temperature_2m_max": [22.0],
            "temperature_2m_min": [11.0],
            "temperature_2m_mean": [16.5],
            "precipitation_sum": [0.4],
            "wind_speed_10m_max": [12.3],
            "relative_humidity_2m_mean": [58],
        },
    }
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.get.return_value = _mock_httpx_response(mock_payload)

    with patch("app.tools.weather_tools.httpx.AsyncClient", return_value=mock_client):
        result = await fetch_current_weather_for_city(
            "Mexico City", 19.4326, -99.1332, "America/Mexico_City"
        )

    assert result["status"] == "success"
    snap = result["snapshot"]
    assert snap["city_name"] == "Mexico City"
    assert snap["observation_kind"] == "current_day"
    assert snap["temperature_current_c"] == 20.5
    assert snap["temperature_mean_c"] == 16.5
    assert snap["condition_description"] == "Partly cloudy"


@pytest.mark.asyncio
async def test_fetch_historical_weather_success_and_out_of_range() -> None:
    too_old = (date.today() - timedelta(days=400)).isoformat()
    out_of_range = await fetch_historical_weather_for_city_and_date(
        "São Paulo", -23.5505, -46.6333, too_old, "America/Sao_Paulo"
    )
    assert out_of_range["status"] == "error"
    assert out_of_range["error_code"] == "DATE_OUT_OF_RANGE"
    assert "1 year" in out_of_range["message"]
    assert "365" in out_of_range["recovery_hint"]

    valid_date = (date.today() - timedelta(days=90)).isoformat()
    mock_archive = {
        "daily": {
            "time": [valid_date],
            "weather_code": [63],
            "temperature_2m_max": [18.0],
            "temperature_2m_min": [12.0],
            "temperature_2m_mean": [15.0],
            "precipitation_sum": [12.5],
            "wind_speed_10m_max": [18.0],
            "relative_humidity_2m_mean": [82],
        }
    }
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.get.return_value = _mock_httpx_response(mock_archive)

    with patch("app.tools.weather_tools.httpx.AsyncClient", return_value=mock_client):
        hist = await fetch_historical_weather_for_city_and_date(
            "São Paulo", -23.5505, -46.6333, valid_date, "America/Sao_Paulo"
        )

    assert hist["status"] == "success"
    assert hist["snapshot"]["observation_date"] == valid_date
    assert hist["snapshot"]["condition_description"] == "Moderate rain"


def test_compute_weather_comparison_metrics_exact_deltas() -> None:
    cdmx_snapshot = {
        "status": "success",
        "snapshot": {
            "city_name": "Mexico City",
            "observation_date": "2026-11-24",
            "observation_kind": "current_day",
            "timezone_name": "America/Mexico_City",
            "condition_description": "Partly cloudy",
            "weather_code": 2,
            "temperature_current_c": 20.0,
            "temperature_mean_c": 16.0,
            "temperature_max_c": 22.0,
            "temperature_min_c": 10.0,
            "relative_humidity_mean_pct": 55.0,
            "precipitation_sum_mm": 0.0,
            "wind_speed_max_kmh": 10.0,
        },
    }
    hist_snapshots = [
        {
            "city_name": "São Paulo",
            "observation_date": "2026-06-15",
            "observation_kind": "historical",
            "timezone_name": "America/Sao_Paulo",
            "condition_description": "Moderate rain",
            "weather_code": 63,
            "temperature_mean_c": 18.0,
            "temperature_max_c": 21.0,
            "temperature_min_c": 15.0,
            "relative_humidity_mean_pct": 85.0,
            "precipitation_sum_mm": 12.0,
            "wind_speed_max_kmh": 14.0,
        },
        {
            "city_name": "New York",
            "observation_date": "2026-06-15",
            "observation_kind": "historical",
            "timezone_name": "America/New_York",
            "condition_description": "Clear sky",
            "weather_code": 0,
            "temperature_mean_c": 24.0,
            "temperature_max_c": 28.0,
            "temperature_min_c": 20.0,
            "relative_humidity_mean_pct": 50.0,
            "precipitation_sum_mm": 0.0,
            "wind_speed_max_kmh": 15.0,
        },
    ]

    res = compute_weather_comparison_metrics(
        json.dumps(cdmx_snapshot), json.dumps(hist_snapshots), "both"
    )
    assert res["status"] == "success"
    assert len(res["deltas"]) == 2
    sp_delta = res["deltas"][0]
    assert sp_delta["mean_temp_delta_c"] == 2.0
    assert sp_delta["warmer_or_cooler"] == "warmer"
    assert sp_delta["wetter_or_drier"] == "wetter"
    assert sp_delta["precipitation_delta_mm"] == 12.0


def test_hitl_confirmation_flow_and_preference_saving() -> None:
    valid_date = (date.today() - timedelta(days=30)).isoformat()
    fake_ctx = SimpleNamespace(
        tool_confirmation=None,
        state={},
        request_confirmation=MagicMock(),
    )

    pending = confirm_historical_comparison_date(
        valid_date, "User asked for 'last month' without an exact day.", fake_ctx
    )
    assert pending["status"] == "pending_confirmation"
    fake_ctx.request_confirmation.assert_called_once()

    fake_ctx.tool_confirmation = SimpleNamespace(
        confirmed=True, payload={"target_date": valid_date}
    )
    approved = confirm_historical_comparison_date(valid_date, "Confirmed", fake_ctx)
    assert approved["status"] == "confirmed"
    assert fake_ctx.state["user:last_confirmed_date"] == valid_date

    saved = save_weather_preference_or_summary("F", "CDMX vs SP/NY summary", fake_ctx)
    assert saved["status"] == "success"
    assert saved["preferred_unit"] == "F"
    assert fake_ctx.state["user:preferred_unit"] == "F"
