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

"""Weather data tools backed by Open-Meteo with strict Pydantic schemas and HITL hooks."""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Any

import httpx
from google.adk.tools import ToolContext
from pydantic import ValidationError

from app.config import get_settings
from app.observability.logging_utils import get_structured_logger, log_structured_event
from app.observability.tracing import traced_span
from app.plugins.agentops_plugin import validate_historical_date_window
from app.schemas.weather import (
    CityComparisonDelta,
    CityCoordinatesInput,
    CityCoordinatesResult,
    CityWeatherSnapshot,
    CurrentWeatherInput,
    HistoricalWeatherInput,
    StructuredToolError,
    WeatherComparisonInput,
    WeatherComparisonResult,
    WeatherFetchResult,
)

logger = get_structured_logger("weather_comparison.tools")

# WMO Weather interpretation codes (Open-Meteo standard)
WMO_WEATHER_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

_CANONICAL_CITIES: dict[str, dict[str, Any]] = {
    "mexico city": {
        "city_name": "Mexico City",
        "country": "Mexico",
        "latitude": 19.4326,
        "longitude": -99.1332,
        "timezone_name": "America/Mexico_City",
    },
    "ciudad de mexico": {
        "city_name": "Mexico City",
        "country": "Mexico",
        "latitude": 19.4326,
        "longitude": -99.1332,
        "timezone_name": "America/Mexico_City",
    },
    "cdmx": {
        "city_name": "Mexico City",
        "country": "Mexico",
        "latitude": 19.4326,
        "longitude": -99.1332,
        "timezone_name": "America/Mexico_City",
    },
    "sao paulo": {
        "city_name": "São Paulo",
        "country": "Brazil",
        "latitude": -23.5505,
        "longitude": -46.6333,
        "timezone_name": "America/Sao_Paulo",
    },
    "são paulo": {
        "city_name": "São Paulo",
        "country": "Brazil",
        "latitude": -23.5505,
        "longitude": -46.6333,
        "timezone_name": "America/Sao_Paulo",
    },
    "new york": {
        "city_name": "New York",
        "country": "United States",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "timezone_name": "America/New_York",
    },
    "new york city": {
        "city_name": "New York",
        "country": "United States",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "timezone_name": "America/New_York",
    },
    "nyc": {
        "city_name": "New York",
        "country": "United States",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "timezone_name": "America/New_York",
    },
}


def _describe_wmo_code(code: int | None) -> str:
    if code is None:
        return "Unknown conditions"
    return WMO_WEATHER_CODES.get(int(code), f"WMO weather code {code}")


def _error_dict(error_code: str, message: str, recovery_hint: str) -> dict[str, Any]:
    return StructuredToolError(
        error_code=error_code,
        message=message,
        recovery_hint=recovery_hint,
    ).model_dump(mode="json")


def _c_to_f(temp_c: float) -> float:
    return round((temp_c * 9.0 / 5.0) + 32.0, 2)


def _c_delta_to_f_delta(delta_c: float) -> float:
    return round(delta_c * 9.0 / 5.0, 2)


async def resolve_city_coordinates(
    city_name: str,
    country_hint: str,
) -> dict[str, Any]:
    """Resolves a city's canonical coordinates and IANA timezone via Open-Meteo Geocoding.

    Use this tool before fetching weather whenever latitude, longitude, or timezone
    are not yet known for a city. Supports canonical fast resolution for Mexico City,
    São Paulo, and New York as well as live Open-Meteo geocoding lookup.

    Args:
        city_name: Human-readable city name (e.g., "Mexico City", "São Paulo", "New York").
        country_hint: Optional country name or ISO code to disambiguate (pass "" if unknown).

    Returns:
        dict with 'status' ("success" or "error"). On success, includes 'city_name',
        'country', 'latitude', 'longitude', and 'timezone_name'. On error, includes
        'error_code', 'message', and 'recovery_hint'.
    """
    try:
        validated = CityCoordinatesInput(city_name=city_name, country_hint=country_hint)
    except ValidationError as exc:
        return _error_dict(
            "INVALID_CITY_INPUT",
            f"Invalid city lookup parameters: {exc}",
            "Provide a non-empty city name (2-100 characters) and optional country hint.",
        )

    normalized = validated.city_name.strip().lower()
    if normalized in _CANONICAL_CITIES:
        return CityCoordinatesResult(**_CANONICAL_CITIES[normalized]).model_dump(
            mode="json"
        )

    settings = get_settings()
    with traced_span(
        "weather_api.open_meteo.geocode",
        attributes={"city_name": validated.city_name},
    ):
        try:
            async with httpx.AsyncClient(
                timeout=settings.http_timeout_seconds
            ) as client:
                resp = await client.get(
                    settings.open_meteo_geocoding_url,
                    params={
                        "name": validated.city_name,
                        "count": 5,
                        "language": "en",
                        "format": "json",
                    },
                )
                resp.raise_for_status()
                payload = resp.json()
        except httpx.HTTPError as exc:
            return _error_dict(
                "GEOCODING_API_ERROR",
                f"Geocoding lookup failed for '{validated.city_name}': {exc}",
                "Retry the geocoding lookup once, or use canonical names 'Mexico City', 'São Paulo', or 'New York'.",
            )

    results = payload.get("results") or []
    if validated.country_hint.strip():
        hint = validated.country_hint.strip().lower()
        filtered = [
            item
            for item in results
            if hint in str(item.get("country", "")).lower()
            or hint == str(item.get("country_code", "")).lower()
        ]
        if filtered:
            results = filtered

    if not results:
        return _error_dict(
            "CITY_NOT_FOUND",
            f"No geographic match found for '{validated.city_name}'.",
            "Check city spelling (e.g. 'Mexico City', 'Sao Paulo', 'New York') and retry.",
        )

    top = results[0]
    return CityCoordinatesResult(
        city_name=str(top.get("name", validated.city_name)),
        country=str(top.get("country", "")),
        latitude=float(top["latitude"]),
        longitude=float(top["longitude"]),
        timezone_name=str(top.get("timezone", "UTC")),
    ).model_dump(mode="json")


async def fetch_current_weather_for_city(
    city_name: str,
    latitude: float,
    longitude: float,
    timezone_name: str,
) -> dict[str, Any]:
    """Fetches today's live and daily summary weather for a city using Open-Meteo Forecast API.

    Use this tool for current-day observations (specifically Mexico City today).
    Retrieves instantaneous temperature alongside today's daily mean, max, min,
    precipitation total, mean humidity, wind speed max, and WMO condition text.

    Args:
        city_name: Canonical display name of the city (e.g., "Mexico City").
        latitude: City latitude in decimal degrees (-90 to 90).
        longitude: City longitude in decimal degrees (-180 to 180).
        timezone_name: IANA timezone name (e.g., "America/Mexico_City").

    Returns:
        dict with 'status' ("success" or "error"). On success, contains 'snapshot'
        conforming to CityWeatherSnapshot. On error, contains guided recovery hints.
    """
    try:
        validated = CurrentWeatherInput(
            city_name=city_name,
            latitude=latitude,
            longitude=longitude,
            timezone_name=timezone_name,
        )
    except ValidationError as exc:
        return _error_dict(
            "INVALID_COORDINATES",
            f"Invalid parameters for current weather fetch: {exc}",
            "Call resolve_city_coordinates first to obtain valid latitude, longitude, and timezone_name.",
        )

    settings = get_settings()
    params = {
        "latitude": validated.latitude,
        "longitude": validated.longitude,
        "timezone": validated.timezone_name,
        "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
        "daily": (
            "weather_code,temperature_2m_max,temperature_2m_min,"
            "temperature_2m_mean,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_mean"
        ),
        "forecast_days": 1,
    }
    with traced_span(
        "weather_api.open_meteo.forecast",
        attributes={
            "city_name": validated.city_name,
            "timezone": validated.timezone_name,
        },
    ):
        try:
            async with httpx.AsyncClient(
                timeout=settings.http_timeout_seconds
            ) as client:
                resp = await client.get(settings.open_meteo_forecast_url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            return _error_dict(
                "WEATHER_API_TIMEOUT",
                f"Timed out fetching current weather for {validated.city_name}.",
                "Retry fetch_current_weather_for_city once after the timeout.",
            )
        except httpx.HTTPError as exc:
            return _error_dict(
                "WEATHER_API_HTTP_ERROR",
                f"Failed to fetch current weather for {validated.city_name}: {exc}",
                "Verify coordinates and retry once, or inform the user that Open-Meteo is temporarily unavailable.",
            )

    current = data.get("current") or {}
    daily = data.get("daily") or {}
    daily_times = daily.get("time") or []
    if not daily_times:
        return _error_dict(
            "INCOMPLETE_WEATHER_DATA",
            f"Open-Meteo returned no daily summary for {validated.city_name}.",
            "Retry the forecast fetch or check the timezone parameter.",
        )

    weather_code = int(
        (daily.get("weather_code") or [current.get("weather_code", 0)])[0]
    )
    temp_max = float(daily["temperature_2m_max"][0])
    temp_min = float(daily["temperature_2m_min"][0])
    mean_list = daily.get("temperature_2m_mean") or []
    temp_mean = (
        float(mean_list[0])
        if mean_list and mean_list[0] is not None
        else round((temp_max + temp_min) / 2.0, 2)
    )
    hum_list = daily.get("relative_humidity_2m_mean") or []
    humidity = (
        float(hum_list[0])
        if hum_list and hum_list[0] is not None
        else (
            float(current["relative_humidity_2m"])
            if current.get("relative_humidity_2m") is not None
            else None
        )
    )

    snapshot = CityWeatherSnapshot(
        city_name=validated.city_name,
        observation_date=str(daily_times[0]),
        observation_kind="current_day",
        timezone_name=validated.timezone_name,
        condition_description=_describe_wmo_code(weather_code),
        weather_code=weather_code,
        temperature_current_c=(
            float(current["temperature_2m"])
            if current.get("temperature_2m") is not None
            else None
        ),
        temperature_mean_c=temp_mean,
        temperature_max_c=temp_max,
        temperature_min_c=temp_min,
        relative_humidity_mean_pct=humidity,
        precipitation_sum_mm=float((daily.get("precipitation_sum") or [0.0])[0] or 0.0),
        wind_speed_max_kmh=float((daily.get("wind_speed_10m_max") or [0.0])[0] or 0.0),
        data_source="open-meteo-forecast",
    )
    return WeatherFetchResult(snapshot=snapshot).model_dump(mode="json")


async def fetch_historical_weather_for_city_and_date(
    city_name: str,
    latitude: float,
    longitude: float,
    target_date: str,
    timezone_name: str,
) -> dict[str, Any]:
    """Fetches historical daily weather for a city on a specific date in the past year.

    Queries Open-Meteo's Historical Archive API (automatically falling back to Forecast
    past-days data for the most recent 7 days if archive ERA5 ingestion lags).
    Enforces that target_date is in ISO YYYY-MM-DD format within the past 365 days.

    Args:
        city_name: Canonical display name of the city (e.g., "São Paulo", "New York").
        latitude: City latitude in decimal degrees (-90 to 90).
        longitude: City longitude in decimal degrees (-180 to 180).
        target_date: Target historical date in YYYY-MM-DD format (strictly within past 365 days).
        timezone_name: IANA timezone name (e.g., "America/Sao_Paulo", "America/New_York").

    Returns:
        dict with 'status' ("success" or "error"). On success, contains 'snapshot'
        conforming to CityWeatherSnapshot. On error, returns actionable recovery instructions.
    """
    try:
        validated = HistoricalWeatherInput(
            city_name=city_name,
            latitude=latitude,
            longitude=longitude,
            target_date=target_date,
            timezone_name=timezone_name,
        )
    except ValidationError as exc:
        return _error_dict(
            "INVALID_HISTORICAL_INPUT",
            f"Invalid historical weather request: {exc}",
            "Format target_date as YYYY-MM-DD within the past 365 days and obtain valid coordinates first.",
        )

    is_valid, reason = validate_historical_date_window(validated.target_date)
    if not is_valid:
        return _error_dict(
            "DATE_OUT_OF_RANGE",
            reason or "Requested date is outside the past-year comparison window.",
            "Ask the user to pick a date between yesterday and 365 days ago in YYYY-MM-DD format.",
        )

    settings = get_settings()
    daily_vars = (
        "weather_code,temperature_2m_max,temperature_2m_min,"
        "temperature_2m_mean,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_mean"
    )
    params = {
        "latitude": validated.latitude,
        "longitude": validated.longitude,
        "timezone": validated.timezone_name,
        "start_date": validated.target_date,
        "end_date": validated.target_date,
        "daily": daily_vars,
    }

    parsed_target = date.fromisoformat(validated.target_date)
    source_name = "open-meteo-archive"
    data: dict[str, Any] = {}

    with traced_span(
        "weather_api.open_meteo.archive",
        attributes={
            "city_name": validated.city_name,
            "target_date": validated.target_date,
        },
    ):
        try:
            async with httpx.AsyncClient(
                timeout=settings.http_timeout_seconds
            ) as client:
                resp = await client.get(settings.open_meteo_archive_url, params=params)
                resp.raise_for_status()
                data = resp.json()

                daily = data.get("daily") or {}
                max_vals = daily.get("temperature_2m_max") or []
                # ERA5 archive can lag 2-5 days behind today; fall back to forecast past_days.
                if (not max_vals or max_vals[0] is None) and (
                    date.today() - parsed_target
                ) <= timedelta(days=10):
                    source_name = "open-meteo-forecast-recent"
                    resp_recent = await client.get(
                        settings.open_meteo_forecast_url, params=params
                    )
                    resp_recent.raise_for_status()
                    data = resp_recent.json()
        except httpx.TimeoutException:
            return _error_dict(
                "HISTORICAL_API_TIMEOUT",
                f"Timed out fetching historical weather for {validated.city_name} on {validated.target_date}.",
                "Retry fetch_historical_weather_for_city_and_date once.",
            )
        except httpx.HTTPError as exc:
            return _error_dict(
                "HISTORICAL_API_HTTP_ERROR",
                f"Historical weather request failed for {validated.city_name}: {exc}",
                "Verify date is YYYY-MM-DD within the past 365 days and retry.",
            )

    daily = data.get("daily") or {}
    daily_times = daily.get("time") or []
    max_list = daily.get("temperature_2m_max") or []
    min_list = daily.get("temperature_2m_min") or []

    if not daily_times or not max_list or max_list[0] is None or min_list[0] is None:
        return _error_dict(
            "HISTORICAL_DATA_UNAVAILABLE",
            f"Weather archive returned incomplete data for {validated.city_name} on {validated.target_date}.",
            "Suggest choosing a date at least 5 days prior to today so ERA5 archive data is finalized.",
        )

    weather_code = int((daily.get("weather_code") or [0])[0] or 0)
    temp_max = float(max_list[0])
    temp_min = float(min_list[0])
    mean_list = daily.get("temperature_2m_mean") or []
    temp_mean = (
        float(mean_list[0])
        if mean_list and mean_list[0] is not None
        else round((temp_max + temp_min) / 2.0, 2)
    )
    hum_list = daily.get("relative_humidity_2m_mean") or []
    humidity = float(hum_list[0]) if hum_list and hum_list[0] is not None else None

    snapshot = CityWeatherSnapshot(
        city_name=validated.city_name,
        observation_date=str(daily_times[0]),
        observation_kind="historical",
        timezone_name=validated.timezone_name,
        condition_description=_describe_wmo_code(weather_code),
        weather_code=weather_code,
        temperature_current_c=None,
        temperature_mean_c=temp_mean,
        temperature_max_c=temp_max,
        temperature_min_c=temp_min,
        relative_humidity_mean_pct=humidity,
        precipitation_sum_mm=float((daily.get("precipitation_sum") or [0.0])[0] or 0.0),
        wind_speed_max_kmh=float((daily.get("wind_speed_10m_max") or [0.0])[0] or 0.0),
        data_source=source_name,
    )
    return WeatherFetchResult(snapshot=snapshot).model_dump(mode="json")


def compute_weather_comparison_metrics(
    baseline_snapshot_json: str,
    historical_snapshots_json: str,
    preferred_unit: str,
) -> dict[str, Any]:
    """Deterministically computes weather deltas comparing Mexico City today against historical cities.

    Accepts serialized JSON strings (from the gatherer tools/state) for the baseline
    current-day snapshot and a JSON array of historical city snapshots, validates them
    with Pydantic, and calculates exact deltas without LLM arithmetic.

    Args:
        baseline_snapshot_json: JSON string representing Mexico City's current-day CityWeatherSnapshot
            (or a WeatherFetchResult envelope containing a 'snapshot' field).
        historical_snapshots_json: JSON array string of historical CityWeatherSnapshot objects
            (or WeatherFetchResult envelopes) for São Paulo and New York.
        preferred_unit: Output temperature emphasis: "C", "F", or "both".

    Returns:
        dict conforming to WeatherComparisonResult with computed 'deltas' and 'summary_highlights',
        or a guided error dict if inputs are malformed.
    """
    try:
        raw_base = json.loads(baseline_snapshot_json)
        if isinstance(raw_base, dict) and "snapshot" in raw_base:
            raw_base = raw_base["snapshot"]
        raw_hist = json.loads(historical_snapshots_json)
        if not isinstance(raw_hist, list):
            raw_hist = [raw_hist]
        normalized_hist = [
            item["snapshot"] if isinstance(item, dict) and "snapshot" in item else item
            for item in raw_hist
        ]
        unit = preferred_unit.strip().upper() if preferred_unit else "both"
        if unit not in ("C", "F", "both"):
            unit = "both"

        validated = WeatherComparisonInput(
            baseline_snapshot=CityWeatherSnapshot.model_validate(raw_base),
            historical_snapshots=[
                CityWeatherSnapshot.model_validate(h) for h in normalized_hist
            ],
            preferred_unit=unit,  # type: ignore[arg-type]
        )
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        return _error_dict(
            "INVALID_COMPARISON_PAYLOAD",
            f"Could not parse weather snapshots for comparison: {exc}",
            "Pass valid JSON matching CityWeatherSnapshot returned by fetch_current_weather_for_city and fetch_historical_weather_for_city_and_date.",
        )

    with traced_span(
        "weather_comparison.compute_metrics",
        attributes={
            "baseline_city": validated.baseline_snapshot.city_name,
            "historical_count": len(validated.historical_snapshots),
        },
    ):
        base = validated.baseline_snapshot
        deltas: list[CityComparisonDelta] = []
        highlights: list[str] = []

        for hist in validated.historical_snapshots:
            mean_delta_c = round(hist.temperature_mean_c - base.temperature_mean_c, 2)
            mean_delta_f = _c_delta_to_f_delta(mean_delta_c)
            max_delta_c = round(hist.temperature_max_c - base.temperature_max_c, 2)
            min_delta_c = round(hist.temperature_min_c - base.temperature_min_c, 2)
            precip_delta = round(
                hist.precipitation_sum_mm - base.precipitation_sum_mm, 2
            )
            wind_delta = round(hist.wind_speed_max_kmh - base.wind_speed_max_kmh, 2)
            humidity_delta = (
                round(
                    hist.relative_humidity_mean_pct - base.relative_humidity_mean_pct,
                    2,
                )
                if hist.relative_humidity_mean_pct is not None
                and base.relative_humidity_mean_pct is not None
                else None
            )

            warmer_or_cooler = (
                "warmer"
                if mean_delta_c > 0.05
                else ("cooler" if mean_delta_c < -0.05 else "identical")
            )
            wetter_or_drier = (
                "wetter"
                if precip_delta > 0.05
                else ("drier" if precip_delta < -0.05 else "identical")
            )

            delta = CityComparisonDelta(
                historical_city=hist.city_name,
                historical_date=hist.observation_date,
                baseline_city=base.city_name,
                baseline_date=base.observation_date,
                mean_temp_delta_c=mean_delta_c,
                mean_temp_delta_f=mean_delta_f,
                max_temp_delta_c=max_delta_c,
                min_temp_delta_c=min_delta_c,
                precipitation_delta_mm=precip_delta,
                wind_speed_delta_kmh=wind_delta,
                humidity_delta_pct=humidity_delta,
                warmer_or_cooler=warmer_or_cooler,
                wetter_or_drier=wetter_or_drier,
            )
            deltas.append(delta)

            temp_dir = (
                f"{abs(mean_delta_c):.1f}°C ({abs(mean_delta_f):.1f}°F) {warmer_or_cooler}"
                if warmer_or_cooler != "identical"
                else "virtually the same mean temperature as"
            )
            highlights.append(
                f"{hist.city_name} on {hist.observation_date} ({hist.temperature_mean_c:.1f}°C / "
                f"{_c_to_f(hist.temperature_mean_c):.1f}°F, {hist.condition_description}) was "
                f"{temp_dir} than {base.city_name} today ({base.temperature_mean_c:.1f}°C / "
                f"{_c_to_f(base.temperature_mean_c):.1f}°F, {base.condition_description})."
            )

        return WeatherComparisonResult(
            preferred_unit=validated.preferred_unit,
            baseline_city=base,
            historical_cities=validated.historical_snapshots,
            deltas=deltas,
            summary_highlights=highlights,
        ).model_dump(mode="json")


def confirm_historical_comparison_date(
    proposed_date: str,
    reason: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Requests explicit human-in-the-loop confirmation for a proposed historical date.

    Call this tool ONLY when the user did not specify an exact date (for example,
    "compare with a day last year") or asked for a relative/ambiguous date, so the
    human user can confirm or adjust the exact YYYY-MM-DD date before running the
    historical comparison pipeline.

    Args:
        proposed_date: Proposed historical date in YYYY-MM-DD format within the past 365 days.
        reason: Brief explanation of why this date was selected for confirmation.

    Returns:
        dict indicating 'status' ("confirmed", "pending_confirmation", "rejected", or "error")
        and the resolved 'target_date'.
    """
    is_valid, err_msg = validate_historical_date_window(proposed_date)
    if not is_valid:
        return _error_dict(
            "INVALID_PROPOSED_DATE",
            err_msg or f"Proposed date '{proposed_date}' is outside the past 365 days.",
            "Choose a proposed_date strictly within the past 365 days (YYYY-MM-DD) before requesting confirmation.",
        )

    confirmation = tool_context.tool_confirmation
    if confirmation is None:
        log_structured_event(
            logger,
            "hitl_confirmation_requested",
            "Requesting human confirmation for proposed historical date",
            proposed_date=proposed_date,
            reason=reason,
        )
        tool_context.request_confirmation(
            hint=(
                f"Confirm historical comparison date {proposed_date} for São Paulo and New York? "
                f"Reason: {reason}"
            ),
            payload={"proposed_date": proposed_date, "reason": reason},
        )
        return {
            "status": "pending_confirmation",
            "proposed_date": proposed_date,
            "message": (
                f"Waiting for user confirmation to compare against {proposed_date} ({reason})."
            ),
        }

    if confirmation.confirmed:
        confirmed_date = proposed_date
        if isinstance(confirmation.payload, dict) and confirmation.payload.get(
            "target_date"
        ):
            confirmed_date = str(confirmation.payload["target_date"])
        tool_context.state["user:last_confirmed_date"] = confirmed_date
        log_structured_event(
            logger,
            "hitl_confirmation_approved",
            "Human approved historical comparison date",
            target_date=confirmed_date,
        )
        return {
            "status": "confirmed",
            "target_date": confirmed_date,
        }

    log_structured_event(
        logger,
        "hitl_confirmation_rejected",
        "Human declined proposed historical comparison date",
        level=logging.INFO,
        proposed_date=proposed_date,
    )
    return {
        "status": "rejected",
        "proposed_date": proposed_date,
        "recovery_hint": "Ask the user which exact YYYY-MM-DD date within the past year they want instead.",
    }


def save_weather_preference_or_summary(
    preferred_unit: str,
    comparison_summary: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Saves user unit preference and latest comparison summary to persistent session state.

    Use this after completing a comparison or when the user specifies a preferred
    temperature unit ("C", "F", or "both") so it persists across conversation turns.

    Args:
        preferred_unit: User temperature preference: "C", "F", or "both".
        comparison_summary: Concise PII-free summary of the weather comparison just performed.

    Returns:
        dict confirming the updated session state keys.
    """
    normalized_unit = preferred_unit.strip().upper()
    if normalized_unit not in ("C", "F", "BOTH"):
        return _error_dict(
            "INVALID_UNIT_PREFERENCE",
            f"Unsupported temperature unit '{preferred_unit}'.",
            "Specify 'C', 'F', or 'both'.",
        )
    canonical_unit = "both" if normalized_unit == "BOTH" else normalized_unit
    tool_context.state["user:preferred_unit"] = canonical_unit
    if comparison_summary.strip():
        history = list(tool_context.state.get("user:recent_comparisons", []))
        history.append(comparison_summary.strip()[:500])
        tool_context.state["user:recent_comparisons"] = history[-5:]
    return {
        "status": "success",
        "preferred_unit": canonical_unit,
        "saved_history_count": len(
            tool_context.state.get("user:recent_comparisons", [])
        ),
    }
