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

"""Strict Pydantic v2 schemas for weather tools, comparisons, and errors."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class StructuredToolError(BaseModel):
    """Guided error response returned to the LLM instead of crashing."""

    status: Literal["error"] = "error"
    error_code: str = Field(description="Machine-readable error identifier.")
    message: str = Field(description="Human-readable explanation of what failed.")
    recovery_hint: str = Field(
        description="Actionable instructions explaining how the agent can recover or respond."
    )


class CityCoordinatesInput(BaseModel):
    """Validated arguments for resolving a city's geographic coordinates."""

    city_name: str = Field(
        min_length=2, max_length=100, description="Target city name."
    )
    country_hint: str = Field(
        default="",
        max_length=80,
        description="Optional ISO-2 country code or country name for disambiguation.",
    )


class CityCoordinatesResult(BaseModel):
    """Canonical coordinate and timezone metadata for a resolved city."""

    status: Literal["success"] = "success"
    city_name: str = Field(description="Canonical resolved city name.")
    country: str = Field(description="Country name or code.")
    latitude: float = Field(
        ge=-90.0, le=90.0, description="Latitude in decimal degrees."
    )
    longitude: float = Field(
        ge=-180.0, le=180.0, description="Longitude in decimal degrees."
    )
    timezone_name: str = Field(
        description="IANA timezone identifier (e.g., America/Mexico_City)."
    )


class CurrentWeatherInput(BaseModel):
    """Validated input for fetching current-day weather."""

    city_name: str = Field(min_length=2, max_length=100)
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    timezone_name: str = Field(min_length=2, max_length=80)


class HistoricalWeatherInput(BaseModel):
    """Validated input for fetching historical weather on a specific date."""

    city_name: str = Field(min_length=2, max_length=100)
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    target_date: str = Field(description="Target historical date in YYYY-MM-DD format.")
    timezone_name: str = Field(min_length=2, max_length=80)

    @field_validator("target_date")
    @classmethod
    def validate_iso_date(cls, value: str) -> str:
        from datetime import date

        try:
            date.fromisoformat(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid target_date '{value}'. Expected strict ISO format YYYY-MM-DD."
            ) from exc
        return value.strip()


class CityWeatherSnapshot(BaseModel):
    """Normalized weather metrics for a single city on a single date."""

    city_name: str
    observation_date: str = Field(description="ISO YYYY-MM-DD date of observation.")
    observation_kind: Literal["current_day", "historical"]
    timezone_name: str
    condition_description: str
    weather_code: int
    temperature_current_c: float | None = Field(
        default=None,
        description="Instantaneous temperature in °C (current-day queries only).",
    )
    temperature_mean_c: float = Field(description="Daily mean temperature in °C.")
    temperature_max_c: float = Field(description="Daily maximum temperature in °C.")
    temperature_min_c: float = Field(description="Daily minimum temperature in °C.")
    relative_humidity_mean_pct: float | None = Field(
        default=None, ge=0.0, le=100.0, description="Mean relative humidity percentage."
    )
    precipitation_sum_mm: float = Field(
        ge=0.0, description="Total daily precipitation in mm."
    )
    wind_speed_max_kmh: float = Field(
        ge=0.0, description="Maximum daily wind speed in km/h."
    )
    data_source: str = Field(default="open-meteo")


class WeatherFetchResult(BaseModel):
    """Tool envelope returning a validated CityWeatherSnapshot."""

    status: Literal["success"] = "success"
    snapshot: CityWeatherSnapshot


class CityComparisonDelta(BaseModel):
    """Computed delta between a historical city snapshot and the Mexico City baseline."""

    historical_city: str
    historical_date: str
    baseline_city: str
    baseline_date: str
    mean_temp_delta_c: float = Field(
        description="historical_mean_c - baseline_mean_c (positive means historical city was warmer)."
    )
    mean_temp_delta_f: float = Field(
        description="Temperature delta expressed in °F difference."
    )
    max_temp_delta_c: float
    min_temp_delta_c: float
    precipitation_delta_mm: float = Field(
        description="historical_precip_mm - baseline_precip_mm."
    )
    wind_speed_delta_kmh: float
    humidity_delta_pct: float | None = None
    warmer_or_cooler: Literal["warmer", "cooler", "identical"]
    wetter_or_drier: Literal["wetter", "drier", "identical"]


class WeatherComparisonInput(BaseModel):
    """Validated input for deterministic comparison math."""

    baseline_snapshot: CityWeatherSnapshot
    historical_snapshots: list[CityWeatherSnapshot] = Field(min_length=1, max_length=5)
    preferred_unit: Literal["C", "F", "both"] = "both"


class WeatherComparisonResult(BaseModel):
    """Structured weather comparison output across cities."""

    status: Literal["success"] = "success"
    preferred_unit: Literal["C", "F", "both"]
    baseline_city: CityWeatherSnapshot
    historical_cities: list[CityWeatherSnapshot]
    deltas: list[CityComparisonDelta]
    summary_highlights: list[str]
