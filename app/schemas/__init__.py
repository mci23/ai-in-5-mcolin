"""Pydantic schemas for Weather Comparison Agent."""

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

__all__ = [
    "CityComparisonDelta",
    "CityCoordinatesInput",
    "CityCoordinatesResult",
    "CityWeatherSnapshot",
    "CurrentWeatherInput",
    "HistoricalWeatherInput",
    "StructuredToolError",
    "WeatherComparisonInput",
    "WeatherComparisonResult",
    "WeatherFetchResult",
]
