# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from typing import Literal

from pydantic import BaseModel, Field

# ==============================================================================
# Pydantic Schemas for Parameter Validation
# ==============================================================================


class WeatherForecastRequest(BaseModel):
    city: str = Field(
        ...,
        description="The city name to check the weather forecast for. E.g. 'Paris', 'Tokyo'.",
    )
    days: int = Field(
        default=3, ge=1, le=7, description="Number of forecast days required (1 to 7)."
    )


class RouteRequest(BaseModel):
    origin: str = Field(
        ..., description="Starting point city or address. E.g. 'San Francisco'."
    )
    destination: str = Field(
        ..., description="Destination city or address. E.g. 'Los Angeles'."
    )
    preference: Literal["greenest", "fastest", "cheapest"] = Field(
        default="greenest", description="Routing optimization preference."
    )


class AttractionRequest(BaseModel):
    city: str = Field(
        ..., description="City name to search attractions in. E.g. 'Seattle'."
    )
    category: Literal["nature", "culture", "food", "sights"] = Field(
        default="sights", description="Category of attractions to search for."
    )


# ==============================================================================
# Tool Implementations
# ==============================================================================


def get_weather_forecast(req: WeatherForecastRequest) -> str:
    """Fetches a multi-day weather forecast with travel recommendations.

    Args:
        req: Validated request containing city name and forecast length.

    Returns:
        A detailed text report of the forecast and weather advisory.
    """
    city = req.city.strip().lower()
    days = req.days

    # Mock weather database
    weather_data = {
        "san francisco": {
            "temp": "58F",
            "condition": "foggy",
            "advisory": "Bring a light jacket; expect cool evenings.",
        },
        "seattle": {
            "temp": "52F",
            "condition": "rainy",
            "advisory": "Rain expected. Bring an umbrella or raincoat.",
        },
        "paris": {
            "temp": "72F",
            "condition": "sunny",
            "advisory": "Excellent walking weather! Wear sunscreen.",
        },
        "tokyo": {
            "temp": "68F",
            "condition": "windy",
            "advisory": "Breezy conditions. Light windbreaker recommended.",
        },
    }

    match_data = weather_data.get(
        city,
        {
            "temp": "65F",
            "condition": "partly cloudy",
            "advisory": "Standard travel gear is sufficient.",
        },
    )

    lines = [f"--- Weather Forecast for {req.city.title()} ({days} Days) ---"]
    for day in range(1, days + 1):
        date_str = (datetime.date.today() + datetime.timedelta(days=day - 1)).strftime(
            "%A, %b %d"
        )
        lines.append(
            f"- {date_str}: Temp: {match_data['temp']}, Condition: {match_data['condition']}"
        )

    lines.append(f"Advisory: {match_data['advisory']}")
    return "\n".join(lines)


def get_eco_routes(req: RouteRequest) -> str:
    """Computes route options between origin and destination with carbon emission stats.

    Args:
        req: Validated routing request with origin, destination, and optimization preference.

    Returns:
        A text summary of transit options, pricing, and carbon emissions.
    """
    origin = req.origin.strip().title()
    destination = req.destination.strip().title()
    pref = req.preference

    # Mock routing engine
    # Outputs format: Option Name | Duration | Cost | Carbon (kg CO2)
    options = [
        {
            "mode": "Train",
            "duration": "4h 30m",
            "cost": 75,
            "carbon": 12.5,
            "note": "Most eco-friendly choice.",
        },
        {
            "mode": "Electric Bus",
            "duration": "6h 15m",
            "cost": 35,
            "carbon": 8.0,
            "note": "Cheapest and highly eco-friendly.",
        },
        {
            "mode": "Flight",
            "duration": "1h 15m",
            "cost": 150,
            "carbon": 145.0,
            "note": "Fastest choice but high carbon footprint.",
        },
        {
            "mode": "Hybrid Car Rental",
            "duration": "5h 45m",
            "cost": 120,
            "carbon": 42.0,
            "note": "Flexible schedule but higher emissions than transit.",
        },
    ]

    # Sort or highlight based on preference
    if pref == "greenest":
        options.sort(key=lambda x: x["carbon"])
    elif pref == "cheapest":
        options.sort(key=lambda x: x["cost"])
    else:  # fastest
        # We manually map mode durations for sorting
        def duration_key(x):
            if "h" in x["duration"]:
                parts = x["duration"].split("h")
                hours = int(parts[0].strip())
                mins = int(parts[1].replace("m", "").strip()) if parts[1] else 0
                return hours * 60 + mins
            return int(x["duration"].replace("m", "").strip())

        options.sort(key=duration_key)

    lines = [f"--- Route Options: {origin} to {destination} (Optimized for {pref}) ---"]
    for idx, opt in enumerate(options, 1):
        rec_marker = "🌟 [Recommended]" if idx == 1 else ""
        lines.append(
            f"{idx}. {opt['mode']} {rec_marker}\n"
            f"   Duration: {opt['duration']} | Cost: ${opt['cost']} | Carbon Footprint: {opt['carbon']} kg CO2\n"
            f"   Note: {opt['note']}"
        )

    return "\n".join(lines)


def get_attractions(req: AttractionRequest) -> str:
    """Finds top local attractions in a city based on category, highlighting eco-ratings.

    Args:
        req: Validated attractions search request.

    Returns:
        A list of attractions with descriptions and eco-ratings.
    """
    city = req.city.strip().title()
    cat = req.category

    # Mock attraction database
    attraction_db = {
        "nature": [
            {
                "name": "Green City Botanical Gardens",
                "cost": "Free",
                "eco_rating": "Green Certified",
                "description": "Lush local flora conservatory run entirely on solar energy.",
            },
            {
                "name": "Echo Forest Trail",
                "cost": "$10 Permit",
                "eco_rating": "Protected Area",
                "description": "Scenic low-impact hiking trails with guided nature tours.",
            },
        ],
        "culture": [
            {
                "name": "Sustainable Art Museum",
                "cost": "$15",
                "eco_rating": "LEED Gold",
                "description": "Showcases local artists using repurposed materials.",
            },
            {
                "name": "Historical Heritage Center",
                "cost": "$12",
                "eco_rating": "Cultural Heritage Site",
                "description": "Interactive museum dedicated to local history and sustainable architecture.",
            },
        ],
        "food": [
            {
                "name": "The Earthy Table",
                "cost": "$$ (Mid-range)",
                "eco_rating": "Zero Waste Kitchen",
                "description": "Farm-to-table organic eatery serving seasonal vegan/vegetarian dishes.",
            },
            {
                "name": "Rooted Brewery",
                "cost": "$",
                "eco_rating": "Local Sourced",
                "description": "Microbrewery using 100% locally harvested grains and ingredients.",
            },
        ],
        "sights": [
            {
                "name": "Skyline Green Tower",
                "cost": "$25",
                "eco_rating": "LEED Platinum",
                "description": "Observation deck with wind turbines generating its own electricity.",
            },
            {
                "name": "Historic Old Town Walk",
                "cost": "Free",
                "eco_rating": "Pedestrian Only Zone",
                "description": "Guided walking tour through historic brick streets closed to vehicle traffic.",
            },
        ],
    }

    matches = attraction_db.get(cat, attraction_db["sights"])

    lines = [f"--- Top Attractions in {city} (Category: {cat.title()}) ---"]
    for idx, attr in enumerate(matches, 1):
        lines.append(
            f"{idx}. {attr['name']} ({attr['eco_rating']})\n"
            f"   Cost: {attr['cost']} | Description: {attr['description']}"
        )
    return "\n".join(lines)
