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

from google.adk.agents import Context
from pydantic import BaseModel, Field

# Supported locations list for guided error recovery
SUPPORTED_CITIES = ["san francisco", "seattle", "paris", "tokyo"]

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
        ..., description="Starting point city. E.g. 'San Francisco', 'Seattle'."
    )
    destination: str = Field(
        ..., description="Destination city. E.g. 'Paris', 'Tokyo'."
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


class BookingRequest(BaseModel):
    route_option_index: int = Field(
        ...,
        description="Index of the route option to book (1-indexed based on search results).",
    )
    confirm_booking: bool = Field(
        default=False,
        description="Set to True ONLY if the user has explicitly confirmed the booking.",
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
    try:
        city = req.city.strip().lower()
        if city not in SUPPORTED_CITIES:
            # Guided recovery instruction for the LLM
            return (
                f"Error: Weather service does not currently support the city '{req.city}'. "
                f"Please instruct the user that only the following cities are supported: "
                f"{', '.join([c.title() for c in SUPPORTED_CITIES])}."
            )

        days = req.days
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

        match_data = weather_data[city]
        lines = [f"--- Weather Forecast for {req.city.title()} ({days} Days) ---"]
        for day in range(1, days + 1):
            date_str = (
                datetime.date.today() + datetime.timedelta(days=day - 1)
            ).strftime("%A, %b %d")
            lines.append(
                f"- {date_str}: Temp: {match_data['temp']}, Condition: {match_data['condition']}"
            )

        lines.append(f"Advisory: {match_data['advisory']}")
        return "\n".join(lines)
    except Exception as e:
        return f"Unexpected error in get_weather_forecast: {e!s}. Please try adjusting parameters."


def get_eco_routes(req: RouteRequest) -> str:
    """Computes route options between origin and destination with carbon emission stats.

    Args:
        req: Validated routing request.

    Returns:
        A text summary of transit options, pricing, and carbon emissions.
    """
    try:
        origin = req.origin.strip().lower()
        destination = req.destination.strip().lower()

        # Guided recovery for unsupported locations
        invalid_locations = []
        if origin not in SUPPORTED_CITIES:
            invalid_locations.append(req.origin)
        if destination not in SUPPORTED_CITIES:
            invalid_locations.append(req.destination)

        if invalid_locations:
            return (
                f"Error: Routing service does not support the location(s): {', '.join(invalid_locations)}. "
                f"Please ask the user to choose from supported locations: "
                f"{', '.join([c.title() for c in SUPPORTED_CITIES])}."
            )

        pref = req.preference
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

        if pref == "greenest":
            options.sort(key=lambda x: x["carbon"])
        elif pref == "cheapest":
            options.sort(key=lambda x: x["cost"])
        else:  # fastest

            def duration_key(x):
                parts = x["duration"].split("h")
                hours = int(parts[0].strip())
                mins = (
                    int(parts[1].replace("m", "").strip())
                    if len(parts) > 1 and parts[1]
                    else 0
                )
                return hours * 60 + mins

            options.sort(key=duration_key)

        lines = [
            f"--- Route Options: {origin.title()} to {destination.title()} (Optimized for {pref}) ---"
        ]
        for idx, opt in enumerate(options, 1):
            rec_marker = "🌟 [Recommended]" if idx == 1 else ""
            lines.append(
                f"{idx}. {opt['mode']} {rec_marker}\n"
                f"   Duration: {opt['duration']} | Cost: ${opt['cost']} | Carbon Footprint: {opt['carbon']} kg CO2\n"
                f"   Note: {opt['note']}"
            )

        return "\n".join(lines)
    except Exception as e:
        return f"Unexpected error in get_eco_routes: {e!s}. Please correct arguments and retry."


def get_attractions(req: AttractionRequest) -> str:
    """Finds top local attractions in a city based on category, highlighting eco-ratings.

    Args:
        req: Validated attractions search request.

    Returns:
        A list of attractions with descriptions and eco-ratings.
    """
    try:
        city = req.city.strip().lower()
        if city not in SUPPORTED_CITIES:
            return (
                f"Error: Attractions database does not support '{req.city}'. "
                f"Please instruct the user to select from: "
                f"{', '.join([c.title() for c in SUPPORTED_CITIES])}."
            )

        cat = req.category
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
        lines = [
            f"--- Top Attractions in {req.city.title()} (Category: {cat.title()}) ---"
        ]
        for idx, attr in enumerate(matches, 1):
            lines.append(
                f"{idx}. {attr['name']} ({attr['eco_rating']})\n"
                f"   Cost: {attr['cost']} | Description: {attr['description']}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Unexpected error in get_attractions: {e!s}. Please correct arguments and retry."


def book_trip(req: BookingRequest, ctx: Context) -> str:
    """Books a travel option. This is a high-stakes action requiring explicit user confirmation.

    Args:
        req: Validated booking request.
        ctx: Automatically injected ADK runtime Context.

    Returns:
        Status description of the booking.
    """
    try:
        # Check if booking index is out of bounds (mock index check)
        if req.route_option_index < 1 or req.route_option_index > 4:
            return (
                f"Error: Invalid route option index {req.route_option_index}. "
                f"Please choose a valid index from 1 to 4 based on route search results."
            )

        # Call request_confirmation if confirm_booking is False
        if not req.confirm_booking:
            ctx.request_confirmation(
                hint=f"Confirm booking for route option {req.route_option_index}. Please reply 'yes' to proceed.",
                payload={"route_option_index": req.route_option_index},
            )
            return "Booking request generated. Pausing execution until user confirms."

        return f"Success! Trip option {req.route_option_index} has been booked successfully."
    except Exception as e:
        return (
            f"Unexpected error in book_trip: {e!s}. Please correct arguments and retry."
        )
