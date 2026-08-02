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

import logging
import os
import re

import google.auth
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.apps.compaction import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models import Gemini
from google.genai import types

from app.tools import get_attractions, get_eco_routes, get_weather_forecast

# Configure Google Cloud credentials and locations
_, project_id = google.auth.default()
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

# Configure logging
logger = logging.getLogger("agent_observability")
logger.setLevel(logging.INFO)

# ==============================================================================
# Observability: PII Redaction & Intent vs. Outcome Callbacks
# ==============================================================================


def redact_pii(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Email pattern
    text = re.sub(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED_EMAIL]", text
    )
    # Phone number pattern
    text = re.sub(
        r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "[REDACTED_PHONE]",
        text,
    )
    # Credit Card/Passport pattern
    text = re.sub(
        r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[REDACTED_CARD]", text
    )
    return text


def before_agent_callback(*args, **kwargs) -> None:
    """Logs the start of an agent invocation with redacted inputs."""
    ctx = kwargs.get("callback_context") or (args[0] if args else None)
    if not ctx:
        return
    user_input = redact_pii(str(ctx.user_content))
    logger.info(
        f"[INTENT] Invoking agent: {ctx.agent_name} | User ID: {ctx.user_id} | Input: {user_input}"
    )


def after_agent_callback(*args, **kwargs) -> None:
    """Logs the end of an agent invocation with redacted outputs."""
    ctx = kwargs.get("callback_context") or (args[0] if args else None)
    if not ctx:
        return
    last_event_str = ""
    if ctx.session.events:
        last_event = ctx.session.events[-1]
        last_event_str = redact_pii(str(last_event.content))
    logger.info(
        f"[OUTCOME] Completed agent: {ctx.agent_name} | User ID: {ctx.user_id} | Last Output: {last_event_str}"
    )


def before_tool_callback(*args, **kwargs) -> None:
    """Logs the initiation of a tool invocation."""
    ctx = kwargs.get("callback_context") or (args[0] if args else None)
    if not ctx:
        return
    logger.info(
        f"[INTENT] Agent {ctx.agent_name} calling tool for function call ID: {ctx.function_call_id}"
    )


def after_tool_callback(*args, **kwargs) -> None:
    """Logs the completion of a tool invocation."""
    ctx = kwargs.get("callback_context") or (args[0] if args else None)
    if not ctx:
        return
    logger.info(
        f"[OUTCOME] Agent {ctx.agent_name} completed tool for function call ID: {ctx.function_call_id}"
    )


# ==============================================================================
# Shared Model Configuration
# ==============================================================================

model_inst = Gemini(
    model="gemini-3-flash-preview",
    retry_options=types.HttpRetryOptions(attempts=3),
)

# ==============================================================================
# Subagents Definitions
# ==============================================================================

weather_advisor = Agent(
    name="weather_advisor",
    model=model_inst,
    instruction=(
        "You are a weather and packing advisor subagent.\n"
        "Your role is to fetch the weather forecast for the requested destination and days, "
        "and provide relevant weather alerts and packing recommendations (e.g. clothing, gear).\n"
        "Always use the get_weather_forecast tool to get accurate forecasts."
    ),
    description="Provides weather forecasts, warnings, and gear/packing suggestions.",
    tools=[get_weather_forecast],
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)

route_optimizer = Agent(
    name="route_optimizer",
    model=model_inst,
    instruction=(
        "You are a routing and transit carbon footprint optimization subagent.\n"
        "Your role is to find transportation options between origin and destination.\n"
        "Analyze the cost, duration, and carbon footprint (in kg CO2).\n"
        "Always prefer lower-emission transit options (like trains or electric buses) and clearly highlight the recommended options.\n"
        "Use the get_eco_routes tool to fetch route details."
    ),
    description="Finds route options with costs, durations, and carbon emissions.",
    tools=[get_eco_routes],
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)

itinerary_planner = Agent(
    name="itinerary_planner",
    model=model_inst,
    instruction=(
        "You are a local itinerary and sightseeing planner subagent.\n"
        "Your role is to discover attractions and activities in the requested city and organize them into daily schedules.\n"
        "Highlight eco-certifications, sustainability, and local culture.\n"
        "Use the get_attractions tool to retrieve attraction categories (nature, culture, food, sights)."
    ),
    description="Generates daily local itineraries and searches attractions.",
    tools=[get_attractions],
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)

# ==============================================================================
# Root Agent (Orchestrator) Definition
# ==============================================================================

root_agent = Agent(
    name="root_agent",
    model=model_inst,
    instruction=(
        "You are the Smart Eco-Travel Concierge Agent, a professional travel assistant designed to make trips green, fun, and efficient.\n"
        "Your job is to greet the user, coordinate their requests, and delegate tasks to your subagents:\n"
        "1. Delegate route, transit, and carbon footprint checks to the 'route_optimizer' subagent.\n"
        "2. Delegate attractions and daily planning tasks to the 'itinerary_planner' subagent.\n"
        "3. Delegate weather lookups and gear recommendations to the 'weather_advisor' subagent.\n\n"
        "Rules:\n"
        "- Never make up weather or route facts; always delegate or call the tools.\n"
        "- Summarize coordinates nicely for the user, presenting structured, readable options.\n"
        "- Keep track of the user's travel preferences (e.g. budget, mode limits) in context."
    ),
    sub_agents=[weather_advisor, route_optimizer, itinerary_planner],
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
)

# ==============================================================================
# Event Compaction (Memory Management)
# ==============================================================================

compaction_summarizer = LlmEventSummarizer(llm=model_inst)
events_compaction_config = EventsCompactionConfig(
    summarizer=compaction_summarizer,
    compaction_interval=5,
    overlap_size=2,
    token_threshold=1500,
    event_retention_size=10,
)

# ==============================================================================
# Application Definition
# ==============================================================================

app = App(
    root_agent=root_agent,
    name="app",
    events_compaction_config=events_compaction_config,
)
