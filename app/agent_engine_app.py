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
from typing import Any

import vertexai
from dotenv import load_dotenv
from google.adk.artifacts import GcsArtifactService, InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.memory.vertex_ai_memory_bank_service import VertexAiMemoryBankService
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService
from google.cloud import logging as google_cloud_logging
from vertexai.agent_engines.templates.adk import AdkApp

from app.agent import app as adk_app
from app.app_utils.telemetry import setup_telemetry
from app.app_utils.typing import Feedback

# Load environment variables from .env file at runtime
load_dotenv()


class AgentEngineApp(AdkApp):
    def set_up(self) -> None:
        """Initialize the agent engine app with logging and telemetry."""
        vertexai.init()
        setup_telemetry()
        super().set_up()
        logging.basicConfig(level=logging.INFO)
        logging_client = google_cloud_logging.Client()
        self.logger = logging_client.logger(__name__)
        if gemini_location:
            os.environ["GOOGLE_CLOUD_LOCATION"] = gemini_location

    def register_feedback(self, feedback: dict[str, Any]) -> None:
        """Collect and log feedback."""
        feedback_obj = Feedback.model_validate(feedback)
        self.logger.log_struct(feedback_obj.model_dump(), severity="INFO")

    def register_operations(self) -> dict[str, list[str]]:
        """Registers the operations of the Agent."""
        operations = super().register_operations()
        operations[""] = [*operations.get("", []), "register_feedback"]
        return operations


def session_service_builder() -> Any:
    """Builds session service. Uses local SQLite during tests or local runs, Vertex AI in cloud."""
    if os.environ.get("INTEGRATION_TEST") == "TRUE" or not os.environ.get(
        "AGENT_ENGINE_ID"
    ):
        db_dir = os.path.join(os.getcwd(), ".session_data")
        os.makedirs(db_dir, exist_ok=True)
        return SqliteSessionService(db_path=os.path.join(db_dir, "sessions.db"))

    return VertexAiSessionService(
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
        agent_engine_id=os.environ.get("AGENT_ENGINE_ID"),
    )


def memory_service_builder() -> Any:
    """Builds memory service. Uses local InMemoryMemoryService during tests or local runs, Vertex AI in cloud."""
    if os.environ.get("INTEGRATION_TEST") == "TRUE" or not os.environ.get(
        "AGENT_ENGINE_ID"
    ):
        return InMemoryMemoryService()

    return VertexAiMemoryBankService(
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
        agent_engine_id=os.environ.get("AGENT_ENGINE_ID"),
    )


gemini_location = os.environ.get("GOOGLE_CLOUD_LOCATION")
logs_bucket_name = os.environ.get("LOGS_BUCKET_NAME")

agent_engine = AgentEngineApp(
    app=adk_app,
    session_service_builder=session_service_builder,
    memory_service_builder=memory_service_builder,
    artifact_service_builder=lambda: (
        GcsArtifactService(bucket_name=logs_bucket_name)
        if logs_bucket_name
        else InMemoryArtifactService()
    ),
)
