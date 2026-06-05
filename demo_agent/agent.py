import os
from dotenv import load_dotenv
load_dotenv()

from vertexai.agent_engines import LanggraphAgent
from demo_agent.tools import call_logging_mcp, get_current_time
from demo_agent.telemetry import init_telemetry

import google.auth
import google.auth.transport.requests
from langchain_google_genai import ChatGoogleGenerativeAI

def custom_model_builder(
    model_name: str,
    project: str,
    location: str,
    model_kwargs: dict | None = None,
) -> ChatGoogleGenerativeAI:
    # Initialize telemetry on the server side
    init_telemetry()

    model_kwargs = model_kwargs or {}

    # Map model name if it starts with google/
    clean_model_name = model_name
    if clean_model_name.startswith("google/"):
        clean_model_name = clean_model_name[len("google/"):]

    # Use the unified google-genai backend pointed at Vertex AI. Per
    # langchain-ai/langchain-google#1422, ChatGoogleGenerativeAI with
    # vertexai=True is the supported path for Gemini-on-Vertex; ChatVertexAI
    # (langchain-google-vertexai) is being deprecated.
    return ChatGoogleGenerativeAI(
        model=clean_model_name,
        project=project,
        location=location,
        vertexai=True,
        **model_kwargs,
    )

def checkpointer_builder(**kwargs):
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()

def create_agent():
    """Creates and returns the LangGraph agent."""
    # Initialize telemetry on the client side during creation/packaging
    init_telemetry()
    agent = LanggraphAgent(
        model="gemini-2.5-flash",
        tools=[call_logging_mcp, get_current_time],
        enable_tracing=False,
        model_builder=custom_model_builder,
        checkpointer_builder=checkpointer_builder,
        runnable_kwargs={"prompt": "You are an expert assistant for Google Cloud Logging. When asked to query or list log entries, use the `list_log_entries` tool provided by the MCP server. IMPORTANT: You MUST provide the `resourceNames` argument as a list of strings (e.g., `['projects/YOUR_PROJECT_ID']`)."},
    )
    return agent
