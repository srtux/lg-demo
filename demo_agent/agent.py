import os
from dotenv import load_dotenv
load_dotenv()

from vertexai.preview.reasoning_engines import LanggraphAgent
from demo_agent.tools import call_logging_mcp
from demo_agent.telemetry import init_telemetry

# Initialize telemetry
init_telemetry()

def create_agent():
    """Creates and returns the LangGraph agent."""
    agent = LanggraphAgent(
        model="gemini-2.5-flash",
        tools=[call_logging_mcp],
        enable_tracing=True,
        runnable_kwargs={"prompt": "You are an expert assistant for Google Cloud Logging. When asked to query or list log entries, use the `list_log_entries` tool provided by the MCP server. IMPORTANT: You MUST provide the `resourceNames` argument as a list of strings (e.g., `['projects/YOUR_PROJECT_ID']`)."},
    )
    return agent
