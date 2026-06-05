import os
import sys
from dotenv import load_dotenv

# Add parent directory to sys.path to find demo_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# 1. Initialize OpenTelemetry with Console exporter to inspect spans
# We do this directly here to see the console output
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

import google.auth
import google.auth.transport.requests
import vertexai
from demo_agent.tools import call_logging_mcp
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Import and instrument using the new package
from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
LangChainInstrumentor().instrument(skip_dep_check=True)

# Configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID")
LOCATION = "us-central1"

def main():
    # 2. Get credentials and refresh token to use as API Key for OpenAI-compatible endpoint
    credentials, project_id = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_request = google.auth.transport.requests.Request()
    credentials.refresh(auth_request)
    api_key = credentials.token

    # 3. Instantiate ChatOpenAI pointing to Vertex AI's OpenAI-compatible endpoint
    base_url = f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{LOCATION}/endpoints/openapi"
    
    print(f"Initializing ChatOpenAI on base_url: {base_url}")
    llm = ChatOpenAI(
        model="google/gemini-2.5-flash",
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
    )

    tools = [call_logging_mcp]

    print("Creating react agent with langgraph...")
    agent_executor = create_react_agent(llm, tools=tools)

    query_input = "Can you call call_logging_mcp with tool_name='list_log_entries' and arguments={'resourceNames': ['projects/YOUR_PROJECT_ID']}?"
    print(f"Querying agent with: {query_input}")
    
    try:
        response = agent_executor.invoke({"messages": [("user", query_input)]})
        print("Agent Messages:")
        for msg in response["messages"]:
            print(f"- {msg.type}: {msg.content or msg.tool_calls}")
    except Exception as e:
        print(f"Error querying agent: {e}")
    finally:
        # Flush spans before this short-lived process exits.
        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()

if __name__ == "__main__":
    main()
