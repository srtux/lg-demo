import os
import sys
from dotenv import load_dotenv

# Add parent directory to sys.path to find demo_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

import vertexai
from vertexai import agent_engines
from demo_agent import create_agent
from rich.console import Console
from rich.panel import Panel

console = Console()

# Configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID")
LOCATION = "us-central1"
STAGING_BUCKET = os.environ.get("STAGING_BUCKET", "gs://agent-engine-staging-YOUR_PROJECT_ID")

REQUIREMENTS = [
    "google-cloud-aiplatform[agent_engines,langchain]",
    "langgraph",
    "langchain-google-genai>=2.0.0",
    "mcp",
    "opentelemetry-exporter-otlp",
    "opentelemetry-sdk",
    "opentelemetry-semantic-conventions>=0.63b1",
    "opentelemetry-util-genai @ git+https://github.com/open-telemetry/opentelemetry-python-genai.git#subdirectory=util/opentelemetry-util-genai",
    "opentelemetry-instrumentation-google-genai @ git+https://github.com/open-telemetry/opentelemetry-python-genai.git#subdirectory=instrumentation/opentelemetry-instrumentation-google-genai",
    "opentelemetry-instrumentation-genai-langchain @ git+https://github.com/open-telemetry/opentelemetry-python-genai.git#subdirectory=instrumentation/opentelemetry-instrumentation-genai-langchain",
    "pyopenssl",
]

def main():
    console.print(Panel("[bold blue]Initializing Vertex AI for deployment...[/bold blue]"))
    vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)
    
    console.print("[bold green]Creating LangGraph agent instance...[/bold green]")
    agent = create_agent()
    
    resource_name = os.environ.get("REASONING_ENGINE_RESOURCE_NAME")
    
    console.print("[bold yellow]Deploying agent to Vertex AI Agent Engine...[/bold yellow]")
    console.print("This may take a few minutes as it packages the code and sets up the environment.")
    
    env_vars = {
        "OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "span_and_event",
        "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY": "true",
    }
    
    try:
        if resource_name:
            console.print(f"Updating existing agent: [cyan]{resource_name}[/cyan]")
            remote_agent = agent_engines.update(
                resource_name=resource_name,
                agent_engine=agent,
                requirements=REQUIREMENTS,
                extra_packages=["demo_agent"],
                display_name="LangGraph Demo Agent",
                env_vars=env_vars,
            )
        else:
            console.print("[bold green]Deploying NEW agent to Vertex AI Agent Engine...[/bold green]")
            remote_agent = agent_engines.create(
                agent_engine=agent,
                requirements=REQUIREMENTS,
                extra_packages=["demo_agent"],
                display_name="LangGraph Demo Agent",
                env_vars=env_vars,
            )
        
        console.print(Panel(f"[bold green]Successfully deployed![/bold green]\nResource name: [yellow]{remote_agent.resource_name}[/yellow]", title="Success"))
    except Exception as e:
        console.print(Panel(f"[bold red]Error during deployment:[/bold red] {e}\nMake sure you have set valid PROJECT_ID and STAGING_BUCKET, and have necessary permissions.", title="Error", border_style="red"))

if __name__ == "__main__":
    main()
