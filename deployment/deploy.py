import os
import sys
from dotenv import load_dotenv

# Add parent directory to sys.path to find demo_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

import vertexai
from vertexai.preview.reasoning_engines import ReasoningEngine
from demo_agent import create_agent
from rich.console import Console
from rich.panel import Panel

console = Console()

# Configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "summitt-gcp")
LOCATION = "us-west1"
STAGING_BUCKET = os.environ.get("STAGING_BUCKET", "gs://agent-engine-staging-summitt-gcp")

REQUIREMENTS = [
    "google-cloud-aiplatform[agent_engines,langchain]",
    "langgraph",
    "mcp",
    "opentelemetry-instrumentation-vertexai",
    "opentelemetry-exporter-otlp",
    "opentelemetry-sdk",
    "opentelemetry-instrumentation-langchain",
]

def main():
    console.print(Panel("[bold blue]Initializing Vertex AI for deployment...[/bold blue]"))
    vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)
    
    console.print("[bold green]Creating LangGraph agent instance...[/bold green]")
    agent = create_agent()
    
    resource_name = os.environ.get("REASONING_ENGINE_RESOURCE_NAME")
    
    console.print("[bold yellow]Deploying agent to Vertex AI Reasoning Engine...[/bold yellow]")
    console.print("This may take a few minutes as it packages the code and sets up the environment.")
    
    try:
        if resource_name:
            console.print(f"Updating existing agent: [cyan]{resource_name}[/cyan]")
            existing_agent = ReasoningEngine(resource_name)
            remote_agent = existing_agent.update(
                reasoning_engine=agent,
                requirements=REQUIREMENTS,
                display_name="LangGraph Demo Agent",
            )
        else:
            console.print("[bold green]Deploying NEW agent to Vertex AI Reasoning Engine...[/bold green]")
            remote_agent = ReasoningEngine.create(
                reasoning_engine=agent,
                requirements=REQUIREMENTS,
                display_name="LangGraph Demo Agent",
            )
        
        console.print(Panel(f"[bold green]Successfully deployed![/bold green]\nResource name: [yellow]{remote_agent.resource_name}[/yellow]", title="Success"))
    except Exception as e:
        console.print(Panel(f"[bold red]Error during deployment:[/bold red] {e}\nMake sure you have set valid PROJECT_ID and STAGING_BUCKET, and have necessary permissions.", title="Error", border_style="red"))

if __name__ == "__main__":
    main()
