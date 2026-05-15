import os
import sys
from dotenv import load_dotenv

# Add parent directory to sys.path to find demo_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

import vertexai
from vertexai.preview.reasoning_engines import ReasoningEngine
from rich.console import Console
from rich.panel import Panel

console = Console()

# Configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "summitt-gcp")
LOCATION = "us-west1"
RESOURCE_NAME = os.environ.get("REASONING_ENGINE_RESOURCE_NAME")

def main():
    if not RESOURCE_NAME:
        raise ValueError("REASONING_ENGINE_RESOURCE_NAME environment variable is not set.")
        
    console.print(Panel(f"[bold blue]Initializing Vertex AI[/bold blue]\nProject: [yellow]{PROJECT_ID}[/yellow]\nLocation: [yellow]{LOCATION}[/yellow]"))
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    console.print(f"[bold blue]Loading Reasoning Engine:[/bold blue] [yellow]{RESOURCE_NAME}[/yellow]")
    re = ReasoningEngine(RESOURCE_NAME)
    
    query_input = "Hello, can you list the logs for project summitt-gcp using list_log_entries tool?"
    console.print(Panel(f"[bold green]Querying agent with:[/bold green]\n{query_input}", title="Query"))
    
    try:
        response = re.query(input=query_input)
        console.print(Panel(f"[bold green]Agent Response:[/bold green]\n{response}", title="Response"))
    except Exception as e:
        console.print(Panel(f"[bold red]Error querying agent:[/bold red] {e}", title="Error", border_style="red"))

if __name__ == "__main__":
    main()
