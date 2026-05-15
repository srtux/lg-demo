import os
import sys
from dotenv import load_dotenv

# Add parent directory to sys.path to find demo_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

import vertexai
from demo_agent import create_agent
from rich.console import Console
from rich.panel import Panel

console = Console()

# Configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "summitt-gcp")
LOCATION = "us-west1"

def main():
    console.print(Panel(f"[bold blue]Initializing Vertex AI[/bold blue]\nProject: [yellow]{PROJECT_ID}[/yellow]\nLocation: [yellow]{LOCATION}[/yellow]"))
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    console.print("[bold green]Creating LangGraph agent...[/bold green]")
    agent = create_agent()
    
    # Some agents require set_up() to be called before querying.
    try:
        console.print("[italic yellow]Setting up agent...[/italic yellow]")
        agent.set_up()
    except Exception as e:
        console.print(f"[dim]Note: set_up() raised an exception (might not be required): {e}[/dim]")
    
    query_input = "Can you call call_logging_mcp with tool_name='list_log_entries' and arguments={'resourceNames': ['projects/summitt-gcp']}?"
    console.print(Panel(f"[bold green]Querying agent with:[/bold green]\n{query_input}", title="Query"))
    
    try:
        response = agent.query(input=query_input)
        console.print(Panel(f"[bold green]Agent Response:[/bold green]\n{response}", title="Response"))
    except Exception as e:
        console.print(Panel(f"[bold red]Error querying agent:[/bold red] {e}\nMake sure you have set a valid PROJECT_ID and are authenticated with Google Cloud.", title="Error", border_style="red"))

if __name__ == "__main__":
    main()
