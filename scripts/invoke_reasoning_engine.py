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
from rich.text import Text
from rich.markdown import Markdown

# Initialize Rich Console
console = Console()

# Configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "summitt-gcp")
LOCATION = "us-west1"
RESOURCE_NAME = os.environ.get("REASONING_ENGINE_RESOURCE_NAME")
if not RESOURCE_NAME:
    raise ValueError("REASONING_ENGINE_RESOURCE_NAME environment variable is not set.")

console.print(f"[bold blue]Initializing Vertex AI...[/bold blue]")
vertexai.init(project=PROJECT_ID, location=LOCATION)

console.print(f"[bold blue]Loading Reasoning Engine:[/bold blue] [yellow]{RESOURCE_NAME}[/yellow]")
re = ReasoningEngine(RESOURCE_NAME)

def parse_agent_response(response):
    """Parses and prints the LangGraph agent response in a readable way."""
    if not isinstance(response, dict) or 'messages' not in response:
        console.print(Panel(f"[bold red]Raw Response (Unexpected format):[/bold red] {response}", title="Error", border_style="red"))
        return
    
    messages = response['messages']
    for msg in messages:
        kwargs = msg.get('kwargs', {})
        msg_type = kwargs.get('type')
        content = kwargs.get('content')
        
        if msg_type == 'human':
            console.print(Panel(Text(content, style="green"), title="👤 User", border_style="green"))
        elif msg_type == 'ai':
            # Handle string or list of content blocks
            if isinstance(content, list):
                text_parts = [block.get('text', '') for block in content if block.get('type') == 'text']
                text = "".join(text_parts)
            else:
                text = content
            
            if text:
                console.print(Panel(Markdown(text), title="🤖 Agent", border_style="blue"))
            
            # Check for tool calls
            tool_calls = kwargs.get('tool_calls', [])
            if tool_calls:
                for tc in tool_calls:
                    tc_text = f"**Tool**: `{tc.get('name')}`\n**Args**: `{tc.get('args')}`"
                    console.print(Panel(Markdown(tc_text), title="🔧 Tool Call Requested", border_style="yellow"))
        elif msg_type == 'tool':
            console.print(Panel(Text(str(content), style="cyan"), title=f"📤 Tool Result ({kwargs.get('name')})", border_style="cyan"))

console.print("\n[bold green]🚀 TUI Chat App Started! 🚀[/bold green]")
console.print("Type [bold red]'exit'[/bold red] or [bold red]'quit'[/bold red] to end the chat. 👋\n")

thread_id = "demo-thread"
config = {"configurable": {"thread_id": thread_id}}

while True:
    try:
        user_input = console.input("[bold green][User][/bold green]: ")
        if user_input.lower() in ['exit', 'quit']:
            console.print("[bold red]Ending chat session. Bye![/bold red] 👋")
            break
            
        if not user_input.strip():
            continue
            
        console.print("[italic yellow]Thinking...[/italic yellow] ⏳")
        response = re.query(input=user_input)
        parse_agent_response(response)
        
    except KeyboardInterrupt:
        console.print("\n[bold red]Chat session interrupted.[/bold red] 👋")
        break
    except Exception as e:
        console.print(Panel(f"[bold red]Error:[/bold red] {e}", title="Error", border_style="red"))
