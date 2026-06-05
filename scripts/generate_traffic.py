import os
import sys
import time
import random
from dotenv import load_dotenv

# Add parent directory to sys.path to find demo_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

import google.auth
import vertexai
from vertexai.agent_engines import AgentEngine
from rich.console import Console
from rich.panel import Panel

console = Console()

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID")
LOCATION = "us-central1"
RESOURCE_NAME = os.environ.get("REASONING_ENGINE_RESOURCE_NAME")

# Variations of conversation prompts to simulate organic traffic
CONVERSATION_TEMPLATES = [
    [
        "Hi, I want to audit the logging activity in project YOUR_PROJECT_ID.",
        "Can you call call_logging_mcp with tool_name='list_log_entries' and arguments={{'resourceNames': ['projects/YOUR_PROJECT_ID']}}?",
        "Are there any log entries mentioning 'system' or 'admin'?",
        "What is the nextPageToken from the logs?",
        "Great, thanks. Please summarize what we audited."
    ],
    [
        "Hello, I need to check GKE cluster logs in project YOUR_PROJECT_ID.",
        "Could you fetch the log entries using list_log_entries for YOUR_PROJECT_ID?",
        "Do you see any K8s controller activity in the retrieved logs?",
        "What principal email did most of these operations run under?",
        "Awesome. Write a one sentence summary of the cluster health based on this."
    ],
    [
        "Hey! Can you help me fetch some audit logs for YOUR_PROJECT_ID?",
        "Please list the log entries by calling the tool call_logging_mcp with list_log_entries.",
        "Is there a nextPageToken returned in the response?",
        "What is the project ID listed in the log resource names?",
        "Thanks for the info, have a good day!"
    ]
]

def main():
    if not RESOURCE_NAME:
        raise ValueError("REASONING_ENGINE_RESOURCE_NAME is not set in environment.")

    console.print(Panel(f"[bold blue]Initializing Traffic Generator[/bold blue]\nProject: [yellow]{PROJECT_ID}[/yellow]\nLocation: [yellow]{LOCATION}[/yellow]\nResource: [yellow]{RESOURCE_NAME}[/yellow]"))
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    console.print("[bold green]Loading Reasoning Engine...[/bold green]")
    re = AgentEngine(RESOURCE_NAME)

    num_conversations = 10
    turns_per_conversation = 5

    for conv_idx in range(num_conversations):
        console.print(Panel(f"[bold magenta]Starting Conversation {conv_idx + 1}/{num_conversations}[/bold magenta]"))
        
        # Select a conversation template
        template = random.choice(CONVERSATION_TEMPLATES)
        
        # Maintain history locally as a list of message dicts
        history = []

        for turn_idx in range(turns_per_conversation):
            # Formulate the query (resolve variable mapping if template contains formatting placeholders)
            query_text = template[turn_idx]
            
            console.print(f"[cyan]Conversation {conv_idx + 1} - Turn {turn_idx + 1} (User):[/cyan] {query_text}")
            
            # Append the user turn to history
            history.append({"role": "user", "content": query_text})
            
            # Query the agent with the full message history
            try:
                # We pass the full history in input
                # LangGraph StateGraph's 'messages' field can accept list of message dicts
                response = re.query(input={"messages": history})
                
                # Retrieve the assistant response
                # The response structure from LanggraphAgent.query is a dictionary containing the state keys
                # We extract the content of the last message returned in the state
                messages = response.get("messages", [])
                
                if messages:
                    # In LangGraph response payload, the message objects are serialized.
                    # We can find the last message (which should be the AI response)
                    last_msg = messages[-1]
                    
                    # Safe retrieval based on serialized structure
                    last_content = ""
                    if isinstance(last_msg, dict):
                        # check if it's langchain serialized JSON format
                        kwargs = last_msg.get("kwargs", {})
                        last_content = kwargs.get("content", "")
                        if not last_content and "tool_calls" in kwargs:
                            last_content = f"[Tool calls: {kwargs['tool_calls']}]"
                    else:
                        last_content = str(last_msg)
                        
                    console.print(f"[green]Agent Response:[/green] {last_content}")
                    
                    # Append assistant turn to history
                    history.append({"role": "assistant", "content": last_content})
                else:
                    console.print("[yellow]Warning: Agent returned empty messages list.[/yellow]")
            except Exception as e:
                console.print(f"[bold red]Error in query:[/bold red] {e}")
            
            # Sleep 3 seconds between turns to prevent hitting API rate limits
            time.sleep(3)
        
        # Sleep 5 seconds between conversations
        time.sleep(5)

    console.print(Panel("[bold green]Traffic Generation Completed Successfully![/bold green]"))

if __name__ == "__main__":
    main()
