import httpx
import google.auth
import google.auth.transport.requests
from langchain_core.tools import tool

@tool
def call_logging_mcp(tool_name: str, arguments: dict) -> str:
    """Calls a tool on the remote Google Cloud Logging MCP server via HTTP POST.
    
    Args:
        tool_name: The name of the tool to call (e.g., 'list_logs', 'query_logs').
        arguments: A dictionary of arguments for the tool.
    """
    # Get default credentials and refresh token
    credentials, project = google.auth.default()
    auth_request = google.auth.transport.requests.Request()
    credentials.refresh(auth_request)
    token = credentials.token

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    
    # The remote MCP endpoint
    url = "https://logging.googleapis.com/mcp"
    
    # Construct JSON-RPC payload for MCP
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }
    
    print(f"Calling remote MCP server at {url} with tool '{tool_name}'...")
    try:
        # We use standard HTTP POST assuming it follows standard JSON-RPC over HTTP
        # rather than persistent SSE which failed with 405.
        response = httpx.post(url, headers=headers, json=payload, timeout=30.0)
        response.raise_for_status()
        return str(response.json())
    except Exception as e:
        import traceback
        return f"Error calling remote MCP server: {str(e)}\n{traceback.format_exc()}"
