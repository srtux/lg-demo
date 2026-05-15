# LangGraph Demo Agent with OpenTelemetry

This project demonstrates a LangGraph agent deployed on Vertex AI Reasoning Engines, instrumented with OpenTelemetry for GenAI semantic conventions.

## Project Structure

*   `demo_agent/agent.py`: The core agent definition, including tools and OpenTelemetry instrumentation.
*   `deployment/deploy.py`: Script to deploy the agent to Vertex AI Reasoning Engine.
*   `scripts/test_local.py`: Script to test the agent locally.
*   `scripts/invoke_reasoning_engine.py`: Script to invoke the deployed agent on Vertex AI.

## Local Development

### Prerequisites

*   Python 3.10+
*   `uv` installed (recommended) or `pip`.
*   Google Cloud SDK authenticated.

### Installation

Since this project uses `pyproject.toml` managed by `uv`, you can install all dependencies by running:

```bash
uv sync
```

### Running Locally

Ensure your `.env` file is set up with your project ID, then run:

```bash
uv run python scripts/test_local.py
```

## Deployment

The agent is deployed to Vertex AI Reasoning Engine.

### Configuration

The project uses a `.env` file to manage environment variables. 

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and fill in your values:
   *   `GOOGLE_CLOUD_PROJECT`: Your Google Cloud Project ID.
   *   `STAGING_BUCKET`: A Cloud Storage bucket to stage the code bundle.
   *   `REASONING_ENGINE_RESOURCE_NAME`: (Optional) The resource name of an existing Reasoning Engine to update.

### Deploying

Run the deployment script:

```bash
uv run python deployment/deploy.py
```

This will package the code and deploy it. The resource name of the deployed engine will be printed.

## Invoking the Deployed Agent

After deployment, you can test the remote agent. Ensure your `.env` file has the `REASONING_ENGINE_RESOURCE_NAME` set correctly, or set it in your shell:

```bash
export REASONING_ENGINE_RESOURCE_NAME="projects/.../locations/.../reasoningEngines/..."
uv run python scripts/invoke_reasoning_engine.py
```

If `REASONING_ENGINE_RESOURCE_NAME` is not set (either in `.env` or in the environment), the script will raise an error.
