# LangGraph Agent with OpenTelemetry GenAI Semantic Conventions

This repository demonstrates how to build, test, and deploy a LangGraph agent on **Vertex AI Agent Engine** (formerly Vertex AI Reasoning Engine) fully instrumented with OpenTelemetry (OTel). It is designed to conform to the **OpenTelemetry GenAI Semantic Conventions**, exporting tracing spans directly to Google Cloud Trace (`telemetry.googleapis.com`) and generating structured JSON logs for auditability and debugging.

## Features

- **LangGraph Integration**: Employs LangGraph's prebuilt React agent using `vertexai.agent_engines.LanggraphAgent` with custom model configurations, MemorySaver state checkpointers, and system instructions.
- **OpenTelemetry Semantic Conventions Conformance**: Traces comply with the latest experimental GenAI standards (e.g., `gen_ai.provider.name` mapped to `gcp.vertex_ai`, `gen_ai.conversation.id` context propagation, and message content capturing).
- **Auto-Patching Instrumentation**: Uses a custom patching strategy to intercept LangChain's callback manager to automatically trace LLM calls and tool executions without requiring manual handler registration across models.
- **Custom Tool Tracing**: Traces tool calls (e.g., `call_logging_mcp` and `get_current_time`) capturing argument payloads, results, and runtime exceptions inside custom OTel spans.
- **Structured JSON Logging**: Standardizes client and server log printouts into Google Cloud Logging compatible JSON payloads.
- **Remote MCP tool connection**: The `call_logging_mcp` tool routes requests to Google Cloud Logging API via a remote JSON-RPC over HTTP MCP backend.

---

## Documentation

For a comprehensive guide on deploying, configuring, and instrumenting LangGraph agents on Google Cloud Agent Platform, see the [Deployment & Instrumentation Guide](./docs/gcp_langgraph_documentation.md).

---

## Directory Structure

- `docs/`
  - [gcp_langgraph_documentation.md](./docs/gcp_langgraph_documentation.md) - Detailed deployment, custom model builder, and telemetry configuration manual.
- `demo_agent/`
  - [agent.py](./demo_agent/agent.py) - Defines the `LanggraphAgent` instance, the custom model builder using `ChatGoogleGenerativeAI`, and MemorySaver checkpointers.
  - [telemetry.py](./demo_agent/telemetry.py) - OpenTelemetry startup logic, compliance span processors, callback handlers, and base instrumentation wrappers.
  - [tools.py](./demo_agent/tools.py) - Tool definitions (`call_logging_mcp`, `get_current_time`).
- `deployment/`
  - [deploy.py](./deployment/deploy.py) - Script to create or update the agent deployment on Vertex AI Agent Engine.
- `scripts/`
  - [test_local.py](./scripts/test_local.py) - Test the LangGraph agent execution flow locally.
  - [invoke_non_interactive.py](./scripts/invoke_non_interactive.py) - Query the deployed Agent Engine in non-interactive mode.
  - [invoke_reasoning_engine.py](./scripts/invoke_reasoning_engine.py) - Run an interactive TUI chat application session against the deployed Agent Engine.
  - [generate_traffic.py](./scripts/generate_traffic.py) - Simulate multi-conversation organic user traffic against the deployed Agent Engine.
  - [test_openai_agent.py](./scripts/test_openai_agent.py) - Tracing demonstration using LangGraph with `ChatOpenAI` and OpenTelemetry GenAI instrumentation.

---

## Getting Started

### Prerequisites

- Python 3.12+ (see `pyproject.toml` / `.python-version`)
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`
- Google Cloud SDK authenticated on your development machine

### Authentication

Set your active Google Cloud project and authenticate:

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### Environment Configuration

1. Copy the example `.env` file:
   ```bash
   cp .env.example .env
   ```

2. Fill in the values inside `.env`:
   - `GOOGLE_CLOUD_PROJECT`: Your Google Cloud Project ID (e.g. `YOUR_PROJECT_ID`).
   - `STAGING_BUCKET`: The GCS bucket (e.g. `gs://agent-engine-staging-bucket`) to store packaged artifacts.
   - `REASONING_ENGINE_RESOURCE_NAME`: The deployed agent's resource path (set this after your first deployment).
   - `CLOUD_RESOURCE_ID`: The unique target cloud resource identifier.

---

## Local Development & Testing

### Installation

Install python dependencies into a local virtual environment managed by `uv`:

```bash
uv sync
```

### Run Locally

Verify the agent creation and run a quick test prompt locally:

```bash
uv run python scripts/test_local.py
```

### Test OpenAI-Compatible Endpoint Tracing

Test tracing a LangGraph agent targeting Vertex AI's OpenAI-compatible endpoint:

```bash
uv run python scripts/test_openai_agent.py
```

---

## Deployment

Deploy or update the agent to **Vertex AI Agent Engine**:

```bash
uv run python deployment/deploy.py
```

After successful deployment, copy the returned resource name (e.g., `projects/YOUR_PROJECT_ID/locations/us-central1/reasoningEngines/YOUR_ENGINE_ID`) and paste it as `REASONING_ENGINE_RESOURCE_NAME` in your `.env` file.

---

## Invoking the Deployed Agent

Ensure your `.env` has the correct `REASONING_ENGINE_RESOURCE_NAME` configured, then use the following scripts to interact with the remote engine.

### Non-Interactive Query

Send a single query to the deployed agent:

```bash
uv run python scripts/invoke_non_interactive.py
```

### Interactive TUI Chat App

Start a stateful, interactive chat session in your terminal with conversation history and pretty formatting:

```bash
uv run python scripts/invoke_reasoning_engine.py
```

### Traffic Generator (Telemetry Ingestion)

Simulate 10 distinct, multi-turn conversations to generate organic usage telemetry on Cloud Trace and Google Cloud Logging:

```bash
uv run python scripts/generate_traffic.py
```

---

## Telemetry Configuration Details

The telemetry layer initializes OpenTelemetry and forces the following behavior:
- **OTLP/gRPC Exporter**: Spans are exported to `telemetry.googleapis.com:443` using credentials loaded from Google Application Default Credentials.
- **Stability and Schema Opt-in**: Runs with `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`.
- **Message Content Capture**: Automatically captures prompt/response inputs/outputs into spans and events by setting `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=span_and_event`.
- **Instrumentation**: Instruments the `google_genai` SDK and `langchain` frameworks. It customizes the LangChain callback manager so that standard LangChain spans work with Gemini endpoints, which the upstream handler otherwise skips (see the implementation note in the [docs](./docs/gcp_langgraph_documentation.md#4-structured-json-cloud-logging) about the `ChatOpenAI` shim).
- **Optional log content control**: Set `OTEL_DEMO_LOG_GENAI_CONTENT=false` to stop mirroring prompt/response content into Cloud Logging (it is still captured on spans).
