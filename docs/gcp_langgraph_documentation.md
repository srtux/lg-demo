# Deploying LangGraph Agents on Google Cloud Agent Platform

Google Cloud Agent Platform (via Vertex AI Agent Engine) provides support for deploying and managing stateful, multi-turn LangGraph agents. By integrating with `vertexai.agent_engines`, you can seamlessly host your LangGraph architectures while leveraging Google Cloud's built-in observability and tracing solutions.

To ensure strict compliance with standard OpenTelemetry GenAI conventions, this setup uses the official OTel community library **`opentelemetry-instrumentation-genai-langchain`** (the official home resulting from the donation of OpenInference to the OpenTelemetry project) rather than the legacy **`openinference-instrumentation-langchain`** package, eliminating the need for transitional semantic convention translation flags.

To ensure a smooth transition from local development to production on Agent Platform—particularly for tracing and memory management—there are several core configuration changes and best practices you should adopt.

## 1. Use the Agent Engine SDK

When defining and deploying your agent, transition to the General Availability (GA) `vertexai.agent_engines` SDK rather than legacy Preview endpoints.

- **Agent Definition**: Wrap your agent configuration using `agent_engines.LanggraphAgent`.
- **Deployment**: Use `agent_engines.create()` or `agent_engines.update()` to deploy your agent. Be sure to include your local agent modules via the `extra_packages` parameter (e.g., `extra_packages=["demo_agent"]`) so they are properly bundled and uploaded to the remote runtime.

## 2. Configure Model and Checkpointer Builders

The Agent Platform manages the runtime lifecycle of your agent in a remote container. To ensure models and state management initialize correctly upon execution, you must define them using builder functions.

- **Custom Model Builder**: Supply a `model_builder` function to `LanggraphAgent`. This function is invoked during runtime. Use it to instantiate your models (e.g., `ChatGoogleGenerativeAI`) and to trigger any necessary server-side initialization (like starting your OpenTelemetry exporters).
- **State Management (Checkpointers)**: Enable stateful, multi-turn conversations by returning a checkpointer (such as LangGraph's `MemorySaver`) through a `checkpointer_builder` function.
- **Thread IDs**: When querying the deployed agent, explicitly pass the thread ID in the query configuration payload so the checkpointer can resume the conversation state.
  ```python
  response = remote_agent.query(
      input="What is my account balance?",
      config={"configurable": {"thread_id": "user_session_123"}}
  )
  ```

## 3. Instrument with OpenTelemetry (OTel)

For production visibility, integrating OpenTelemetry with GenAI Semantic Conventions is critical.

- **Disable Native Tracing**: Set `enable_tracing=False` in your `LanggraphAgent` configuration. This prevents LangGraph from emitting default traces that may duplicate or conflict with your custom OpenTelemetry configuration.
- **Dual Initialization**: Initialize your OpenTelemetry configuration twice:
  1.  On the **client side** right before packaging and deploying the agent.
  2.  On the **server side** inside your `custom_model_builder` function to ensure it runs in the deployed container.
- **Agent Span Classification**: By default, LangGraph graph executions are categorized as generic `invoke_workflow` spans. To classify your execution as an agent span (`invoke_agent`), supply agent signal metadata during execution:
  ```python
  config = {
      "configurable": {"thread_id": "thread-123"},
      "metadata": {
          "otel_agent_span": True,             # Triggers invoke_agent classification
          "agent_name": "LoggingAssistant",    # Populates gen_ai.agent.name
          "agent_id": "logging-assistant-v1",  # Populates gen_ai.agent.id
          "agent_description": "Expert GCP logs assistant"
      }
  }
  response = remote_agent.query(input=..., config=config)
  ```
- **Context-Local Conversation ID Propagation**: LangGraph passes thread information dynamically via config metadata. In your custom callbacks, capture the active thread ID (`thread_id`, `session_id`, or `conversation_id`) and store it in a context-local variable (`contextvars.ContextVar`). Use a custom `SpanProcessor` to intercept spans at `on_end` and inject the `gen_ai.conversation.id` attribute.
- **OTel Provider Name Normalization**: The unified `google-genai` SDK and LangChain integrations may output non-standard `gen_ai.provider.name` values (e.g. `google_genai` or `vertex_ai`). Use a custom `SpanProcessor` to intercept spans and normalize the provider name to the standard **`gcp.vertex_ai`**.
- **Tool Tracing Compliance**: To capture tool activities:
  - Create internal spans using `SpanKind.INTERNAL` named `execute_tool {tool_name}`.
  - Set `gen_ai.operation.name = "execute_tool"`.
  - Always stringify tool arguments (`gen_ai.tool.call.arguments`) and results (`gen_ai.tool.call.result`) to avoid OTel SDK serialization warnings.
  - Record errors using the `error.type` attribute containing the Python exception class name.
- **Synchronous Trace Flushing**: If executing short-lived CLI tasks or ephemeral containers, call `trace.get_tracer_provider().shutdown()` before the Python process exits. Standard background trace exporters (`BatchSpanProcessor`) write asynchronously; shutdown ensures all remaining spans are flushed synchronously.

## 4. Structured JSON Cloud Logging

Ensure prompt messages and responses are written to Cloud Logging in a structured format:
- Implement custom callback hooks for `on_chain_start`, `on_chain_end`, `on_chat_model_start`, and `on_llm_end`.
- Format log entries as structured JSON objects containing severity, message, list of prompt messages (using LangChain's `messages_to_dict`), output responses, and the correlation `gen_ai_conversation_id`.
- Output these JSON payloads directly to `stdout` / `stderr`. Google Cloud Logging automatically ingests stdout JSON objects into structured `jsonPayload` fields.

## 5. Enable Telemetry in the Deployment Environment

When deploying the agent to the cloud, explicitly configure the remote runtime container to export OpenTelemetry data to Google Cloud Trace and Logging.

Pass the following environment variables via the `env_vars` argument when calling `agent_engines.create()`:

- `GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY="true"`: Activates the native telemetry ingestion pipeline on the Agent Platform.
- `OTEL_SEMCONV_STABILITY_OPT_IN="gen_ai_latest_experimental"`: Opts into the latest experimental GenAI semantic conventions to ensure your spans are formatted correctly.
- `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="span_and_event"`: Ensures that full prompt inputs and LLM response outputs are captured in your traces for effective debugging and quality auditing.

---

By adopting these core configurations, your LangGraph agents will run efficiently on the Google Cloud Agent Platform, featuring fully managed state checkpointers, structured logs in Cloud Logging, and deeply compliant, semantic-convention-aligned observability across complex, multi-turn reasoning workflows.
