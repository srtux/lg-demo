# Deploying LangGraph Agents on Gemini Enterprise Agent Platform

Gemini Enterprise Agent Platform provides support for deploying and managing stateful, multi-turn LangGraph agents. By integrating with `vertexai.agent_engines`, you can seamlessly host your LangGraph architectures while leveraging Google Cloud's built-in Agent Observability and tracing capabilities.

To ensure strict compliance with standard OpenTelemetry GenAI conventions, this setup uses the official OTel community library **`opentelemetry-instrumentation-genai-langchain`** (the official home resulting from the donation of OpenInference to the OpenTelemetry project) rather than the legacy **`openinference-instrumentation-langchain`** package, eliminating the need for transitional semantic convention translation flags.

There are several core configuration changes and best practices to be aware of:

## 1. Use the Agent Engine SDK

When defining and deploying your agent, transition to the General Availability (GA) `vertexai.agent_engines` SDK rather than legacy Preview endpoints.

- **Agent Definition**: Wrap your agent configuration using `agent_engines.LanggraphAgent`.
- **Deployment**: Use `agent_engines.create()` or `agent_engines.update()` to deploy your agent. Be sure to include your local agent modules via the `extra_packages` parameter (e.g., `extra_packages=["demo_agent"]`) so they are properly bundled and uploaded to the remote runtime.

## 2. Configure Model and Checkpointer Builders

The Agent Platform manages the runtime lifecycle of your agent in a remote container. To ensure models and state management initialize correctly upon execution, you must define them using builder functions.

- **Custom Model Builder**: Supply a `model_builder` function to `LanggraphAgent`. This function is invoked during runtime. Use it to instantiate your models and to trigger any necessary server-side initialization (like starting your OpenTelemetry exporters). For Gemini on Vertex AI, use `ChatGoogleGenerativeAI` from `langchain-google-genai` with `vertexai=True` (plus `project`/`location`). Per [langchain-google#1422](https://github.com/langchain-ai/langchain-google/discussions/1422), this unified path supersedes `ChatVertexAI` (from `langchain-google-vertexai`), which is being deprecated.
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
- **Context-Local Conversation ID Propagation**: LangGraph passes thread information dynamically via config metadata. In your custom callbacks, capture the active thread ID (`thread_id`, `session_id`, or `conversation_id`) from the **root** chain's metadata and store it in a context-local variable (`contextvars.ContextVar`), retaining the reset token so the value is restored when the root workflow ends (preventing the id from leaking across requests that share a worker context). A custom `SpanProcessor` then intercepts spans at `on_end` and injects the `gen_ai.conversation.id` attribute.
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
- **Avoid duplicating sensitive content**: prompt/response content is already captured on spans/events via `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`. Mirroring it into logs as well doubles storage of potentially large or sensitive payloads. In this repo the log-side content mirroring can be turned off by setting `OTEL_DEMO_LOG_GENAI_CONTENT=false` (structural log entries are still emitted).

> **Implementation note (Gemini ↔ LangChain shim)**: the upstream `opentelemetry-instrumentation-genai-langchain` callback handler only instruments chat models whose serialized class name is `ChatOpenAI`/`ChatBedrock`, and derives the provider from the `ls_provider` metadata. To trace Gemini models, the custom handler temporarily renames the serialized class to `ChatOpenAI` so it passes that gate; the provider stays `google_*` (metadata-derived) and is normalized to `gcp.vertex_ai` by the compliance span processor. If you upgrade that upstream package, re-verify this gate still exists, or Gemini spans may silently stop being emitted.

## 5. Enable Telemetry in the Deployment Environment

When deploying the agent to the cloud, explicitly configure the remote runtime container to export OpenTelemetry data to Google Cloud Trace and Logging.

Pass the following environment variables via the `env_vars` argument when calling `agent_engines.create()`:

- `GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY="true"`: Activates the native telemetry ingestion pipeline on the Agent Platform.
- `OTEL_SEMCONV_STABILITY_OPT_IN="gen_ai_latest_experimental"`: Opts into the latest experimental GenAI semantic conventions to ensure your spans are formatted correctly.
- `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="span_and_event"`: Ensures that full prompt inputs and LLM response outputs are captured in your traces for effective debugging and quality auditing.

---

By adopting these core configurations, your LangGraph agents will run efficiently on the Google Cloud Agent Platform, featuring fully managed state checkpointers, structured logs in Cloud Logging, and deeply compliant, semantic-convention-aligned observability across complex, multi-turn reasoning workflows.
