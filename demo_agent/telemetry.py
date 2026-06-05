import os
import json
from uuid import UUID
from contextvars import ContextVar
from typing import Any, Optional

import google.auth
import google.auth.transport.requests
import grpc
from google.auth.transport.grpc import AuthMetadataPlugin
from wrapt import wrap_function_wrapper

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.genai.langchain.callback_handler import (
    OpenTelemetryLangChainCallbackHandler,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.util.genai.handler import get_telemetry_handler

from langchain_core.callbacks import BaseCallbackHandler


conversation_id_var = ContextVar("conversation_id", default=None)


def _build_resource(project_id: str | None) -> Resource:
    attributes = {
        "cloud.provider": "gcp",
        "service.name": os.environ.get("OTEL_SERVICE_NAME", "demo-langgraph"),
    }
    if project_id:
        attributes["gcp.project_id"] = project_id

    cloud_resource_id = os.environ.get("CLOUD_RESOURCE_ID")
    if cloud_resource_id:
        attributes["cloud.resource_id"] = cloud_resource_id

    return Resource(attributes=attributes)


def _genai_content_logging_enabled() -> bool:
    """Whether to mirror prompt/response content into Cloud Logging.

    Message content is already captured on spans/events via
    ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``. Set
    ``OTEL_DEMO_LOG_GENAI_CONTENT=false`` to avoid duplicating (potentially
    large or sensitive) payloads into Cloud Logging as well.
    """
    return os.environ.get("OTEL_DEMO_LOG_GENAI_CONTENT", "true").lower() == "true"


class SafeJSONEncoder(json.JSONEncoder):
    def default(self, o):
        try:
            return super().default(o)
        except TypeError:
            return str(o)


def log_structured(severity: str, message: str, **kwargs):
    entry = {
        "severity": severity,
        "message": message,
        **kwargs,
    }
    try:
        print(json.dumps(entry, cls=SafeJSONEncoder), flush=True)
    except Exception as e:
        print(
            f"[{severity}] {message} - {kwargs} (Failed to serialize structured logs: {e})",
            flush=True,
        )


class ComplianceSpanProcessor(SpanProcessor):
    def on_start(self, span, parent_context=None) -> None:
        pass

    def on_end(self, span) -> None:
        attributes = getattr(span, "_attributes", None)
        if attributes is None:
            return

        # 1. Map provider names to standard gcp.vertex_ai
        provider = attributes.get("gen_ai.provider.name")
        if provider in ("google_genai", "vertex_ai", "google", "google_vertexai"):
            attributes["gen_ai.provider.name"] = "gcp.vertex_ai"

        # 2. Inject conversation ID from ContextVar if not already present
        conv_id = conversation_id_var.get()
        if conv_id and "gen_ai.conversation.id" not in attributes:
            attributes["gen_ai.conversation.id"] = conv_id


class GoogleSupportedOTelLangChainCallbackHandler(OpenTelemetryLangChainCallbackHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Track ContextVar reset tokens per root run so the conversation id does
        # not leak across requests sharing a worker context.
        self._conv_tokens: dict[UUID, Any] = {}

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if not parent_run_id:
            conv_id = None
            if metadata:
                for key in ("thread_id", "session_id", "conversation_id"):
                    if metadata.get(key):
                        conv_id = metadata[key]
                        break
            if conv_id is not None:
                self._conv_tokens[run_id] = conversation_id_var.set(conv_id)

            if _genai_content_logging_enabled():
                try:
                    log_structured(
                        severity="INFO",
                        message="Workflow Input Received",
                        gen_ai_operation="invoke_workflow",
                        gen_ai_conversation_id=conversation_id_var.get(),
                        inputs=inputs,
                    )
                except Exception:
                    pass

        return super().on_chain_start(
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if not parent_run_id and _genai_content_logging_enabled():
            try:
                log_structured(
                    severity="INFO",
                    message="Workflow Output Generated",
                    gen_ai_operation="invoke_workflow",
                    gen_ai_conversation_id=conversation_id_var.get(),
                    outputs=outputs,
                )
            except Exception:
                pass

        # End the upstream span (which triggers ComplianceSpanProcessor.on_end
        # while the conversation id is still set) before resetting the ContextVar.
        result = super().on_chain_end(
            outputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )

        token = self._conv_tokens.pop(run_id, None)
        if token is not None:
            try:
                conversation_id_var.reset(token)
            except ValueError:
                # Reset attempted from a different context (e.g. async); ignore.
                pass
        return result

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if _genai_content_logging_enabled():
            try:
                from langchain_core.messages import messages_to_dict
                messages_list = []
                for sub_messages in messages:
                    messages_list.extend(messages_to_dict(sub_messages))

                log_structured(
                    severity="INFO",
                    message="GenAI Prompt Sent",
                    gen_ai_operation="chat",
                    gen_ai_conversation_id=conversation_id_var.get(),
                    model=serialized.get("name") or "unknown",
                    prompts=messages_list,
                )
            except Exception:
                log_structured(
                    severity="INFO",
                    message="GenAI Prompt Sent",
                    gen_ai_operation="chat",
                    gen_ai_conversation_id=conversation_id_var.get(),
                    model=serialized.get("name") or "unknown",
                    prompts=str(messages),
                )

        # The upstream handler only instruments models whose serialized name is
        # "ChatOpenAI"/"ChatBedrock" and derives the provider from the
        # "ls_provider" metadata. Spoofing the class name lets Gemini models pass
        # that gate; the provider stays google_* (metadata-derived) and is
        # normalized to gcp.vertex_ai by ComplianceSpanProcessor.
        class_name = serialized.get("name")
        if class_name in ("ChatGoogleGenerativeAI", "ChatVertexAI"):
            # Clone serialized to avoid mutating caller's data
            serialized = serialized.copy()
            serialized["name"] = "ChatOpenAI"

            # Map invocation parameter 'model' or 'model_name'
            if "invocation_params" in kwargs and kwargs["invocation_params"]:
                params = kwargs["invocation_params"]
                if "model" in params and "model_name" not in params:
                    params["model_name"] = params["model"]
            else:
                if "model" in kwargs and "model_name" not in kwargs:
                    kwargs["model_name"] = kwargs["model"]

        super().on_chat_model_start(
            serialized,
            messages,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if _genai_content_logging_enabled():
            try:
                from langchain_core.messages import message_to_dict
                response_list = []
                for generation in getattr(response, "generations", []):
                    for chat_generation in generation:
                        if chat_generation.message:
                            response_list.append(message_to_dict(chat_generation.message))

                log_structured(
                    severity="INFO",
                    message="GenAI Response Received",
                    gen_ai_operation="chat",
                    gen_ai_conversation_id=conversation_id_var.get(),
                    responses=response_list,
                )
            except Exception:
                log_structured(
                    severity="INFO",
                    message="GenAI Response Received",
                    gen_ai_operation="chat",
                    gen_ai_conversation_id=conversation_id_var.get(),
                    responses=str(response),
                )
        super().on_llm_end(
            response,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )


class CustomToolTracingCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self._tracer = trace.get_tracer("demo-langgraph")
        self._active_spans: dict[UUID, Any] = {}

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name") or "unknown_tool"
        span = self._tracer.start_span(
            name=f"execute_tool {tool_name}",
            kind=SpanKind.INTERNAL,
        )
        span.set_attribute("gen_ai.operation.name", "execute_tool")
        span.set_attribute("gen_ai.tool.name", tool_name)
        span.set_attribute(
            "gen_ai.tool.type",
            "extension" if tool_name == "call_logging_mcp" else "function",
        )

        # Add tool description if available
        desc = serialized.get("description")
        if desc:
            span.set_attribute("gen_ai.tool.description", desc)

        if input_str:
            span.set_attribute("gen_ai.tool.call.arguments", input_str)

        self._active_spans[run_id] = span

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        span = self._active_spans.pop(run_id, None)
        if span:
            if output:
                span.set_attribute("gen_ai.tool.call.result", str(output))
            span.end()

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        span = self._active_spans.pop(run_id, None)
        if span:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.set_attribute("error.type", error.__class__.__name__)
            span.end()


class _CustomCallbackManagerInitWrapper:
    def __init__(
        self,
        google_handler: GoogleSupportedOTelLangChainCallbackHandler,
        tool_handler: CustomToolTracingCallbackHandler,
    ):
        self._google_handler = google_handler
        self._tool_handler = tool_handler

    def __call__(
        self,
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        wrapped(*args, **kwargs)

        # Remove the default OpenTelemetryLangChainCallbackHandler handler (if any)
        # to prevent it from ignoring Google models or emitting duplicates. We
        # match the exact type so our subclass instance is preserved.
        instance.handlers = [
            h for h in instance.handlers
            if type(h) is not OpenTelemetryLangChainCallbackHandler
        ]
        if hasattr(instance, "inheritable_handlers"):
            instance.inheritable_handlers = [
                h for h in instance.inheritable_handlers
                if type(h) is not OpenTelemetryLangChainCallbackHandler
            ]

        # Register our custom handlers if they aren't already registered
        handlers_to_check = getattr(
            instance, "inheritable_handlers", getattr(instance, "handlers", [])
        )

        for handler in handlers_to_check:
            if isinstance(handler, GoogleSupportedOTelLangChainCallbackHandler):
                break
        else:
            instance.add_handler(self._google_handler, inherit=True)

        for handler in handlers_to_check:
            if isinstance(handler, CustomToolTracingCallbackHandler):
                break
        else:
            instance.add_handler(self._tool_handler, inherit=True)


def init_telemetry() -> None:
    """Initialize OpenTelemetry with OTLP/gRPC export and GenAI instrumentation."""
    # Disable LangSmith OTel integration to avoid double instrumentation.
    os.environ["LANGSMITH_OTEL_ENABLED"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"

    # Opt-in to GenAI semantic conventions for compatible instrumentations.
    os.environ.setdefault("OTEL_SERVICE_NAME", "demo-langgraph")
    os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental")

    # If set to 'true', map it to 'span_and_event' to prevent OTel SDK enum
    # warnings in experimental mode.
    capture_content = os.environ.get("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")
    if not capture_content or capture_content.lower() == "true":
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "span_and_event"

    os.environ.setdefault("GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY", "true")

    # Provider setup and instrumentation are not idempotent on their own, so run
    # them exactly once even if init_telemetry() is called again (it is invoked
    # both client-side during packaging and server-side in the model builder).
    if getattr(init_telemetry, "_initialized", False):
        return
    init_telemetry._initialized = True

    current_provider = trace.get_tracer_provider()
    if isinstance(current_provider, TracerProvider):
        # A real TracerProvider already exists (e.g. installed by the Agent
        # Engine runtime). Reuse it and its exporter, but attach our compliance
        # processor so provider normalization and conversation-id injection still
        # happen. We deliberately do NOT add a second exporter here.
        provider = current_provider
        provider.add_span_processor(ComplianceSpanProcessor())
    else:
        credentials, project_id = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        auth_request = google.auth.transport.requests.Request()
        auth_metadata_plugin = AuthMetadataPlugin(credentials=credentials, request=auth_request)
        channel_creds = grpc.composite_channel_credentials(
            grpc.ssl_channel_credentials(),
            grpc.metadata_call_credentials(auth_metadata_plugin),
        )

        provider = TracerProvider(resource=_build_resource(project_id))
        exporter = OTLPSpanExporter(
            endpoint="telemetry.googleapis.com:443",
            credentials=channel_creds,
        )
        provider.add_span_processor(ComplianceSpanProcessor())
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

    # Install the upstream GenAI instrumentation (idempotent within a process).
    LangChainInstrumentor().instrument(skip_dep_check=True)
    GoogleGenAiSdkInstrumentor().instrument(skip_dep_check=True)

    # Initialize our custom Google-supported and tool tracing callback handlers,
    # then patch the LangChain callback manager so they are auto-registered.
    telemetry_handler = get_telemetry_handler()
    google_callback = GoogleSupportedOTelLangChainCallbackHandler(
        telemetry_handler=telemetry_handler
    )
    tool_callback = CustomToolTracingCallbackHandler()

    wrap_function_wrapper(
        "langchain_core.callbacks",
        "BaseCallbackManager.__init__",
        _CustomCallbackManagerInitWrapper(google_callback, tool_callback),
    )
