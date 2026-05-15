import os
import google.auth
import google.auth.transport.requests
import grpc
from google.auth.transport.grpc import AuthMetadataPlugin
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


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


def init_telemetry() -> None:
    """Initialize OpenTelemetry with OTLP/gRPC export and GenAI instrumentation."""
    # Disable LangSmith OTel integration to avoid double instrumentation.
    os.environ["LANGSMITH_OTEL_ENABLED"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"

    # Opt-in to GenAI semantic conventions for compatible instrumentations.
    os.environ.setdefault("OTEL_SERVICE_NAME", "demo-langgraph")
    os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental")

    # Avoid duplicate instrumentation if init_telemetry() is called more than once.
    current_provider = trace.get_tracer_provider()
    if isinstance(current_provider, TracerProvider):
        return

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
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    LangchainInstrumentor().instrument()
    GoogleGenAiSdkInstrumentor().instrument()
