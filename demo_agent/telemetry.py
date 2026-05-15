import os
import grpc
import google.auth
import google.auth.transport.requests
from google.auth.transport.grpc import AuthMetadataPlugin
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor

def init_telemetry():
    """Initializes OpenTelemetry with Google Cloud exporter and LangChain instrumentation."""
    # Disable LangSmith OTel integration to avoid double instrumentation
    os.environ["LANGSMITH_OTEL_ENABLED"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["OTEL_SERVICE_NAME"] = "demo-langgraph"
    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"


    # Get default credentials with broad cloud-platform scope
    credentials, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    auth_request = google.auth.transport.requests.Request()
    
    # Set up OTLP gRPC auth as per official docs
    auth_metadata_plugin = AuthMetadataPlugin(credentials=credentials, request=auth_request)
    channel_creds = grpc.composite_channel_credentials(
        grpc.ssl_channel_credentials(),
        grpc.metadata_call_credentials(auth_metadata_plugin),
    )

    cloud_resource_id = os.environ.get("CLOUD_RESOURCE_ID")
    resource_attributes = {
        "gcp.project_id": project,
        "cloud.provider": "gcp",
        "service.name": "demo-langgraph",
    }
    if cloud_resource_id:
        resource_attributes["cloud.resource_id"] = cloud_resource_id
    resource = Resource(attributes=resource_attributes)
    
    provider = TracerProvider(resource=resource)
    otlp_exporter = OTLPSpanExporter(
        credentials=channel_creds,
        endpoint="telemetry.googleapis.com:443",
    )
    processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    # Instrument LangChain
    LangchainInstrumentor().instrument()

    # Instrument Google GenAI for official gen_ai.* attributes
    GoogleGenAiSdkInstrumentor().instrument()
