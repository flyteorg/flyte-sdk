"""A unary call through the SDK's real clients and the installed connectrpc, against a local Connect server.

Mocked request contexts cannot tell whether the installed connectrpc marshals flyteidl2's google.protobuf
messages or hands the interceptors the headers they expect; only a real call does.
"""

import gzip
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from flyteidl2.auth import auth_service_pb2
from flyteidl2.project import project_service_pb2

from flyte._initialize import _init_for_testing
from flyte.remote._client.controlplane import ClientSet

_ISSUER = "https://issuer.example.com"


class _ConnectHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        if self.headers.get("content-encoding") == "gzip":
            body = gzip.decompress(body)
        if self.path.endswith("/GetOAuth2Metadata"):
            response = auth_service_pb2.GetOAuth2MetadataResponse(issuer=_ISSUER).SerializeToString()
        else:
            request = project_service_pb2.GetProjectRequest.FromString(body)
            response = project_service_pb2.GetProjectResponse(
                project=project_service_pb2.Project(id=request.id, name=self.headers.get("x-request-id", ""))
            ).SerializeToString()
        self.send_response(200)
        self.send_header("content-type", "application/proto")
        self.send_header("content-length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


@pytest.fixture
def connect_endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ConnectHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.asyncio
async def test_unary_call_round_trips_protobuf_and_request_id(connect_endpoint):
    clients = await ClientSet.for_endpoint(connect_endpoint, insecure=True)

    response = await clients.project_domain_service.get_project(project_service_pb2.GetProjectRequest(id="p1"))

    assert isinstance(response, project_service_pb2.GetProjectResponse)
    assert response.project.id == "p1"
    # The server echoes the x-request-id header the default metadata interceptor injected.
    assert response.project.name


@pytest.mark.asyncio
async def test_mcp_issuer_discovery_round_trips_protobuf(connect_endpoint):
    from flyte.ai.mcp._protected_resource import discover_authorization_server

    clients = await ClientSet.for_endpoint(connect_endpoint, insecure=True)
    await _init_for_testing(client=clients, project="test", domain="test")

    assert await discover_authorization_server() == _ISSUER
