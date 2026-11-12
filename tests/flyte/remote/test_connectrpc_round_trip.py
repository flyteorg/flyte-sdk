"""A unary call through the SDK's real clients and the installed connectrpc, against a local Connect server.

Mocked request contexts cannot tell whether the installed connectrpc marshals flyteidl2's google.protobuf
messages or hands the interceptors the headers they expect; only a real call does.
"""

import gzip
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from flyteidl2.project import project_service_pb2

from flyte.remote._client.controlplane import ClientSet


class _GetProjectHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        if self.headers.get("content-encoding") == "gzip":
            body = gzip.decompress(body)
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
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GetProjectHandler)
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
