"""Local serving example using the @app_env.server decorator.

This example demonstrates how to serve an app locally using the
``@app_env.server`` decorator pattern, and then call it from a task.

Usage (SDK):
    python examples/apps/local_server_decorator_app.py

Usage (CLI):
    flyte serve --local examples/apps/local_server_decorator_app.py app_env
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import httpx

import flyte
from flyte.app import AppEnvironment

app_env = AppEnvironment(
    name="local-multiply",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("httpx"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    port=8081,
    requires_auth=False,
)

task_env = flyte.TaskEnvironment(
    name="local-multiply-task-env",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("httpx"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


class MultiplyHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler that multiplies two numbers."""

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if parsed.path == "/":
            x = int(params.get("x", [0])[0])
            y = int(params.get("y", [1])[0])
            result = x * y
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"result": result}).encode())
        elif parsed.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default logging."""


@app_env.server
def serve():
    """Start the HTTP server on the configured port."""
    port = app_env.get_port().port
    server = HTTPServer(("127.0.0.1", port), MultiplyHandler)
    print(f"Serving on http://127.0.0.1:{port}")
    server.serve_forever()


@task_env.task
async def multiply_task(x: int, y: int) -> int:
    """Task that calls the local multiply endpoint."""
    print(f"Calling app at {app_env.endpoint}")
    async with httpx.AsyncClient() as client:
        response = await client.get(app_env.endpoint, params={"x": x, "y": y})
        response.raise_for_status()
        return response.json()["result"]


if __name__ == "__main__":
    # Serve the app locally (non-blocking)
    local_app = flyte.with_servecontext(mode="local").serve(app_env)

    # Wait for the app to be ready
    assert local_app.is_ready(path="/health"), "App failed to become ready"
    print(f"App is ready at {local_app.endpoint}")

    # Run a task that calls the local app
    result = flyte.with_runcontext(mode="local").run(multiply_task, x=7, y=6)
    print(f"Result: {result.outputs()[0]}")
    assert result.outputs()[0] == 42

    # Shut down the local app
    local_app.shutdown()
    print("Done!")
