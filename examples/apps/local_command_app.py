"""Local serving example using command-based AppEnvironment.

This example demonstrates how to serve an app locally using the ``command``
specification pattern. The app is run as a subprocess.

Usage (CLI):
    flyte serve --local examples/apps/local_command_app.py app_env

Note: This example uses ``python -m http.server`` as the command, which only
serves static files. For a real application, you would use a proper web server
command like ``uvicorn main:app --host 0.0.0.0 --port 8082``.
"""

import flyte
from flyte.app import AppEnvironment

app_env = AppEnvironment(
    name="local-static-server",
    image=flyte.Image.from_debian_base(python_version=(3, 12)),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    command="python -m http.server 8082",
    port=8082,
    requires_auth=False,
)


if __name__ == "__main__":
    import httpx

    # Serve the app locally (non-blocking)
    local_app = flyte.with_servecontext(mode="local").serve(app_env)

    # Wait for the app to be ready
    assert local_app.is_ready(path="/"), "App failed to become ready"
    print(f"App is ready at {local_app.endpoint}")

    # Test the endpoint
    response = httpx.get(local_app.endpoint)
    print(f"Response status: {response.status_code}")

    # Shut down the local app
    local_app.shutdown()
    print("Done!")
