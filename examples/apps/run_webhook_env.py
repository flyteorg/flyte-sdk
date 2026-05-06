# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "httpx",
#     "flyte>=2.0.0",
# ]
# ///

"""
Example demonstrating the FlyteWebhookAppEnvironment.

This example shows how to use the pre-built FlyteWebhookAppEnvironment which provides
a ready-to-use FastAPI application with endpoints for common Flyte operations:

- POST /run-task/{project}/{domain}/{name} - Run a task
- GET /run/{name} - Get run metadata
- GET /run/{name}/io - Get run inputs/outputs
- POST /run/{name}/abort - Abort a run
- GET /task/{project}/{domain}/{name} - Get task metadata
- GET /app/{name} - Get app status
- POST /app/{name}/activate - Activate an app
- POST /app/{name}/deactivate - Deactivate an app
- POST /app/{name}/call - Call another app's endpoint
- POST /trigger/{task_name}/{trigger_name}/activate - Activate a trigger
- POST /trigger/{task_name}/{trigger_name}/deactivate - Deactivate a trigger
- GET /me - Get current user info
- GET /health - Health check

All endpoints use FastAPIPassthroughAuthMiddleware for authentication.
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

import os

import flyte
import flyte.app
from flyte.app.extras import FlyteWebhookAppEnvironment

logger = logging.getLogger(__name__)


webhook_env = FlyteWebhookAppEnvironment(
    name="webhook-env-example",
    title="Flyte Webhook Environment Example",
    description="A pre-built webhook service for Flyte operations",
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=True,
    scaling=flyte.app.Scaling(replicas=1),
)

# You can also create a task environment to test the webhook
task_env = flyte.TaskEnvironment(
    name="webhook-env-task",
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@task_env.task
async def example_task(x: int, y: str) -> str:
    """A simple task to test the webhook."""
    return f"Result: {x} - {y}"


if __name__ == "__main__":
    import httpx

    flyte.init_from_config(log_level=logging.DEBUG)

    # Deploy the task environment first
    flyte.deploy(task_env)

    # Serve the webhook environment
    served_app = flyte.serve(webhook_env)
    url = served_app.url
    endpoint = served_app.endpoint
    print(f"Webhook is served on {url}")
    print(f"OpenAPI docs available at: {endpoint}/docs")

    # Use a Flyte user token for passthrough auth
    token = os.getenv("FLYTE_API_KEY")
    if not token:
        raise ValueError("FLYTE_API_KEY not set. Obtain with: flyte get api-key")

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "flyte-webhook-client/1.0",
    }

    served_app.activate()

    with httpx.Client(headers=headers) as client:
        # Test /health endpoint (no auth required)
        health_resp = client.get(endpoint.rstrip("/") + "/health")
        health_resp.raise_for_status()
        print(f"/health response: {health_resp.text}")

        # Test /me endpoint to verify passthrough auth works
        me_resp = client.get(endpoint.rstrip("/") + "/me")
        me_resp.raise_for_status()
        print(f"/me response: {me_resp.text}")

        # Test /run-task endpoint
        data = {"x": 42, "y": "hello"}
        route = "/run-task/development/flytesnacks/webhook-env-task.example_task"
        print(f"Running task at: {endpoint.rstrip('/')}{route}")

        resp = client.post(endpoint.rstrip("/") + route, json=data)
        if resp.is_success:
            result = resp.json()
            print(f"Run task response: {result}")

            # Test /run/{name} endpoint to get run status
            run_name = result["name"]
            run_resp = client.get(endpoint.rstrip("/") + f"/run/{run_name}")
            run_resp.raise_for_status()
            print(f"/run/{run_name} response: {run_resp.text}")
        else:
            print(f"HTTP Error: {resp.status_code} - {resp.text}")
