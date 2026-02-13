# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "httpx",
#     "flyte==2.0.0b56",
# ]
# ///

"""
Exhaustive example demonstrating all FlyteWebhookAppEnvironment endpoints.

This example tests every endpoint available in the FlyteWebhookAppEnvironment:

Health & User:
- GET /health - Health check
- GET /me - Get current user info

Task Operations:
- POST /run-task/{domain}/{project}/{name} - Run a task
- GET /task/{domain}/{project}/{name} - Get task metadata

Run Operations:
- GET /run/{name} - Get run metadata
- GET /run/{name}/io - Get run inputs/outputs
- POST /run/{name}/abort - Abort a run

App Operations:
- GET /app/{name} - Get app status
- POST /app/{name}/activate - Activate an app
- POST /app/{name}/deactivate - Deactivate an app
- POST /app/{name}/call - Call another app's endpoint

Trigger Operations:
- POST /trigger/{task_name}/{trigger_name}/activate - Activate a trigger
- POST /trigger/{task_name}/{trigger_name}/deactivate - Deactivate a trigger

Image Build Operations:
- POST /build-image - Build a container image

HuggingFace Prefetch Operations:
- POST /prefetch/hf-model - Prefetch a HuggingFace model
- GET /prefetch/hf-model/{run_name} - Get prefetch run status
- GET /prefetch/hf-model/{run_name}/io - Get prefetch run I/O
- POST /prefetch/hf-model/{run_name}/abort - Abort a prefetch run

All endpoints use FastAPIPassthroughAuthMiddleware for authentication.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import flyte
import flyte.app
from fastapi import FastAPI
from flyte.app.extras import FlyteWebhookAppEnvironment, FastAPIAppEnvironment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single endpoint test."""

    endpoint: str
    method: str
    success: bool
    status_code: int | None
    response: Any
    error: str | None = None


class WebhookEndpointTester:
    """
    Test harness for exhaustively testing all FlyteWebhookAppEnvironment endpoints.
    """

    def __init__(self, endpoint: str, headers: dict[str, str]):
        import httpx

        self.endpoint = endpoint.rstrip("/")
        self.headers = headers
        self.results: list[TestResult] = []
        self.client = httpx.Client(headers=headers, timeout=60.0)

    def _make_request(
        self,
        method: str,
        path: str,
        json: dict | None = None,
        params: dict | None = None,
    ) -> TestResult:
        """Make an HTTP request and return the result."""
        url = f"{self.endpoint}{path}"
        try:
            if method.upper() == "GET":
                resp = self.client.get(url, params=params)
            elif method.upper() == "POST":
                resp = self.client.post(url, json=json, params=params)
            else:
                raise ValueError(f"Unsupported method: {method}")

            result = TestResult(
                endpoint=path,
                method=method,
                success=resp.is_success,
                status_code=resp.status_code,
                response=resp.json() if resp.is_success else resp.text,
            )
        except Exception as e:
            result = TestResult(
                endpoint=path,
                method=method,
                success=False,
                status_code=None,
                response=None,
                error=str(e),
            )

        self.results.append(result)
        return result

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    # ==================== Health & User Endpoints ====================

    def test_health(self) -> TestResult:
        """Test GET /health endpoint."""
        logger.info("Testing GET /health")
        return self._make_request("GET", "/health")

    def test_me(self) -> TestResult:
        """Test GET /me endpoint."""
        logger.info("Testing GET /me")
        return self._make_request("GET", "/me")

    # ==================== Task Endpoints ====================

    def test_run_task(
        self,
        domain: str,
        project: str,
        name: str,
        inputs: dict,
        version: str | None = None,
    ) -> TestResult:
        """Test POST /run-task/{domain}/{project}/{name} endpoint."""
        path = f"/run-task/{domain}/{project}/{name}"
        params = {"version": version} if version else None
        logger.info(f"Testing POST {path}")
        return self._make_request("POST", path, json=inputs, params=params)

    def test_get_task_metadata(
        self,
        domain: str,
        project: str,
        name: str,
        version: str | None = None,
    ) -> TestResult:
        """Test GET /task/{domain}/{project}/{name} endpoint."""
        path = f"/task/{domain}/{project}/{name}"
        params = {"version": version} if version else None
        logger.info(f"Testing GET {path}")
        return self._make_request("GET", path, params=params)

    # ==================== Run Endpoints ====================

    def test_get_run(self, run_name: str) -> TestResult:
        """Test GET /run/{name} endpoint."""
        path = f"/run/{run_name}"
        logger.info(f"Testing GET {path}")
        return self._make_request("GET", path)

    def test_get_run_io(self, run_name: str) -> TestResult:
        """Test GET /run/{name}/io endpoint."""
        path = f"/run/{run_name}/io"
        logger.info(f"Testing GET {path}")
        return self._make_request("GET", path)

    def test_abort_run(self, run_name: str, reason: str = "Test abort") -> TestResult:
        """Test POST /run/{name}/abort endpoint."""
        path = f"/run/{run_name}/abort"
        logger.info(f"Testing POST {path}")
        return self._make_request("POST", path, json={"reason": reason})

    # ==================== App Endpoints ====================

    def test_get_app_status(
        self,
        app_name: str,
        domain: str | None = None,
        project: str | None = None,
    ) -> TestResult:
        """Test GET /app/{name} endpoint."""
        path = f"/app/{app_name}"
        params = {}
        if domain:
            params["domain"] = domain
        if project:
            params["project"] = project
        logger.info(f"Testing GET {path}")
        return self._make_request("GET", path, params=params or None)

    def test_activate_app(
        self,
        app_name: str,
        domain: str | None = None,
        project: str | None = None,
        wait: bool = False,
    ) -> TestResult:
        """Test POST /app/{name}/activate endpoint."""
        path = f"/app/{app_name}/activate"
        params = {"wait": wait}
        if domain:
            params["domain"] = domain
        if project:
            params["project"] = project
        logger.info(f"Testing POST {path}")
        return self._make_request("POST", path, params=params)

    def test_deactivate_app(
        self,
        app_name: str,
        domain: str | None = None,
        project: str | None = None,
        wait: bool = False,
    ) -> TestResult:
        """Test POST /app/{name}/deactivate endpoint."""
        path = f"/app/{app_name}/deactivate"
        params = {"wait": wait}
        if domain:
            params["domain"] = domain
        if project:
            params["project"] = project
        logger.info(f"Testing POST {path}")
        return self._make_request("POST", path, params=params)

    def test_call_app_endpoint(
        self,
        app_name: str,
        path: str,
        method: str = "GET",
        domain: str | None = None,
        project: str | None = None,
        payload: dict | None = None,
        query_params: dict | None = None,
    ) -> TestResult:
        """Test POST /app/{name}/call endpoint."""
        endpoint_path = f"/app/{app_name}/call"
        params = {"path": path, "method": method}
        if domain:
            params["domain"] = domain
        if project:
            params["project"] = project
        body = {}
        if payload:
            body["payload"] = payload
        if query_params:
            body["query_params"] = query_params
        logger.info(f"Testing POST {endpoint_path}")
        return self._make_request("POST", endpoint_path, json=body or None, params=params)

    # ==================== Trigger Endpoints ====================

    def test_activate_trigger(self, task_name: str, trigger_name: str) -> TestResult:
        """Test POST /trigger/{task_name}/{trigger_name}/activate endpoint."""
        path = f"/trigger/{task_name}/{trigger_name}/activate"
        logger.info(f"Testing POST {path}")
        return self._make_request("POST", path)

    def test_deactivate_trigger(self, task_name: str, trigger_name: str) -> TestResult:
        """Test POST /trigger/{task_name}/{trigger_name}/deactivate endpoint."""
        path = f"/trigger/{task_name}/{trigger_name}/deactivate"
        logger.info(f"Testing POST {path}")
        return self._make_request("POST", path)

    # ==================== Image Build Endpoints ====================

    def test_build_image(
        self,
        base_image: str | None = None,
        pip_packages: list[str] | None = None,
        apt_packages: list[str] | None = None,
        python_version: str | None = None,
        flyte_version: str | None = None,
        name: str | None = None,
        pre: bool = False,
    ) -> TestResult:
        """Test POST /build-image endpoint."""
        path = "/build-image"
        body = {}
        if base_image:
            body["base_image"] = base_image
        if pip_packages:
            body["pip_packages"] = pip_packages
        if apt_packages:
            body["apt_packages"] = apt_packages
        if python_version:
            body["python_version"] = python_version
        if flyte_version:
            body["flyte_version"] = flyte_version
        if name:
            body["name"] = name
        if pre:
            body["pre"] = pre
        logger.info(f"Testing POST {path}")
        return self._make_request("POST", path, json=body or None)

    # ==================== HuggingFace Prefetch Endpoints ====================

    def test_prefetch_hf_model(
        self,
        repo: str,
        raw_data_path: str | None = None,
        artifact_name: str | None = None,
        architecture: str | None = None,
        task: str = "auto",
        modality: list[str] | None = None,
        serial_format: str | None = None,
        model_type: str | None = None,
        short_description: str | None = None,
        hf_token_key: str = "HF_TOKEN",
        cpu: str = "2",
        memory: str = "8Gi",
        disk: str = "50Gi",
        force: int = 0,
    ) -> TestResult:
        """Test POST /prefetch/hf-model endpoint."""
        path = "/prefetch/hf-model"
        body = {
            "repo": repo,
            "task": task,
            "hf_token_key": hf_token_key,
            "cpu": cpu,
            "memory": memory,
            "disk": disk,
            "force": force,
        }
        if raw_data_path:
            body["raw_data_path"] = raw_data_path
        if artifact_name:
            body["artifact_name"] = artifact_name
        if architecture:
            body["architecture"] = architecture
        if modality:
            body["modality"] = modality
        if serial_format:
            body["serial_format"] = serial_format
        if model_type:
            body["model_type"] = model_type
        if short_description:
            body["short_description"] = short_description
        logger.info(f"Testing POST {path}")
        return self._make_request("POST", path, json=body)

    def test_get_prefetch_hf_model_status(self, run_name: str) -> TestResult:
        """Test GET /prefetch/hf-model/{run_name} endpoint."""
        path = f"/prefetch/hf-model/{run_name}"
        logger.info(f"Testing GET {path}")
        return self._make_request("GET", path)

    def test_get_prefetch_hf_model_io(self, run_name: str) -> TestResult:
        """Test GET /prefetch/hf-model/{run_name}/io endpoint."""
        path = f"/prefetch/hf-model/{run_name}/io"
        logger.info(f"Testing GET {path}")
        return self._make_request("GET", path)

    def test_abort_prefetch_hf_model(
        self,
        run_name: str,
        reason: str = "Test abort",
    ) -> TestResult:
        """Test POST /prefetch/hf-model/{run_name}/abort endpoint."""
        path = f"/prefetch/hf-model/{run_name}/abort"
        logger.info(f"Testing POST {path}")
        return self._make_request("POST", path, json={"reason": reason})

    # ==================== Summary ====================

    def print_summary(self):
        """Print a summary of all test results."""
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)

        passed = sum(1 for r in self.results if r.success)
        failed = len(self.results) - passed

        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"{status} | {result.method:4} {result.endpoint}")
            if not result.success:
                if result.error:
                    print(f"       Error: {result.error}")
                elif result.status_code:
                    print(f"       Status: {result.status_code}, Response: {result.response[:200] if isinstance(result.response, str) else result.response}")

        print("=" * 80)
        print(f"Total: {len(self.results)} | Passed: {passed} | Failed: {failed}")
        print("=" * 80)


# ==================== Environment Setup ====================

# Create the webhook environment
webhook_env = FlyteWebhookAppEnvironment(
    name="webhook-env-exhaustive-test",
    title="Flyte Webhook Exhaustive Test",
    description="A webhook service for exhaustively testing all Flyte operations",
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=True,
    scaling=flyte.app.Scaling(replicas=1),
)

# Create a task environment for testing task-related endpoints
task_env = flyte.TaskEnvironment(
    name="webhook-test-task-env",
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@task_env.task(
    triggers=flyte.Trigger(
        name="example-trigger",
        automation=flyte.FixedRate(3),
        inputs={"x": 1, "y": "hello"},
    )
)
async def example_task(x: int, y: str) -> str:
    """A simple task to test the webhook."""
    return f"Result: {x} - {y}"


@task_env.task
async def long_running_task(duration: int) -> str:
    """A long-running task that can be aborted."""
    import asyncio

    await asyncio.sleep(duration)
    return f"Completed after {duration} seconds"


# Create a second app environment for testing app-to-app calls
app = FastAPI()

@app.get("/ping")
async def ping():
    """Simple ping endpoint for testing app-to-app calls."""
    return {"message": "pong"}

@app.post("/echo")
async def echo(data: dict):
    """Echo endpoint for testing POST calls."""
    return {"echo": data}

helper_app_env = FastAPIAppEnvironment(
    name="webhook-test-helper-app",
    app=app,
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn"),
)


if __name__ == "__main__":
    flyte.init_from_config(log_level=logging.DEBUG)

    # Deploy the task environment first
    print("Deploying task environment...")
    flyte.deploy(task_env)

    # Deploy the helper app for app-to-app testing
    print("Deploying helper app environment...")
    served_helper = flyte.serve(helper_app_env)
    served_helper.activate(wait=True)
    print(f"Helper app served at: {served_helper.endpoint}")

    # Serve the webhook environment
    print("Serving webhook environment...")
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

    served_app.activate(wait=True)

    # Create the tester
    tester = WebhookEndpointTester(endpoint, headers)

    try:
        # ==================== Test Health & User Endpoints ====================
        print("\n--- Testing Health & User Endpoints ---")
        tester.test_health()
        tester.test_me()

        # ==================== Test Task Endpoints ====================
        print("\n--- Testing Task Endpoints ---")

        # Get task metadata
        task_result = tester.test_get_task_metadata(
            domain="development",
            project="flytesnacks",
            name="webhook-test-task-env.example_task",
        )

        # Run a task
        run_result = tester.test_run_task(
            domain="development",
            project="flytesnacks",
            name="webhook-test-task-env.example_task",
            inputs={"x": 42, "y": "hello"},
        )

        # ==================== Test Run Endpoints ====================
        print("\n--- Testing Run Endpoints ---")

        run_name = None
        if run_result.success and isinstance(run_result.response, dict):
            run_name = run_result.response.get("name")

            # Get run status
            tester.test_get_run(run_name)

            # Get run I/O
            tester.test_get_run_io(run_name)

        # Start a long-running task to test abort
        abort_run_result = tester.test_run_task(
            domain="development",
            project="flytesnacks",
            name="webhook-test-task-env.long_running_task",
            inputs={"duration": 300},  # 5 minutes
        )

        if abort_run_result.success and isinstance(abort_run_result.response, dict):
            abort_run_name = abort_run_result.response.get("name")
            if abort_run_name:
                # Give it a moment to start
                time.sleep(2)
                # Test abort
                tester.test_abort_run(abort_run_name, reason="Testing abort endpoint")

        # ==================== Test App Endpoints ====================
        print("\n--- Testing App Endpoints ---")

        # Get helper app status
        tester.test_get_app_status(app_name="webhook-test-helper-app")

        # Test deactivate app (then reactivate)
        tester.test_deactivate_app(app_name="webhook-test-helper-app", wait=True)

        # Test activate app
        tester.test_activate_app(app_name="webhook-test-helper-app", wait=True)

        # Test call app endpoint - GET
        tester.test_call_app_endpoint(
            app_name="webhook-test-helper-app",
            path="/ping",
            method="GET",
        )

        # Test call app endpoint - POST
        tester.test_call_app_endpoint(
            app_name="webhook-test-helper-app",
            path="/echo",
            method="POST",
            payload={"test": "data"},
        )

        # ==================== Test Trigger Endpoints ====================
        print("\n--- Testing Trigger Endpoints ---")

        # Note: These will likely fail if no triggers exist, but we test the endpoints anyway
        # to verify the API is working correctly
        tester.test_activate_trigger(
            task_name="webhook-test-task-env.example_task",
            trigger_name="test-trigger",
        )

        tester.test_deactivate_trigger(
            task_name="webhook-test-task-env.example_task",
            trigger_name="test-trigger",
        )

        # ==================== Test Image Build Endpoints ====================
        print("\n--- Testing Image Build Endpoints ---")

        tester.test_build_image(
            pip_packages=["requests", "pandas"],
            python_version="3.12",
            name="test-webhook-image",
        )

        # ==================== Test HuggingFace Prefetch Endpoints ====================
        print("\n--- Testing HuggingFace Prefetch Endpoints ---")

        # Note: This requires HF_TOKEN secret to be configured
        # Using a small model for testing
        prefetch_result = tester.test_prefetch_hf_model(
            repo="hf-internal-testing/tiny-random-gpt2",
            task="auto",
            cpu="1",
            memory="2Gi",
            disk="10Gi",
        )

        if prefetch_result.success and isinstance(prefetch_result.response, dict):
            prefetch_run_name = prefetch_result.response.get("name")
            if prefetch_run_name:
                # Give it a moment to start
                time.sleep(2)

                # Get prefetch status
                tester.test_get_prefetch_hf_model_status(prefetch_run_name)

                # Get prefetch I/O
                tester.test_get_prefetch_hf_model_io(prefetch_run_name)

                # Abort the prefetch (since we're just testing)
                tester.test_abort_prefetch_hf_model(
                    prefetch_run_name,
                    reason="Testing abort endpoint",
                )

        # ==================== Test Error Cases ====================
        print("\n--- Testing Error Cases ---")

        # Test 404 - non-existent run
        tester.test_get_run("non-existent-run-name-12345")

        # Test 404 - non-existent task
        tester.test_get_task_metadata(
            domain="development",
            project="flytesnacks",
            name="non-existent-task",
        )

        # Test 404 - non-existent app
        tester.test_get_app_status(app_name="non-existent-app-12345")

        # Test 400 - self-reference (should fail)
        tester.test_get_app_status(app_name="webhook-env-exhaustive-test")
        tester.test_activate_app(app_name="webhook-env-exhaustive-test")
        tester.test_deactivate_app(app_name="webhook-env-exhaustive-test")
        tester.test_call_app_endpoint(
            app_name="webhook-env-exhaustive-test",
            path="/health",
            method="GET",
        )

        # ==================== Print Summary ====================
        tester.print_summary()

    finally:
        tester.close()

        # Cleanup: deactivate apps
        print("\n--- Cleanup ---")
        try:
            served_helper.deactivate()
            print("Helper app deactivated")
        except Exception as e:
            print(f"Failed to deactivate helper app: {e}")

        try:
            served_app.deactivate()
            print("Webhook app deactivated")
        except Exception as e:
            print(f"Failed to deactivate webhook app: {e}")
