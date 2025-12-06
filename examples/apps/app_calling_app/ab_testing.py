import os
import typing
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

import flyte
from flyte.app.extras import FastAPIAppEnvironment


class StatsigClient:
    """Singleton to manage Statsig client lifecycle."""

    _instance: "StatsigClient | None" = None
    _statsig = None

    @classmethod
    def initialize(cls, api_key: str):
        """Initialize Statsig client (call during lifespan startup)."""
        if cls._instance is None:
            cls._instance = cls()

        # Import statsig at runtime (only available in container)
        from statsig_python_core import Statsig

        cls._statsig = Statsig(api_key)
        cls._statsig.initialize().wait()

    @classmethod
    def get_client(cls):
        """Get the initialized Statsig instance."""
        if cls._statsig is None:
            raise RuntimeError("StatsigClient not initialized. Call initialize() first.")
        return cls._statsig

    @classmethod
    def shutdown(cls):
        """Shutdown Statsig client (call during lifespan shutdown)."""
        if cls._statsig is not None:
            cls._statsig.shutdown()
            cls._statsig = None
            cls._instance = None


# Image with statsig-python-core for A/B testing
image = flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn", "httpx", "statsig-python-core")

# App A - First variant
app_a = FastAPI(
    title="App A",
    description="Variant A for A/B testing",
)

# App B - Second variant
app_b = FastAPI(
    title="App B",
    description="Variant B for A/B testing",
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize and shutdown Statsig for A/B testing."""
    # Startup: Initialize Statsig using singleton
    api_key = os.getenv("STATSIG_API_KEY", None)
    if api_key is None:
        raise RuntimeError(f"StatsigClient API Key not set. ENV vars {os.environ}")
    StatsigClient.initialize(api_key)

    yield

    # Shutdown: Cleanup Statsig
    StatsigClient.shutdown()


# Root App - Performs A/B testing and routes to A or B
root_app = FastAPI(
    title="Root App - A/B Testing",
    description="Routes requests to App A or App B based on Statsig A/B test",
    lifespan=lifespan,
)

env_a = FastAPIAppEnvironment(
    name="app-a-variant",
    app=app_a,
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)

env_b = FastAPIAppEnvironment(
    name="app-b-variant",
    app=app_b,
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)

env_root = FastAPIAppEnvironment(
    name="root-ab-testing-app",
    app=root_app,
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    depends_on=[env_a, env_b],
    secrets=flyte.Secret("statsig-api-key", as_env_var="STATSIG_API_KEY"),
)


# App A endpoints
@app_a.get("/process/{message}")
async def process_a(message: str) -> dict[str, str]:
    return {
        "variant": "A",
        "message": f"App A processed: {message}",
        "algorithm": "fast-processing",
    }


# App B endpoints
@app_b.get("/process/{message}")
async def process_b(message: str) -> dict[str, str]:
    return {
        "variant": "B",
        "message": f"App B processed: {message}",
        "algorithm": "enhanced-processing",
    }


# Root app A/B testing endpoint
@root_app.get("/process/{message}")
async def process_with_ab_test(message: str, user_key: str) -> dict[str, typing.Any]:
    """
    Process a message using A/B testing to determine which app to call.

    Args:
        message: The message to process
        user_key: User identifier for A/B test bucketing (e.g., user_id, session_id)

    Returns:
        Response from either App A or App B, plus metadata about which variant was used
    """
    # Import StatsigUser at runtime (only available in container)
    from statsig_python_core import StatsigUser

    # Get statsig client from singleton
    statsig = StatsigClient.get_client()

    # Create Statsig user with the provided key
    user = StatsigUser(user_id=user_key)

    # Check the feature gate "variant_b" to determine which variant
    # If gate is enabled, use App B; otherwise use App A
    use_variant_b = statsig.check_gate(user, "variant_b")

    # Call the appropriate app based on A/B test result
    async with httpx.AsyncClient() as client:
        if use_variant_b:
            endpoint = f"{env_b.endpoint}/process/{message}"
            response = await client.get(endpoint)
            result = response.json()
        else:
            endpoint = f"{env_a.endpoint}/process/{message}"
            response = await client.get(endpoint)
            result = response.json()

    # Add A/B test metadata to response
    return {
        "ab_test_result": {
            "user_key": user_key,
            "selected_variant": "B" if use_variant_b else "A",
            "gate_name": "variant_b",
        },
        "response": result,
    }


@root_app.get("/endpoints")
async def get_endpoints() -> dict[str, str]:
    """Get the endpoints for App A and App B."""
    return {
        "app_a_endpoint": env_a.endpoint,
        "app_b_endpoint": env_b.endpoint,
    }


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(env_root)
    print("Deployed A/B Testing Root App")
    print("\nUsage:")
    print("  curl '<endpoint>/process/hello?user_key=user123'")
    print("  curl '<endpoint>/endpoints'")
    print("\nNote: Set STATSIG_API_KEY secret to use real Statsig A/B testing.")
    print("      Create a feature gate named 'variant_b' in your Statsig dashboard.")
