"""Local serving example where one FastAPI app calls another FastAPI app.

This example demonstrates how to serve two FastAPI apps locally and have one
app proxy requests to the other.  Both apps are started via
``flyte.with_servecontext(mode="local")`` and communicate over HTTP.

Usage (SDK):
    python examples/apps/local_app_calling_app.py
"""

import httpx
from fastapi import FastAPI

import flyte
from flyte.app.extras import FastAPIAppEnvironment

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("fastapi", "uvicorn", "httpx")

# ---------------------------------------------------------------------------
# App 1 - a simple "square" service
# ---------------------------------------------------------------------------

app1 = FastAPI(
    title="Square Service",
    description="A FastAPI app that squares a number",
    version="1.0.0",
)

app1_env = FastAPIAppEnvironment(
    name="local-square-service",
    app=app1,
    description="Squares a number",
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    port=8091,
    requires_auth=False,
)


@app1.get("/")
async def square(x: int) -> dict[str, int]:
    """Return x squared."""
    return {"result": x * x}


@app1.get("/health")
async def app1_health() -> dict[str, str]:
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# App 2 - proxies to App 1 and adds its own logic
# ---------------------------------------------------------------------------

app2 = FastAPI(
    title="Square-Plus-One Service",
    description="A FastAPI app that calls the square service and adds one",
    version="1.0.0",
)

app2_env = FastAPIAppEnvironment(
    name="local-square-plus-one",
    app=app2,
    description="Calls the square service and adds one",
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    port=8092,
    requires_auth=False,
    depends_on=[app1_env],
)


@app2.get("/")
async def square_plus_one(x: int) -> dict[str, int]:
    """Call app1 to square x, then add one."""
    async with httpx.AsyncClient() as client:
        response = await client.get(app1_env.endpoint, params={"x": x})
        response.raise_for_status()
        squared = response.json()["result"]
    return {"result": squared + 1}


@app2.get("/health")
async def app2_health() -> dict[str, str]:
    return {"status": "healthy"}


if __name__ == "__main__":
    serve_ctx = flyte.with_servecontext(mode="local")

    # Serve app1 first (app2 depends on it)
    local_app1 = serve_ctx.serve(app1_env)
    local_app1.activate()
    print(f"App 1 (square) is ready at {local_app1.endpoint}")

    # Serve app2
    local_app2 = serve_ctx.serve(app2_env)
    local_app2.activate()
    print(f"App 2 (square+1) is ready at {local_app2.endpoint}")

    # Call app2 which internally calls app1
    response = httpx.get(f"{local_app2.endpoint}", params={"x": 5})
    response.raise_for_status()
    data = response.json()
    print(f"Response: {data}")
    assert data["result"] == 26  # 5^2 + 1 = 26

    # Shut down both apps
    local_app2.deactivate()
    local_app1.deactivate()
    print("Done!")
