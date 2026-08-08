"""App-serving check — deploy a FastAPI app, hit it over its public endpoint,
then deactivate.

Exercises a Union-platform feature (app serving); on a backend without app
support, skip this scenario (``--skip-app``). Its own module so the basic-task
pods never import fastapi — only the app-tester pod loads this.
"""

from __future__ import annotations

import os
import typing

import flyte  # type: ignore
import flyte.app.extras  # type: ignore

_suffix = os.environ.get("FLYTE_FUNCTIONAL_SUFFIX") or os.environ.get("ENV_SUFFIX") or "smoke"


def _make_fastapi_app():
    import fastapi  # type: ignore

    app = fastapi.FastAPI()

    @app.get("/")
    async def root() -> str:
        return "functional-app-ok"

    @app.get("/health")
    async def health() -> dict:
        return {"status": "healthy"}

    return app


_app_env = flyte.app.extras.FastAPIAppEnvironment(
    name=f"functional-app-{_suffix}",
    app=_make_fastapi_app(),
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn", "httpx"),
    resources=flyte.Resources(cpu="250m", memory="256Mi"),
    env_vars={"FLYTE_FUNCTIONAL_SUFFIX": _suffix},
    requires_auth=False,
)

_app_task_env = flyte.TaskEnvironment(
    name=f"functional-app-tester-{_suffix}",
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn", "httpx"),
    resources=flyte.Resources(cpu="250m", memory="256Mi"),
    depends_on=[_app_env],
    cache="disable",
    env_vars={"FLYTE_FUNCTIONAL_SUFFIX": _suffix},
)


class AppDeployResult(typing.NamedTuple):
    endpoint: str


# Upper bound on the teardown deactivate. deactivate(wait=True) blocks until the
# app reaches the deactivated state; on a cold-starting app that wait can outlast
# the run budget. Bounding it keeps a genuine assertion failure from being masked
# as a generic timeout (a hung deactivate in a finally swallows the real error).
_TEARDOWN_TIMEOUT = 60  # seconds


async def _teardown_app(deployed, log) -> None:  # type: ignore[no-untyped-def]
    """Best-effort, bounded deactivate. Never raises, never hangs."""
    import asyncio

    try:
        await asyncio.wait_for(deployed.deactivate.aio(wait=True), timeout=_TEARDOWN_TIMEOUT)
    except asyncio.TimeoutError:
        log.warning(
            f"app teardown: deactivate did not confirm stopped within "
            f"{_TEARDOWN_TIMEOUT}s; leaving best-effort (stop already requested)"
        )
    except Exception as exc:
        log.warning(f"app teardown: deactivate failed (best-effort): {exc}")


@_app_task_env.task
async def app_deploy_test() -> AppDeployResult:
    import asyncio
    import logging

    import httpx  # type: ignore

    log = logging.getLogger("functional.app")
    await flyte.init_in_cluster.aio()
    # serve()/deactivate() are @syncify wrappers — call the .aio variants from
    # this async task (the sync wrapper inside a running loop can deadlock).
    deployed = await flyte.serve.aio(_app_env)
    endpoint = deployed.endpoint
    log.info(f"app: endpoint={endpoint}")

    # Keep the app UP through every assertion; deactivate only afterwards as a
    # separate bounded teardown. On failure still run teardown, then re-raise the
    # ORIGINAL error untouched.
    try:
        # The serving layer may still be pulling the image / cold-starting the
        # revision when serve() returns, so poll "/" until it answers 200 instead
        # of firing a single un-timed request that hangs on a not-yet-ready
        # endpoint. Bounded so a genuinely broken deploy fails fast with detail.
        deadline = 300  # seconds
        interval = 5
        async with httpx.AsyncClient(timeout=10.0) as client:
            last_err = "no attempt made"
            for _ in range(deadline // interval):
                try:
                    resp = await client.get(f"{endpoint}/")
                    if resp.status_code == 200 and "functional-app-ok" in resp.text:
                        break
                    last_err = f"/ returned {resp.status_code}: {resp.text[:80]}"
                except Exception as exc:
                    last_err = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(interval)
            else:
                raise RuntimeError(f"app / endpoint not ready within {deadline}s: {last_err}")
            log.info("app: / is ready, checking /health")
            resp = await client.get(f"{endpoint}/health")
            assert resp.status_code == 200, f"/health returned {resp.status_code}"
            assert resp.json().get("status") == "healthy"
    except Exception:
        await _teardown_app(deployed, log)
        raise

    await _teardown_app(deployed, log)
    return AppDeployResult(endpoint=endpoint)
