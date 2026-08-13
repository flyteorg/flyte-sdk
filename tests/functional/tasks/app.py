"""App-serving check — deploy a FastAPI app, hit its endpoint, then deactivate.

Exercises a Union-platform feature (app serving); on a backend without app
support, skip this scenario (``--skip-app``). Its own module so the basic-task
pods never import fastapi — only the app-tester pod loads this.
"""

from __future__ import annotations

import os
import typing

import flyte  # type: ignore
import flyte.app.extras  # type: ignore

from . import image_cache_bust

_suffix = os.environ.get("FLYTE_FUNCTIONAL_SUFFIX") or os.environ.get("ENV_SUFFIX") or "smoke"

# Optional consumer overrides, propagated into the tester pod ONLY when set (an
# unset flag stays unset in-pod → SDK-default behaviour), so the suite stays
# backend-agnostic. A consumer with a multi-dataplane topology (e.g. a shared CI
# tenant) sets these; a single-backend user leaves them unset:
#   INTERNAL_APP_ENDPOINT_PATTERN     flyte SDK env — makes _app_env.endpoint resolve
#                                     to an internal per-cluster URL instead of the
#                                     shared public wildcard (last-writer-wins).
#   FLYTE_FUNCTIONAL_APP_CLUSTER_POOL serve() cluster_pool override, for per-dataplane
#                                     isolation on a shared tenant.
_APP_POD_OVERRIDES = {
    k: v for k in ("INTERNAL_APP_ENDPOINT_PATTERN", "FLYTE_FUNCTIONAL_APP_CLUSTER_POOL") if (v := os.environ.get(k))
}


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
    image=flyte.Image.from_debian_base()
    .with_pip_packages("fastapi", "uvicorn", "httpx")
    .with_env_vars(image_cache_bust()),
    resources=flyte.Resources(cpu="250m", memory="256Mi"),
    env_vars={"FLYTE_FUNCTIONAL_SUFFIX": _suffix},
    requires_auth=False,
)

_app_task_env = flyte.TaskEnvironment(
    name=f"functional-app-tester-{_suffix}",
    image=flyte.Image.from_debian_base()
    .with_pip_packages("fastapi", "uvicorn", "httpx")
    .with_env_vars(image_cache_bust()),
    resources=flyte.Resources(cpu="250m", memory="256Mi"),
    depends_on=[_app_env],
    cache="disable",
    # The tester pod re-imports this module and calls flyte.serve(_app_env), which
    # BUILDS _app_env's image in-pod and resolves _app_env.endpoint from the pod's
    # OWN env — so everything that resolution reads must be injected here (baked
    # from the runner at registration), not merely set on the runner:
    #   FLYTE_FUNCTIONAL_IMAGE_CACHE_BUST  image_cache_bust() key; empty in-pod → the
    #     build gets the stable tag and can hit a purged-output cache entry, so
    #     serve() deploys an image never pushed to this run's registry.
    #   _APP_POD_OVERRIDES                 the optional endpoint/pool overrides above.
    env_vars={
        "FLYTE_FUNCTIONAL_SUFFIX": _suffix,
        "FLYTE_FUNCTIONAL_IMAGE_CACHE_BUST": os.environ.get("FLYTE_FUNCTIONAL_IMAGE_CACHE_BUST", ""),
        **_APP_POD_OVERRIDES,
    },
)


class AppDeployResult(typing.NamedTuple):
    endpoint: str


# Upper bound on the teardown delete-retry loop. Delete only succeeds once the app
# has quiesced to a stopped state after deactivate, which can take ~100s; bound the
# retries so a stuck teardown can't outlast the run budget or mask a real failure.
_TEARDOWN_DELETE_DEADLINE = 150  # seconds

# Upper bound on serve(). serve() deploys the app then blocks in
# watch(wait_for="activated") with NO timeout of its own — if the app's revision
# never becomes ready (backend can't schedule/pull it) serve() hangs until the CI
# job is killed, and nothing downstream runs. Bound it so the hang becomes a fast,
# explained failure instead.
# Cold-start budget for serve(): a fresh k3d does a real (cache-miss) in-pod image
# build + a knative revision cold start, which together can run ~3min. Give margin
# over that so a slow-but-healthy revision isn't cut off, while still bounding a
# genuinely stuck one.
_SERVE_TIMEOUT = 360  # seconds


async def _dump_app_status(app_name: str, log) -> None:  # type: ignore[no-untyped-def]
    """Best-effort dump of the app's server-side deployment status — for when serve()
    times out waiting on a revision that never becomes ready. Turns a silent hang into
    an actionable log line (RevisionMissing / pending / image error / ...). Never raises."""
    try:
        from flyte.remote import App  # type: ignore

        app = await App.get.aio(name=app_name)
        pb = app.pb2
        log.error(
            f"app status: name={app_name!r} is_active={app.is_active()} "
            f"desired_state={pb.spec.desired_state} cluster_pool={pb.spec.cluster_pool!r} "
            f"revision={app.revision}"
        )
        for c in list(pb.status.conditions)[-8:]:
            log.error(f"  cond: deployment_status={c.deployment_status} rev={c.revision} msg={c.message!r}")
    except Exception as exc:
        log.error(f"app status dump failed (best-effort): {exc}")


async def _teardown_app(deployed, log) -> None:  # type: ignore[no-untyped-def]
    """Best-effort stop-then-delete. Never raises, never hangs.

    The app MUST be deleted (not merely deactivated) or it poisons the next run:
      * A *deactivated* app lingers on the CP; the next run's serve() re-activates
        that stale registration, which races with this teardown's scale-down and
        never reaches ``is_active`` (it sticks in "pending deletion" even after
        knative reports RevisionReady), so the next serve() times out. This is the
        app-lifecycle form of the one-run lag.
      * A still-*active* app re-materializes on the next run's fresh dataplane and
        fails to pull its now-absent image, tripping the Health gate.

    Delete requires a stopped app ("must be in a stopped state"), and confirming
    the stop via ``deactivate(wait=True)`` is slow (its watch can take ~100s). So
    fire the stop without blocking on the watch, then RETRY delete until the app
    has quiesced enough to be deletable. Bounded; best-effort (never raises)."""
    import asyncio

    from flyte.remote import App  # type: ignore

    name = getattr(deployed, "name", None) or _app_env.name
    # Fire the stop; don't block on the (slow) deactivated-watch.
    try:
        await asyncio.wait_for(deployed.deactivate.aio(wait=False), timeout=30)
    except Exception as exc:
        log.warning(f"app teardown: deactivate(request) of {name!r} failed (best-effort): {exc}")
    # Retry delete until the app is stopped enough to accept it.
    interval, deadline = 8, _TEARDOWN_DELETE_DEADLINE
    last = "no attempt"
    for _ in range(deadline // interval):
        try:
            await asyncio.wait_for(App.delete.aio(name=name), timeout=20)
            log.info(f"app teardown: deleted {name!r}")
            return
        except Exception as exc:
            last = str(exc)
        await asyncio.sleep(interval)
    log.warning(f"app teardown: could not delete {name!r} within {deadline}s (best-effort); last: {last[:160]}")


@_app_task_env.task
async def app_deploy_test() -> AppDeployResult:
    import asyncio
    import logging

    import httpx  # type: ignore

    log = logging.getLogger("functional.app")
    await flyte.init_in_cluster.aio()

    # Echo what this pod actually resolved, so an app-serving failure is debuggable
    # from the run log rather than a bare "did not reach terminal state" timeout.
    # These are the inputs that decide the served image tag, where it deploys, and
    # which URL we poll — the exact things that silently diverge between the CI
    # runner and this in-pod re-import.
    pool = os.environ.get("FLYTE_FUNCTIONAL_APP_CLUSTER_POOL")
    log.info(
        "app config resolved in-pod: "
        f"app_name={_app_env.name!r} "
        f"cluster_pool={pool or '<sdk-default>'} "
        f"image_cache_bust={os.environ.get('FLYTE_FUNCTIONAL_IMAGE_CACHE_BUST', '') or '<unset>'!r} "
        f"internal_endpoint_pattern={os.environ.get('INTERNAL_APP_ENDPOINT_PATTERN') or '<unset>'!r}"
    )

    # serve()/deactivate() are @syncify wrappers — call the .aio variants from this
    # async task (the sync wrapper inside a running loop can deadlock). Pin a cluster
    # pool only when the consumer asked for one (multi-dataplane isolation); unset →
    # the SDK's default pool.
    serve_coro = flyte.with_servecontext(cluster_pool=pool).serve.aio(_app_env) if pool else flyte.serve.aio(_app_env)
    # We poll the app env's OWN endpoint (not serve()'s return): _app_env.endpoint is
    # the public URL by default, or an internal per-cluster URL when
    # INTERNAL_APP_ENDPOINT_PATTERN is set (multi-dataplane safe). With the pattern it
    # is a pure string format (no network), so log the poll target UP FRONT — before
    # serve() can block — so the run log always shows which URL will be hit even if
    # serve() never returns.
    if os.environ.get("INTERNAL_APP_ENDPOINT_PATTERN"):
        log.info(f"app poll target (pre-serve, internal pattern): {_app_env.endpoint!r}")
    # Bound serve(): it blocks in watch(wait_for="activated") with no timeout, so a
    # revision that never becomes ready would hang here forever. On timeout, dump the
    # app's server-side status (the reason it's stuck) and fail explicitly.
    try:
        deployed = await asyncio.wait_for(serve_coro, timeout=_SERVE_TIMEOUT)
    except asyncio.TimeoutError:
        log.error(
            f"serve() did not reach 'activated' within {_SERVE_TIMEOUT}s — the app "
            f"deployed but its revision never became ready; dumping app status:"
        )
        await _dump_app_status(_app_env.name, log)
        raise RuntimeError(
            f"app serve() timed out after {_SERVE_TIMEOUT}s waiting for the revision "
            f"to become ready (see app status dump above)"
        )
    endpoint = _app_env.endpoint
    log.info(
        f"app deployed: polling endpoint={endpoint!r} "
        f"(serve-handle public endpoint={getattr(deployed, 'endpoint', None)!r})"
    )

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
