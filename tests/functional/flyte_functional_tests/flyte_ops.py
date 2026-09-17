"""Shared helpers for the functional (environment-validation) suite.

These are thin wrappers over the flyte v2 SDK — client init, submission retry,
and terminal-state assertion — used by every scenario. Nothing here is specific
to any one deployment; the suite is meant to run against *your* Flyte backend to
show it can build images, run tasks, cache, serve apps, and schedule triggers.

Connection is resolved once (``init_client``): by default from your standard
Flyte config (``flyte.init_from_config()`` → ``~/.flyte/config.yaml`` or the
``FLYTE_CONFIG`` env var). For CI or ad-hoc runs you can point it at an endpoint
explicitly with the ``FLYTE_FUNCTIONAL_*`` env vars (see ``env_config``).
"""

from __future__ import annotations

import asyncio
import logging
import os

# Quiet the HTTP client stack: at DEBUG it floods the log and echoes auth headers.
for _noisy in ("httpx", "httpcore", "urllib3", "hpack", "h2"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

log = logging.getLogger("functional")


def _first_env(*names: str, default: str = "") -> str:
    """First non-empty value among env var ``names`` (later names are fallbacks
    for the helm-charts CI contract, so that workflow can reuse this suite)."""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def _int_env(name: str, default: int) -> int:
    """Integer tunable from ``name`` (env), falling back to ``default`` if unset or
    unparsable — so backends that need a longer/shorter budget can override it."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("ignoring non-integer %s=%r; using default %d", name, raw, default)
        return default


def env_config() -> dict:
    """Resolve the suite's connection + naming knobs from the environment.

    All optional — with nothing set the suite uses your default Flyte config and
    a ``smoke`` naming suffix.

    | Purpose            | Preferred var                | CI fallback        |
    | ------------------ | ---------------------------- | ------------------ |
    | Endpoint           | ``FLYTE_FUNCTIONAL_ENDPOINT``| ``CONTROL_PLANE_URL`` |
    | API key            | ``FLYTE_FUNCTIONAL_API_KEY`` | ``FLYTE_API_KEY``  |
    | Org                | ``FLYTE_FUNCTIONAL_ORG``     | ``ORG_NAME``       |
    | Project            | ``FLYTE_FUNCTIONAL_PROJECT`` | ``CLUSTER_NAME``   |
    | Domain             | ``FLYTE_FUNCTIONAL_DOMAIN``  | (``development``)  |
    | Env-name suffix    | ``FLYTE_FUNCTIONAL_SUFFIX``  | ``ENV_SUFFIX``     |
    | Queue pin (opt.)   | ``FLYTE_FUNCTIONAL_QUEUE``   | ``CLUSTER_NAME``   |
    | Backend flavour    | ``FLYTE_FUNCTIONAL_BACKEND`` | (``oss``)          |

    ``backend`` (``oss``, the default, or ``union``) only selects which submission
    errors count as transient/retryable (see ``_transient_markers``); it does not
    change how the scenarios run. Set it to ``union`` against a managed data plane.

    The api key is read directly by ``init_client`` and intentionally NOT returned
    here: pytest prints fixture values in failure tracebacks (and truncates them, so
    ``::add-mask::`` can miss), so keeping the secret out of the config dict avoids
    leaking it into public CI logs.
    """
    return {
        "endpoint": _first_env("FLYTE_FUNCTIONAL_ENDPOINT", "CONTROL_PLANE_URL"),
        "org": _first_env("FLYTE_FUNCTIONAL_ORG", "ORG_NAME"),
        "project": _first_env("FLYTE_FUNCTIONAL_PROJECT", "CLUSTER_NAME"),
        "domain": _first_env("FLYTE_FUNCTIONAL_DOMAIN", default="development"),
        "suffix": _first_env("FLYTE_FUNCTIONAL_SUFFIX", "ENV_SUFFIX", default="smoke"),
        # Optional: pin runs to a specific queue/cluster. Unset => default routing.
        "queue": _first_env("FLYTE_FUNCTIONAL_QUEUE", "CLUSTER_NAME"),
        "backend": _first_env("FLYTE_FUNCTIONAL_BACKEND", default="oss").lower(),
    }


async def init_client(cfg: dict | None = None) -> None:
    """Initialise the flyte client for the whole suite.

    With ``FLYTE_FUNCTIONAL_ENDPOINT``/``CONTROL_PLANE_URL`` set, connect to that
    endpoint explicitly (``flyte.init``); otherwise fall back to your standard
    config (``flyte.init_from_config``). Images build on the backend's builder
    (``image_builder="remote"``) so no local Docker is required.
    """
    import flyte

    cfg = cfg or env_config()
    endpoint = cfg["endpoint"]
    project = cfg["project"] or None
    domain = cfg["domain"] or "development"
    org = cfg["org"] or None

    if endpoint:
        if not endpoint.startswith(("https://", "http://")):
            endpoint = "https://" + endpoint
        kwargs: dict = {
            "endpoint": endpoint,
            "project": project,
            "domain": domain,
            "image_builder": "remote",
        }
        if org:
            kwargs["org"] = org
        # Read the api key from the env here (never via the config dict) so it can't
        # surface in pytest's fixture-value output.
        api_key = _first_env("FLYTE_FUNCTIONAL_API_KEY", "FLYTE_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        await flyte.init.aio(**kwargs)  # type: ignore[attr-defined]
    else:
        # Standard config resolution (config file / env) — the common user path.
        await flyte.init_from_config.aio(  # type: ignore[attr-defined]
            org=org,
            project=project,
            domain=domain,
            image_builder="remote",
        )


def _phase_name(run) -> str:  # type: ignore[no-untyped-def]
    return str(run.phase).rsplit(".", 1)[-1].lower()


# Bound each wait so a stuck run fails instead of hanging. Configurable — a slower
# backend (cold image builds, cluster spin-up) may need a longer ceiling.
_ASSERT_TIMEOUT = _int_env("FLYTE_FUNCTIONAL_WAIT_TIMEOUT", 600)  # seconds


async def dump_run_logs(run, label: str, max_lines: int = 200) -> None:  # type: ignore[no-untyped-def]
    """Best-effort: stream a run's task logs — where the task's own diagnostics land
    (e.g. the app tester's in-pod resolved-config echo). The driver only sees the run
    phase, so on any failure this is what turns a bare timeout into a debuggable
    trace of what the task actually did. Never raises — diagnostics must not mask the
    result."""
    try:
        print(f"[functional] {label}: --- task logs for run {run.name} ---", flush=True)
        n = 0
        async for line in run.get_logs.aio():  # type: ignore
            print(f"[functional] {label}| {line}", flush=True)
            n += 1
            if n >= max_lines:
                print(f"[functional] {label}: ...(truncated at {max_lines} lines)", flush=True)
                break
    except Exception as exc:
        print(f"[functional] {label}: could not fetch task logs: {exc}", flush=True)


async def assert_succeeded(run, label: str, timeout: float = _ASSERT_TIMEOUT) -> None:  # type: ignore[no-untyped-def]
    """Wait for a run to reach a terminal state and assert it Succeeded.

    On timeout, abort the run on the backend (so it stops holding resources) and
    raise. On a non-Succeeded terminal phase, surface the run's error detail. Both
    failure paths dump the run's task logs first (see dump_run_logs).
    """
    try:
        await asyncio.wait_for(run.wait.aio(wait_for="terminal"), timeout=timeout)  # type: ignore
    except asyncio.TimeoutError:
        # Dump task logs while the run is still up (before the abort) so the in-task
        # diagnostics survive.
        await dump_run_logs(run, label)
        try:
            await asyncio.wait_for(
                run.abort.aio(reason=f"{label}: exceeded {timeout:.0f}s wait"),  # type: ignore
                timeout=30,
            )
            log.info("%s: aborted run %s after %.0fs timeout", label, run.name, timeout)
        except Exception as exc:
            log.warning("%s: abort after timeout failed: %s", label, exc)
        run.sync()
        raise RuntimeError(
            f"{label}: run {run.name} did not reach a terminal state within "
            f"{timeout:.0f}s (last phase={run.phase}) — aborted"
        )
    run.sync()
    p = _phase_name(run)
    if p != "succeeded":
        detail = ""
        try:
            details = await run.details.aio()  # type: ignore
            err = details.action_details.error_info
            if err is not None:
                detail = f": {err.kind}: {err.message}"
        except Exception:
            pass
        if not detail:
            try:
                act = run.pb2.action
                if act.HasField("error_info"):
                    detail = f": {act.error_info.kind}: {act.error_info.message}"
            except Exception:
                pass
        await dump_run_logs(run, label)
        raise RuntimeError(f"{label}: run {run.name} ended in phase={run.phase}{detail}")


# Submission retry budget while the backend is in a transient state
# (_SUBMIT_MAX_ATTEMPTS x _SUBMIT_RETRY_DELAY seconds). Both configurable — a fresh
# or managed backend may need a longer propagation window than the default 20 min.
_SUBMIT_MAX_ATTEMPTS = _int_env("FLYTE_FUNCTIONAL_SUBMIT_ATTEMPTS", 40)
_SUBMIT_RETRY_DELAY = _int_env("FLYTE_FUNCTIONAL_SUBMIT_RETRY_DELAY", 30)  # seconds

# Which submission-time errors count as transient (retry) rather than real failures
# is backend-dependent, so the set is assembled from a common base + a per-backend
# group selected by FLYTE_FUNCTIONAL_BACKEND, plus any extra substrings supplied in
# FLYTE_FUNCTIONAL_TRANSIENT_MARKERS (comma-separated) for backends not special-cased.
#
# Common: network / gRPC blips any backend can emit.
_TRANSIENT_COMMON = (
    "unavailable",  # gRPC UNAVAILABLE
    "deadline exceeded",
    "connection refused",
    "connection reset",
    "temporarily unavailable",
    "too many requests",
    "try again",
)
# Union (managed data plane): the DP<->CP tunnel / cluster-pool routing can flap
# during node churn — the CP briefly reports the pool unhealthy, no cluster
# selectable, or the enabled-clusters cache empty/lagging on a fresh data plane.
_TRANSIENT_UNION = (
    "no clusters found",
    "no cluster",
    "could not select a cluster",
    "unhealthy",
    "failed to get proxy",
    "failed to get data proxy",
)
# Flyte OSS (single flyteadmin + propeller, no DP tunnel). Empty placeholder: the
# OSS-specific intermittent submission errors aren't enumerated yet. Until they are,
# OSS retries only the common blips above (plus anything in
# FLYTE_FUNCTIONAL_TRANSIENT_MARKERS). Add markers here as they're observed.
_TRANSIENT_OSS: tuple[str, ...] = ()  # TODO: populate as OSS transient errors surface


def _transient_markers(backend: str) -> tuple[str, ...]:
    per_backend = _TRANSIENT_OSS if backend == "oss" else _TRANSIENT_UNION
    extra = tuple(
        m.strip().lower() for m in os.environ.get("FLYTE_FUNCTIONAL_TRANSIENT_MARKERS", "").split(",") if m.strip()
    )
    return _TRANSIENT_COMMON + per_backend + extra


def _is_transient_submit_error(msg: str) -> bool:
    m = msg.lower()
    backend = env_config()["backend"]
    if any(marker in m for marker in _transient_markers(backend)):
        return True
    # Union: 'cluster "<name>" not found' — enabled-clusters cache miss on a fresh
    # data plane. The name sits between the two tokens, so match both rather than a
    # single substring; not treated as transient on OSS.
    if backend != "oss":
        return "cluster" in m and "not found" in m
    return False


async def submit_with_retry(task_fn, label: str, **kwargs):  # type: ignore[no-untyped-def]
    """Submit a task, retrying while the backend is in a transient state.

    Retries the transient submission errors for the configured backend — see
    ``_is_transient_submit_error`` / ``_transient_markers`` — for up to
    ``_SUBMIT_MAX_ATTEMPTS`` x ``_SUBMIT_RETRY_DELAY`` s (default 40 x 30s = 20 min,
    all overridable). A deterministic error (bad task, real config problem) is not
    retried. A queue pin is applied only if configured (``FLYTE_FUNCTIONAL_QUEUE``);
    otherwise the backend routes by default.
    """
    import flyte  # type: ignore

    queue = env_config()["queue"] or None
    # When the image cache is busted per run (ephemeral registries — k3d destroys its
    # RustFS/registry each teardown), pin the run version to that same per-run value so
    # every task RE-REGISTERS each run. Otherwise the version defaults to the
    # code-bundle hash, which is stable across runs (the source doesn't change), so the
    # backend reuses a prior run's registration and the pod inherits that run's baked
    # env — including a STALE FLYTE_FUNCTIONAL_IMAGE_CACHE_BUST. An in-pod serve() then
    # resolves the app image to a prior run's tag, pushed to a now-gone registry
    # (ImagePullBackOff). Unset (a persistent store) => default versioning, which
    # caches correctly across runs.
    version = os.environ.get("FLYTE_FUNCTIONAL_IMAGE_CACHE_BUST") or None
    rc_kwargs: dict = {}
    if queue:
        rc_kwargs["queue"] = queue
    if version:
        rc_kwargs["version"] = version
    rc = flyte.with_runcontext(**rc_kwargs)

    run = None
    last_err = ""
    for attempt in range(1, _SUBMIT_MAX_ATTEMPTS + 1):
        try:
            run = await rc.run.aio(task_fn, **kwargs)  # type: ignore
            break
        except Exception as exc:
            last_err = str(exc)
            if _is_transient_submit_error(last_err):
                if attempt < _SUBMIT_MAX_ATTEMPTS:
                    log.info(
                        "%s: attempt %d/%d — %s — retrying in %ds …",
                        label,
                        attempt,
                        _SUBMIT_MAX_ATTEMPTS,
                        last_err[:160],
                        _SUBMIT_RETRY_DELAY,
                    )
                    await asyncio.sleep(_SUBMIT_RETRY_DELAY)
            else:
                raise
    if run is None:
        raise RuntimeError(
            f"{label}: submission failed after {_SUBMIT_MAX_ATTEMPTS} attempts (last error: {last_err[:300]})"
        )
    return run


async def dump_app_state(app_name: str) -> None:
    """Best-effort diagnostic: print an app's spec + status from the control
    plane (no credentials). Never raises — diagnostics must not mask the result."""
    try:
        import flyte.remote  # type: ignore

        app = await flyte.remote.App.get.aio(name=app_name)  # type: ignore
        pb = app.pb2
        log.info("app %r state: spec.cluster_pool=%r", app_name, pb.spec.cluster_pool)
        for line in str(pb.status).splitlines():
            log.info("app %r status| %s", app_name, line)
    except Exception as exc:
        log.info("could not fetch app state for %r: %s", app_name, exc)
