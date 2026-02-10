from __future__ import annotations

import asyncio
import atexit
import contextvars
import hashlib
import os
import pathlib
import signal
import subprocess
import threading
import time
from contextlib import AbstractAsyncContextManager, AbstractContextManager, asynccontextmanager, contextmanager
from dataclasses import replace
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
)

import cloudpickle

from flyte._initialize import get_init_config
from flyte._logging import LogFormat, logger
from flyte._tools import ipython_check
from flyte.models import SerializationContext
from flyte.syncify import syncify

if TYPE_CHECKING:
    from types import FrameType

    import flyte.io
    from flyte.app import AppEnvironment
    from flyte.remote import App

    from ._code_bundle import CopyFiles

ServeMode = Literal["local", "remote"]

# Module-level registry for local app endpoints so AppEnvironment.endpoint
# can resolve them when running locally.
_LOCAL_HOST: str = "localhost"
_LOCAL_APP_ENDPOINTS: dict[str, str] = {}
_LOCAL_DEACTIVATE_TIMEOUT: float = 6.0
_LOCAL_IS_ACTIVE_TOTAL_TIMEOUT: float = 60.0
_LOCAL_IS_ACTIVE_RESPONSE_TIMEOUT: float = 2.0
_LOCAL_IS_ACTIVE_INTERVAL: float = 1.0
_LOCAL_HEALTH_CHECK_PATH: str = "/health"

# Environment variable to indicate the serve mode for subprocess-based apps.
# This allows apps started via command/args to detect they're running locally.
_FSERVE_MODE_ENV_VAR: str = "_FSERVE_MODE"


# Serve mode context variable to allow the framework to distinguish between
# 'local' and 'remote' server modes, even across async call contexts.
serve_mode_var: contextvars.ContextVar[ServeMode] = contextvars.ContextVar("serve_mode", default="remote")


# ---------------------------------------------------------------------------
# Protocol for the shared App / _LocalApp interface
# ---------------------------------------------------------------------------


@runtime_checkable
class AppHandle(Protocol):
    """Protocol defining the common interface between local and remote app handles.

    Both ``_LocalApp`` (local serving) and ``App`` (remote serving) satisfy this
    protocol, enabling calling code to work uniformly regardless of the serving mode.
    """

    @property
    def name(self) -> str: ...

    @property
    def endpoint(self) -> str: ...

    def is_active(self) -> bool: ...

    def is_deactivated(self) -> bool: ...

    def activate(self, wait: bool = False) -> AppHandle: ...

    def deactivate(self, wait: bool = False) -> AppHandle: ...

    def ephemeral_ctx(self) -> AbstractAsyncContextManager[None]: ...

    def ephemeral_ctx_sync(self) -> AbstractContextManager[None]: ...


# ---------------------------------------------------------------------------
# Global registry of active _LocalApp instances and signal-based cleanup
# ---------------------------------------------------------------------------
_ACTIVE_LOCAL_APPS: set[_LocalApp] = set()
_SIGNAL_HANDLERS_INSTALLED: bool = False
_ORIGINAL_SIGINT_HANDLER: Callable[[int, FrameType | None], Any] | int | None = None
_ORIGINAL_SIGTERM_HANDLER: Callable[[int, FrameType | None], Any] | int | None = None


def _cleanup_local_apps() -> None:
    """Deactivate every registered _LocalApp.

    Called automatically via signal handlers (SIGINT / SIGTERM) and via
    ``atexit`` so that child processes and daemon threads are torn down
    even if the user Ctrl-C's or kills the parent process.
    """
    # Iterate over a copy because deactivate() removes from the set.
    for app in list(_ACTIVE_LOCAL_APPS):
        try:
            app.deactivate()
        except Exception:
            pass


def _signal_handler(signum: int, frame) -> None:
    """Signal handler that cleans up local apps then re-raises the signal."""
    _cleanup_local_apps()

    # Re-invoke the original handler so normal behaviour is preserved
    # (e.g. KeyboardInterrupt for SIGINT).
    original = _ORIGINAL_SIGINT_HANDLER if signum == signal.SIGINT else _ORIGINAL_SIGTERM_HANDLER
    if callable(original):
        original(signum, frame)
    elif original == signal.SIG_DFL:
        # Re-raise with default disposition
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


def _install_signal_handlers() -> None:
    """Install SIGINT / SIGTERM handlers (idempotent, main-thread only)."""
    global _SIGNAL_HANDLERS_INSTALLED, _ORIGINAL_SIGINT_HANDLER, _ORIGINAL_SIGTERM_HANDLER  # noqa: PLW0603

    if _SIGNAL_HANDLERS_INSTALLED:
        return

    # signal.signal() can only be called from the main thread.
    if threading.current_thread() is not threading.main_thread():
        return

    try:
        _ORIGINAL_SIGINT_HANDLER = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _signal_handler)

        _ORIGINAL_SIGTERM_HANDLER = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _signal_handler)
    except (OSError, ValueError):
        # Some environments (e.g. certain embedded interpreters) do not
        # allow changing signal handlers — fall through silently.
        pass

    atexit.register(_cleanup_local_apps)
    _SIGNAL_HANDLERS_INSTALLED = True


class _LocalApp:
    """
    Represents a locally-served app environment.

    Provides an interface similar to the remote ``App`` object so that callers
    of ``_Serve.serve`` can use the result uniformly.
    """

    def __init__(
        self,
        app_env: "AppEnvironment",
        host: str,
        port: int,
        _serve_obj: "_Serve",
        process: subprocess.Popen | None = None,
        thread: threading.Thread | None = None,
    ):
        self._app_env = app_env
        self._host = host
        self._port = port
        self._process = process
        self._thread = thread
        self._serve_obj = _serve_obj
        self._stop_event = threading.Event()
        self._thread_loop: asyncio.AbstractEventLoop | None = None

        # Register this instance so it can be cleaned up on process exit.
        _ACTIVE_LOCAL_APPS.add(self)
        _install_signal_handlers()

    @property
    def name(self) -> str:
        return self._app_env.name

    @property
    def endpoint(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def url(self) -> str:
        return self.endpoint

    def is_active(self) -> bool:
        """
        Check if the app is currently active or started.
        """
        from urllib.error import URLError
        from urllib.request import urlopen

        health_check_path = self._serve_obj._health_check_path if self._serve_obj else _LOCAL_HEALTH_CHECK_PATH
        health_check_timeout = (
            self._serve_obj._health_check_timeout if self._serve_obj else _LOCAL_IS_ACTIVE_RESPONSE_TIMEOUT
        )
        url = f"{self.endpoint}{health_check_path}"
        try:
            resp = urlopen(url, timeout=health_check_timeout)
            if resp.status < 500:
                return True
        except (URLError, OSError):
            pass

        return False

    def is_deactivated(self) -> bool:
        """
        Check if the app is currently deactivated.
        """
        if self._thread is not None:
            return not self._thread.is_alive()
        if self._process is not None:
            return self._process.poll() is not None
        return True

    def _is_running(self) -> bool:
        """Check whether the underlying server thread or process is still alive."""
        if self._thread is not None and self._thread.is_alive():
            return True
        if self._process is not None and self._process.poll() is None:
            return True
        return False

    @syncify
    async def activate(self, wait: bool = False) -> _LocalApp:
        """Activate the locally-served app.

        :param wait: Wait for the app to reach activated state
        """
        if self.is_active():
            return self

        activate_timeout = self._serve_obj._activate_timeout if self._serve_obj else _LOCAL_IS_ACTIVE_TOTAL_TIMEOUT
        health_check_interval = self._serve_obj._health_check_interval if self._serve_obj else _LOCAL_IS_ACTIVE_INTERVAL

        # Only start a new server if one isn't already running (it may just
        # not be ready to accept connections yet).
        if not self._is_running():
            self._serve_obj._serve_local(self._app_env)

        if wait:
            deadline = time.monotonic() + activate_timeout
            while time.monotonic() < deadline:
                if self.is_active():
                    return self
                await asyncio.sleep(health_check_interval)
            raise TimeoutError(f"App '{self._app_env.name}' failed to become active within {activate_timeout} seconds")
        return self

    @syncify
    async def deactivate(self, wait: bool = False) -> _LocalApp:
        """Deactivate the locally-served app.

        :param wait: Wait for the app to reach deactivated state
        """
        deactivate_timeout = self._serve_obj._deactivate_timeout if self._serve_obj else _LOCAL_DEACTIVATE_TIMEOUT
        if self._process is not None:
            try:
                self._process.terminate()
            except (ProcessLookupError, PermissionError, OSError):
                pass

            if wait:
                try:
                    self._process.wait(timeout=deactivate_timeout)
                except subprocess.TimeoutExpired:
                    try:
                        self._process.kill()
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
        elif self._thread is not None and self._thread.is_alive():
            # Signal the thread to stop
            self._stop_event.set()
            # Try to stop the event loop running in the thread so that
            # blocking ``loop.run_until_complete()`` calls are interrupted.
            if self._thread_loop is not None and self._thread_loop.is_running():
                try:
                    self._thread_loop.call_soon_threadsafe(self._thread_loop.stop)
                except RuntimeError:
                    # Loop already closed or not running
                    pass
            if wait:
                self._thread.join(timeout=deactivate_timeout)
        # Remove from the local endpoint registry
        _LOCAL_APP_ENDPOINTS.pop(self._app_env.name, None)
        # Unregister from the global active-apps set.
        _ACTIVE_LOCAL_APPS.discard(self)
        return self

    @asynccontextmanager
    async def ephemeral_ctx(self) -> AsyncGenerator[None, None]:
        """
        Async context manager that activates the app and deactivates it when the context is exited.
        """
        try:
            await self.activate.aio(wait=True)
            yield
        finally:
            await self.deactivate.aio(wait=True)

    @contextmanager
    def ephemeral_ctx_sync(self) -> Generator[None, None, None]:
        """
        Context manager that activates the app and deactivates it when the context is exited.
        """
        try:
            self.activate(wait=True)
            yield
        finally:
            self.deactivate(wait=True)


class _Serve:
    """
    Context manager for serving apps with custom configuration.

    Similar to _Runner for tasks, but specifically for AppEnvironment serving.
    """

    def __init__(
        self,
        mode: ServeMode | None = None,
        *,
        version: Optional[str] = None,
        copy_style: CopyFiles = "loaded_modules",
        dry_run: bool = False,
        project: str | None = None,
        domain: str | None = None,
        env_vars: dict[str, str] | None = None,
        parameter_values: dict[str, dict[str, str | flyte.io.File | flyte.io.Dir]] | None = None,
        cluster_pool: str | None = None,
        log_level: int | None = None,
        log_format: LogFormat = "console",
        interactive_mode: bool | None = None,
        copy_bundle_to: pathlib.Path | None = None,
        deactivate_timeout: float | None = None,
        activate_timeout: float | None = None,
        health_check_timeout: float | None = None,
        health_check_interval: float | None = None,
        health_check_path: str | None = None,
    ):
        """
        Initialize serve context.

        Args:
            mode: Serve mode - "local" to run on localhost, "remote" to deploy to
                  the Flyte backend.  When ``None`` the mode is inferred: if a Flyte
                  client is configured the mode defaults to "remote", otherwise "local".
            version: Optional version override for the app deployment
            copy_style: Code bundle copy style (default: "loaded_modules")
            dry_run: If True, don't actually deploy (default: False)
            project: Optional project override
            domain: Optional domain override
            env_vars: Optional environment variables to inject into the app
            parameter_values: Optional parameter values to inject into the app
            cluster_pool: Optional cluster pool override
            log_level: Optional log level to set for the app (e.g., logging.INFO)
            log_format: Optional log format ("console" or "json", default: "console")
            interactive_mode: If True, raises NotImplementedError (apps don't support interactive/notebook mode)
            copy_bundle_to: When dry_run is True, the bundle will be copied to this location if specified
            deactivate_timeout: Timeout in seconds for waiting for the app to stop during
                `deactivate(wait=True)`. Defaults to `6` seconds.
            activate_timeout: Total timeout in seconds when polling the health-check endpoint
                during `activate(wait=True)`. Defaults to `60` seconds.
            health_check_timeout: Per-request timeout in seconds for each health-check HTTP
                request. Defaults to `2` seconds.
            health_check_interval: Interval in seconds between consecutive health-check polls.
                Defaults to `1` second.
            health_check_path: URL path used for the local health-check probe (e.g. `"/healthz"`).
                Defaults to `"/health"`.
        """
        from flyte._initialize import _get_init_config

        if mode is None:
            init_config = _get_init_config()
            client = init_config.client if init_config else None
            mode = "remote" if client is not None else "local"

        self._mode: ServeMode = mode
        self._version = version
        self._copy_style = copy_style
        self._dry_run = dry_run
        self._project = project
        self._domain = domain
        self._env_vars = env_vars or {}
        self._parameter_values = parameter_values or {}
        self._cluster_pool = cluster_pool
        self._log_level = log_level
        self._log_format = log_format
        self._interactive_mode = interactive_mode if interactive_mode is not None else ipython_check()
        self._copy_bundle_to = copy_bundle_to

        # Local-serving configuration (fall back to module-level defaults)
        self._deactivate_timeout = deactivate_timeout if deactivate_timeout is not None else _LOCAL_DEACTIVATE_TIMEOUT
        self._activate_timeout = activate_timeout if activate_timeout is not None else _LOCAL_IS_ACTIVE_TOTAL_TIMEOUT
        self._health_check_timeout = (
            health_check_timeout if health_check_timeout is not None else _LOCAL_IS_ACTIVE_RESPONSE_TIMEOUT
        )
        self._health_check_interval = (
            health_check_interval if health_check_interval is not None else _LOCAL_IS_ACTIVE_INTERVAL
        )
        self._health_check_path = health_check_path if health_check_path is not None else _LOCAL_HEALTH_CHECK_PATH

    # ------------------------------------------------------------------
    # Local serving
    # ------------------------------------------------------------------

    def _serve_local(self, app_env: "AppEnvironment") -> _LocalApp:
        """
        Serve an AppEnvironment locally in a background thread or subprocess.

        The method is **non-blocking**: it starts the app in the background and
        returns a ``_LocalApp`` handle immediately after verifying readiness.
        """

        port = app_env.get_port().port

        # Materialise parameters (simple string values only for local mode)
        materialized_parameters: dict[str, str] = {}
        for parameter in app_env.parameters:
            if app_env_param_values := self._parameter_values.get(app_env.name):
                value = app_env_param_values.get(parameter.name, parameter.value)
            else:
                value = parameter.value
            if isinstance(value, str):
                materialized_parameters[parameter.name] = value
            else:
                materialized_parameters[parameter.name] = str(value)

        # Set env_vars from parameters
        for parameter in app_env.parameters:
            if parameter.env_var:
                val = materialized_parameters.get(parameter.name)
                if val is not None:
                    os.environ[parameter.env_var] = val

        # Set user-supplied env_vars
        for k, v in self._env_vars.items():
            os.environ[k] = v

        if app_env._server is not None:
            # Use the @app_env.server decorator function - run in a background thread
            return self._serve_local_with_server_func(app_env, _LOCAL_HOST, port, materialized_parameters)
        elif app_env.command is not None:
            # Use the command / args specification - run as a subprocess
            return self._serve_local_with_command(app_env, _LOCAL_HOST, port)
        elif app_env.args is not None:
            return self._serve_local_with_command(app_env, _LOCAL_HOST, port)
        else:
            raise ValueError(
                f"AppEnvironment '{app_env.name}' has no server function, command, or args defined. "
                "Cannot serve locally."
            )

    def _serve_local_with_server_func(
        self,
        app_env: "AppEnvironment",
        host: str,
        port: int,
        materialized_parameters: dict[str, str],
    ) -> _LocalApp:
        """Start the app via the ``@app_env.server`` decorated function in a daemon thread."""
        from flyte._bin.serve import _bind_parameters

        assert app_env._server is not None

        # Create the _LocalApp handle first so the thread closure can store
        # its event-loop reference on it for graceful shutdown.
        local_app = _LocalApp(app_env=app_env, _serve_obj=self, host=host, port=port)

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            local_app._thread_loop = loop
            serve_mode_var.set("local")

            # Copy the context after setting serve_mode_var
            ctx = contextvars.copy_context()

            # Install a custom task factory that ensures all tasks created in this loop
            # inherit the context with serve_mode_var set to "local". This is necessary
            # because ASGI servers like uvicorn create new tasks for each request, and
            # those tasks would otherwise get a fresh context without our variable.
            def _context_task_factory(loop, coro, context=None):
                # Use our captured context if no explicit context is provided
                return asyncio.Task(coro, loop=loop, context=context or ctx)

            loop.set_task_factory(_context_task_factory)

            try:
                # Run on_startup if defined
                if app_env._on_startup is not None:
                    bound_params = _bind_parameters(app_env._on_startup, materialized_parameters)
                    if asyncio.iscoroutinefunction(app_env._on_startup):
                        loop.run_until_complete(app_env._on_startup(**bound_params))
                    else:
                        app_env._on_startup(**bound_params)

                # Run the server function
                bound_params = _bind_parameters(app_env._server, materialized_parameters)
                if asyncio.iscoroutinefunction(app_env._server):
                    loop.run_until_complete(app_env._server(**bound_params))
                else:
                    app_env._server(**bound_params)
            except RuntimeError as e:
                # When deactivate() stops the event loop via loop.stop(),
                # run_until_complete raises "Event loop stopped before Future
                # completed." — this is expected and not an error.
                if "Event loop stopped before Future completed" not in str(e):
                    logger.exception("Local app server raised an exception")
            except Exception:
                logger.exception("Local app server raised an exception")
            finally:
                # Cancel all pending tasks so lifespan handlers and other
                # coroutines can clean up before the loop is closed.  Without
                # this, tasks are garbage-collected after loop.close() which
                # triggers "Task was destroyed but it is pending!" warnings and
                # RuntimeError("Event loop is closed") in cleanup callbacks.
                try:
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        # Temporarily suppress logging during task cleanup to
                        # avoid noisy CancelledError tracebacks from ASGI
                        # server internals (e.g. uvicorn/starlette lifespan
                        # handlers) that are expected during forced shutdown.
                        import logging as _logging

                        _prev_disable = _logging.root.manager.disable
                        _logging.disable(_logging.CRITICAL)
                        try:
                            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                        finally:
                            _logging.disable(_prev_disable)
                except Exception:
                    pass
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                local_app._thread_loop = None
                loop.close()
                serve_mode_var.set("remote")

        thread = threading.Thread(target=_run, daemon=True, name=f"flyte-local-app-{app_env.name}")
        thread.start()
        local_app._thread = thread

        # Register the endpoint so AppEnvironment.endpoint can resolve it
        _LOCAL_APP_ENDPOINTS[app_env.name] = local_app.endpoint

        return local_app

    def _serve_local_with_command(
        self,
        app_env: "AppEnvironment",
        host: str,
        port: int,
    ) -> _LocalApp:
        """Start the app via its ``command`` or ``args`` as a subprocess."""
        import shlex

        if app_env.command is not None:
            if isinstance(app_env.command, str):
                cmd = shlex.split(app_env.command)
            else:
                cmd = list(app_env.command)
        elif app_env.args is not None:
            if isinstance(app_env.args, str):
                cmd = shlex.split(app_env.args)
            else:
                cmd = list(app_env.args)
        else:
            raise ValueError("No command or args to run")

        # Set up environment for the subprocess, including the serve mode
        # so that AppEnvironment.endpoint can detect local mode
        subprocess_env = os.environ.copy()
        subprocess_env[_FSERVE_MODE_ENV_VAR] = "local"

        logger.info(f"Starting local app '{app_env.name}' with command: {cmd}")
        process = subprocess.Popen(cmd, env=subprocess_env, start_new_session=True)

        local_app = _LocalApp(app_env=app_env, _serve_obj=self, host=host, port=port, process=process)

        # Register the endpoint so AppEnvironment.endpoint can resolve it
        _LOCAL_APP_ENDPOINTS[app_env.name] = local_app.endpoint

        return local_app

    # ------------------------------------------------------------------
    # Remote serving (unchanged logic, extracted for clarity)
    # ------------------------------------------------------------------

    async def _serve_remote(self, app_env: "AppEnvironment") -> "App":
        """Deploy an AppEnvironment to the remote Flyte backend."""
        from copy import deepcopy

        from flyte.app import _deploy
        from flyte.app._app_environment import AppEnvironment

        from ._code_bundle import build_code_bundle, build_pkl_bundle
        from ._deploy import build_images, plan_deploy

        cfg = get_init_config()
        project = self._project or cfg.project
        domain = self._domain or cfg.domain

        # Configure logging env vars (similar to _run.py)
        env = self._env_vars.copy()
        if env.get("LOG_LEVEL") is None:
            if self._log_level:
                env["LOG_LEVEL"] = str(self._log_level)
            else:
                env["LOG_LEVEL"] = str(logger.getEffectiveLevel())
        env["LOG_FORMAT"] = self._log_format

        # Update env_vars with logging configuration
        self._env_vars = env

        # Plan deployment (discovers all dependent environments)
        deployments = plan_deploy(app_env)
        assert deployments
        app_deployment = deployments[0]

        # Build images
        image_cache = await build_images.aio(app_env)
        assert image_cache

        # Build code bundle (tgz style)
        if self._interactive_mode:
            code_bundle = await build_pkl_bundle(
                app_env,
                upload_to_controlplane=not self._dry_run,
                copy_bundle_to=self._copy_bundle_to,
            )
        else:
            code_bundle = await build_code_bundle(
                from_dir=cfg.root_dir,
                dryrun=self._dry_run,
                copy_style=self._copy_style,
                copy_bundle_to=self._copy_bundle_to,
            )

        # Compute version
        if self._version:
            version = self._version
        elif app_deployment.version:
            version = app_deployment.version
        else:
            h = hashlib.md5()
            h.update(cloudpickle.dumps(app_deployment.envs))
            h.update(code_bundle.computed_version.encode("utf-8"))
            h.update(cloudpickle.dumps(image_cache))
            version = h.hexdigest()

        # Create serialization context
        sc = SerializationContext(
            project=project,
            domain=domain,
            org=cfg.org,
            code_bundle=code_bundle,
            version=version,
            image_cache=image_cache,
            root_dir=cfg.root_dir,
        )

        # Deploy all AppEnvironments in the deployment plan (including dependencies)
        deployment_coros = []
        app_envs_to_deploy = []
        for env_name, dep_env in app_deployment.envs.items():
            if isinstance(dep_env, AppEnvironment):
                # Inject parameter overrides from the serve for this specific app
                parameter_overrides = None
                if app_env_parameter_values := self._parameter_values.get(dep_env.name):
                    parameter_overrides = []
                    for parameter in dep_env.parameters:
                        value = app_env_parameter_values.get(parameter.name, parameter.value)
                        parameter_overrides.append(replace(parameter, value=value))

                logger.info(f"Deploying app {env_name}")
                deployment_coros.append(_deploy._deploy_app(dep_env, sc, parameter_overrides=parameter_overrides))
                app_envs_to_deploy.append(dep_env)

        # Deploy all apps concurrently
        deployed_apps = await asyncio.gather(*deployment_coros)

        # Find the deployed app corresponding to the requested app_env
        deployed_app = None
        for dep_env, deployed in zip(app_envs_to_deploy, deployed_apps):
            logger.warning(f"Deployed App {dep_env.name}, you can check the console at {deployed.url}")
            if dep_env.name == app_env.name:
                deployed_app = deployed

        assert deployed_app, f"Failed to find deployed app for {app_env.name}"
        # Mutate app_idl if env_vars or cluster_pool are provided
        # This is a temporary solution until the update/create APIs support these attributes
        if self._env_vars or self._cluster_pool:
            from flyteidl2.core import literals_pb2

            app_idl = deepcopy(deployed_app.pb2)

            # TODO This should be part of the params!
            # Update env_vars
            if self._env_vars:
                if app_idl.spec.container:
                    # Merge with existing env vars
                    if app_idl.spec.container.env:
                        existing_env = {kv.key: kv.value for kv in app_idl.spec.container.env}
                    else:
                        existing_env = {}
                    existing_env.update(self._env_vars)
                    app_idl.spec.container.env.extend(
                        [literals_pb2.KeyValuePair(key=k, value=v) for k, v in existing_env.items()]
                    )
                elif app_idl.spec.pod:
                    # For pod specs, we'd need to update the containers in the pod
                    # This is more complex as it requires modifying the serialized pod_spec
                    raise NotImplementedError(
                        "Env var override for pod-based apps is not yet supported. "
                        "Please use container-based apps or set env_vars in the AppEnvironment definition."
                    )

            # Update cluster_pool
            if self._cluster_pool:
                app_idl.spec.cluster_pool = self._cluster_pool

            # Update the deployed app with mutated IDL
            # Note: This is a workaround. Ideally, the API would support these fields directly
            deployed_app = type(deployed_app)(pb2=app_idl)

        # Watch for activation
        return await deployed_app.watch.aio(wait_for="activated")

    @syncify
    async def serve(self, app_env: "AppEnvironment") -> AppHandle:
        """
        Serve an app with the configured context.

        Args:
            app_env: The app environment to serve

        Returns:
            An :class:`AppHandle` — either a ``_LocalApp`` (local mode) or a
            remote ``App`` (remote mode).  Both satisfy the same protocol so
            callers can use them interchangeably.

        Raises:
            NotImplementedError: If interactive mode is detected (remote only)
        """
        if self._mode == "local":
            return self._serve_local(app_env)
        return await self._serve_remote(app_env)


def with_servecontext(
    mode: ServeMode | None = None,
    *,
    version: Optional[str] = None,
    copy_style: CopyFiles = "loaded_modules",
    dry_run: bool = False,
    project: str | None = None,
    domain: str | None = None,
    env_vars: dict[str, str] | None = None,
    parameter_values: dict[str, dict[str, str | flyte.io.File | flyte.io.Dir]] | None = None,
    cluster_pool: str | None = None,
    log_level: int | None = None,
    log_format: LogFormat = "console",
    interactive_mode: bool | None = None,
    copy_bundle_to: pathlib.Path | None = None,
    # Local-serving parameters
    deactivate_timeout: float | None = None,
    activate_timeout: float | None = None,
    health_check_timeout: float | None = None,
    health_check_interval: float | None = None,
    health_check_path: str | None = None,
) -> _Serve:
    """
    Create a serve context with custom configuration.

    This function allows you to customize how an app is served, including
    overriding environment variables, cluster pool, logging, and other deployment settings.

    Use ``mode="local"`` to serve the app on localhost (non-blocking) so you can
    immediately invoke tasks that call the app endpoint:

    ```python
    import flyte

    local_app = flyte.with_servecontext(mode="local").serve(app_env)
    local_app.is_active()  # wait for the server to start
    # ... call tasks that use app_env.endpoint ...
    local_app.deactivate()
    ```

    Use ``mode="remote"`` (or omit *mode* when a Flyte client is configured) to
    deploy the app to the Flyte backend:

    ```python
    app = flyte.with_servecontext(
        env_vars={"DATABASE_URL": "postgresql://..."},
        log_level=logging.DEBUG,
        log_format="json",
        cluster_pool="gpu-pool",
        project="my-project",
        domain="development",
    ).serve(env)

    print(f"App URL: {app.url}")
    ```

    Args:
        mode: "local" to run on localhost, "remote" to deploy to the Flyte backend.
            When ``None`` the mode is inferred from the current configuration.
        version: Optional version override for the app deployment
        copy_style: Code bundle copy style. Options: "loaded_modules", "all", "none" (default: "loaded_modules")
        dry_run: If True, don't actually deploy (default: False)
        project: Optional project override
        domain: Optional domain override
        env_vars: Optional environment variables to inject/override in the app container
        parameter_values: Optional parameter values to inject/override in the app container. Must be a dictionary that
            maps app environment names to a dictionary of parameter names to values.
        cluster_pool: Optional cluster pool to deploy the app to
        log_level: Optional log level (e.g., logging.DEBUG, logging.INFO). If not provided, uses init config or default
        log_format: Optional log format ("console" or "json", default: "console")
        interactive_mode: Optional, can be forced to True or False.
            If not provided, it will be set based on the current environment. For example Jupyter notebooks are
            considered interactive mode, while scripts are not. This is used to determine how the code bundle is
            created. This is used to determine if the app should be served in interactive mode or not.
        copy_bundle_to: When dry_run is True, the bundle will be copied to this location if specified
        deactivate_timeout: Timeout in seconds for waiting for the app to stop during
            ``deactivate(wait=True)``. Defaults to 6 s.
        activate_timeout: Total timeout in seconds when polling the health-check endpoint
            during ``activate(wait=True)``. Defaults to 60 s.
        health_check_timeout: Per-request timeout in seconds for each health-check HTTP
            request. Defaults to 2 s.
        health_check_interval: Interval in seconds between consecutive health-check polls.
            Defaults to 1 s.
        health_check_path: URL path used for the local health-check probe (e.g. ``"/healthz"``).
            Defaults to ``"/health"``.

    Returns:
        _Serve: Serve context manager with configured settings

    Raises:
        NotImplementedError: If called from a notebook/interactive environment (remote mode only)

    Notes:
        - Apps do not support pickle-based bundling (interactive mode)
        - LOG_LEVEL and LOG_FORMAT are automatically set as env vars if not explicitly provided in env_vars
        - The env_vars and cluster_pool overrides mutate the app IDL after creation
        - This is a temporary solution until the API natively supports these fields
    """
    return _Serve(
        mode=mode,
        version=version,
        copy_style=copy_style,
        dry_run=dry_run,
        project=project,
        domain=domain,
        env_vars=env_vars,
        parameter_values=parameter_values,
        cluster_pool=cluster_pool,
        log_level=log_level,
        log_format=log_format,
        interactive_mode=interactive_mode,
        copy_bundle_to=copy_bundle_to,
        deactivate_timeout=deactivate_timeout,
        activate_timeout=activate_timeout,
        health_check_timeout=health_check_timeout,
        health_check_interval=health_check_interval,
        health_check_path=health_check_path,
    )


@syncify
async def serve(app_env: "AppEnvironment") -> AppHandle:
    """
    Serve a Flyte app using an AppEnvironment.

    This is the simple, direct way to serve an app. For more control over
    deployment settings (env vars, cluster pool, etc.), use with_servecontext().

    Example:
    ```python
    import flyte
    from flyte.app.extras import FastAPIAppEnvironment

    env = FastAPIAppEnvironment(name="my-app", ...)

    # Simple serve
    app = flyte.serve(env)
    print(f"App URL: {app.url}")
    ```

    Args:
        app_env: The app environment to serve

    Returns:
        An :class:`AppHandle` — either a ``_LocalApp`` (local) or ``App`` (remote)

    See Also:
        with_servecontext: For customizing deployment settings
    """
    # Use default serve context
    return await _Serve().serve.aio(app_env)
