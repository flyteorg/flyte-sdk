from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import TYPE_CHECKING, Optional

import cloudpickle

from flyte._initialize import get_init_config
from flyte._logging import LogFormat, logger
from flyte._tools import ipython_check
from flyte.models import SerializationContext
from flyte.syncify import syncify

if TYPE_CHECKING:
    import flyte.io
    from flyte.app import AppEnvironment
    from flyte.remote import App

    from ._code_bundle import CopyFiles


class _Serve:
    """
    Context manager for serving apps with custom configuration.

    Similar to _Runner for tasks, but specifically for AppEnvironment serving.
    """

    def __init__(
        self,
        version: Optional[str] = None,
        copy_style: CopyFiles = "loaded_modules",
        dry_run: bool = False,
        project: str | None = None,
        domain: str | None = None,
        env_vars: dict[str, str] | None = None,
        input_values: dict[str, dict[str, str | flyte.io.File | flyte.io.Dir]] | None = None,
        cluster_pool: str | None = None,
        log_level: int | None = None,
        log_format: LogFormat = "console",
        interactive_mode: bool | None = None,
    ):
        """
        Initialize serve context.

        Args:
            version: Optional version override for the app deployment
            copy_style: Code bundle copy style (default: "loaded_modules")
            dry_run: If True, don't actually deploy (default: False)
            project: Optional project override
            domain: Optional domain override
            env_vars: Optional environment variables to inject into the app
            input_values: Optional input values to inject into the app
            cluster_pool: Optional cluster pool override
            log_level: Optional log level to set for the app (e.g., logging.INFO)
            log_format: Optional log format ("console" or "json", default: "console")
            interactive_mode: If True, raises NotImplementedError (apps don't support interactive/notebook mode)
        """
        self._version = version
        self._copy_style = copy_style
        self._dry_run = dry_run
        self._project = project
        self._domain = domain
        self._env_vars = env_vars or {}
        self._input_values = input_values or {}
        self._cluster_pool = cluster_pool
        self._log_level = log_level
        self._log_format = log_format
        self._interactive_mode = interactive_mode if interactive_mode is not None else ipython_check()

        if self._interactive_mode:
            raise NotImplementedError(
                "Apps do not support running from notebooks or interactive mode yet. "
                "Please run apps from a Python script file."
            )

    @syncify
    async def serve(self, app_env: "AppEnvironment") -> "App":
        """
        Serve an app with the configured context.

        Args:
            app_env: The app environment to serve

        Returns:
            Deployed and activated App instance

        Raises:
            NotImplementedError: If interactive mode is detected
        """
        from copy import deepcopy

        from flyte.app import _deploy

        from ._code_bundle import build_code_bundle
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

        # Plan deployment
        deployments = plan_deploy(app_env)
        assert deployments
        app_deployment = deployments[0]

        # Build images
        image_cache = await build_images.aio(app_env)
        assert image_cache

        # Build code bundle (tgz style)
        code_bundle = await build_code_bundle(
            from_dir=cfg.root_dir,
            dryrun=self._dry_run,
            copy_style=self._copy_style,
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

        # Inject input overrides from the serve
        input_overrides = None
        if app_env_input_values := self._input_values.get(app_env.name):
            input_overrides = []
            for _input in app_env.inputs:
                value = app_env_input_values.get(_input.name, _input.value)
                input_overrides.append(replace(_input, value=value))

        # Deploy app
        deployed_app = await _deploy._deploy_app(app_env, sc, input_overrides=input_overrides)
        assert deployed_app

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


def with_servecontext(
    version: Optional[str] = None,
    copy_style: CopyFiles = "loaded_modules",
    dry_run: bool = False,
    project: str | None = None,
    domain: str | None = None,
    env_vars: dict[str, str] | None = None,
    input_values: dict[str, dict[str, str | flyte.io.File | flyte.io.Dir]] | None = None,
    cluster_pool: str | None = None,
    log_level: int | None = None,
    log_format: LogFormat = "console",
) -> _Serve:
    """
    Create a serve context with custom configuration.

    This function allows you to customize how an app is served, including
    overriding environment variables, cluster pool, logging, and other deployment settings.

    Example:
    ```python
    import logging
    import flyte
    from flyte.app.extras import FastAPIAppEnvironment

    env = FastAPIAppEnvironment(name="my-app", ...)

    # Serve with custom env vars, logging, and cluster pool
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
        version: Optional version override for the app deployment
        copy_style: Code bundle copy style. Options: "loaded_modules", "all", "none" (default: "loaded_modules")
        dry_run: If True, don't actually deploy (default: False)
        project: Optional project override
        domain: Optional domain override
        env_vars: Optional environment variables to inject/override in the app container
        input_values: Optional input values to inject/override in the app container. Must be a dictionary that maps
            app environment names to a dictionary of input names to values.
        cluster_pool: Optional cluster pool to deploy the app to
        log_level: Optional log level (e.g., logging.DEBUG, logging.INFO). If not provided, uses init config or default
        log_format: Optional log format ("console" or "json", default: "console")

    Returns:
        _Serve: Serve context manager with configured settings

    Raises:
        NotImplementedError: If called from a notebook/interactive environment

    Notes:
        - Apps do not support pickle-based bundling (interactive mode)
        - LOG_LEVEL and LOG_FORMAT are automatically set as env vars if not explicitly provided in env_vars
        - The env_vars and cluster_pool overrides mutate the app IDL after creation
        - This is a temporary solution until the API natively supports these fields
    """
    return _Serve(
        version=version,
        copy_style=copy_style,
        dry_run=dry_run,
        project=project,
        domain=domain,
        env_vars=env_vars,
        input_values=input_values,
        cluster_pool=cluster_pool,
        log_level=log_level,
        log_format=log_format,
    )


@syncify
async def serve(app_env: "AppEnvironment") -> "App":
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
        Deployed and activated App instance

    Raises:
        NotImplementedError: If called from a notebook/interactive environment

    See Also:
        with_servecontext: For customizing deployment settings
    """
    # Use default serve context
    return await _Serve().serve.aio(app_env)
