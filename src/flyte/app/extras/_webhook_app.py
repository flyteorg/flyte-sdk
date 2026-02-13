"""
FlyteWebhookAppEnvironment - A pre-built FastAPI app environment for common Flyte operations.

This module provides a ready-to-use FastAPI application environment that exposes
endpoints for interacting with the Flyte control plane, including:
- Running tasks
- Getting run information and I/O
- Aborting runs
- Managing apps (activate/deactivate/status)
- Managing triggers (activate/deactivate)
- Building images
- Calling other app endpoints
- Prefetching HuggingFace models

Example:
    Basic usage:

    ```python
    from flyte.app.extras import FlyteWebhookAppEnvironment

    webhook_env = FlyteWebhookAppEnvironment(
        name="my-webhook",
        resources=flyte.Resources(cpu=1, memory="512Mi"),
        # Optional: set custom subdomain
        domain=flyte.app.Domain(subdomain="my-webhook-subdomain"),
    )
    ```
"""

from __future__ import annotations

import inspect
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import rich.repr

import flyte
import flyte.app
from flyte.app.extras import FastAPIAppEnvironment
from flyte.models import SerializationContext

if TYPE_CHECKING:
    import fastapi
    import uvicorn

logger = logging.getLogger(__name__)


def _create_webhook_app(
    webhook_env: "FlyteWebhookAppEnvironment",
) -> "fastapi.FastAPI":
    """
    Create a FastAPI app with all webhook endpoints.

    Args:
        webhook_env: The FlyteWebhookAppEnvironment instance to configure the app for.

    Returns:
        A configured FastAPI application with all webhook endpoints.
    """
    from fastapi import FastAPI, HTTPException, Query
    from starlette import status

    from flyte.app.extras import FastAPIPassthroughAuthMiddleware

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        FastAPI lifespan context manager to initialize Flyte with passthrough auth.
        """
        await flyte.init_passthrough.aio(
            project=flyte.current_project(),
            domain=flyte.current_domain(),
        )
        logger.info("Initialized Flyte passthrough auth for FlyteWebhookAppEnvironment")
        yield

    app = FastAPI(
        title=webhook_env.title or f"Flyte Webhook: {webhook_env.name}",
        description=webhook_env.description or "A webhook service for Flyte operations",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add auth middleware - automatically extracts auth headers and sets Flyte context
    excluded_paths = {"/health", "/docs", "/openapi.json", "/redoc"}
    app.add_middleware(FastAPIPassthroughAuthMiddleware, excluded_paths=excluded_paths)

    # ==================== Health Check ====================

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    # ==================== User Info ====================

    @app.get("/me")
    async def get_current_user():
        """
        Get information about the currently authenticated user.

        Verifies passthrough authentication by fetching user info from the
        Flyte control plane using the caller's credentials.
        """
        import flyte.remote as remote

        try:
            user = await remote.User.get.aio()
            return {
                "subject": user.subject(),
                "name": user.name(),
            }
        except Exception as e:
            logger.error(f"Failed to get user info: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials or unauthorized",
            )

    # ==================== Task Operations ====================

    @app.post("/run-task/{domain}/{project}/{name}")
    async def run_task(
        domain: str,
        project: str,
        name: str,
        inputs: dict,
        version: str | None = None,
    ):
        """
        Trigger a Flyte task run with the caller's credentials.

        Args:
            domain: Flyte domain (e.g., development, staging, production)
            project: Flyte project name
            name: Task name
            inputs: Dictionary of input parameters for the task
            version: Task version (optional, defaults to "latest")

        Returns:
            Dictionary containing the launched run information:
            - url: URL to view the run in the Flyte UI
            - name: Name of the run
        """
        import flyte.errors
        import flyte.remote as remote

        logger.info(f"Running task: {domain}/{project}/{name} version={version}")

        try:
            auto_version = "latest" if version is None else None

            tk = remote.Task.get(
                project=project,
                domain=domain,
                name=name,
                version=version,
                auto_version=auto_version,
            )
            r = await flyte.run.aio(tk, **inputs)

            return {"url": r.url, "name": r.name}

        except flyte.errors.RemoteTaskNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {name} with version {version or 'latest'} in {domain}/{project} not found",
            )
        except flyte.errors.RemoteTaskUsageError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        except Exception as e:
            logger.error(f"Failed to run task: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    @app.get("/task/{domain}/{project}/{name}")
    async def get_task_metadata(
        domain: str,
        project: str,
        name: str,
        version: str | None = None,
    ):
        """
        Get metadata for a task.

        Args:
            domain: Flyte domain
            project: Flyte project name
            name: Task name
            version: Task version (optional, defaults to "latest")

        Returns:
            Task metadata including interface, resources, and configuration.
        """
        import flyte.errors
        import flyte.remote as remote

        try:
            auto_version = "latest" if version is None else None

            lazy_task = remote.Task.get(
                project=project,
                domain=domain,
                name=name,
                version=version,
                auto_version=auto_version,
            )
            task_details = await lazy_task.fetch.aio()

            return {
                "name": task_details.name,
                "version": task_details.version,
                "task_type": task_details.task_type,
                "required_args": task_details.required_args,
                "default_input_args": task_details.default_input_args,
                "cache": {
                    "behavior": task_details.cache.behavior,
                    "version_override": task_details.cache.version_override,
                    "serialize": task_details.cache.serialize,
                },
                "secrets": task_details.secrets,
            }

        except flyte.errors.RemoteTaskNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {name} with version {version or 'latest'} in {domain}/{project} not found",
            )
        except Exception as e:
            logger.error(f"Failed to get task metadata: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    # ==================== Run Operations ====================

    @app.get("/run/{name}")
    async def get_run(name: str):
        """
        Get run metadata (status, phase, etc.).

        Args:
            name: The name of the run

        Returns:
            Run metadata including phase, URL, and completion status.
        """
        import flyte.remote as remote

        try:
            run = await remote.Run.get.aio(name=name)
            return {
                "name": run.name,
                "phase": run.phase,
                "url": run.url,
                "done": run.done(),
            }
        except Exception as e:
            logger.error(f"Failed to get run: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run {name} not found: {e}",
            )

    @app.get("/run/{name}/io")
    async def get_run_io(name: str):
        """
        Get run inputs and outputs.

        Args:
            name: The name of the run

        Returns:
            Dictionary containing run inputs and outputs.
        """
        import flyte.remote as remote

        try:
            run = await remote.Run.get.aio(name=name)
            inputs = await run.inputs.aio()
            outputs = None
            if run.done():
                try:
                    outputs = await run.outputs.aio()
                except Exception:
                    # Outputs may not be available if run failed
                    pass

            # Convert inputs/outputs to serializable format
            inputs_dict = None
            if inputs is not None:
                # ActionInputs is a UserDict, so we can access its data directly
                inputs_dict = dict(inputs) if hasattr(inputs, "keys") else inputs.to_dict()

            outputs_dict = None
            if outputs is not None:
                # ActionOutputs is a tuple with named_outputs property
                if hasattr(outputs, "named_outputs"):
                    outputs_dict = outputs.named_outputs
                else:
                    outputs_dict = outputs.to_dict() if hasattr(outputs, "to_dict") else list(outputs)

            return {
                "name": run.name,
                "inputs": inputs_dict,
                "outputs": outputs_dict,
            }
        except Exception as e:
            logger.error(f"Failed to get run I/O: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run {name} not found or I/O unavailable: {e}",
            )

    @app.post("/run/{name}/abort")
    async def abort_run(
        name: str,
        reason: str = "Aborted via webhook API",
    ):
        """
        Abort a running task.

        Args:
            name: The name of the run to abort
            reason: Reason for aborting the run

        Returns:
            Confirmation of abort request.
        """
        import flyte.remote as remote

        try:
            run = await remote.Run.get.aio(name=name)
            await run.abort.aio(reason=reason)
            return {
                "name": name,
                "status": "abort_requested",
                "reason": reason,
            }
        except Exception as e:
            logger.error(f"Failed to abort run: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to abort run {name}: {e}",
            )

    # ==================== App Operations ====================

    @app.get("/app/{name}")
    async def get_app_status(
        name: str,
        domain: str | None = None,
        project: str | None = None,
    ):
        """
        Get app status.

        Args:
            name: App name
            domain: Domain name (optional, uses current domain if not specified)
            project: Project name (optional, uses current project if not specified)

        Returns:
            App status information.
        """
        import flyte.remote as remote

        # Prevent self-reference
        if name == webhook_env.name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot get status of self",
            )

        try:
            app = await remote.App.get.aio(name=name, project=project, domain=domain)
            return {
                "name": app.name,
                "revision": app.revision,
                "endpoint": app.endpoint,
                "is_active": app.is_active(),
                "is_deactivated": app.is_deactivated(),
                "url": app.url,
            }
        except Exception as e:
            logger.error(f"Failed to get app status: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"App {name} not found: {e}",
            )

    @app.post("/app/{name}/activate")
    async def activate_app(
        name: str,
        domain: str | None = None,
        project: str | None = None,
        wait: bool = False,
    ):
        """
        Activate an app.

        Args:
            name: App name
            domain: Domain name (optional)
            project: Project name (optional)
            wait: Whether to wait for activation to complete

        Returns:
            Updated app status.
        """
        import flyte.remote as remote

        # Prevent self-reference
        if name == webhook_env.name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot activate self",
            )

        try:
            app = await remote.App.get.aio(name=name, project=project, domain=domain)
            updated_app = await app.activate.aio(wait=wait)
            return {
                "name": updated_app.name,
                "is_active": updated_app.is_active(),
                "endpoint": updated_app.endpoint,
                "url": updated_app.url,
            }
        except Exception as e:
            logger.error(f"Failed to activate app: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to activate app {name}: {e}",
            )

    @app.post("/app/{name}/deactivate")
    async def deactivate_app(
        name: str,
        domain: str | None = None,
        project: str | None = None,
        wait: bool = False,
    ):
        """
        Deactivate an app.

        Args:
            name: App name
            domain: Domain name (optional)
            project: Project name (optional)
            wait: Whether to wait for deactivation to complete

        Returns:
            Updated app status.
        """
        import flyte.remote as remote

        # Prevent self-reference
        if name == webhook_env.name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot deactivate self",
            )

        try:
            app = await remote.App.get.aio(name=name, project=project, domain=domain)
            updated_app = await app.deactivate.aio(wait=wait)
            return {
                "name": updated_app.name,
                "is_active": updated_app.is_active(),
                "is_deactivated": updated_app.is_deactivated(),
                "url": updated_app.url,
            }
        except Exception as e:
            logger.error(f"Failed to deactivate app: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to deactivate app {name}: {e}",
            )

    @app.post("/app/{name}/call")
    async def call_app_endpoint(
        name: str,
        path: str = Query(..., description="The endpoint path to call on the target app"),
        method: str = Query("GET", description="HTTP method (GET, POST, PUT, DELETE)"),
        domain: str | None = None,
        project: str | None = None,
        payload: dict | None = None,
        query_params: dict | None = None,
    ):
        """
        Call another app's endpoint.

        Args:
            name: Target app name
            path: Endpoint path to call (e.g., "/predict")
            method: HTTP method
            domain: Domain name (optional)
            project: Project name (optional)
            payload: JSON payload for POST/PUT requests
            query_params: Query parameters to include

        Returns:
            Response from the target app endpoint.
        """
        import json
        import urllib.error
        import urllib.parse
        import urllib.request

        import flyte.remote as remote

        # Prevent self-reference
        if name == webhook_env.name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot call self",
            )

        try:
            app = await remote.App.get.aio(name=name, project=project, domain=domain)
            endpoint = app.endpoint

            # Build URL with query params
            url = endpoint.rstrip("/") + "/" + path.lstrip("/")
            if query_params:
                url += "?" + urllib.parse.urlencode(query_params)

            # Prepare request
            data = None
            if payload and method.upper() in ("POST", "PUT", "PATCH"):
                data = json.dumps(payload).encode("utf-8")

            # Get current auth context to forward
            # Note: This requires the auth_metadata context to be set by the middleware
            headers = {
                "Content-Type": "application/json",
                "User-Agent": f"flyte-webhook/{webhook_env.name}",
            }

            req = urllib.request.Request(
                url,
                data=data,
                headers=headers,
                method=method.upper(),
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                response_data = resp.read().decode("utf-8")
                try:
                    return json.loads(response_data)
                except json.JSONDecodeError:
                    return {"response": response_data}

        except urllib.error.HTTPError as http_err:
            raise HTTPException(
                status_code=http_err.code,
                detail=f"Target app returned error: {http_err.read().decode('utf-8')}",
            )
        except Exception as e:
            logger.error(f"Failed to call app endpoint: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to call app {name} endpoint {path}: {e}",
            )

    # ==================== Trigger Operations ====================

    @app.post("/trigger/{task_name}/{trigger_name}/activate")
    async def activate_trigger(
        task_name: str,
        trigger_name: str,
    ):
        """
        Activate a trigger.

        Args:
            task_name: Name of the task the trigger is associated with
            trigger_name: Name of the trigger

        Returns:
            Confirmation of activation.
        """
        import flyte.remote as remote

        try:
            await remote.Trigger.update.aio(name=trigger_name, task_name=task_name, active=True)
            return {
                "task_name": task_name,
                "trigger_name": trigger_name,
                "active": True,
            }
        except Exception as e:
            logger.error(f"Failed to activate trigger: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to activate trigger {trigger_name}: {e}",
            )

    @app.post("/trigger/{task_name}/{trigger_name}/deactivate")
    async def deactivate_trigger(
        task_name: str,
        trigger_name: str,
    ):
        """
        Deactivate a trigger.

        Args:
            task_name: Name of the task the trigger is associated with
            trigger_name: Name of the trigger

        Returns:
            Confirmation of deactivation.
        """
        import flyte.remote as remote

        try:
            await remote.Trigger.update.aio(name=trigger_name, task_name=task_name, active=False)
            return {
                "task_name": task_name,
                "trigger_name": trigger_name,
                "active": False,
            }
        except Exception as e:
            logger.error(f"Failed to deactivate trigger: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to deactivate trigger {trigger_name}: {e}",
            )

    # ==================== Image Build Operations ====================

    @app.post("/build-image")
    async def build_image(
        base_image: str | None = None,
        pip_packages: list[str] | None = None,
        apt_packages: list[str] | None = None,
        python_version: str | None = None,
        flyte_version: str | None = None,
        platform: list[str] | None = None,
        name: str | None = None,
        pre: bool = False,
    ):
        """
        Build a container image.

        This endpoint allows building images by specifying packages and configuration.
        The image is built asynchronously and returns build information.

        Args:
            base_image: Base image to use (optional, defaults to debian base)
            pip_packages: List of pip packages to install
            apt_packages: List of apt packages to install
            python_version: Python version to use (e.g., "3.12")
            name: Name for the image
            pre: Whether to use pre-release packages

        Returns:
            Image build information.
        """
        try:
            # Start with base image
            if base_image:
                image = flyte.Image(base_image).clone(
                    name=name,
                    python_version=python_version,
                )
            else:
                image = flyte.Image.from_debian_base(
                    python_version=python_version,
                    flyte_version=flyte_version,
                    platform=platform,
                    name=name,
                )

            # Add apt packages
            if apt_packages:
                image = image.with_apt_packages(*apt_packages)

            # Add pip packages
            if pip_packages:
                image = image.with_pip_packages(*pip_packages, pre=pre)

            # Build the image
            image_build = await flyte.build.aio(image, wait=False)

            return {
                "image_build_run_url": image_build.remote_run,
                "image_name": name,
                "base_image": base_image or "flyte-debian-base",
                "pip_packages": pip_packages,
                "apt_packages": apt_packages,
            }

        except Exception as e:
            logger.error(f"Failed to build image: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to build image: {e}",
            )

    # ==================== Prefetch Operations ====================

    @app.post("/prefetch/hf-model")
    async def prefetch_hf_model(
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
    ):
        """
        Prefetch a HuggingFace model to remote storage.

        This endpoint downloads a model from the HuggingFace Hub and stores it
        in remote storage. The operation runs asynchronously and returns run metadata.

        Args:
            repo: The HuggingFace repository ID (e.g., 'meta-llama/Llama-2-7b-hf')
            raw_data_path: Optional path for raw data storage
            artifact_name: Optional name for the stored artifact
            architecture: Model architecture from HuggingFace config.json
            task: Model task (e.g., 'generate', 'classify', 'embed')
            modality: Modalities supported by the model (e.g., ['text', 'image'])
            serial_format: Model serialization format (e.g., 'safetensors', 'onnx')
            model_type: Model type (e.g., 'transformer', 'custom')
            short_description: Short description of the model
            hf_token_key: Name of the secret containing the HuggingFace token
            cpu: CPU request for the prefetch task
            memory: Memory request for the prefetch task
            disk: Disk storage request
            force: Force re-prefetch (increment to force a new prefetch)

        Returns:
            Run metadata including name, URL, and phase.
        """
        import flyte.prefetch

        try:
            resources = flyte.Resources(cpu=cpu, memory=memory, disk=disk)
            run = flyte.prefetch.hf_model(
                repo=repo,
                raw_data_path=raw_data_path,
                artifact_name=artifact_name,
                architecture=architecture,
                task=task,
                modality=tuple(modality) if modality else ("text",),
                serial_format=serial_format,
                model_type=model_type,
                short_description=short_description,
                hf_token_key=hf_token_key,
                resources=resources,
                force=force,
            )

            return {
                "name": run.name,
                "url": run.url,
                "phase": run.phase,
                "done": run.done(),
            }

        except Exception as e:
            logger.error(f"Failed to prefetch HF model: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to prefetch HF model: {e}",
            )

    @app.get("/prefetch/hf-model/{run_name}")
    async def get_prefetch_hf_model_status(run_name: str):
        """
        Get the status of a HuggingFace model prefetch run.

        Args:
            run_name: The name of the prefetch run

        Returns:
            Run status including phase, URL, and completion status.
        """
        import flyte.remote as remote

        try:
            run = await remote.Run.get.aio(name=run_name)
            return {
                "name": run.name,
                "phase": run.phase,
                "url": run.url,
                "done": run.done(),
            }
        except Exception as e:
            logger.error(f"Failed to get prefetch run status: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prefetch run {run_name} not found: {e}",
            )

    @app.get("/prefetch/hf-model/{run_name}/io")
    async def get_prefetch_hf_model_io(run_name: str):
        """
        Get the inputs and outputs of a HuggingFace model prefetch run.

        Args:
            run_name: The name of the prefetch run

        Returns:
            Dictionary containing run inputs and outputs.
        """
        import flyte.remote as remote

        try:
            run = await remote.Run.get.aio(name=run_name)
            inputs = await run.inputs.aio()
            outputs = None
            if run.done():
                try:
                    outputs = await run.outputs.aio()
                except Exception:
                    # Outputs may not be available if run failed
                    pass

            # Convert inputs/outputs to serializable format
            inputs_dict = None
            if inputs is not None:
                inputs_dict = dict(inputs) if hasattr(inputs, "keys") else inputs.to_dict()

            outputs_dict = None
            if outputs is not None:
                if hasattr(outputs, "named_outputs"):
                    outputs_dict = outputs.named_outputs
                else:
                    outputs_dict = outputs.to_dict() if hasattr(outputs, "to_dict") else list(outputs)

            return {
                "name": run.name,
                "phase": run.phase,
                "inputs": inputs_dict,
                "outputs": outputs_dict,
            }
        except Exception as e:
            logger.error(f"Failed to get prefetch run I/O: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prefetch run {run_name} not found or I/O unavailable: {e}",
            )

    @app.post("/prefetch/hf-model/{run_name}/abort")
    async def abort_prefetch_hf_model(
        run_name: str,
        reason: str = "Aborted via webhook API",
    ):
        """
        Abort a HuggingFace model prefetch run.

        Args:
            run_name: The name of the prefetch run to abort
            reason: Reason for aborting the run

        Returns:
            Confirmation of abort request.
        """
        import flyte.remote as remote

        try:
            run = await remote.Run.get.aio(name=run_name)
            await run.abort.aio(reason=reason)
            return {
                "name": run_name,
                "status": "abort_requested",
                "reason": reason,
            }
        except Exception as e:
            logger.error(f"Failed to abort prefetch run: {type(e).__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to abort prefetch run {run_name}: {e}",
            )

    return app


@rich.repr.auto
@dataclass(kw_only=True, repr=True)
class FlyteWebhookAppEnvironment(FastAPIAppEnvironment):
    """
    A pre-built FastAPI app environment for common Flyte webhook operations.

    This environment provides a ready-to-use FastAPI application with endpoints for:
    - Running tasks in a specific domain/project/version
    - Getting run I/O and metadata
    - Aborting runs
    - Getting task metadata
    - Building images
    - Activating/deactivating apps (except itself)
    - Getting app status
    - Calling other app endpoints
    - Activating/deactivating triggers
    - Prefetching HuggingFace models (run, status, I/O, abort)

    All endpoints use FastAPIPassthroughAuthMiddleware for authentication.

    Args:
        name: Name of the webhook app environment
        image: Docker image to use for the environment
        title: Title for the FastAPI app (optional)
        description: Description for the FastAPI app (optional)
        resources: Resources to allocate for the environment
        requires_auth: Whether the app requires authentication (default: True)
        scaling: Scaling configuration for the app environment
        depends_on: Environment dependencies
        secrets: Secrets to inject into the environment

    Example:
        Basic usage:

        ```python
        import flyte
        from flyte.app.extras import FlyteWebhookAppEnvironment

        webhook_env = FlyteWebhookAppEnvironment(
            name="my-webhook",
            image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn"),
            resources=flyte.Resources(cpu=1, memory="512Mi"),
        )

        # Deploy the webhook
        flyte.serve(webhook_env)
        ```
    """

    title: str | None = None
    type: str = "FlyteWebhookApp"
    app: fastapi.FastAPI | None = field(default=None, init=False)
    uvicorn_config: "uvicorn.Config | None" = None
    image: flyte.Image = field(
        default_factory=lambda: flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn")
    )
    _caller_frame: inspect.FrameInfo | None = None

    def __post_init__(self):
        self.app = _create_webhook_app(self)
        super().__post_init__()

        # Capture the frame where this environment was instantiated
        # This helps us find the module where the app variable is defined
        frame = inspect.currentframe()
        if frame and frame.f_back:
            # Go up the call stack to find the user's module
            # Skip the dataclass __init__ frame
            caller_frame = frame.f_back
            if caller_frame and caller_frame.f_back:
                self._caller_frame = inspect.getframeinfo(caller_frame.f_back)

    async def _fastapi_app_server(self):
        try:
            import uvicorn
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "uvicorn is not installed. Please install 'uvicorn' to use FlyteWebhookAppEnvironment."
            )

        if self.uvicorn_config is None:
            self.uvicorn_config = uvicorn.Config(self.app, port=self.port.port)
        elif self.uvicorn_config is not None:
            if self.uvicorn_config.port is None:
                self.uvicorn_config.port = self.port.port

        await uvicorn.Server(self.uvicorn_config).serve()

    def container_command(self, serialization_context: SerializationContext) -> list[str]:
        return []

    def __rich_repr__(self) -> rich.repr.Result:
        yield "name", self.name
        yield "title", self.title
        yield "type", self.type
