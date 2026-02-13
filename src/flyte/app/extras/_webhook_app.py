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

The module also provides configuration options for:
- Endpoint group filtering: Enable groups of endpoints using `endpoint_groups` parameter
- Endpoint filtering: Enable only specific endpoints using the `endpoints` parameter
- Task allow-listing: Restrict access to specific tasks using `TaskAllowList`
- App allow-listing: Restrict access to specific apps using `AppAllowList`

Available endpoint groups (WebhookEndpointGroup):
    - "all": All available endpoints
    - "core": Health check and user info endpoints ("health", "me")
    - "task": Task-related endpoints ("run_task", "get_task")
    - "run": Run-related endpoints ("get_run", "get_run_io", "abort_run")
    - "app": App-related endpoints ("get_app", "activate_app", "deactivate_app", "call_app")
    - "trigger": Trigger-related endpoints ("activate_trigger", "deactivate_trigger")
    - "build": Image build endpoints ("build_image")
    - "prefetch": HuggingFace prefetch endpoints ("prefetch_hf_model", "get_prefetch_hf_model",
                  "get_prefetch_hf_model_io", "abort_prefetch_hf_model")

Available endpoint types (WebhookEndpoint):
    "health", "me", "run_task", "get_task", "get_run", "get_run_io", "abort_run",
    "get_app", "activate_app", "deactivate_app", "call_app", "activate_trigger",
    "deactivate_trigger", "build_image", "prefetch_hf_model", "get_prefetch_hf_model",
    "get_prefetch_hf_model_io", "abort_prefetch_hf_model"

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

    With endpoint group filtering:

    ```python
    from flyte.app.extras import FlyteWebhookAppEnvironment

    # Only enable core, task, and run endpoint groups
    webhook_env = FlyteWebhookAppEnvironment(
        name="task-runner-webhook",
        endpoint_groups=["core", "task", "run"],
    )
    ```

    With individual endpoint filtering:

    ```python
    from flyte.app.extras import FlyteWebhookAppEnvironment

    # Only enable specific endpoints
    webhook_env = FlyteWebhookAppEnvironment(
        name="minimal-webhook",
        endpoints=["health", "run_task", "get_run"],
    )
    ```

    With task allow-listing:

    ```python
    from flyte.app.extras import FlyteWebhookAppEnvironment, TaskAllowList

    webhook_env = FlyteWebhookAppEnvironment(
        name="restricted-webhook",
        task_allowlist=TaskAllowList(
            tasks=["production/my-project/my-task", "another-task"]
        ),
    )
    ```
"""

from __future__ import annotations

import inspect
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, get_args

import rich.repr

import flyte
import flyte.app
from flyte.app.extras import FastAPIAppEnvironment
from flyte.models import SerializationContext

if TYPE_CHECKING:
    import fastapi
    import uvicorn

logger = logging.getLogger(__name__)


# ==================== Endpoint Type Definitions ====================

# All available individual endpoints
WebhookEndpoint = Literal[
    "health",
    "me",
    "run_task",
    "get_task",
    "get_run",
    "get_run_io",
    "abort_run",
    "get_app",
    "activate_app",
    "deactivate_app",
    "call_app",
    "activate_trigger",
    "deactivate_trigger",
    "build_image",
    "prefetch_hf_model",
    "get_prefetch_hf_model",
    "get_prefetch_hf_model_io",
    "abort_prefetch_hf_model",
]

# All available endpoints as a tuple for easy reference
ALL_WEBHOOK_ENDPOINTS: tuple[WebhookEndpoint, ...] = get_args(WebhookEndpoint)

# ==================== Endpoint Group Definitions ====================

# Available endpoint groups
WebhookEndpointGroup = Literal[
    "all",
    "core",
    "task",
    "run",
    "app",
    "trigger",
    "build",
    "prefetch",
]

# All available endpoint groups as a tuple for easy reference
ALL_WEBHOOK_ENDPOINT_GROUPS: tuple[WebhookEndpointGroup, ...] = get_args(WebhookEndpointGroup)

# Mapping from endpoint group names to their constituent endpoints
ENDPOINT_GROUP_MAPPING: dict[WebhookEndpointGroup, tuple[WebhookEndpoint, ...]] = {
    "all": ALL_WEBHOOK_ENDPOINTS,
    "core": ("health", "me"),
    "task": ("run_task", "get_task"),
    "run": ("get_run", "get_run_io", "abort_run"),
    "app": ("get_app", "activate_app", "deactivate_app", "call_app"),
    "trigger": ("activate_trigger", "deactivate_trigger"),
    "build": ("build_image",),
    "prefetch": (
        "prefetch_hf_model",
        "get_prefetch_hf_model",
        "get_prefetch_hf_model_io",
        "abort_prefetch_hf_model",
    ),
}


@dataclass
class TaskAllowList:
    """
    Configuration for task allow-listing.

    When configured, only tasks matching the specified criteria will be accessible.
    If a field is None, that field is not filtered.

    Args:
        tasks: List of allowed task identifiers in the format "domain/project/name"
               or just "name" (matches any domain/project).
               If None, all tasks are allowed.

    Example:
        ```python
        # Allow only specific tasks
        task_allowlist = TaskAllowList(
            tasks=["production/my-project/my-task", "staging/my-project/another-task"]
        )

        # Allow tasks by name only (any domain/project)
        task_allowlist = TaskAllowList(tasks=["my-task", "another-task"])
        ```
    """

    tasks: list[str] | None = None

    def is_allowed(self, domain: str, project: str, name: str) -> bool:
        """Check if a task is allowed based on the allowlist."""
        if self.tasks is None:
            return True

        full_path = f"{domain}/{project}/{name}"
        for allowed in self.tasks:
            # Check for exact match (domain/project/name)
            if allowed == full_path:
                return True
            # Check for name-only match
            if "/" not in allowed and allowed == name:
                return True
            # Check for project/name match
            if allowed.count("/") == 1:
                proj_name = f"{project}/{name}"
                if allowed == proj_name:
                    return True
        return False


@dataclass
class AppAllowList:
    """
    Configuration for app allow-listing.

    When configured, only apps matching the specified criteria will be accessible.
    If a field is None, that field is not filtered.

    Args:
        apps: List of allowed app names.
              If None, all apps are allowed.

    Example:
        ```python
        # Allow only specific apps
        app_allowlist = AppAllowList(apps=["my-app", "another-app"])
        ```
    """

    apps: list[str] | None = None

    def is_allowed(self, name: str) -> bool:
        """Check if an app is allowed based on the allowlist."""
        if self.apps is None:
            return True
        return name in self.apps


@dataclass
class TriggerAllowList:
    """
    Configuration for trigger allow-listing.

    When configured, only triggers matching the specified criteria will be accessible.
    If a field is None, that field is not filtered.

    Args:
        triggers: List of allowed trigger identifiers in the format "task_name/trigger_name"
                  or just "trigger_name" (matches any task).
                  If None, all triggers are allowed.

    Example:
        ```python
        # Allow only specific triggers with their task names
        trigger_allowlist = TriggerAllowList(
            triggers=["my-task/my-trigger", "another-task/another-trigger"]
        )

        # Allow triggers by name only (any task)
        trigger_allowlist = TriggerAllowList(triggers=["my-trigger", "another-trigger"])
        ```
    """

    triggers: list[str] | None = None

    def is_allowed(self, task_name: str, trigger_name: str) -> bool:
        """Check if a trigger is allowed based on the allowlist."""
        if self.triggers is None:
            return True

        full_path = f"{task_name}/{trigger_name}"
        for allowed in self.triggers:
            # Check for exact match (task_name/trigger_name)
            if allowed == full_path:
                return True
            # Check for trigger_name-only match
            if "/" not in allowed and allowed == trigger_name:
                return True
        return False


@dataclass
class _EndpointConfig:
    """Configuration for a single endpoint."""

    method: str
    path: str
    handler: callable
    name: str | None = None


def _resolve_endpoints(
    endpoint_groups: list[WebhookEndpointGroup] | tuple[WebhookEndpointGroup, ...] | None,
    endpoints: list[WebhookEndpoint] | tuple[WebhookEndpoint, ...] | None,
) -> set[WebhookEndpoint]:
    """
    Resolve endpoint groups and individual endpoints to a set of enabled endpoints.

    Args:
        endpoint_groups: List of endpoint groups to enable.
        endpoints: List of individual endpoints to enable.

    Returns:
        A set of all enabled endpoints.

    Note:
        - If both are None, all endpoints are enabled.
        - If only endpoint_groups is specified, those groups are enabled.
        - If only endpoints is specified, those individual endpoints are enabled.
        - If both are specified, the union of both is enabled.
    """
    enabled: set[WebhookEndpoint] = set()

    # If neither is specified, enable all endpoints
    if endpoint_groups is None and endpoints is None:
        return set(ALL_WEBHOOK_ENDPOINTS)

    # Add endpoints from groups
    if endpoint_groups is not None:
        for group in endpoint_groups:
            if group in ENDPOINT_GROUP_MAPPING:
                enabled.update(ENDPOINT_GROUP_MAPPING[group])
            else:
                logger.warning(f"Unknown endpoint group: {group}, skipping")

    # Add individual endpoints
    if endpoints is not None:
        enabled.update(endpoints)

    return enabled


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

    # Resolve endpoint groups and individual endpoints
    enabled_endpoints = _resolve_endpoints(webhook_env.endpoint_groups, webhook_env.endpoints)
    task_allowlist: TaskAllowList | None = webhook_env.task_allowlist
    app_allowlist: AppAllowList | None = webhook_env.app_allowlist
    trigger_allowlist: TriggerAllowList | None = webhook_env.trigger_allowlist

    def _check_task_allowed(domain: str, project: str, name: str):
        """Check if a task is allowed and raise HTTPException if not."""
        if task_allowlist is not None and not task_allowlist.is_allowed(domain, project, name):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Task {domain}/{project}/{name} is not in the allowlist",
            )

    def _check_app_allowed(name: str):
        """Check if an app is allowed and raise HTTPException if not."""
        if app_allowlist is not None and not app_allowlist.is_allowed(name):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"App {name} is not in the allowlist",
            )

    def _check_trigger_allowed(task_name: str, trigger_name: str):
        """Check if a trigger is allowed and raise HTTPException if not."""
        if trigger_allowlist is not None and not trigger_allowlist.is_allowed(task_name, trigger_name):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Trigger {task_name}/{trigger_name} is not in the allowlist",
            )

    # ==================== Endpoint Handler Definitions ====================

    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

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

        # Check task allowlist
        _check_task_allowed(domain, project, name)

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

        # Check task allowlist
        _check_task_allowed(domain, project, name)

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

        # Check app allowlist
        _check_app_allowed(name)

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

        # Check app allowlist
        _check_app_allowed(name)

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

        # Check app allowlist
        _check_app_allowed(name)

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

        # Check app allowlist
        _check_app_allowed(name)

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

        # Check trigger allowlist
        _check_trigger_allowed(task_name, trigger_name)

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

        # Check trigger allowlist
        _check_trigger_allowed(task_name, trigger_name)

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

    # ==================== Endpoint Registry ====================

    # Map endpoint names to their configurations
    endpoint_registry: dict[WebhookEndpoint, _EndpointConfig] = {
        "health": _EndpointConfig(
            method="GET",
            path="/health",
            handler=health_check,
        ),
        "me": _EndpointConfig(
            method="GET",
            path="/me",
            handler=get_current_user,
        ),
        "run_task": _EndpointConfig(
            method="POST",
            path="/run-task/{domain}/{project}/{name}",
            handler=run_task,
        ),
        "get_task": _EndpointConfig(
            method="GET",
            path="/task/{domain}/{project}/{name}",
            handler=get_task_metadata,
        ),
        "get_run": _EndpointConfig(
            method="GET",
            path="/run/{name}",
            handler=get_run,
        ),
        "get_run_io": _EndpointConfig(
            method="GET",
            path="/run/{name}/io",
            handler=get_run_io,
        ),
        "abort_run": _EndpointConfig(
            method="POST",
            path="/run/{name}/abort",
            handler=abort_run,
        ),
        "get_app": _EndpointConfig(
            method="GET",
            path="/app/{name}",
            handler=get_app_status,
        ),
        "activate_app": _EndpointConfig(
            method="POST",
            path="/app/{name}/activate",
            handler=activate_app,
        ),
        "deactivate_app": _EndpointConfig(
            method="POST",
            path="/app/{name}/deactivate",
            handler=deactivate_app,
        ),
        "call_app": _EndpointConfig(
            method="POST",
            path="/app/{name}/call",
            handler=call_app_endpoint,
        ),
        "activate_trigger": _EndpointConfig(
            method="POST",
            path="/trigger/{task_name}/{trigger_name}/activate",
            handler=activate_trigger,
        ),
        "deactivate_trigger": _EndpointConfig(
            method="POST",
            path="/trigger/{task_name}/{trigger_name}/deactivate",
            handler=deactivate_trigger,
        ),
        "build_image": _EndpointConfig(
            method="POST",
            path="/build-image",
            handler=build_image,
        ),
        "prefetch_hf_model": _EndpointConfig(
            method="POST",
            path="/prefetch/hf-model",
            handler=prefetch_hf_model,
        ),
        "get_prefetch_hf_model": _EndpointConfig(
            method="GET",
            path="/prefetch/hf-model/{run_name}",
            handler=get_prefetch_hf_model_status,
        ),
        "get_prefetch_hf_model_io": _EndpointConfig(
            method="GET",
            path="/prefetch/hf-model/{run_name}/io",
            handler=get_prefetch_hf_model_io,
        ),
        "abort_prefetch_hf_model": _EndpointConfig(
            method="POST",
            path="/prefetch/hf-model/{run_name}/abort",
            handler=abort_prefetch_hf_model,
        ),
    }

    # ==================== Create FastAPI App ====================

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

    # ==================== Register Enabled Endpoints ====================

    for endpoint_name in enabled_endpoints:
        if endpoint_name not in endpoint_registry:
            logger.warning(f"Unknown endpoint: {endpoint_name}, skipping")
            continue

        config = endpoint_registry[endpoint_name]

        # Use add_api_route to programmatically register endpoints
        app.add_api_route(
            path=config.path,
            endpoint=config.handler,
            methods=[config.method],
            name=config.name or config.handler.__name__,
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
        endpoint_groups: List of endpoint groups to enable. If None (and endpoints is None),
            all endpoints are enabled. Available groups (see WebhookEndpointGroup type):
            - "all": All available endpoints
            - "core": Health check and user info ("health", "me")
            - "task": Task operations ("run_task", "get_task")
            - "run": Run operations ("get_run", "get_run_io", "abort_run")
            - "app": App operations ("get_app", "activate_app", "deactivate_app", "call_app")
            - "trigger": Trigger operations ("activate_trigger", "deactivate_trigger")
            - "build": Image build operations ("build_image")
            - "prefetch": HuggingFace prefetch operations ("prefetch_hf_model",
                         "get_prefetch_hf_model", "get_prefetch_hf_model_io", "abort_prefetch_hf_model")
        endpoints: List of individual endpoints to enable. Can be used alone or combined
            with endpoint_groups. Available endpoints (see WebhookEndpoint type):
            - "health": Health check endpoint
            - "me": Get current user info
            - "run_task": Run a task
            - "get_task": Get task metadata
            - "get_run": Get run status
            - "get_run_io": Get run inputs/outputs
            - "abort_run": Abort a run
            - "get_app": Get app status
            - "activate_app": Activate an app
            - "deactivate_app": Deactivate an app
            - "call_app": Call another app's endpoint
            - "activate_trigger": Activate a trigger
            - "deactivate_trigger": Deactivate a trigger
            - "build_image": Build a container image
            - "prefetch_hf_model": Prefetch a HuggingFace model
            - "get_prefetch_hf_model": Get prefetch run status
            - "get_prefetch_hf_model_io": Get prefetch run I/O
            - "abort_prefetch_hf_model": Abort a prefetch run
        task_allowlist: Configuration for task allow-listing. When set, only tasks
            matching the allowlist can be accessed via task endpoints.
        app_allowlist: Configuration for app allow-listing. When set, only apps
            matching the allowlist can be accessed via app endpoints.
        trigger_allowlist: Configuration for trigger allow-listing. When set, only triggers
            matching the allowlist can be accessed via trigger endpoints.

    Example:
        Basic usage (all endpoints enabled):

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

        With endpoint group filtering:

        ```python
        from flyte.app.extras import FlyteWebhookAppEnvironment

        # Only enable core, task, and run endpoint groups
        webhook_env = FlyteWebhookAppEnvironment(
            name="task-runner-webhook",
            endpoint_groups=["core", "task", "run"],
        )
        ```

        With individual endpoint filtering:

        ```python
        from flyte.app.extras import FlyteWebhookAppEnvironment

        # Only enable specific endpoints
        webhook_env = FlyteWebhookAppEnvironment(
            name="minimal-webhook",
            endpoints=["health", "run_task", "get_run"],
        )
        ```

        Combining endpoint groups and individual endpoints:

        ```python
        from flyte.app.extras import FlyteWebhookAppEnvironment

        # Enable core group plus specific additional endpoints
        webhook_env = FlyteWebhookAppEnvironment(
            name="custom-webhook",
            endpoint_groups=["core"],
            endpoints=["run_task", "get_run"],
        )
        ```

        With task allow-listing:

        ```python
        from flyte.app.extras import FlyteWebhookAppEnvironment, TaskAllowList

        # Only allow specific tasks
        webhook_env = FlyteWebhookAppEnvironment(
            name="restricted-webhook",
            endpoint_groups=["core", "task", "run"],
            task_allowlist=TaskAllowList(
                tasks=["production/my-project/allowed-task", "my-other-task"]
            ),
        )
        ```

        With app allow-listing:

        ```python
        from flyte.app.extras import FlyteWebhookAppEnvironment, AppAllowList

        # Only allow specific apps
        webhook_env = FlyteWebhookAppEnvironment(
            name="app-manager-webhook",
            endpoint_groups=["core", "app"],
            app_allowlist=AppAllowList(apps=["my-app", "another-app"]),
        )
        ```

        With trigger allow-listing:

        ```python
        from flyte.app.extras import FlyteWebhookAppEnvironment, TriggerAllowList

        # Only allow specific triggers
        webhook_env = FlyteWebhookAppEnvironment(
            name="trigger-manager-webhook",
            endpoint_groups=["core", "trigger"],
            trigger_allowlist=TriggerAllowList(
                triggers=["my-task/my-trigger", "another-trigger"]
            ),
        )
        ```
    """

    title: str | None = None
    type: str = "FlyteWebhookApp"
    app: fastapi.FastAPI | None = field(default=None, init=False)
    uvicorn_config: "uvicorn.Config | None" = None
    image: flyte.Image = field(
        default_factory=lambda: flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn")
    )
    endpoint_groups: list[WebhookEndpointGroup] | tuple[WebhookEndpointGroup, ...] | None = None
    endpoints: list[WebhookEndpoint] | tuple[WebhookEndpoint, ...] | None = None
    task_allowlist: TaskAllowList | None = None
    app_allowlist: AppAllowList | None = None
    trigger_allowlist: TriggerAllowList | None = None
    _caller_frame: inspect.FrameInfo | None = None

    def __post_init__(self):
        if self.endpoints is not None and self.endpoint_groups is not None:
            raise ValueError("Cannot specify both endpoints and endpoint_groups")

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
        if self.endpoint_groups is not None:
            yield "endpoint_groups", list(self.endpoint_groups)
        if self.endpoints is not None:
            yield "endpoints", list(self.endpoints)
        if self.task_allowlist is not None:
            yield "task_allowlist", self.task_allowlist
        if self.app_allowlist is not None:
            yield "app_allowlist", self.app_allowlist
        if self.trigger_allowlist is not None:
            yield "trigger_allowlist", self.trigger_allowlist
