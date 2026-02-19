"""
Event class for Human-in-the-Loop (HITL) workflows.

This module provides the Event class that encapsulates the HITL functionality:
- Creates and serves a FastAPI app for receiving human input
- Provides endpoints for form-based and JSON-based submission
- Polls object storage for responses using durable sleep (crash-resilient)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, ClassVar, Generic, Literal, Type, TypeVar

import flyte
import flyte.app
import flyte.report
import flyte.storage as storage
from flyte.app.extras import FastAPIAppEnvironment
from flyte.syncify import syncify

from ._app import app
from ._helpers import _get_request_path, _get_response_path, _get_type_name
from flyte._image import DIST_FOLDER, PythonWheels

# Type variable for generic Event
T = TypeVar("T")

# Scope type for events
EventScope = Literal["run"]

logger = logging.getLogger(__name__)


# ============================================================================
# App and Task Environment for HITL events (module-level)
# ============================================================================

event_image = (
    flyte.Image.from_debian_base(python_version=(3, 12))
    .with_apt_packages("git")
    .with_pip_packages("fastapi", "uvicorn", "python-multipart", "aiofiles")
    .with_pip_packages("flyte>=2.0.0")
    .with_pip_packages("git+https://github.com/flyteorg/flyte-sdk.git@4e8bf5cf#subdirectory=plugins/hitl")
)

event_app_env = FastAPIAppEnvironment(
    name="hitl-event-app",
    app=app,
    domain=flyte.app.Domain(subdomain="hitl-event-app"),
    description="Human-in-the-loop event service for Flyte workflows",
    image=event_image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=True,
    env_vars={"LOG_LEVEL": "10"},
)

event_task_env = flyte.TaskEnvironment(
    name="hitl-event-task-env",
    image=event_image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    depends_on=[event_app_env],
)


# ============================================================================
# Event-based HITL API
# ============================================================================


class Event(Generic[T]):
    """
    An event that waits for human input via an embedded FastAPI app.

    This class encapsulates the entire HITL functionality:
    - Creates and serves a FastAPI app for receiving human input
    - Provides endpoints for form-based and JSON-based submission
    - Polls object storage for responses using durable sleep (crash-resilient)

    The app is automatically served when the Event is created via `Event.create()`.
    All infrastructure details (AppEnvironment, deployment) are abstracted away.

    Example:
        # Create an event (serves the app) and wait for input
        event = await Event.create.aio(
            "proceed_event",
            scope="run",
            prompt="What should I add to x?",
            data_type=int,
        )
        result = await event.wait.aio()

        # Or synchronously
        event = Event.create("my_event", scope="run", prompt="Enter value", data_type=str)
        value = event.wait()
    """

    # Class-level app handle (shared across all events)
    _app_handle: ClassVar[flyte.AppHandle | None] = None
    _app_served: ClassVar[bool] = False

    def __init__(
        self,
        name: str,
        scope: EventScope,
        data_type: Type[T],
        prompt: str,
        request_id: str,
        endpoint: str,
        request_path: str,
        response_path: str,
        timeout_seconds: int = 3600,
        poll_interval_seconds: int = 5,
    ):
        self.name = name
        self.scope = scope
        self.data_type = data_type
        self.prompt = prompt
        self.request_id = request_id
        self._endpoint = endpoint
        self._request_path = request_path
        self._response_path = response_path
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._type_name = _get_type_name(data_type)
        self._report_flushed = False

    @property
    def form_url(self) -> str:
        """URL where humans can submit input for this event."""
        from urllib.parse import urlencode

        params = urlencode({"request_path": self._request_path})
        return f"{self._endpoint}/form/{self.request_id}?{params}"

    @property
    def api_url(self) -> str:
        """API endpoint for programmatic submission."""
        return f"{self._endpoint}/submit/json"

    @property
    def endpoint(self) -> str:
        """Base endpoint of the HITL app."""
        return self._endpoint

    @classmethod
    async def _serve_app(cls) -> flyte.AppHandle:
        """Serve the app and return the app handle."""
        from flyteplugins.hitl import __version__

        await flyte.init_in_cluster.aio()
        return await flyte.with_servecontext(
            copy_style="none",
            version=__version__,
            interactive_mode=True,
        ).serve.aio(event_app_env)

    @classmethod
    @syncify
    async def create(
        cls,
        name: str,
        data_type: Type[T],
        scope: EventScope = "run",
        prompt: str = "Please provide a value",
        timeout_seconds: int = 3600,
        poll_interval_seconds: int = 5,
    ) -> "Event[T]":
        """
        Create a new human-in-the-loop event and serve the app.

        This method creates an event that waits for human input via the FastAPI app.
        The app is automatically served if not already running. All infrastructure
        details are abstracted away - you just get an event to wait on.

        Args:
            name: A descriptive name for the event (used in logs and UI)
            scope: The scope of the event. Currently only "run" is supported.
            prompt: The prompt to display to the human
            data_type: The expected type of the input (int, float, str, bool)
            timeout_seconds: Maximum time to wait for human input (default: 1 hour)
            poll_interval_seconds: How often to check for a response (default: 5 seconds)

        Returns:
            An Event object that can be used to wait for the human input

        Example:
            # Async usage
            event = await Event.create.aio(
                "approval_event",
                scope="run",
                prompt="Do you approve this action?",
                data_type=bool,
            )
            approved = await event.wait.aio()

            # Sync usage
            event = Event.create("value_event", scope="run", prompt="Enter a number", data_type=int)
            value = event.wait()
        """
        # Serve the app if not already served
        if not cls._app_served or cls._app_handle is None:
            logger.info("Serving HITL Event app...")
            cls._app_handle = await cls._serve_app()
            cls._app_served = True
            logger.info(f"HITL Event app served at: {cls._app_handle.endpoint}")

        # Get the endpoint from the app handle
        endpoint = cls._app_handle.endpoint

        # Generate request ID and create request metadata
        request_id = str(uuid.uuid4())
        type_name = _get_type_name(data_type)

        # Get the full storage paths (these will be blob storage paths when running in cluster)
        request_path = _get_request_path(request_id)
        response_path = _get_response_path(request_id)

        # Write the request metadata to storage, including the full paths
        # so the app can read/write to blob storage
        data = {
            "request_id": request_id,
            "event_name": name,
            "scope": scope,
            "prompt": prompt,
            "data_type": type_name,
            "status": "pending",
            "app_endpoint": endpoint,
            "request_path": request_path,
            "response_path": response_path,
        }
        logger.info(f"Creating event with data: {data}")
        request_data = json.dumps(data).encode()

        await storage.put_stream(request_data, to_path=request_path)
        logger.info(f"Created event '{name}' at {request_path}")

        # Create the event object
        event: Event[T] = cls(
            name=name,
            scope=scope,
            data_type=data_type,
            prompt=prompt,
            request_id=request_id,
            endpoint=endpoint,
            request_path=request_path,
            response_path=response_path,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        return event

    @syncify
    async def wait(self) -> T:
        """
        Wait for human input and return the result.

        This method polls object storage for a response using durable sleep,
        making it crash-resilient. If the task crashes and restarts, it will
        resume polling from where it left off.

        Returns:
            The value provided by the human, converted to the event's data_type

        Raises:
            TimeoutError: If no response is received within the timeout
        """
        # Generate Flyte report with URLs and instructions
        import html as html_module

        curl_body = json.dumps(
            {
                "request_id": self.request_id,
                "response_path": self._response_path,
                "value": "<your_value>",
                "data_type": self._type_name,
            },
            indent=2,
        )

        report_html = f"""
        <style>
            .hitl-container {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 20px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 12px;
                color: white;
            }}
            .hitl-header {{
                text-align: center;
                margin-bottom: 20px;
            }}
            .hitl-header h1 {{
                margin: 0;
                font-size: 1.8em;
            }}
            .hitl-section {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 15px;
                color: #333;
            }}
            .hitl-section h2 {{
                margin-top: 0;
                color: #667eea;
                font-size: 1.2em;
                border-bottom: 2px solid #667eea;
                padding-bottom: 8px;
            }}
            .hitl-url {{
                background: #f5f5f5;
                padding: 12px;
                border-radius: 6px;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 0.9em;
                word-break: break-all;
                border-left: 4px solid #667eea;
            }}
            .hitl-url a {{
                color: #667eea;
                text-decoration: none;
            }}
            .hitl-url a:hover {{
                text-decoration: underline;
            }}
            .hitl-code {{
                background: #1e1e1e;
                color: #d4d4d4;
                padding: 15px;
                border-radius: 6px;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 0.85em;
                overflow-x: auto;
                white-space: pre-wrap;
                word-break: break-all;
            }}
            .hitl-info {{
                display: grid;
                grid-template-columns: 120px 1fr;
                gap: 8px;
                margin-bottom: 15px;
            }}
            .hitl-info-label {{
                font-weight: bold;
                color: #667eea;
            }}
            .hitl-info-value {{
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 0.9em;
            }}
        </style>
        <div class="hitl-container">
            <div class="hitl-header">
                <h1>Event Input Required</h1>
            </div>

            <div class="hitl-section">
                <h2>Option 1: Web Form</h2>
                <p>
                Submit your input using <a href="{html_module.escape(self.form_url)}" target="_blank">this form</a>:
                </p>
            </div>

            <div class="hitl-section">
                <h2>Option 2: Programmatic API (curl)</h2>
                <p>Use the following curl command to submit input programmatically:</p>
                <div class="hitl-code">curl -X POST "{html_module.escape(self.api_url)}" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {{FLYTE_API_KEY}}" \\
  -d '{html_module.escape(curl_body)}'</div>
                <p style="margin-top: 15px; font-size: 0.9em; color: #666;">
                    <strong>Note:</strong> Replace <code>&lt;your_value&gt;</code> with the actual value you want to
                    submit. The value should match the expected type: <code>{html_module.escape(self._type_name)}</code>

                    <p>Replace <code>{{FLYTE_API_KEY}}</code> with your Flyte API key.</p>
                </p>
            </div>
        </div>
        """

        await show_form.override(
            short_name=self.name,
            links=[
                EventFormLink(
                    endpoint=self.endpoint,
                    request_id=self.request_id,
                    request_path=self._request_path,
                    name=self.name,
                )
            ]
        )(report_html)
        return await wait_for_input_event(
            name=self.name,
            request_id=self.request_id,
            response_path=self._response_path,
            timeout_seconds=self.timeout_seconds,
            poll_interval_seconds=self.poll_interval_seconds,
        )

    def __repr__(self) -> str:
        return (
            f"Event(name={self.name!r}, scope={self.scope!r}, "
            f"data_type={self._type_name}, request_id={self.request_id!r})"
        )


class EventFormLink(flyte.Link):
    """
    A link to the event form.
    """

    endpoint: str
    request_id: str
    request_path: str
    name: str = "Event Form"

    def get_link(
        self,
        run_name: str,
        project: str,
        domain: str,
        context: dict[str, str],
        parent_action_name: str,
        action_name: str,
        pod_name: str,
        **kwargs,
    ) -> str:
        from urllib.parse import urlencode

        params = urlencode({"request_path": self.request_path})
        return f"{self.endpoint}/form/{self.request_id}?{params}"


@event_task_env.task(report=True)
async def show_form(html_report: str):
    """
    Task that serves the event app.
    """
    await flyte.report.replace.aio(html_report)
    await flyte.report.flush.aio()


@flyte.trace
async def wait_for_input_event(
    name: str,
    request_id: str,
    response_path: str,
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> Any:
    """
    Task that waits for input from the event app.
    """
    # Use the stored response path (which includes the full blob storage path)
    elapsed = 0

    while elapsed < timeout_seconds:
        # Check if response exists
        if await storage.exists(response_path):
            async for chunk in storage.get_stream(response_path):
                response = json.loads(chunk.decode())
                if response.get("status") == "completed":
                    value = response["value"]
                    logger.info(f"Event '{name}' received human input: {value}")
                    print(f"\nReceived human input for '{name}': {value}")
                    return value

            logger.info(f"Event '{name}' waiting for human input... ({elapsed}/{timeout_seconds}s elapsed)")
        await asyncio.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds

    raise TimeoutError(
        f"Event '{name}' (request_id={request_id}) timed out after "
        f"{timeout_seconds} seconds. No human input was received."
    )


@syncify
async def new_event(
    name: str,
    data_type: Type[T],
    scope: EventScope = "run",
    prompt: str = "Please provide a value",
    timeout_seconds: int = 3600,
    poll_interval_seconds: int = 5,
) -> Event[T]:
    """
    Create a new human-in-the-loop event.

    This is a convenience function that wraps Event.create().

    Args:
        name: A descriptive name for the event (used in logs and UI)
        data_type: The expected type of the input (int, float, str, bool)
        scope: The scope of the event. Currently only "run" is supported.
        prompt: The prompt to display to the human
        timeout_seconds: Maximum time to wait for human input (default: 1 hour)
        poll_interval_seconds: How often to check for a response (default: 5 seconds)

    Returns:
        An Event object that can be used to wait for the human input

    Example:
        # Async usage
        event = await new_event.aio(
            "approval_event",
            data_type=bool,
            scope="run",
            prompt="Do you approve this action?",
        )
        approved = await event.wait.aio()

        # Sync usage
        event = new_event("value_event", data_type=int, scope="run", prompt="Enter a number")
        value = event.wait()
    """
    return await Event.create.aio(
        name=name,
        data_type=data_type,
        scope=scope,
        prompt=prompt,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
