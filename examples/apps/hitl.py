"""
Human-in-the-Loop (HITL) Pattern Example

This example demonstrates how to implement a human-in-the-loop workflow using
Flyte tasks and apps. The pattern allows a workflow to pause and wait for human
input before continuing execution.

Architecture:
1. The Event class encapsulates a FastAPI app that provides endpoints for human input
2. When an Event is created, it automatically serves the app using flyte.serve
3. The workflow orchestrates automated tasks with HITL checkpoints

The HITL functionality uses an event-based API:
- `Event.create(...)` creates an event and starts the FastAPI app
- `event.wait()` or `await event.wait.aio()` blocks until input is received
- Supports different data types (int, float, str, bool)
- Supports different scopes ("run" for run-level events)

Usage:
    python examples/apps/hitl.py

The workflow will:
1. Run task1() which returns an integer
2. Create an Event (which starts the app) and wait for human input
3. Print a URL where the human can submit their input
4. Once input is received, continue to task2() with both values

Example event-based API usage:
    # Create an event (starts the app) and wait for input
    event = await Event.create.aio(
        "my_event",
        scope="run",
        prompt="Enter a number",
        data_type=int,
    )
    value = await event.wait.aio()
"""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, ClassVar, Generic, Literal, Type, TypeVar

import aiofiles
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import flyte
import flyte.app
import flyte.report
import flyte.storage as storage
from flyte.app.extras import FastAPIAppEnvironment
from flyte.syncify import syncify

# Type variable for generic Event
T = TypeVar("T")

# Scope type for events
EventScope = Literal["run"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================


def _get_type_name(data_type: Type) -> str:
    """Get a string name for a type that can be used in the form."""
    if data_type is int:
        return "int"
    elif data_type is float:
        return "float"
    elif data_type is bool:
        return "bool"
    elif data_type is str:
        return "str"
    else:
        # For complex types, default to string (JSON serialized)
        return "str"


def _convert_value(value: str, data_type: str) -> Any:
    """Convert a string value to the specified data type."""
    if data_type == "int":
        return int(value)
    elif data_type == "float":
        return float(value)
    elif data_type == "bool":
        return value.lower() in ("true", "1", "yes")
    else:
        # Default to string
        return value


def _get_hitl_base_path() -> str:
    """Get the base path for HITL requests in object storage."""
    return "hitl-requests"


def _get_request_path(request_id: str) -> str:
    """Get the storage path for a HITL request."""
    from flyte._context import internal_ctx

    ctx = internal_ctx()
    if ctx.has_raw_data:
        base = ctx.raw_data.path
    elif raw_data_path_env_var := os.getenv("RAW_DATA_PATH"):
        base = raw_data_path_env_var
    else:
        # Fallback for local development
        base = "/tmp/flyte/hitl"
    return f"{base}/{_get_hitl_base_path()}/{request_id}/request.json"


def _get_response_path(request_id: str) -> str:
    """Get the storage path for a HITL response."""
    from flyte._context import internal_ctx

    ctx = internal_ctx()
    if ctx.has_raw_data:
        base = ctx.raw_data.path
    elif raw_data_path_env_var := os.getenv("RAW_DATA_PATH"):
        base = raw_data_path_env_var
    else:
        # Fallback for local development
        base = "/tmp/flyte/hitl"
    return f"{base}/{_get_hitl_base_path()}/{request_id}/response.json"


# ============================================================================
# FastAPI App for Human Input
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Flyte on app startup."""
    logger.info("HITL Event App starting up...")
    await flyte.init_in_cluster.aio()
    yield
    logger.info("HITL Event App shutting down...")


app = FastAPI(
    title="Human-in-the-Loop Event Service",
    description="Provides endpoints for humans to submit input to Flyte workflow events",
    version="1.0.0",
    lifespan=lifespan,
)


# Pydantic models for submissions
class HITLSubmissionTyped(BaseModel):
    """Schema for HITL input submission with explicit type."""

    request_id: str
    value: Any
    data_type: str = "str"
    response_path: str = ""  # Full storage path for the response (e.g., s3://bucket/path/response.json)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Landing page with instructions."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HITL Event Service</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
            .endpoint { margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Human-in-the-Loop Event Service</h1>
        <p>This service allows humans to provide input to paused Flyte workflow events.</p>

        <div class="endpoint">
            <h3>Submit Input</h3>
            <p>To submit input for a pending event, visit:</p>
            <code>GET /form/{request_id}</code>
            <p>Or POST directly to:</p>
            <code>POST /submit</code>
        </div>

        <div class="endpoint">
            <h3>Check Event Status</h3>
            <code>GET /status/{request_id}</code>
        </div>
    </body>
    </html>
    """


@app.get("/form/{request_id}", response_class=HTMLResponse)
async def input_form(request_id: str) -> str:
    """Render an HTML form for human input.

    Args:
        request_id: The unique identifier for this HITL request
        request_path: Optional full storage path to the request metadata (e.g., s3://bucket/path/request.json).
                     If not provided, falls back to local path construction.
    """
    # Use provided request_path or fall back to local path construction
    request_path = _get_request_path(request_id)

    prompt = "Please enter a value"
    data_type_name = "str"
    event_name = "Unknown"
    response_path = ""

    print(f"Request path: {request_path}")
    try:
        if await storage.exists(request_path):
            request_data = await storage.get(request_path)
            async with aiofiles.open(request_data, "r") as f:
                request_data = json.loads(await f.read())
                prompt = request_data.get("prompt", prompt)
                data_type_name = request_data.get("data_type", "str")
                event_name = request_data.get("event_name", "Unknown")
                response_path = request_data.get("response_path", "")
    except Exception as e:
        print(f"Could not fetch request metadata: {e}")

    # Determine input type based on data type
    if data_type_name == "int":
        input_type = "number"
        placeholder = "Enter integer value"
        input_attrs = 'step="1"'
    elif data_type_name == "float":
        input_type = "number"
        placeholder = "Enter decimal value"
        input_attrs = 'step="any"'
    elif data_type_name == "bool":
        input_type = "select"
        placeholder = ""
        input_attrs = ""
    else:
        input_type = "text"
        placeholder = "Enter value"
        input_attrs = ""

    if input_type == "select":
        input_element = """
            <select
                name="value"
                required
                style="
                    width: 100%;
                    padding: 12px;
                    font-size: 16px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    box-sizing: border-box;
                    margin-bottom: 15px;"
                onchange="this.form.submit()"
            >
                <option value="">-- Select --</option>
                <option value="true">True</option>
                <option value="false">False</option>
            </select>
        """
    else:
        input_element = f'<input type="{input_type}" name="value" placeholder="{placeholder}" {input_attrs} required>'

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Event Input Required</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 500px;
                margin: 25px auto;
                padding: 20px;
            }}
            .card {{
                background: #fff;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 30px;
            }}
            h1 {{ color: #333; margin-top: 0; }}
            .event-name {{ color: #4CAF50; font-weight: bold; margin-bottom: 10px; }}
            .prompt {{ color: #666; margin-bottom: 20px; }}
            .request-id {{ font-size: 12px; color: #999; margin-bottom: 20px; }}
            .data-type {{ font-size: 12px; color: #666; margin-bottom: 10px; }}
            input[type="number"], input[type="text"] {{
                width: 100%;
                padding: 12px;
                font-size: 16px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
                margin-bottom: 15px;
            }}
            button {{
                width: 100%;
                padding: 12px;
                font-size: 16px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }}
            button:hover {{ background: #45a049; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Event Input Required</h1>
            <p class="event-name">Event: {event_name}</p>
            <p class="prompt">{prompt}</p>
            <p class="data-type">Expected type: <code>{data_type_name}</code></p>
            <p class="request-id">Request ID: {request_id}</p>
            <form action="/submit" method="post">
                <input type="hidden" name="request_id" value="{request_id}">
                <input type="hidden" name="data_type" value="{data_type_name}">
                <input type="hidden" name="response_path" value="{response_path}">
                {input_element}
                <button type="submit">Submit</button>
            </form>
        </div>
    </body>
    </html>
    """


@app.post("/submit")
async def submit_input(
    request_id: str = Form(...),
    value: str = Form(...),
    data_type: str = Form("str"),
    response_path: str = Form(""),
) -> dict:
    """Submit human input for a pending event.

    Args:
        request_id: The unique identifier for this HITL request
        value: The value submitted by the user
        data_type: The expected data type (int, float, bool, str)
        response_path: Optional full storage path to write the response (e.g., s3://bucket/path/response.json).
                      If not provided, falls back to local path construction.
    """
    try:
        converted_value = _convert_value(value, data_type)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Failed to convert value '{value}' to type '{data_type}': {e}")

    logger.info(f"Received event submission: request_id={request_id}, value={converted_value} (type={data_type})")

    # Use provided response_path or fall back to local path construction
    if not response_path:
        response_path = _get_response_path(request_id)

    response_data = json.dumps(
        {
            "value": converted_value,
            "status": "completed",
            "request_id": request_id,
            "data_type": data_type,
        }
    ).encode()

    try:
        await storage.put_stream(response_data, to_path=response_path)
        logger.info(f"Wrote response to {response_path}")
        return {
            "status": "submitted",
            "request_id": request_id,
            "value": converted_value,
            "data_type": data_type,
            "message": "Input received successfully. The workflow will continue.",
        }
    except Exception as e:
        logger.error(f"Failed to write response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save response: {e}")


@app.post("/submit/json")
async def submit_input_json(submission: HITLSubmissionTyped) -> dict:
    """Submit human input via JSON."""
    return await submit_input(
        request_id=submission.request_id,
        value=str(submission.value),
        data_type=submission.data_type,
        response_path=submission.response_path,
    )


@app.get("/status/{request_id}")
async def get_status(
    request_id: str,
    request_path: str | None = None,
    response_path: str | None = None,
) -> dict:
    """Check the status of an event.

    Args:
        request_id: The unique identifier for this HITL request
        request_path: Optional full storage path to the request metadata
        response_path: Optional full storage path to the response metadata
    """
    # Use provided paths or fall back to local path construction
    if not request_path:
        request_path = _get_request_path(request_id)
    if not response_path:
        response_path = _get_response_path(request_id)

    result = {"request_id": request_id}

    if await storage.exists(request_path):
        async for chunk in storage.get_stream(request_path):
            result["request"] = json.loads(chunk.decode())
            # If response_path wasn't provided, try to get it from request metadata
            if not response_path:
                response_path = result["request"].get("response_path", _get_response_path(request_id))
            break
    else:
        result["request"] = None

    if await storage.exists(response_path):
        async for chunk in storage.get_stream(response_path):
            result["response"] = json.loads(chunk.decode())
            result["status"] = "completed"
            break
    else:
        result["response"] = None
        result["status"] = "pending" if result["request"] else "not_found"

    return result


# ============================================================================
# App and Task Environment for HITL events (module-level)
# ============================================================================

event_image = (
    flyte.Image.from_debian_base(python_version=(3, 12))
    .with_pip_packages("fastapi", "uvicorn", "python-multipart")
    .with_pip_packages("flyte==2.0.0b56", pre=True)
)

event_app_env = FastAPIAppEnvironment(
    name="event-app-test0",
    app=app,
    domain=flyte.app.Domain(subdomain="event-app-test0"),
    description="Human-in-the-loop event service for Flyte workflows",
    image=event_image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=True,
    parameters=[
        flyte.app.Parameter(name="raw_data_path", value="placeholder", env_var="RAW_DATA_PATH"),
    ],
    env_vars={"LOG_LEVEL": "10"},
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
        await flyte.init_in_cluster.aio()

        ctx = flyte.ctx()

        # Use flyte.serve to start the app (uses module-level app_env)
        # Set the RAW_DATA_PATH environment variable to the raw data path since
        # the backend doesn't inject the raw data path into the flyte serve binary
        print(f"Serving app with RAW_DATA_PATH: {ctx.raw_data_path.path}")
        app_handle = await flyte.with_servecontext(
            parameter_values={event_app_env.name: {"raw_data_path": ctx.raw_data_path.path}}
        ).serve.aio(event_app_env)
        return app_handle

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
  -H "Authorization: Bearer <flyte_api_key>" \\
  -d '{html_module.escape(curl_body)}'</div>
                <p style="margin-top: 15px; font-size: 0.9em; color: #666;">
                    <strong>Note:</strong> Replace <code>&lt;your_value&gt;</code> with the actual value you want to
                    submit. The value should match the expected type: <code>{html_module.escape(self._type_name)}</code>

                    <p>Replace <code>&lt;flyte_api_key&gt;</code> with your Flyte API key.</p>
                </p>
            </div>
        </div>
        """

        await show_form.override(short_name=self.name).aio(report_html)
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


event_task_env = flyte.TaskEnvironment(
    name="event-task-env",
    image=event_image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    depends_on=[event_app_env],
)


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
    return await Event.create.aio(
        name=name,
        data_type=data_type,
        scope=scope,
        prompt=prompt,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )


# ============================================================================
# Task Environment and Tasks
# ============================================================================

task_env = flyte.TaskEnvironment(
    name="hitl-workflow",
    image=(
        flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("fastapi", "uvicorn", "python-multipart")
    ),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    depends_on=[event_task_env],
)


@task_env.task(report=True)
async def task1() -> int:
    """
    First task in the workflow - returns an automated value.
    """
    logger.info("task1: Computing automated value...")
    result = 42
    logger.info(f"task1: Returning {result}")
    await flyte.report.replace.aio(f"task1: Returning {result}")
    await flyte.report.flush.aio()
    return result


@task_env.task
async def task2(x: int, y: int) -> int:
    """
    Second task that combines automated and human input.
    """
    logger.info(f"task2: Received x={x}, y={y}")
    result = x + y
    logger.info(f"task2: Returning {result}")
    return result


@task_env.task(report=True)
async def main() -> int:
    """
    Main workflow that orchestrates automated and human-in-the-loop tasks.

    Flow:
    1. task1() runs and returns an automated value (x)
    2. Create an Event (serves the app) and wait for human input (y)
    3. task2(x, y) combines both values and returns the result
    """
    print("Starting HITL workflow...")

    # Step 1: Automated task
    x = await task1()
    print(f"task1 completed: x = {x}")

    # Step 2: Human-in-the-loop using the Event-based API
    # Create an event (this serves the app if not already running)
    event = await new_event.aio(
        "integer_input_event",
        data_type=int,
        scope="run",
        prompt="What should I add to x?",
    )
    y = await event.wait.aio()
    print(f"Event completed: y = {y}")

    # Step 3: Combine results
    result = await task2(x, y)
    print(f"task2 completed: result = {result}")

    return result


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HITL Example")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run in local mode (for testing)",
    )
    args = parser.parse_args()

    flyte.init_from_config(log_level=logging.DEBUG)

    print("\nStarting HITL workflow...")
    run = flyte.run(main)
    print(f"Run URL: {run.url}")
    print(f"Run name: {run.name}")

    print("\nWaiting for workflow to complete...")
    print("(Remember to submit human input when prompted!)")
    run.wait()

    outputs = run.outputs()
    print(f"\nWorkflow completed! Result: {outputs}")
