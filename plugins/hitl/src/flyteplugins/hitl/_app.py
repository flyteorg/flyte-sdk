"""
FastAPI app for Human-in-the-Loop (HITL) events.

This module provides the web interface for humans to submit input to paused Flyte workflows.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import aiofiles
import flyte
import flyte.storage as storage
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ._helpers import _convert_value, _get_request_path, _get_response_path

logger = logging.getLogger(__name__)


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
async def input_form(request_id: str, request_path: str | None = None) -> str:
    """Render an HTML form for human input.

    Args:
        request_id: The unique identifier for this HITL request
    """
    # Use provided request_path or fall back to local path construction
    original_request_path = request_path
    if request_path is None:
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
            <p class="original-request-path">Original request path: {original_request_path}</p>
            <p class="request-path">Request path: {request_path}</p>
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
