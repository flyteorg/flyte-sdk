# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "websockets",
#     "flyte>=2.0.0b29"
# ]
# ///

"""
FastAPI WebSocket Example for Flyte

This example demonstrates how to create a FastAPI application with:
- Static file serving for frontend assets
- WebSocket support for real-time communication
- Health check endpoints
- Dynamic HTML injection for configuration

"""

import asyncio
import json
import pathlib
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

import flyte
from flyte.app.extras import FastAPIAppEnvironment

# Create the FastAPI app
app = FastAPI(
    title="Flyte WebSocket Demo",
    description="A FastAPI app with WebSocket support and frontend serving",
    version="1.0.0",
)

# Setup frontend directory
FRONTEND_DIR = Path(__file__).parent / "frontend"
FRONTEND_DIR.mkdir(exist_ok=True)

# Mount static files if frontend directory exists and has files
if FRONTEND_DIR.exists() and any(FRONTEND_DIR.iterdir()):
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Create FastAPI environment
env = FastAPIAppEnvironment(
    name="websocket-app",
    app=app,
    description="A FastAPI app with WebSocket and frontend serving capabilities",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("fastapi", "uvicorn", "websockets"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    requires_auth=False,
)


# Store active WebSocket connections
class ConnectionManager:
    """Manages WebSocket connections and broadcasts messages."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.remove(websocket)
        print(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        """Broadcast a message to all active connections."""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"Error broadcasting to client: {e}")


manager = ConnectionManager()


@env.app.get("/")
async def index():
    """Serve the main HTML page with dynamic WebSocket URL injection."""
    # Check if custom frontend exists
    html_file = FRONTEND_DIR / "index.html"
    if html_file.exists():
        html_content = html_file.read_text()

        # Inject WebSocket configuration
        # For local development, use relative URL so it auto-detects the port
        # In production, this would be the actual WebSocket endpoint URL
        ws_url = None  # Let the client auto-detect
        if hasattr(env, "endpoint") and env.endpoint:
            ws_url = env.endpoint.replace("http", "ws") + "/ws"

        # Only inject WS_URL if we have a specific endpoint (production)
        if ws_url:
            script_tag = f'<script>window.WS_URL = "{ws_url}";</script>'
            # Inject before closing head tag or at the start of body
            if "</head>" in html_content:
                html_content = html_content.replace("</head>", f"{script_tag}\n</head>")
            elif "<body>" in html_content:
                html_content = html_content.replace("<body>", f"<body>\n{script_tag}")
            else:
                html_content = script_tag + "\n" + html_content

        return HTMLResponse(content=html_content)

    # Default embedded HTML if no frontend directory
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html>
<head>
    <title>Flyte WebSocket Demo</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .status {
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            font-weight: bold;
        }
        .connected {
            background-color: #d4edda;
            color: #155724;
        }
        .disconnected {
            background-color: #f8d7da;
            color: #721c24;
        }
        #messages {
            height: 300px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 10px;
            margin: 20px 0;
            background: #fafafa;
            border-radius: 4px;
        }
        .message {
            padding: 8px;
            margin: 5px 0;
            border-radius: 4px;
            background: white;
            border-left: 3px solid #4CAF50;
        }
        .message.sent {
            border-left-color: #2196F3;
            background: #e3f2fd;
        }
        .message.received {
            border-left-color: #4CAF50;
            background: #e8f5e9;
        }
        .message .timestamp {
            font-size: 0.8em;
            color: #666;
        }
        input[type="text"] {
            width: 70%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-left: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Flyte WebSocket Demo</h1>
        <div id="status" class="status disconnected">Disconnected</div>

        <div id="messages"></div>

        <div>
            <input type="text" id="messageInput" placeholder="Type a message..." disabled>
            <button id="sendButton" onclick="sendMessage()" disabled>Send</button>
            <button id="connectButton" onclick="toggleConnection()">Connect</button>
        </div>
    </div>

    <script>
        let ws = null;
        const messagesDiv = document.getElementById('messages');
        const statusDiv = document.getElementById('status');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const connectButton = document.getElementById('connectButton');

        function addMessage(text, type = 'received') {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            const timestamp = new Date().toLocaleTimeString();
            messageDiv.innerHTML = `<div class="timestamp">${timestamp}</div><div>${text}</div>`;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function updateStatus(connected) {
            if (connected) {
                statusDiv.textContent = 'Connected';
                statusDiv.className = 'status connected';
                messageInput.disabled = false;
                sendButton.disabled = false;
                connectButton.textContent = 'Disconnect';
            } else {
                statusDiv.textContent = 'Disconnected';
                statusDiv.className = 'status disconnected';
                messageInput.disabled = true;
                sendButton.disabled = true;
                connectButton.textContent = 'Connect';
            }
        }

        function connect() {
            const wsUrl = window.WS_URL || `wss://${window.location.host}/ws`;
            ws = new WebSocket(wsUrl);

            ws.onopen = function(event) {
                addMessage('Connected to server', 'system');
                updateStatus(true);
            };

            ws.onmessage = function(event) {
                addMessage(event.data, 'received');
            };

            ws.onerror = function(error) {
                addMessage('WebSocket error occurred', 'system');
                console.error('WebSocket error:', error);
            };

            ws.onclose = function(event) {
                addMessage('Disconnected from server', 'system');
                updateStatus(false);
                ws = null;
            };
        }

        function disconnect() {
            if (ws) {
                ws.close();
            }
        }

        function toggleConnection() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                disconnect();
            } else {
                connect();
            }
        }

        function sendMessage() {
            const message = messageInput.value.trim();
            if (message && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(message);
                addMessage(message, 'sent');
                messageInput.value = '';
            }
        }

        messageInput.addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        });

        // Auto-connect on page load
        // connect();
    </script>
</body>
</html>
    """
    )


@env.app.get("/status")
async def status():
    """Health check endpoint."""
    return Response(status_code=200, content="OK")


@env.app.get("/health")
async def health():
    """Detailed health check with connection info."""
    return {
        "status": "healthy",
        "active_connections": len(manager.active_connections),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@env.app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time bidirectional communication.

    This endpoint:
    - Accepts WebSocket connections
    - Echoes received messages back to the sender
    - Broadcasts messages to all connected clients
    - Handles disconnections gracefully
    """
    await manager.connect(websocket)

    try:
        # Send welcome message
        await manager.send_personal_message(
            json.dumps(
                {
                    "type": "system",
                    "message": "Welcome! You are now connected to the Flyte WebSocket server.",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            ),
            websocket,
        )

        # Listen for messages
        while True:
            data = await websocket.receive_text()

            # Echo back to sender
            await manager.send_personal_message(
                json.dumps({"type": "echo", "message": f"Echo: {data}", "timestamp": datetime.now(UTC).isoformat()}),
                websocket,
            )

            # Broadcast to all clients
            await manager.broadcast(
                json.dumps(
                    {
                        "type": "broadcast",
                        "message": f"Broadcast: {data}",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "connections": len(manager.active_connections),
                    }
                )
            )

            # Simulate some processing delay
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(
            json.dumps(
                {
                    "type": "system",
                    "message": "A client disconnected",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "connections": len(manager.active_connections),
                }
            )
        )
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@env.app.get("/connections")
async def active_connections():
    """Return the number of active WebSocket connections."""
    return {"active_connections": len(manager.active_connections), "timestamp": datetime.now(UTC).isoformat()}


if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=pathlib.Path(__file__).parent,
    )

    print("🚀 Starting FastAPI WebSocket app locally...")
    print("=" * 60)

    # Serve the app on remote
    app = flyte.serve(env)
    print(f"App serving Deployment Info: {app.url}\n {app.endpoint}")
