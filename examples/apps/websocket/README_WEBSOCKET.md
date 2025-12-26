# Flyte FastAPI WebSocket Example

This example demonstrates how to build a FastAPI application with WebSocket support and frontend serving in Flyte, inspired by Modal's speech-to-text example pattern.

## Features

- ✅ **WebSocket Support**: Real-time bidirectional communication
- ✅ **Static File Serving**: Serve HTML, CSS, and JavaScript frontend
- ✅ **Connection Management**: Track and manage multiple WebSocket connections
- ✅ **Message Broadcasting**: Send messages to all connected clients
- ✅ **Auto-Reconnect**: Client automatically reconnects on disconnect
- ✅ **Health Endpoints**: Monitor application and connection status
- ✅ **Dynamic Configuration**: Inject WebSocket URLs into frontend

## Project Structure

```
examples/apps/
├── fastapi_websocket.py      # Main FastAPI application
├── frontend/                  # Frontend assets (optional)
│   ├── index.html            # Main HTML page
│   ├── styles.css            # Styling
│   └── app.js                # WebSocket client logic
└── README_WEBSOCKET.md       # This file
```

## Quick Start

### 1. Run Locally

```bash
# Navigate to the examples directory
cd examples/apps

# Run the application
python fastapi_websocket.py
```

### 2. Deploy to Flyte

```bash
# Deploy using the script
python fastapi_websocket.py

# Or use flyte CLI
flyte deploy fastapi_websocket.py
```

### 3. Access the Application

After deployment, you'll see output like:

```
✅ Deployed FastAPI WebSocket app:
<deployment details>

WebSocket URL: https://your-app.flyte.dev/ws
Frontend URL: https://your-app.flyte.dev/
Health check: https://your-app.flyte.dev/health
```

## API Endpoints

### HTTP Endpoints

- **`GET /`** - Main HTML page with WebSocket client
- **`GET /status`** - Simple health check (returns 200 OK)
- **`GET /health`** - Detailed health info with connection count
- **`GET /connections`** - Current active WebSocket connections
- **`GET /static/{file}`** - Serve static files from frontend directory

### WebSocket Endpoint

- **`WS /ws`** - WebSocket endpoint for real-time communication

## WebSocket Message Flow

1. **Client connects** → Server sends welcome message
2. **Client sends message** → Server echoes back to sender
3. **Server broadcasts** → All clients receive the message
4. **Client disconnects** → Cleanup and notify others

## Message Format

Messages are sent as JSON objects:

```json
{
  "type": "echo|broadcast|system",
  "message": "The message content",
  "timestamp": "2025-01-15T10:30:00",
  "connections": 3
}
```

## Frontend Usage

The frontend automatically connects to the WebSocket endpoint and provides:

- Real-time message display
- Send/receive messages
- Connection status indicator
- Active connection count
- Auto-reconnect on disconnect

### Using Custom Frontend

To use the custom frontend, ensure the `frontend/` directory exists with:
- `index.html` - Main page
- `styles.css` - Styling
- `app.js` - WebSocket client

The application will automatically mount these files at `/static/`.

### Using Embedded Frontend

If no `frontend/` directory is found, the application serves an embedded HTML page with all necessary functionality.

## Connection Manager

The `ConnectionManager` class handles:

```python
- connect(websocket)           # Accept new connection
- disconnect(websocket)         # Remove connection
- send_personal_message(msg, ws) # Send to specific client
- broadcast(msg)                # Send to all clients
```

## Configuration

### Environment Resources

```python
env = FastAPIAppEnvironment(
    name="websocket-app",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12))
        .with_pip_packages("fastapi", "uvicorn", "websockets"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    requires_auth=False,
)
```

### WebSocket URL Injection

The server automatically injects the WebSocket URL into the frontend:

```javascript
// Injected by server
window.WS_URL = "wss://your-app.flyte.dev/ws";

// Used by client
const ws = new WebSocket(window.WS_URL);
```

## Testing Locally

1. Run the application:
   ```bash
   uvicorn fastapi_websocket:app --reload
   ```

2. Open http://localhost:8000 in your browser

3. Click "Connect" to establish WebSocket connection

4. Type messages and see them echo back and broadcast

## Production Deployment

When deploying to production:

1. The WebSocket URL is automatically configured
2. HTTPS → WSS protocol upgrade is handled
3. Multiple concurrent connections are supported
4. Connection cleanup is automatic

## Customization

### Add Custom Message Handlers

```python
@env.app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()

            # Custom processing
            if data.startswith("/command"):
                await handle_command(data, websocket)
            else:
                await manager.broadcast(data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### Add Authentication

```python
from fastapi import Header, HTTPException

@env.app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Header(None)
):
    if not verify_token(token):
        await websocket.close(code=1008)
        return
    # ... rest of handler
```

## Troubleshooting

### WebSocket Connection Fails

- Check that the WebSocket URL uses `ws://` (local) or `wss://` (production)
- Verify firewall rules allow WebSocket connections
- Check browser console for connection errors

### Messages Not Broadcasting

- Verify `ConnectionManager` is shared across requests (module-level)
- Check that connections are properly registered
- Look for exceptions in server logs

### Static Files Not Loading

- Ensure `frontend/` directory exists and has correct structure
- Check file permissions
- Verify StaticFiles mounting path

## Advanced Features

### Add Message Persistence

```python
from collections import deque

# Store last 100 messages
message_history = deque(maxlen=100)

# On new connection, send history
for msg in message_history:
    await manager.send_personal_message(msg, websocket)
```

### Add Rate Limiting

```python
from time import time

class RateLimiter:
    def __init__(self, max_per_minute=60):
        self.messages = {}
        self.limit = max_per_minute

    def check(self, client_id):
        now = time()
        if client_id not in self.messages:
            self.messages[client_id] = []

        # Remove old messages
        self.messages[client_id] = [
            t for t in self.messages[client_id]
            if now - t < 60
        ]

        if len(self.messages[client_id]) >= self.limit:
            return False

        self.messages[client_id].append(now)
        return True
```

## References

- [FastAPI WebSocket Documentation](https://fastapi.tiangolo.com/advanced/websockets/)
- [Flyte FastAPI Environment](https://docs.flyte.org/)

## License

This example is part of the Flyte SDK examples and follows the same license.
