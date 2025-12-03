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
Speech Transcription Web Frontend

This FastAPI application provides a web interface for real-time speech
transcription. It receives audio streams via WebSocket and calls the
Parakeet transcription service using app-to-app calling.

Based on Modal's speech-to-text example pattern.
"""

import asyncio
import json
import logging
import pathlib
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

# Import the transcriber (for app-to-app calling)
from transcriber import Transcriber

import flyte
from flyte.app.extras import FastAPIAppEnvironment

# Create the FastAPI app
app = FastAPI(
    title="Flyte Speech Transcription",
    description="Real-time speech transcription using Parakeet multi-talker model",
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
    name="asr-web-frontend",
    app=app,
    description="Web frontend for speech transcription with WebSocket support",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("fastapi", "uvicorn", "websockets"),
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    requires_auth=False,
)


class TranscriptionManager:
    """Manages WebSocket connections and transcription sessions."""

    def __init__(self):
        self.active_connections: dict[int, WebSocket] = {}
        self.stream_counter = 0

    async def connect(self, websocket: WebSocket) -> int:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        stream_id = self.stream_counter
        self.stream_counter += 1
        self.active_connections[stream_id] = websocket
        print(f"Client connected (stream {stream_id}). Total connections: {len(self.active_connections)}")
        return stream_id

    def disconnect(self, stream_id: int):
        """Remove a WebSocket connection."""
        if stream_id in self.active_connections:
            del self.active_connections[stream_id]
            print(f"Client disconnected (stream {stream_id}). Total connections: {len(self.active_connections)}")

    async def send_message(self, message: dict, stream_id: int):
        """Send a JSON message to a specific WebSocket connection."""
        if stream_id in self.active_connections:
            websocket = self.active_connections[stream_id]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                print(f"Error sending message to stream {stream_id}: {e}")


manager = TranscriptionManager()


@env.app.get("/")
async def index():
    """Serve the main HTML page with audio recording and transcription UI."""
    html_file = FRONTEND_DIR / "index.html"
    if html_file.exists():
        html_content = html_file.read_text()

        # Inject WebSocket configuration
        ws_url = "ws://localhost:8000/ws"
        if hasattr(env, "endpoint") and env.endpoint:
            ws_url = env.endpoint.replace("http", "ws") + "/ws"

        script_tag = f'<script>window.WS_URL = "{ws_url}";</script>'

        if "</head>" in html_content:
            html_content = html_content.replace("</head>", f"{script_tag}\n</head>")
        else:
            html_content = script_tag + "\n" + html_content

        return HTMLResponse(content=html_content)

    # Default embedded HTML with audio recording
    return HTMLResponse(content=get_embedded_html())


def get_embedded_html() -> str:
    """Return embedded HTML with audio recording interface."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Flyte Speech Transcription</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 30px auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }
        .status {
            padding: 15px;
            margin: 15px 0;
            border-radius: 8px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }
        .connected {
            background-color: #d4edda;
            color: #155724;
        }
        .connected .status-indicator {
            background: #28a745;
            animation: pulse 2s infinite;
        }
        .disconnected {
            background-color: #f8d7da;
            color: #721c24;
        }
        .disconnected .status-indicator {
            background: #dc3545;
        }
        .recording {
            background-color: #fff3cd;
            color: #856404;
        }
        .recording .status-indicator {
            background: #ffc107;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.2); }
            100% { opacity: 1; transform: scale(1); }
        }
        #transcription {
            min-height: 300px;
            max-height: 400px;
            overflow-y: auto;
            border: 2px solid #e9ecef;
            padding: 20px;
            margin: 20px 0;
            background: #f8f9fa;
            border-radius: 8px;
            font-size: 16px;
            line-height: 1.6;
        }
        .transcript-line {
            padding: 10px;
            margin: 5px 0;
            background: white;
            border-radius: 6px;
            border-left: 3px solid #667eea;
        }
        .transcript-line .time {
            font-size: 0.8em;
            color: #6c757d;
            margin-bottom: 4px;
        }
        .controls {
            display: flex;
            gap: 15px;
            margin: 20px 0;
        }
        button {
            padding: 15px 30px;
            font-size: 16px;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .btn-primary {
            background-color: #667eea;
            color: white;
        }
        .btn-danger {
            background-color: #dc3545;
            color: white;
        }
        .btn-secondary {
            background-color: #6c757d;
            color: white;
        }
        .info {
            background: #e7f3ff;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #0066cc;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Flyte Speech Transcription</h1>
        <p>Real-time speech transcription powered by NVIDIA Parakeet multi-talker model</p>

        <div id="status" class="status disconnected">
            <span class="status-indicator"></span>
            <span id="statusText">Disconnected</span>
        </div>

        <div class="controls">
            <button id="connectBtn" class="btn-primary" onclick="toggleConnection()">
                Connect
            </button>
            <button id="recordBtn" class="btn-danger" onclick="toggleRecording()" disabled>
                Start Recording
            </button>
            <button class="btn-secondary" onclick="clearTranscription()">
                Clear
            </button>
        </div>

        <div class="info">
            <strong>Instructions:</strong>
            <ol>
                <li>Click "Connect" to establish WebSocket connection</li>
                <li>Allow microphone access when prompted</li>
                <li>Click "Start Recording" to begin transcription</li>
                <li>Speak clearly into your microphone</li>
                <li>Click "Stop Recording" when done</li>
            </ol>
        </div>

        <div id="transcription">
            <p style="color: #6c757d; text-align: center;">Transcription will appear here...</p>
        </div>
    </div>

    <script>
        let ws = null;
        let mediaRecorder = null;
        let audioContext = null;
        let isRecording = false;

        const statusDiv = document.getElementById('status');
        const statusText = document.getElementById('statusText');
        const connectBtn = document.getElementById('connectBtn');
        const recordBtn = document.getElementById('recordBtn');
        const transcriptionDiv = document.getElementById('transcription');

        function updateStatus(state, text) {
            statusDiv.className = 'status ' + state;
            statusText.textContent = text;
        }

        function toggleConnection() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                disconnect();
            } else {
                connect();
            }
        }

        function connect() {
            const wsUrl = window.WS_URL || `ws://${window.location.host}/ws`;
            ws = new WebSocket(wsUrl);
            ws.binaryType = 'arraybuffer';

            ws.onopen = function() {
                updateStatus('connected', 'Connected - Ready to record');
                connectBtn.textContent = 'Disconnect';
                connectBtn.className = 'btn-danger';
                recordBtn.disabled = false;
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'transcription' && data.text) {
                    addTranscription(data.text);
                } else if (data.type === 'error') {
                    console.error('Transcription error:', data.message);
                }
            };

            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                updateStatus('disconnected', 'Connection error');
            };

            ws.onclose = function() {
                updateStatus('disconnected', 'Disconnected');
                connectBtn.textContent = 'Connect';
                connectBtn.className = 'btn-primary';
                recordBtn.disabled = true;
                if (isRecording) {
                    stopRecording();
                }
            };
        }

        function disconnect() {
            if (isRecording) {
                stopRecording();
            }
            if (ws) {
                ws.close();
            }
        }

        async function toggleRecording() {
            if (isRecording) {
                stopRecording();
            } else {
                await startRecording();
            }
        }

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        channelCount: 1,
                        sampleRate: 16000,
                        echoCancellation: true,
                        noiseSuppression: true,
                    }
                });

                audioContext = new AudioContext({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(stream);
                const processor = audioContext.createScriptProcessor(4096, 1, 1);

                processor.onaudioprocess = function(e) {
                    if (!isRecording || !ws || ws.readyState !== WebSocket.OPEN) return;

                    const inputData = e.inputBuffer.getChannelData(0);
                    // Convert float32 to int16
                    const int16Data = new Int16Array(inputData.length);
                    for (let i = 0; i < inputData.length; i++) {
                        int16Data[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
                    }
                    ws.send(int16Data.buffer);
                };

                source.connect(processor);
                processor.connect(audioContext.destination);

                isRecording = true;
                updateStatus('recording', 'Recording...');
                recordBtn.textContent = 'Stop Recording';
                recordBtn.className = 'btn-danger';

            } catch (error) {
                console.error('Error starting recording:', error);
                alert('Could not access microphone. Please grant permission.');
            }
        }

        function stopRecording() {
            isRecording = false;
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            updateStatus('connected', 'Connected - Ready to record');
            recordBtn.textContent = 'Start Recording';
            recordBtn.className = 'btn-primary';
        }

        function addTranscription(text) {
            if (!text.trim()) return;

            const lineDiv = document.createElement('div');
            lineDiv.className = 'transcript-line';

            const timeDiv = document.createElement('div');
            timeDiv.className = 'time';
            timeDiv.textContent = new Date().toLocaleTimeString();

            const textDiv = document.createElement('div');
            textDiv.textContent = text;

            lineDiv.appendChild(timeDiv);
            lineDiv.appendChild(textDiv);

            // Remove placeholder if it exists
            if (transcriptionDiv.children.length === 1 &&
                transcriptionDiv.children[0].tagName === 'P') {
                transcriptionDiv.innerHTML = '';
            }

            transcriptionDiv.appendChild(lineDiv);
            transcriptionDiv.scrollTop = transcriptionDiv.scrollHeight;
        }

        function clearTranscription() {
            transcriptionDiv.innerHTML = (
                '<p style="color: #6c757d; text-align: center;">Transcription will appear here...</p>'
            );
        }
    </script>
</body>
</html>
    """


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
        "timestamp": datetime.utcnow().isoformat(),
    }


@env.app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time speech transcription.

    This endpoint:
    - Accepts WebSocket connections
    - Receives audio bytes from the client
    - Calls the Transcriber service (app-to-app calling)
    - Streams transcription results back to the client
    """
    stream_id = await manager.connect(websocket)

    # Create a queue for audio chunks
    audio_queue = asyncio.Queue()

    # Get reference to transcriber (app-to-app calling)
    try:
        transcriber = Transcriber()
    except Exception as e:
        await manager.send_message({"type": "error", "message": f"Failed to connect to transcriber: {e}"}, stream_id)
        manager.disconnect(stream_id)
        await websocket.close()
        return

    # Send welcome message
    await manager.send_message(
        {
            "type": "system",
            "message": "Connected! Start recording to begin transcription.",
            "timestamp": datetime.utcnow().isoformat(),
        },
        stream_id,
    )

    async def receive_audio():
        """Receive audio bytes from client and queue them."""
        try:
            audio_buffer = bytearray()
            chunk_size = 13 * 16000 * 2  # 13 frames at 16kHz, 2 bytes per sample

            while True:
                data = await websocket.receive_bytes()
                audio_buffer.extend(data)

                # Send chunks of the right size
                while len(audio_buffer) >= chunk_size:
                    chunk = bytes(audio_buffer[:chunk_size])
                    audio_buffer = audio_buffer[chunk_size:]
                    await audio_queue.put(chunk)

        except WebSocketDisconnect:
            await audio_queue.put(None)  # Signal end of stream
        except Exception as e:
            print(f"Error receiving audio: {e}")
            await audio_queue.put(None)

    async def process_audio():
        """Process audio chunks and send transcriptions."""
        try:
            while True:
                chunk = await audio_queue.get()

                if chunk is None:  # End of stream
                    break

                # Call transcriber service (app-to-app calling)
                result = transcriber.transcribe(chunk, stream_id)

                if result.get("success") and result.get("text"):
                    await manager.send_message(
                        {
                            "type": "transcription",
                            "text": result["text"],
                            "is_final": result.get("is_final", False),
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                        stream_id,
                    )

        except Exception as e:
            print(f"Error processing audio: {e}")
            await manager.send_message({"type": "error", "message": str(e)}, stream_id)

    # Run both tasks concurrently
    try:
        await asyncio.gather(
            receive_audio(),
            process_audio(),
        )
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Cleanup
        transcriber.reset(stream_id)
        manager.disconnect(stream_id)


if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=pathlib.Path(__file__).parent,
        log_level=logging.DEBUG,
    )

    print("Deploying Speech Transcription Web App...")
    deployments = flyte.deploy(env)

    if deployments:
        d = deployments[0]
        print("\n✅ Deployed Web App:")
        print(f"{d.table_repr()}")
        print(f"\nFrontend URL: {d.endpoint}/")
        print(f"WebSocket URL: {d.endpoint}/ws")
        print(f"Health check: {d.endpoint}/health")
    else:
        print("\n❌ Deployment failed")
