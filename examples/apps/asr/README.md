# 🎤 Flyte Speech Transcription with Parakeet

Real-time multi-speaker speech transcription using NVIDIA's Parakeet model, demonstrating Flyte's app-to-app calling pattern.

## Overview

This example demonstrates a complete speech transcription system with two Flyte apps:

1. **Transcriber Service** (`transcriber.py`) - GPU-accelerated backend running Parakeet model
2. **Web Frontend** (`web_app.py`) - FastAPI app with WebSocket for audio streaming

The web frontend calls the transcriber service using Flyte's app-to-app calling, similar to Modal's function calling pattern.

## Architecture

```
┌─────────────────┐
│                 │
│   Web Browser   │
│  (Microphone)   │
│                 │
└────────┬────────┘
         │ WebSocket
         │ (Audio Stream)
         ▼
┌─────────────────┐
│                 │
│   Web Frontend  │
│  FastAPI + WS   │
│  (CPU Only)     │
│                 │
└────────┬────────┘
         │ App Call
         │ (Flyte)
         ▼
┌─────────────────┐
│                 │
│  Transcriber    │
│  Parakeet +GPU  │
│  (A10G/T4/A100) │
│                 │
└─────────────────┘
```

## Features

✅ **Multi-speaker transcription** - Identifies and labels different speakers
✅ **Real-time streaming** - Low-latency transcription as you speak
✅ **GPU acceleration** - Uses NVIDIA NeMo and Parakeet models
✅ **App-to-app calling** - Web frontend calls GPU backend seamlessly
✅ **WebSocket support** - Full-duplex communication for audio
✅ **Browser-based** - No installation needed, works in any modern browser

## Quick Start

### 1. Deploy Both Apps

```bash
cd examples/apps/asr

# Deploy the transcriber service (GPU)
python transcriber.py

# Deploy the web frontend (CPU)
python web_app.py
```

### 2. Access the Web Interface

After deployment, open the frontend URL in your browser:

```
Frontend: https://your-app.flyte.dev/
```

### 3. Use the Application

1. Click "Connect" to establish WebSocket connection
2. Grant microphone permission when prompted
3. Click "Start Recording"
4. Speak clearly into your microphone
5. See real-time transcription appear

## File Structure

```
asr/
├── transcriber.py          # GPU transcription service
├── web_app.py             # Web frontend with WebSocket
├── frontend/              # Optional custom frontend files
│   ├── index.html        # Custom UI (if desired)
│   ├── styles.css        # Custom styles
│   └── app.js            # Custom client logic
└── README.md             # This file
```

## How It Works

### Transcriber Service

The transcriber service (`transcriber.py`):

1. **Loads Models** on GPU:
   - Speaker diarization model: `nvidia/diar_streaming_sortformer_4spk-v2.1`
   - ASR model: `nvidia/multitalker-parakeet-streaming-0.6b-v1`

2. **Processes Audio Chunks**:
   - Receives int16 PCM audio at 16kHz
   - Converts to float32 tensors
   - Performs speaker-tagged transcription
   - Returns text with speaker labels

3. **Manages Streaming**:
   - Uses `CacheAwareStreamingAudioBuffer` for chunked processing
   - Maintains state across multiple chunks
   - Supports multiple concurrent streams

### Web Frontend

The web frontend (`web_app.py`):

1. **Serves HTML Interface**:
   - Embeds audio recording UI
   - Injects WebSocket URL dynamically
   - Handles microphone access

2. **WebSocket Handler**:
   - Accepts binary audio data
   - Buffers and chunks audio appropriately
   - Calls transcriber via app-to-app calling
   - Streams transcription results back

3. **Audio Processing**:
   - Captures microphone using Web Audio API
   - Converts float32 to int16 PCM
   - Streams in ~13 frame chunks
   - Maintains 16kHz sample rate

## Configuration

### GPU Resources

In `transcriber.py`, configure GPU resources:

```python
@app.function(
    resources=flyte.Resources(
        cpu=4,
        memory="16Gi",
        gpu=1,
        gpu_kind="A10G"  # Options: T4, A10G, A100
    ),
    keep_warm=1,  # Keep instances warm for low latency
)
```

### Audio Parameters

In `transcriber.py`, adjust streaming parameters:

```python
@dataclass
class TranscriptionConfig:
    speaker_cache_len: int = 188       # Speaker embedding cache
    asr_context_size: int = 10         # ASR context frames
    asr_decoder_context_len: int = 3   # Decoder context
    asr_batch_size: int = 4            # Batch size
    sample_rate: int = 16000           # Must be 16kHz for Parakeet
    chunk_size_frames: int = 13        # Frames per chunk
```

### Frontend Resources

In `web_app.py`, configure CPU resources:

```python
env = FastAPIAppEnvironment(
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    requires_auth=False,  # Set to True for authentication
)
```

## App-to-App Calling

The key integration point is in `web_app.py`:

```python
# Import the transcriber
from transcriber import Transcriber

# In the WebSocket handler:
transcriber = Transcriber()  # Get reference to deployed service

# Call the transcriber
result = transcriber.transcribe(audio_chunk, stream_id)

# Use the result
if result["success"]:
    await send_transcription(result["text"])
```

Flyte handles:
- Service discovery
- Load balancing
- Error handling
- Retries

## Audio Format Requirements

The system expects:
- **Format**: PCM int16
- **Sample Rate**: 16,000 Hz (16 kHz)
- **Channels**: Mono (1 channel)
- **Byte Order**: Little-endian

Browser audio is automatically converted from float32 to int16.

## Performance Optimization

### Keep-Warm Instances

```python
@app.function(
    keep_warm=1,  # Keeps 1 GPU instance always ready
)
```

Benefits:
- Eliminates cold start latency
- Model stays loaded in GPU memory
- First request is just as fast as subsequent ones

### Chunking Strategy

Audio is processed in chunks of 13 frames:
- Chunk size: 13 frames × 16000 Hz × 2 bytes = ~416 KB
- Provides good balance between latency and accuracy
- Allows streaming transcription without waiting for full audio

## Troubleshooting

### No Audio Being Received

- Check browser permissions for microphone
- Ensure HTTPS is used (required for getUserMedia)
- Verify WebSocket connection is established
- Check browser console for errors

### Poor Transcription Quality

- Speak clearly and at moderate pace
- Reduce background noise
- Ensure good microphone quality
- Check that sample rate is 16kHz

### High Latency

- Increase `keep_warm` instances
- Use faster GPU (A100 > A10G > T4)
- Reduce network latency (deploy closer to users)
- Check chunk sizes aren't too large

### GPU Out of Memory

- Reduce batch sizes in config
- Use smaller GPU if possible
- Ensure only necessary models are loaded
- Check for memory leaks in long sessions

## API Endpoints

### Web Frontend

- `GET /` - Main HTML page with audio recorder
- `GET /status` - Simple health check
- `GET /health` - Detailed health with connection count
- `WS /ws` - WebSocket endpoint for audio streaming

### Transcriber Service

The transcriber is called programmatically via Flyte's app-to-app calling:

```python
result = transcriber.transcribe(
    audio_bytes=<bytes>,
    stream_id=<int>
)
# Returns: {"success": bool, "text": str, "is_final": bool}

transcriber.reset(stream_id=<int>)
# Resets the streaming buffer
```

## Customization

### Add Custom Frontend

Create files in `frontend/` directory:

```bash
frontend/
├── index.html    # Your custom HTML
├── styles.css    # Your custom styles
└── app.js        # Your custom JavaScript
```

The web app will automatically serve these instead of the embedded HTML.

### Add Speaker Separation

Modify `transcriber.py` to track speakers:

```python
def _perform_inference(self, audio, audio_length):
    # ... existing code ...

    # Extract speaker IDs
    speaker_ids = self._extract_speaker_ids(diar_logits)

    # Format with speaker labels
    return f"[Speaker {speaker_ids[0]}] {text}"
```

### Add Language Support

The Parakeet model supports English by default. For other languages:

1. Use a different NeMo ASR model
2. Update model loading in `transcriber.py`
3. Adjust preprocessing parameters as needed

## Comparison with Modal

This example closely follows Modal's pattern:

| Feature | Modal | Flyte |
|---------|-------|-------|
| GPU Service | `@app.cls(gpu="A10G")` | `@app.function(gpu=1, gpu_kind="A10G")` |
| Web Frontend | `@app.asgi_app()` | `FastAPIAppEnvironment` |
| Service Calling | `cls_instance.method()` | `Transcriber().transcribe()` |
| WebSocket | Native FastAPI | Native FastAPI |
| Keep Warm | `concurrency_limit` | `keep_warm` |

## Cost Optimization

To reduce costs while maintaining performance:

1. **Use keep_warm strategically**:
   ```python
   keep_warm=1  # During business hours
   keep_warm=0  # Off-hours
   ```

2. **Right-size GPU**:
   - Development: T4 ($0.35/hr)
   - Production: A10G ($1.00/hr)
   - High-performance: A100 ($3.00/hr)

3. **Set timeouts**:
   ```python
   timeout=3600  # Auto-shutdown after 1 hour idle
   ```

## Production Considerations

### Authentication

Enable authentication in `web_app.py`:

```python
env = FastAPIAppEnvironment(
    requires_auth=True,  # Enable auth
)
```

### Rate Limiting

Add rate limiting to prevent abuse:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@env.app.websocket("/ws")
@limiter.limit("10/minute")
async def websocket_endpoint(websocket: WebSocket):
    ...
```

### Monitoring

Add logging and metrics:

```python
import time

start_time = time.time()
result = transcriber.transcribe(chunk, stream_id)
latency = time.time() - start_time

logger.info(f"Transcription latency: {latency:.3f}s")
```

### Error Handling

Implement robust error handling:

```python
try:
    result = transcriber.transcribe(chunk, stream_id)
except Exception as e:
    logger.error(f"Transcription failed: {e}")
    await send_error_message(websocket, str(e))
    # Optionally retry or gracefully degrade
```

## References

- [NVIDIA Parakeet Model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/parakeet)
- [NeMo Toolkit Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)
- [Flyte FastAPI Apps](https://docs.flyte.org/)
- [Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)

## License

This example is part of the Flyte SDK and follows the same license.
