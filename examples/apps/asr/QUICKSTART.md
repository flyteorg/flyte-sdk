# 🚀 Quick Start Guide

Get your speech transcription app running in minutes!

## Prerequisites

- Python 3.12+
- Flyte SDK 2.0+
- Access to GPU resources (A10G, T4, or A100)
- Modern web browser with microphone

## 1-Minute Deploy

```bash
# Navigate to the asr directory
cd examples/apps/asr

# Deploy both apps
python deploy.py
```

That's it! The script will:
1. Deploy the GPU transcriber service
2. Deploy the web frontend
3. Display your application URL

## Access Your App

Open the provided URL in your browser:
```
https://your-app.flyte.dev/
```

## Use the App

1. **Connect** - Click the connect button
2. **Allow Microphone** - Grant browser permission
3. **Start Recording** - Click to begin
4. **Speak** - Talk clearly into your mic
5. **See Transcription** - Watch it appear in real-time!

## Deploy Individual Apps

If you prefer to deploy separately:

```bash
# Deploy transcriber (GPU)
python transcriber.py

# Deploy web frontend (CPU)
python web_app.py
```

## What's Happening?

```
Your Browser
    ↓ (audio stream via WebSocket)
Web Frontend (CPU)
    ↓ (app-to-app call)
Transcriber Service (GPU + Parakeet model)
    ↓ (transcription result)
Web Frontend
    ↓ (WebSocket message)
Your Browser
```

## Key Features

- ✅ **Real-time**: See transcription as you speak
- ✅ **Multi-speaker**: Identifies different speakers
- ✅ **GPU-accelerated**: Fast inference with NVIDIA Parakeet
- ✅ **No setup**: Works in any modern browser

## Troubleshooting

### Can't access microphone
- Ensure you're using HTTPS (required for microphone access)
- Check browser permissions in settings

### No transcription appearing
- Check WebSocket connection status
- Verify transcriber service is running
- Look at browser console for errors

### Poor quality transcription
- Speak clearly and at normal pace
- Reduce background noise
- Check microphone is working properly

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Customize the frontend in `frontend/` directory
- Adjust model parameters in `transcriber.py`
- Add authentication in `web_app.py`

## Need Help?

Check the comprehensive README or file an issue on GitHub!
