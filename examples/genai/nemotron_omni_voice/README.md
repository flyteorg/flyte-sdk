# Nemotron-Omni voice chat (low-latency)

A Flyte app that serves NVIDIA's **Nemotron-3-Nano-Omni-30B-A3B** multimodal
model and a microphone client for real-time spoken conversation.

The Omni model ingests audio natively (audio/video/image/text → text), so the
client sends raw mic audio straight to the model — **no separate speech-to-text
stage**. Replies stream back as text and are spoken sentence-by-sentence on the
client.

```
mic ──► [client] capture audio ──► base64 WAV in an OpenAI chat message
                                      │
                                      ▼
                    [Flyte app] vLLM serving Nemotron-3-Nano-Omni-30B-A3B
                                      │  (audio-in → text-out, streamed)
                                      ▼
                      [client] sentence-by-sentence TTS ──► speaker
```

## Files

| File | What it is |
|------|-----------|
| `serve.py` | Flyte app: prefetch weights → serve with `flyteplugins-vllm` (`VLLMAppEnvironment`) |
| `voice_client.py` | Microphone client: VAD capture → streamed reply → local TTS |

## Why it's low-latency

**Server** (`serve.py`, flags passed to `vllm serve`):
- FP8 weights + `--kv-cache-dtype fp8` — ~33 GB, fits one **L40S 48 GB**, faster decode.
- `--max-num-seqs 4` — small batch ceiling minimizes scheduling/queueing latency for a 1-user session.
- `--max-model-len 32768` — smaller KV cache, lower per-token latency (raise for long audio context).
- `--async-scheduling` — NVIDIA's recommended best-latency scheduler.
- `Scaling(replicas=(1, 1))` — one **warm** replica, so there is no cold start on the first word.

**Client** (`voice_client.py`):
- Energy-based VAD ends the turn the instant you stop talking.
- Response consumed as a token **stream**; TTS speaks each sentence as it completes, overlapping synthesis with generation.
- `enable_thinking=false` — skips `<think>` reasoning traces that would otherwise add seconds of latency.

## Deploy

1. **Create the HF token secret** (the Nemotron weights are license-gated):

   ```bash
   flyte create secret HF_TOKEN <your-hf-token>
   ```

2. **Prefetch + deploy** (one command — prefetch is cached and reused):

   ```bash
   python examples/genai/nemotron_omni_voice/serve.py
   ```

   This runs `flyte.prefetch.hf_model(...)` to cache the weights in the Flyte
   object store, then `flyte.serve(...)`. It prints the app URL and the
   OpenAI-compatible `/v1` endpoint.

   To redeploy against an already-prefetched model without re-running prefetch:

   ```bash
   flyte deploy examples/genai/nemotron_omni_voice/serve.py vllm_app
   ```

## Talk to it

```bash
pip install sounddevice numpy httpx       # core client deps
pip install pyttsx3                        # TTS on Linux/Windows (macOS uses built-in `say`)

python examples/genai/nemotron_omni_voice/voice_client.py \
    --endpoint <app-url> --model nemotron-omni
```

Speak, pause, and the assistant answers out loud. `Ctrl-C` to quit.

Flags: `--no-speak` (print instead of speak), `--silence 0.8` (trailing-silence
cutoff in seconds), `--api-key <token>` (if you enable auth on the app).

## Sanity-check the endpoint without a mic

The app is a standard OpenAI-compatible server, so you can send a `.wav` directly:

```python
import base64, httpx

wav = base64.b64encode(open("hello.wav", "rb").read()).decode()
r = httpx.post(
    "<app-url>/v1/chat/completions",
    json={
        "model": "nemotron-omni",
        "messages": [{"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{wav}"}},
            {"type": "text", "text": "Transcribe and answer."},
        ]}],
        "chat_template_kwargs": {"enable_thinking": False},
    },
    timeout=120,
)
print(r.json()["choices"][0]["message"]["content"])
```

## Notes / knobs

- **Version pins are deliberate.** The Omni checkpoint needs **vLLM ≥ 0.20** and
  `--trust-remote-code`; both are set in `serve.py`. Bump `VLLM_VERSION` there if
  NVIDIA's recipe moves.
- **Want lower cost over lower latency?** Set `Scaling(replicas=(0, 1), scaledown_after=300)`
  to scale to zero when idle — at the cost of a cold start (multi-GB load) on the next request.
- **BF16 instead of FP8** needs ~64 GB → switch `MODEL_REPO` to the BF16 variant
  and request an H100/H200 or 2×L40S with tensor parallel.
- **Production:** set `requires_auth=True` and pass `--api-key` from the client.
