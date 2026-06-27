# Voice customer-service agent (browser → Qwen → voice)

Talk to a customer-support agent in your browser and hear it talk back. **Speech
recognition (STT) runs in the browser** (Web Speech API). **TTS is switchable**: the
browser's built-in voice, or a real neural voice (**Kokoro-82M**) served on Flyte. A
small **Qwen** instruct model served with **vLLM** on a single **L4** GPU is the brain.

```
🎤 browser STT ─► /api/chat (FastAPI proxy) ─► vLLM /v1 (Qwen 3B on L4)
                                                   │ streamed tokens
🔊 voice ◄─ [ Browser TTS ]  OR  [ /api/tts ─► Kokoro-82M (CPU) ] ◄─ streamed text
```

Why this shape:
- **STT is free in the browser** — no audio model to host for input.
- **TTS is a live A/B** — flip between instant browser TTS and natural Kokoro TTS, and
  the page **measures the latency of each** (see below).
- **Served over HTTPS from Flyte** — required for the browser to grant mic access.
- **FastAPI proxy** — the browser only talks to its own origin (no CORS); the LLM and
  TTS stay internal.

## Two apps

| App env | What | Hardware |
|---------|------|----------|
| `llm_app` | `Qwen/Qwen2.5-3B-Instruct` via `flyteplugins-vllm` (`VLLMAppEnvironment`), OpenAI `/v1` | L4:1 |
| `ui_app`  | FastAPI: voice page + `/api/chat` proxy to `llm_app` + `/api/tts` (Kokoro-82M) | CPU |

## TTS: switch + measure

The footer has a **TTS engine** toggle: **Browser (instant)** vs **Server · Kokoro**.
After each reply the page shows a metrics line, e.g.:

```
Server · Kokoro — first audio 840ms (avg 910ms) · LLM first token 240ms · synth 450ms
Browser — first audio 70ms (avg 80ms) · LLM first token 240ms
```

- `synth` = server-measured Kokoro time (returned via the `X-Synth-Ms` header), ~0.4–0.5s/clause.
- `first audio` = time from a clause being ready to audio actually playing (TTFA).
- Replies are spoken clause-by-clause and the **next clause is prefetched while the current
  one plays**, so only the first clause's latency is felt.
- Barge-in works in both modes (interrupting aborts the TTS fetch + stops playback, or cancels
  browser speech).

Knobs (env vars on `ui_app`):
- `TTS_MODE` = `both` (default, shows the switch) | `browser` | `server` — force one mode so you
  can instead deploy two clearly-labeled single-mode instances.
- `TTS_VOICE` = a Kokoro voice id (default `af_heart`).

Kokoro runs on **CPU** here (it's only 82M params). If you want lower synth latency, give the UI
app a GPU (`resources=flyte.Resources(..., gpu="L4:1")`) — Kokoro will use it automatically.

## Deploy (demo cluster)

`config.yaml` targets `demo.hosted.unionai.cloud` (project `flyteexamples`,
domain `development`) and the example uses the **remote** image builder, so no
local Docker is needed.

```bash
# 1. Bring up the GPU model server (long pole: provisions an L4, pulls weights)
python app.py llm
#    -> prints:  LLM app: https://<llm-url>

# 2. Bring up the voice UI, pointed at the LLM
python app.py ui --llm-url https://<llm-url>
#    -> prints:  Voice UI: https://<ui-url>
```

Open `https://<ui-url>` in **Google Chrome**, click the mic, and start talking.

## Test without a mic

```bash
# LLM directly (OpenAI-compatible):
curl -s https://<llm-url>/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"qwen",
  "messages":[{"role":"user","content":"My order hasn'\''t arrived. Help?"}]
}' | jq -r .choices[0].message.content

# Through the UI proxy (streams plain text):
curl -N https://<ui-url>/api/chat -H 'content-type: application/json' -d '{
  "messages":[{"role":"user","content":"Do you ship to Canada?"}]
}'
```

## Knobs

- **Latency vs. quality:** swap `model_hf_path` to `Qwen/Qwen2.5-7B-Instruct` (still
  fits L4 24 GB) for better answers; keep 3B for snappiest responses.
- **Cost:** `Scaling(replicas=(1, 1))` keeps the GPU warm. Use `(0, 1)` +
  `scaledown_after` to release it when idle (adds a cold start).
- **T4 instead of L4:** T4 (Turing) has no bf16 — add `--dtype float16` to the vLLM
  `extra_args` and set `gpu="T4:1"`.
- **The agent's persona** lives in `SYSTEM_PROMPT` in `app.py`.

## Browser support

Speech recognition (`webkitSpeechRecognition`) is Chrome/Edge only. TTS
(`speechSynthesis`) works broadly. Use Chrome for the full experience.

## Third engine: Omni (Qwen2.5-Omni) — experimental, currently blocked

`app.py` also defines `build_omni_app()` / `python app.py omni`: a single combined
**LLM+TTS** model — `Qwen/Qwen2.5-Omni-3B` served via **`vllm-omni`** — exposed in the
UI as a third "Omni" engine (`/api/omni`, one call returns text **and** speech). It
only appears in the UI when `OMNI_BASE_URL` is wired up.

**Status: does not serve yet.** It builds, schedules, loads the model, and Stage 0
(the thinker/LLM) initializes — but vllm-omni runs each stage as a separate engine on
one GPU and **Stage 1 (the talker/speech stage) crashes during init** (low-level kill,
no recoverable error), even on a 48 GB L40S with a shared `--gpu-memory-utilization`.
This is a bleeding-edge `vllm-omni` limitation, not a problem with the Flyte wiring.

If you pursue it, known requirements baked into `build_omni_app()`:
- `--torch-backend=cu130` (not `auto` — the GPU-less remote builder otherwise installs CPU torch).
- `bash -lc` + `export PATH=/opt/venv/bin:/usr/local/bin` (custom `command` runs off-PATH).
- `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_USE_DEEP_GEMM=0` (those JIT-compile and need an
  nvcc/CUDA toolkit absent from the runtime image — a CUDA-devel base image would be the
  proper fix and may also unblock the talker stage).
- `--gpu-memory-utilization ~0.45` so the two stages share the GPU; needs an L40S (48 GB).
