"""
Voice customer-service agent — talk in the browser, it talks back.

A two-app Flyte demo:

  * ``llm_app``  — a small, fast Qwen instruct model served with vLLM on an L4
                   GPU (OpenAI-compatible API). This is the "brain".
  * ``ui_app``   — a tiny FastAPI app that serves a single-page voice UI and
                   proxies chat requests to ``llm_app``.

Speech-to-text and text-to-speech happen **in the browser** via the Web Speech
API, so there is no audio model to host: the mic is transcribed locally, the
text goes to the LLM, and the reply is spoken locally. That keeps latency low
and the GPU footprint tiny (a 3B model on one L4).

    🎤 browser STT ──► /api/chat (FastAPI proxy) ──► vLLM /v1 (Qwen on L4)
                                                          │ streamed tokens
    🔊 browser TTS ◄── streamed text ◄────────────────────┘

The UI is served over HTTPS from the Flyte app, which is what lets the browser
grant microphone access and use speech recognition (both require a secure
context). The proxy means the browser only ever talks to its own origin, so
there are no CORS headaches.

Deploy (against the demo cluster)
---------------------------------
    # 1. Bring up the GPU model server (long pole: provisions an L4 + pulls weights)
    python app.py llm

    # 2. Bring up the voice UI, pointed at the LLM from step 1
    python app.py ui --llm-url <llm-app-url>

Then open the printed UI url in Chrome and click the mic.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import time

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse

import flyte
import flyte.app
from flyte.app.extras import FastAPIAppEnvironment

# NOTE: `flyteplugins.vllm` is imported lazily inside build_llm_app() rather than
# at module top. This module is loaded by BOTH app containers; the lightweight UI
# image does not install flyteplugins-vllm, so a top-level import would crash the
# UI app on startup.

# ---------------------------------------------------------------------------
# 1. The LLM: small, fast Qwen instruct model on vLLM / L4
#
# Qwen2.5-3B-Instruct is a good "quality is OK, latency matters" pick: ~6 GB in
# bf16, trivially fits a 24 GB L4, and decodes fast enough that the browser's
# TTS is the pacing factor, not the model. vLLM downloads the weights straight
# from the Hugging Face hub (the model is public — no token needed).
# ---------------------------------------------------------------------------

MODEL_ID = "qwen"

# Pin the serving image. The plugin's default image pins vllm==0.11.0 but not
# transformers, and the newest transformers breaks vllm 0.11's tokenizer caching
# (AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended).
# transformers==4.57.6 is the version the repo's own vLLM example uses.
vllm_image = (
    flyte.Image.from_debian_base(name="vllm-app-image", install_flyte=False)
    .with_pip_packages("flashinfer-python", "flashinfer-cubin")
    .with_pip_packages("flashinfer-jit-cache", index_url="https://flashinfer.ai/whl/cu129")
    .with_pip_packages("flyteplugins-vllm")
    .with_pip_packages("vllm==0.11.0", "transformers==4.57.6")
)

# `llm_app` MUST be a module-level variable: the default fserve entrypoint resolves
# the app by `module:attr`, so a function-local instance would be unresolvable and the
# resolver would silently fall back to `ui_app` (the vLLM pod would then run the UI app
# and 404 on /v1). We still keep the flyteplugins import guarded so the lightweight UI
# container — which imports this module but never installs flyteplugins-vllm — loads
# fine (it resolves to `ui_app`, leaving `llm_app` as None, which is never used there).
try:
    from flyteplugins.vllm import VLLMAppEnvironment

    llm_app = VLLMAppEnvironment(
        name="cs-qwen-llm",
        model_id=MODEL_ID,
        model_hf_path="Qwen/Qwen2.5-3B-Instruct",
        image=vllm_image,
        resources=flyte.Resources(cpu="6", memory="20Gi", gpu="L4:1", disk="40Gi"),
        # One warm replica so there's no cold start mid-demo. Flip to (0, 1) +
        # scaledown_after to save the GPU when idle, at the cost of a cold start.
        scaling=flyte.app.Scaling(replicas=(1, 1)),
        requires_auth=False,
        extra_args=[
            # Short context keeps the KV cache small and latency low; a customer
            # service turn is tiny.
            "--max-model-len",
            "8192",
            "--max-num-seqs",
            "16",
        ],
    )
except ImportError:
    llm_app = None  # flyteplugins-vllm not installed (e.g. the UI container)

# ---------------------------------------------------------------------------
# 1b. The combined app: ONE model that does LLM + speech (Qwen2.5-Omni-3B)
#
# Qwen2.5-Omni uses a Thinker-Talker architecture: a single
# /v1/chat/completions call with "modalities": ["audio"] returns BOTH the text
# reply and synthesized speech. Served by vllm-omni (a separate vLLM project that
# adds omni-modality output) — NOT the flyteplugins-vllm plugin, which pins an
# older vLLM without omni support. We run the OpenAI server via a custom
# container `command`, which bypasses Flyte's default fserve entrypoint.
# ---------------------------------------------------------------------------

OMNI_HF_MODEL = "Qwen/Qwen2.5-Omni-3B"
OMNI_MODEL_ID = "omni"

# vllm-omni installs from source on top of vLLM 0.23.0 (see its quickstart).
# CRITICAL: pin --torch-backend=cu130 (NOT auto). The remote image builder has no
# GPU, so `auto` resolves to CPU torch (torch+cpu) and vllm._C then fails with
# `libcudart.so.13: cannot open shared object file`. The demo L4 nodes run driver
# 580 / CUDA 13, so cu130 is the right GPU build. No separate flashinfer (the old
# cu129 wheels are CUDA 12.9 and conflict with the CUDA-13 stack).
omni_image = (
    flyte.Image.from_debian_base(name="vllm-omni-server", install_flyte=False)
    .with_apt_packages("git")
    .with_commands(
        [
            "uv pip install --system vllm==0.23.0 --torch-backend=cu130",
            "git clone https://github.com/vllm-project/vllm-omni.git /opt/vllm-omni",
            "uv pip install --system -e /opt/vllm-omni",
        ]
    )
)


def build_omni_app():
    """A single model that returns text + speech (Qwen2.5-Omni-3B via vllm-omni)."""
    return flyte.app.AppEnvironment(
        name="cs-omni",
        image=omni_image,
        # Raw vllm OpenAI server with omni audio output enabled.
        # vllm-omni runs each stage (thinker + talker) as a SEPARATE engine on the
        # SAME GPU, and each applies --gpu-memory-utilization to the whole device. So
        # the stages must share: 0.45 each (~0.90 total) leaves room for both. The
        # thinker model alone is ~8.8 GB, so the 24 GB L4 is too tight for two stages
        # with usable KV cache — the 48 GB L40S fits both comfortably.
        command=[
            "bash",
            "-lc",
            "export PATH=/opt/venv/bin:/usr/local/bin:$PATH; "
            f"exec vllm serve {OMNI_HF_MODEL} --omni --trust-remote-code "
            f"--served-model-name {OMNI_MODEL_ID} --port 8080 "
            "--gpu-memory-utilization 0.45 --max-model-len 8192",
        ],
        # This runtime image has the CUDA *runtime* libs (from torch) but no CUDA
        # *toolkit* (nvcc / CUDA_HOME). Several vLLM kernels JIT-compile at startup and
        # assert a toolkit is present, killing the engine core. Disable those so they
        # use prebuilt/native paths: the flashinfer sampler and deep_gemm. (The crash
        # was never RAM/GPU size — L4 and L40S failed identically — so we use the L4.)
        env_vars={"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_DEEP_GEMM": "0"},
        port=8080,
        # L40S (g6e.12xlarge): 48 GB GPU fits both omni stages; big node so cpu/mem/disk
        # requests schedule freely. (Earlier L40S attempt failed only at the now-fixed
        # flashinfer error, before reaching this two-stage memory split.)
        resources=flyte.Resources(cpu="12", memory="48Gi", gpu="L40s:1", disk="60Gi"),
        requires_auth=False,
        scaling=flyte.app.Scaling(replicas=(1, 1)),
    )


# ---------------------------------------------------------------------------
# 2. The voice UI: FastAPI serving the page + proxying to the LLM
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are Ava, a warm, efficient customer-support agent for 'Northwind', a "
    "consumer electronics company. Your replies are spoken aloud in a live phone-"
    "like call, so keep them very short (1-2 sentences), natural, and free of "
    "markdown, lists, or emoji. Get to the point in the first sentence. Ask one "
    "clarifying question at a time. The caller may interrupt you at any moment; if "
    "they do, stop and listen. If you don't know an account-specific detail, say "
    "you'll look into it rather than inventing facts."
)

# The LLM endpoint is injected at deploy time (see __main__) via this env var.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")

# TTS configuration.
#   TTS_MODE:  "both" (show the in-UI switch) | "browser" | "server" (lock one mode)
#   TTS_VOICE: a Kokoro voice id; af_heart is a warm female voice that fits "Ava".
TTS_MODE = os.environ.get("TTS_MODE", "both")
TTS_VOICE = os.environ.get("TTS_VOICE", "af_heart")

# Omni (combined LLM+TTS) backend — Qwen2.5-Omni via vllm-omni. Injected at
# deploy time; when set, the UI exposes an "Omni" engine that does chat+speech in
# one call. OMNI_SAMPLE_RATE is used only if the model returns raw PCM (no header).
OMNI_BASE_URL = os.environ.get("OMNI_BASE_URL", "")
OMNI_MODEL_ID = os.environ.get("OMNI_MODEL_ID", "omni")
OMNI_SAMPLE_RATE = int(os.environ.get("OMNI_SAMPLE_RATE", "24000"))

# Kokoro is loaded lazily/once at startup (heavy torch import) and only when the
# server-side TTS path is enabled. Stored on app state so requests reuse it.
_tts_state: dict = {"pipeline": None, "error": None}

# Kokoro synthesis is CPU-bound; running several at once just thrashes the cores
# and makes each one slower. Serialize so every clause stays fast (~0.5s) even if
# the client's prefetch ever overlaps two requests.
_synth_sem = asyncio.Semaphore(1)


def _load_kokoro():
    """Build the Kokoro pipeline once and warm it. Returns the pipeline or raises."""
    from kokoro import KPipeline  # heavy (torch); imported only when serving TTS

    pipeline = KPipeline(lang_code="a")  # 'a' = American English
    # Warm-up: the first synth compiles/caches; do it now so real calls are fast.
    for _ in pipeline("Hello.", voice=TTS_VOICE):
        pass
    return pipeline


def _synth(text: str):
    """Run Kokoro and return concatenated 24 kHz float32 audio (numpy)."""
    import numpy as np

    pipeline = _tts_state["pipeline"]
    chunks = [audio for _, _, audio in pipeline(text, voice=TTS_VOICE)]
    if not chunks:
        return np.zeros(1, dtype="float32")
    return np.concatenate(chunks).astype("float32")


def _wav_bytes(audio, sr: int = 24000) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


fastapi_app = FastAPI(title="Northwind Voice Support")


@fastapi_app.on_event("startup")
async def _startup():
    # Load Kokoro unless TTS is browser-only (then we skip the heavy import).
    if TTS_MODE == "browser":
        return
    try:
        _tts_state["pipeline"] = await asyncio.to_thread(_load_kokoro)
    except Exception as e:  # keep the app up; server-TTS just stays unavailable
        _tts_state["error"] = f"{type(e).__name__}: {e}"


@fastapi_app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "llm": LLM_BASE_URL or "unset",
        "tts_mode": TTS_MODE,
        "tts_ready": _tts_state["pipeline"] is not None,
        "tts_error": _tts_state["error"],
        "omni": OMNI_BASE_URL or "unset",
    }


@fastapi_app.get("/api/config")
async def config():
    """Tells the browser which TTS modes / engines are available."""
    return {
        "tts_mode": TTS_MODE,
        "tts_ready": _tts_state["pipeline"] is not None,
        "omni_ready": bool(OMNI_BASE_URL),
    }


@fastapi_app.post("/api/tts")
async def tts(req: Request):
    """Synthesize speech for one clause with Kokoro; returns a 24 kHz WAV.

    The X-Synth-Ms response header carries the measured server-side synthesis
    time so the client can display/compare latency.
    """
    body = await req.json()
    text = (body.get("text") or "").strip()
    if not text:
        return Response(status_code=204)
    if _tts_state["pipeline"] is None:
        return Response(status_code=503, content=_tts_state["error"] or "TTS not ready")

    t0 = time.perf_counter()
    async with _synth_sem:
        audio = await asyncio.to_thread(_synth, text)
    wav = await asyncio.to_thread(_wav_bytes, audio)
    synth_ms = int((time.perf_counter() - t0) * 1000)
    return Response(content=wav, media_type="audio/wav", headers={"X-Synth-Ms": str(synth_ms)})


@fastapi_app.post("/api/chat")
async def chat(req: Request):
    """Proxy a chat turn to the vLLM backend and stream the text reply back."""
    body = await req.json()
    history = body.get("messages", [])

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *history],
        "stream": True,
        "max_tokens": 200,
        "temperature": 0.3,
    }

    async def gen():
        base = os.environ.get("LLM_BASE_URL", LLM_BASE_URL).rstrip("/")
        url = f"{base}/v1/chat/completions"
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if delta:
                        yield delta

    return StreamingResponse(gen(), media_type="text/plain")


def _omni_extract(data: dict) -> tuple[str, bytes]:
    """Pull (reply_text, wav_bytes) out of a Qwen2.5-Omni chat-completion response.

    The omni audio field shape isn't fully documented, so be defensive: text is in
    choices[0]; audio is in some later choice's message.audio, as either a base64
    string or a dict with a base64 ``data`` field. If the decoded bytes are already
    a WAV (RIFF) we pass them through; otherwise we assume raw PCM16 and add a header.
    """
    choices = data.get("choices") or []
    text = ""
    audio_b64 = None
    for ch in choices:
        msg = ch.get("message") or {}
        if not text and msg.get("content"):
            text = msg["content"]
        aud = msg.get("audio")
        if aud is not None and audio_b64 is None:
            audio_b64 = aud.get("data") if isinstance(aud, dict) else aud
            if isinstance(aud, dict) and not text and aud.get("transcript"):
                text = aud["transcript"]
    if not audio_b64:
        raise ValueError("no audio in omni response")

    raw = base64.b64decode(audio_b64)
    if raw[:4] == b"RIFF":
        return text, raw  # already a WAV container
    # Raw PCM16 -> wrap in a WAV header at the configured sample rate.
    import numpy as np

    pcm = np.frombuffer(raw, dtype="<i2")
    return text, _wav_bytes(pcm, OMNI_SAMPLE_RATE)


@fastapi_app.post("/api/omni")
async def omni(req: Request):
    """Combined LLM+TTS via Qwen2.5-Omni: one call returns text + spoken audio.

    Returns a WAV body; the reply text rides in the X-Reply-Text header and the
    server round-trip time in X-Synth-Ms (for the latency comparison).
    """
    base = os.environ.get("OMNI_BASE_URL", OMNI_BASE_URL).rstrip("/")
    if not base:
        return Response(status_code=503, content="omni backend not configured")
    history = (await req.json()).get("messages", [])
    payload = {
        "model": OMNI_MODEL_ID,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *history],
        "modalities": ["text", "audio"],
        "max_tokens": 200,
        "temperature": 0.3,
    }
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{base}/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
    synth_ms = int((time.perf_counter() - t0) * 1000)
    try:
        text, wav = _omni_extract(data)
    except Exception as e:
        return Response(status_code=502, content=f"omni parse error: {e}")
    # Header values must be latin-1 safe; keep the transcript ASCII-ish.
    safe_text = text.encode("ascii", "replace").decode("ascii")[:800]
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"X-Synth-Ms": str(synth_ms), "X-Reply-Text": safe_text},
    )


@fastapi_app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


# The UI image bundles the FastAPI server AND Kokoro TTS (CPU torch + espeak-ng).
# Kokoro is tiny (82M) and runs comfortably on CPU; no GPU on this app.
ui_image = (
    flyte.Image.from_debian_base(name="cs-voice-ui")
    .with_apt_packages("espeak-ng")  # Kokoro's grapheme->phoneme runtime dep
    # CPU torch wheel keeps the image far smaller than the default CUDA build.
    .with_pip_packages("torch", index_url="https://download.pytorch.org/whl/cpu")
    .with_pip_packages("fastapi", "uvicorn", "httpx", "kokoro>=0.9.2", "soundfile", "numpy")
)

ui_app = FastAPIAppEnvironment(
    name="cs-voice-ui",
    app=fastapi_app,
    description="Browser voice UI for the Qwen customer-service agent (browser + Kokoro TTS)",
    image=ui_image,
    # Bumped for torch + the Kokoro model living in memory.
    resources=flyte.Resources(cpu="6", memory="8Gi"),
    requires_auth=False,
    scaling=flyte.app.Scaling(replicas=(1, 1)),
)


# ---------------------------------------------------------------------------
# Single-page voice UI (Web Speech API: SpeechRecognition + speechSynthesis)
# ---------------------------------------------------------------------------

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Northwind Voice Support</title>
<style>
  :root { --purple:#7652a2; --bg:#f7f5fd; --ink:#171020; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background:var(--bg); color:var(--ink); height:100vh; display:flex; flex-direction:column; }
  header { padding:18px 24px; background:var(--purple); color:#fff; font-weight:600; font-size:18px; }
  #log { flex:1; overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:12px; }
  .msg { max-width:75%; padding:12px 16px; border-radius:16px; line-height:1.4; white-space:pre-wrap; }
  .user { align-self:flex-end; background:var(--purple); color:#fff; border-bottom-right-radius:4px; }
  .bot  { align-self:flex-start; background:#fff; border:1px solid #e6e0f2; border-bottom-left-radius:4px; }
  .interim { opacity:.5; font-style:italic; }
  footer { padding:18px; display:flex; flex-direction:column; align-items:center; gap:8px;
           border-top:1px solid #e6e0f2; background:#fff; }
  #mic { width:84px; height:84px; border-radius:50%; border:none; cursor:pointer; font-size:34px;
         background:var(--purple); color:#fff; transition:transform .1s, background .2s; }
  #mic:active { transform:scale(.94); }
  #mic.listening { background:#d4366b; animation:pulse 1.2s infinite; }
  @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(212,54,107,.5);} 70%{box-shadow:0 0 0 18px rgba(212,54,107,0);} }
  #status { font-size:13px; color:#6b6480; min-height:18px; }
  #hint { font-size:12px; color:#9a93ad; }
  #unsupported { color:#b00; padding:20px; }
  /* orb reflects state: idle / listening / thinking / speaking */
  #orb { width:84px; height:84px; border-radius:50%; border:none; cursor:pointer; font-size:30px;
         color:#fff; background:#b8b0cc; transition:transform .1s, background .25s; }
  #orb:active { transform:scale(.94); }
  #orb.listening { background:#7652a2; animation:pulse 1.4s infinite; }
  #orb.thinking  { background:#e0972b; }
  #orb.speaking  { background:#2f9e6f; animation:pulse 1.0s infinite; }
  @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(118,82,162,.45);} 70%{box-shadow:0 0 0 18px rgba(118,82,162,0);} }
  #controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap; justify-content:center; }
  .seg-label { font-size:12px; color:#6b6480; }
  .seg { display:inline-flex; border:1px solid #d8d0ea; border-radius:999px; overflow:hidden; }
  .seg button { border:none; background:#fff; color:#6b6480; padding:6px 13px; font-size:12px; cursor:pointer; }
  .seg button.on { background:#7652a2; color:#fff; }
  .seg button:disabled { opacity:.5; cursor:not-allowed; }
  #metrics { font-size:12px; color:#564f6b; font-variant-numeric:tabular-nums; min-height:16px; text-align:center; }
</style>
</head>
<body>
<header>🎧 Northwind Voice Support — talk to Ava</header>
<div id="log"></div>
<footer>
  <button id="orb" title="Start / end call">📞</button>
  <div id="status">Press the button to start the call</div>
  <div id="controls">
    <span class="seg-label">TTS engine:</span>
    <span class="seg" id="ttsSeg">
      <button id="ttsBrowser" class="on">Browser (instant)</button>
      <button id="ttsServer">Server · Kokoro</button>
      <button id="ttsOmni">Omni · Qwen (1 model)</button>
    </span>
  </div>
  <div id="metrics">Latency will appear here after the first reply.</div>
  <div id="hint">Tip: use headphones so you can interrupt Ava without echo.</div>
</footer>
<div id="unsupported" hidden>
  Your browser doesn't support the Web Speech API. Please use Google Chrome.
</div>

<script>
// ===========================================================================
// Always-listening, barge-in voice client.
//
//  * SpeechRecognition runs CONTINUOUSLY (auto-restarts) so you never have to
//    click to talk — and so you can speak *while* Ava is talking.
//  * A Web Audio energy meter (with echo cancellation) detects when YOU are
//    actually speaking. The moment it does while Ava is talking, we BARGE IN:
//    cancel her TTS and abort the in-flight generation instantly.
//  * The meter also gates which transcripts count as real user turns, so Ava's
//    own voice (leaking into the mic on speakers) doesn't talk to itself.
//  * Replies are spoken at clause boundaries (commas too), so the first words
//    come out fast instead of waiting for a whole sentence.
// ===========================================================================

// --- Tunables (raise if Ava interrupts herself on speakers) ----------------
const SPEECH_RMS = 0.025;   // mic energy that counts as "the user is speaking"
const BARGE_RMS  = 0.045;   // louder energy that triggers an interruption
const MIN_SPEAK_CHARS = 12; // don't speak a clause shorter than this

const log = document.getElementById('log');
const orb = document.getElementById('orb');
const statusEl = document.getElementById('status');
const metricsEl = document.getElementById('metrics');
const btnBrowser = document.getElementById('ttsBrowser');
const btnServer = document.getElementById('ttsServer');
const btnOmni = document.getElementById('ttsOmni');

const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
if (!SR) { document.getElementById('unsupported').hidden = false; orb.disabled = true; }

let messages = [];        // conversation history sent to the LLM
let sessionOn = false;
let rec = null;
let controller = null;    // AbortController for the in-flight /api/chat fetch
let generating = false;   // a generation is streaming
let pending = 0;          // unfinished BROWSER TTS utterances
let userSpoke = false;    // mic energy crossed SPEECH_RMS since the last turn
let interimEl = null;
let loud = 0;
let audioCtx = null;      // shared (unlocked on Start) for meter + server audio

let omniReady = false;   // is the combined Qwen2.5-Omni backend wired up?
// TTS mode: 'browser' | 'server' | 'omni'. ready flags/availableMode from /api/config.
let ttsMode = 'browser';
let availableMode = 'both';
let serverReady = false;

// Server-TTS audio pipeline -------------------------------------------------
let ttsGen = 0;           // bumped on barge-in to invalidate in-flight TTS work
let ttsQueue = [];        // pending clause texts to synthesize
let ttsRunning = false;   // the consumer loop is active
let ttsPlaying = false;   // a server audio buffer is currently playing
let currentSource = null; // the playing AudioBufferSourceNode
const ttsControllers = new Set();  // in-flight /api/tts fetch aborts

// Per-turn latency measurement ----------------------------------------------
let turn = null;          // {sentAt, firstTokenAt, firstClauseAt, firstAudioAt}
const stats = { browser: [], server: [] };

function isAiActive() {
  return generating || pending > 0 || speechSynthesis.speaking ||
         ttsPlaying || ttsQueue.length > 0 || ttsRunning;
}

function setState(s) {
  orb.classList.remove('listening', 'thinking', 'speaking');
  if (s) orb.classList.add(s);
  statusEl.textContent =
    s === 'listening' ? 'Listening… (just start talking; interrupt anytime)' :
    s === 'thinking'  ? 'Ava is thinking…' :
    s === 'speaking'  ? 'Ava is speaking… (talk over her to interrupt)' :
    'Press the button to start the call';
}

function bubble(cls, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}
function showInterim(t) {
  if (!t) { if (interimEl) { interimEl.remove(); interimEl = null; } return; }
  if (!interimEl) interimEl = bubble('user interim', '');
  interimEl.textContent = t;
  log.scrollTop = log.scrollHeight;
}
function clearInterim() { if (interimEl) { interimEl.remove(); interimEl = null; } }

// --- Perf: record time-to-first-audio + show a rolling comparison ----------
function markFirstAudio(mode, synthMs) {
  if (!turn || turn.firstAudioAt) return;
  turn.firstAudioAt = performance.now();
  const ttfa = Math.round(turn.firstAudioAt - (turn.firstClauseAt || turn.sentAt));
  const llm = turn.firstTokenAt ? Math.round(turn.firstTokenAt - turn.sentAt) : null;
  const arr = stats[mode]; arr.push(ttfa); if (arr.length > 10) arr.shift();
  const avg = Math.round(arr.reduce((a, b) => a + b, 0) / arr.length);
  const label = mode === 'server' ? 'Server · Kokoro' : 'Browser';
  let s = label + ' — first audio ' + ttfa + 'ms (avg ' + avg + 'ms)';
  if (llm != null) s += ' · LLM first token ' + llm + 'ms';
  if (synthMs != null) s += ' · synth ' + synthMs + 'ms';
  metricsEl.textContent = s;
  console.log('[perf] ' + s);
}

// --- Speech: route each clause to the selected engine ----------------------
function enqueueSpeech(text) {
  if (!text.trim()) return;
  if (turn && !turn.firstClauseAt) turn.firstClauseAt = performance.now();
  if (ttsMode === 'server' && serverReady) ttsEnqueueServer(text);
  else browserSpeak(text);
}

// Browser TTS (speechSynthesis) ---------------------------------------------
let voice = null;
function pickVoice() {
  const vs = speechSynthesis.getVoices();
  voice = vs.find(v => /en[-_]US/i.test(v.lang) && /female|samantha|zira|aria|jenny/i.test(v.name))
       || vs.find(v => /en[-_]US/i.test(v.lang)) || vs[0] || null;
}
speechSynthesis.onvoiceschanged = pickVoice; pickVoice();
function browserSpeak(text) {
  if (!text.trim()) return;
  const u = new SpeechSynthesisUtterance(text);
  u.lang = 'en-US'; u.rate = 1.06; if (voice) u.voice = voice;
  pending++;
  u.onstart = () => { setState('speaking'); markFirstAudio('browser', null); };
  u.onend = u.onerror = () => {
    pending = Math.max(0, pending - 1);
    if (!isAiActive() && sessionOn) setState('listening');
  };
  speechSynthesis.speak(u);
}

// Server TTS (Kokoro) — fetch+decode per clause, play gaplessly, prefetch ----
function ttsEnqueueServer(text) {
  ttsQueue.push(text);
  if (!ttsRunning) ttsRun();
}
async function ttsFetchDecode(text, gen) {
  const ctrl = new AbortController(); ttsControllers.add(ctrl);
  try {
    const res = await fetch('/api/tts', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }), signal: ctrl.signal,
    });
    if (!res.ok) throw new Error('tts ' + res.status);
    const synthMs = parseInt(res.headers.get('X-Synth-Ms') || '0', 10);
    const arr = await res.arrayBuffer();
    if (gen !== ttsGen) return null;
    const buffer = await audioCtx.decodeAudioData(arr);
    return { buffer, synthMs };
  } catch (e) {
    if (e.name !== 'AbortError') { browserSpeak(text); }  // never go mute
    return null;
  } finally { ttsControllers.delete(ctrl); }
}
function ttsPlay(item, gen, mode) {
  return new Promise((resolve) => {
    if (gen !== ttsGen) return resolve();
    const src = audioCtx.createBufferSource();
    src.buffer = item.buffer; src.connect(audioCtx.destination);
    currentSource = src; ttsPlaying = true;
    setState('speaking');
    markFirstAudio(mode || 'server', item.synthMs);
    src.onended = () => {
      if (gen === ttsGen) {
        ttsPlaying = false; currentSource = null;
        if (!isAiActive() && sessionOn) setState('listening');
      }
      resolve();
    };
    src.start();
  });
}
async function ttsRun() {
  ttsRunning = true;
  const gen = ttsGen;
  let prefetch = null;   // Promise of the next decoded clause
  while (sessionOn && gen === ttsGen && (ttsQueue.length || prefetch)) {
    let item = prefetch ? await prefetch : await ttsFetchDecode(ttsQueue.shift(), gen);
    prefetch = null;
    if (gen !== ttsGen) break;
    if (ttsQueue.length) prefetch = ttsFetchDecode(ttsQueue.shift(), gen);  // overlap with playback
    if (item) await ttsPlay(item, gen);
  }
  ttsRunning = false;
  if (!isAiActive() && sessionOn) setState('listening');
}

// --- Barge-in: stop whichever engine is talking ----------------------------
function bargeIn() {
  if (!isAiActive()) return;
  if (controller) controller.abort();           // stop the LLM stream
  ttsGen++;                                      // invalidate all server-TTS work
  ttsQueue = []; ttsRunning = false;
  for (const c of ttsControllers) c.abort(); ttsControllers.clear();
  if (currentSource) { try { currentSource.stop(); } catch (e) {} currentSource = null; }
  ttsPlaying = false;
  speechSynthesis.cancel();                      // stop browser TTS
  pending = 0; generating = false;
  setState('listening');
}

// --- Mic energy meter (echo-cancelled) -------------------------------------
async function initMeter() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === 'suspended') await audioCtx.resume();
  const src = audioCtx.createMediaStreamSource(stream);
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  const buf = new Float32Array(analyser.fftSize);
  src.connect(analyser);
  (function loop() {
    analyser.getFloatTimeDomainData(buf);
    let s = 0; for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
    const rms = Math.sqrt(s / buf.length);
    if (rms > SPEECH_RMS) userSpoke = true;
    if (isAiActive() && rms > BARGE_RMS) { if (++loud >= 2) bargeIn(); } else { loud = 0; }
    requestAnimationFrame(loop);
  })();
}

// --- Continuous speech recognition -----------------------------------------
function initRec() {
  rec = new SR();
  rec.lang = 'en-US';
  rec.interimResults = true;
  rec.continuous = true;
  rec.onerror = (e) => {
    if (e.error !== 'no-speech' && e.error !== 'aborted') statusEl.textContent = 'Mic: ' + e.error;
  };
  rec.onend = () => { if (sessionOn) { try { rec.start(); } catch (e) {} } };
  rec.onresult = (e) => {
    let interim = '', finalText = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const r = e.results[i];
      if (r.isFinal) finalText += r[0].transcript; else interim += r[0].transcript;
    }
    // Interrupt as soon as the user clearly starts talking over Ava.
    if (interim.trim() && userSpoke && isAiActive()) bargeIn();
    showInterim(interim);
    if (finalText.trim()) {
      clearInterim();
      if (userSpoke) { userSpoke = false; handleUser(finalText.trim()); }
      // else: likely Ava's own voice echoed into the mic — ignore it.
    }
  };
  try { rec.start(); } catch (e) {}
}

// --- One conversational turn ------------------------------------------------
async function handleUser(text) {
  if (!text) return;
  bargeIn();                       // cut off anything still playing
  bubble('user', text);
  messages.push({ role: 'user', content: text });
  setState('thinking');

  turn = { sentAt: performance.now(), firstTokenAt: null, firstClauseAt: null, firstAudioAt: null };
  controller = new AbortController();
  generating = true;
  const div = bubble('bot', '');
  try {
    if (ttsMode === 'omni' && omniReady) await omniTurn(div);
    else await streamTurn(div);
  } catch (err) {
    if (err.name !== 'AbortError') div.textContent = '⚠️ ' + err;
  } finally {
    generating = false;
    if (!isAiActive() && sessionOn) setState('listening');
  }
}

// Streaming path: LLM text streams in, spoken clause-by-clause (browser or Kokoro).
async function streamTurn(div) {
  let full = '', spoken = 0;
  const res = await fetch('/api/chat', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }), signal: controller.signal,
  });
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!turn.firstTokenAt) turn.firstTokenAt = performance.now();   // LLM first-token latency
    full += dec.decode(value, { stream: true });
    div.textContent = full;
    log.scrollTop = log.scrollHeight;
    const tail = full.slice(spoken);
    const m = [...tail.matchAll(/[^.!?,;:]*[.!?,;:]+/g)];
    if (m.length) {
      const last = m[m.length - 1];
      const upto = last.index + last[0].length;
      if (upto >= MIN_SPEAK_CHARS) { enqueueSpeech(tail.slice(0, upto)); spoken += upto; }
    }
  }
  if (spoken < full.length) enqueueSpeech(full.slice(spoken));
  if (full.trim()) messages.push({ role: 'assistant', content: full });
}

// Omni path: ONE model returns text + speech together; play the whole reply.
async function omniTurn(div) {
  const gen = ttsGen;
  const res = await fetch('/api/omni', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }), signal: controller.signal,
  });
  if (!res.ok) { div.textContent = '⚠️ omni ' + res.status; return; }
  const synthMs = parseInt(res.headers.get('X-Synth-Ms') || '0', 10);
  const replyText = res.headers.get('X-Reply-Text') || '(spoken reply)';
  div.textContent = replyText;
  if (replyText.trim() && replyText !== '(spoken reply)') messages.push({ role: 'assistant', content: replyText });
  const arr = await res.arrayBuffer();
  if (gen !== ttsGen) return;
  const buffer = await audioCtx.decodeAudioData(arr);
  await ttsPlay({ buffer, synthMs }, gen, 'omni');   // TTFA here = whole-reply (LLM+speech)
}

// --- TTS engine toggle ------------------------------------------------------
function setTtsMode(mode) {
  if (mode === 'server' && !serverReady) return;
  if (mode === 'omni' && !omniReady) return;
  ttsMode = mode;
  btnBrowser.classList.toggle('on', mode === 'browser');
  btnServer.classList.toggle('on', mode === 'server');
  btnOmni.classList.toggle('on', mode === 'omni');
}
btnBrowser.onclick = () => setTtsMode('browser');
btnServer.onclick = () => setTtsMode('server');
btnOmni.onclick = () => setTtsMode('omni');

async function loadConfig() {
  try {
    const cfg = await (await fetch('/api/config')).json();
    availableMode = cfg.tts_mode || 'both';
    serverReady = !!cfg.tts_ready;
    omniReady = !!cfg.omni_ready;
  } catch (e) { availableMode = 'browser'; serverReady = false; omniReady = false; }

  // The Omni engine only appears if a combined backend is wired up.
  if (!omniReady) btnOmni.style.display = 'none';

  // Apply forced modes / readiness to the toggle.
  if (availableMode === 'browser') {
    btnServer.style.display = 'none'; setTtsMode('browser');
  } else if (availableMode === 'server') {
    btnBrowser.style.display = 'none';
    if (serverReady) setTtsMode('server');
    else { btnServer.textContent = 'Server · Kokoro (loading…)'; pollServerReady(true); }
  } else {
    // both: default to Browser; enable Server once the model finishes loading.
    setTtsMode('browser');
    if (!serverReady) {
      btnServer.disabled = true;
      btnServer.textContent = 'Server · Kokoro (loading…)';
      pollServerReady(false);
    }
  }
}
async function pollServerReady(forceSelect) {
  for (let i = 0; i < 60 && !serverReady; i++) {
    await new Promise(r => setTimeout(r, 3000));
    try { serverReady = !!(await (await fetch('/api/config')).json()).tts_ready; } catch (e) {}
  }
  if (serverReady) {
    btnServer.disabled = false; btnServer.textContent = 'Server · Kokoro';
    if (forceSelect) setTtsMode('server');
  }
}
loadConfig();

// --- Session start/stop -----------------------------------------------------
orb.onclick = async () => {
  if (!SR) return;
  if (sessionOn) {                 // end call
    sessionOn = false;
    bargeIn();
    if (rec) rec.stop();
    orb.textContent = '📞';
    setState(null);
    return;
  }
  try {
    await initMeter();             // mic permission + echo-cancelled energy meter
    initRec();                     // start always-on recognition
    sessionOn = true;
    orb.textContent = '⏹';
    setState('listening');
  } catch (e) {
    statusEl.textContent = 'Mic permission needed: ' + e;
  }
};
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Deploy driver
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["llm", "omni", "ui"])
    parser.add_argument("--llm-url", default=os.environ.get("LLM_BASE_URL", ""))
    parser.add_argument("--omni-url", default=os.environ.get("OMNI_BASE_URL", ""))
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()

    # Use the hosted remote image builder (no local Docker needed).
    flyte.init_from_config(args.config, image_builder="remote")

    if args.target == "llm":
        if llm_app is None:
            sys.exit("flyteplugins-vllm not importable; run `uv pip install -e plugins/vllm --no-deps`")
        # GPU provisioning + image build + weight download can take a while.
        app = flyte.with_servecontext(activate_timeout=1800.0).serve(llm_app)
        print(f"LLM app: {app.url}")
    elif args.target == "omni":
        # vllm-omni builds from source + downloads a multimodal model — be patient.
        app = flyte.with_servecontext(activate_timeout=1800.0).serve(build_omni_app())
        print(f"Omni app: {app.url}")
    else:
        if not args.llm_url:
            sys.exit("--llm-url (or LLM_BASE_URL) is required for the ui target")
        # Bake the backend endpoints into the app's container env so the proxies can reach them.
        env = {**(ui_app.env_vars or {}), "LLM_BASE_URL": args.llm_url}
        if args.omni_url:
            env["OMNI_BASE_URL"] = args.omni_url
        ui_app.env_vars = env
        app = flyte.serve(ui_app)
        print(f"Voice UI: {app.url}")
