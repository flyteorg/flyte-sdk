#!/usr/bin/env python3
"""
Microphone voice-chat client for the Nemotron-3-Nano-Omni Flyte app.

Talk into your mic; the audio is sent straight to the Omni model (no local
speech-to-text), the text reply is streamed back token-by-token, and each
sentence is spoken as soon as it is complete -- so you hear the answer begin
while the model is still generating the rest of it.

Latency-oriented design choices
--------------------------------
* End-of-utterance is detected with simple energy-based VAD, so a turn is sent
  the moment you stop talking instead of on a fixed timer.
* The HTTP response is consumed as a token stream (``stream=true``); we do not
  wait for the full completion.
* ``enable_thinking=false`` is sent so the model does not emit <think> reasoning
  traces -- those are great for hard problems but ruinous for conversation latency.
* TTS runs on a worker thread and is driven sentence-by-sentence, overlapping
  speech synthesis with ongoing token generation.

Usage
-----
    pip install sounddevice numpy httpx        # plus an optional TTS engine below
    python voice_client.py --endpoint https://<your-app-url> --model nemotron-omni

TTS engine: on macOS the built-in ``say`` command is used (zero install). On
Linux/Windows install ``pip install pyttsx3``. Use ``--no-speak`` to just print.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import queue
import re
import subprocess
import sys
import threading
import wave

import httpx
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000  # Omni accepts 8 kHz+; 16 kHz is a good speech default.
CHANNELS = 1
BLOCK_MS = 30  # VAD granularity.

SYSTEM_PROMPT = (
    "You are a friendly, concise voice assistant. Keep answers short and "
    "conversational since they will be spoken aloud. Do not use markdown or emoji."
)


# ---------------------------------------------------------------------------
# Microphone capture with energy-based voice-activity detection
# ---------------------------------------------------------------------------


def record_utterance(
    silence_secs: float = 0.8,
    start_timeout: float = 8.0,
    threshold: float = 0.012,
) -> bytes | None:
    """Record from the mic until ~`silence_secs` of silence follows speech.

    Returns 16-bit PCM mono WAV bytes, or None if nothing was said.
    """
    block = int(SAMPLE_RATE * BLOCK_MS / 1000)
    silence_blocks = int(silence_secs * 1000 / BLOCK_MS)
    start_blocks = int(start_timeout * 1000 / BLOCK_MS)

    frames: list[np.ndarray] = []
    started = False
    trailing_silence = 0
    waited = 0

    print("🎙️  Listening… (speak, then pause)")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32", blocksize=block) as stream:
        while True:
            chunk, _ = stream.read(block)
            chunk = chunk[:, 0]
            rms = float(np.sqrt(np.mean(chunk**2)) + 1e-9)
            voiced = rms > threshold

            if voiced:
                started = True
                trailing_silence = 0
            elif started:
                trailing_silence += 1

            if started:
                frames.append(chunk.copy())
                if trailing_silence >= silence_blocks:
                    break
            else:
                waited += 1
                if waited >= start_blocks:
                    return None  # nobody spoke

    audio = np.concatenate(frames) if frames else np.zeros(0, dtype="float32")
    if audio.size < SAMPLE_RATE // 4:  # < 0.25 s of speech -> ignore
        return None

    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def wav_to_data_url(wav_bytes: bytes) -> str:
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


# ---------------------------------------------------------------------------
# Text-to-speech worker (sentence-by-sentence, off the main thread)
# ---------------------------------------------------------------------------


class Speaker:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._q: queue.Queue[str | None] = queue.Queue()
        self._engine = None
        if enabled and sys.platform != "darwin":
            try:
                import pyttsx3

                self._engine = pyttsx3.init()
            except Exception:
                print("(pyttsx3 not available; install it or use --no-speak; printing only)")
                self.enabled = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            text = self._q.get()
            if text is None:
                return
            if not self.enabled or not text.strip():
                continue
            try:
                if sys.platform == "darwin":
                    subprocess.run(["say", text], check=False)
                elif self._engine is not None:
                    self._engine.say(text)
                    self._engine.runAndWait()
            except Exception as e:
                print(f"(tts error: {e})")

    def say(self, text: str) -> None:
        self._q.put(text)


_SENTENCE_END = re.compile(r"(.+?[.!?…]+[\s\"')\]]*)", re.DOTALL)


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Pull complete sentences out of `buffer`; return (sentences, remainder)."""
    sentences = []
    pos = 0
    for m in _SENTENCE_END.finditer(buffer):
        sentences.append(m.group(1).strip())
        pos = m.end()
    return sentences, buffer[pos:]


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


# ---------------------------------------------------------------------------
# Streaming chat call
# ---------------------------------------------------------------------------


def stream_reply(
    client: httpx.Client,
    base_url: str,
    model: str,
    messages: list[dict],
    api_key: str | None,
    speaker: Speaker,
) -> str:
    """POST a streaming chat completion, speak sentences as they arrive, return full text."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": 512,
        "temperature": 0.2,
        # vLLM passthrough: greedy-ish + reasoning OFF for low latency.
        "top_k": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    full = ""
    pending = ""  # un-spoken tail
    print("🤖 ", end="", flush=True)
    with client.stream("POST", f"{base_url}/chat/completions", json=payload, headers=headers, timeout=120.0) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0]["delta"].get("content") or ""
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if not delta:
                continue
            full += delta
            print(delta, end="", flush=True)

            pending += delta
            sentences, pending = split_sentences(pending)
            for s in sentences:
                clean = _THINK.sub("", s).strip()
                if clean:
                    speaker.say(clean)

    print()
    tail = _THINK.sub("", pending).strip()
    if tail:
        speaker.say(tail)
    return _THINK.sub("", full).strip()


# ---------------------------------------------------------------------------
# Main conversation loop
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Mic voice chat for Nemotron-Omni on Flyte")
    ap.add_argument("--endpoint", required=True, help="App URL, e.g. https://<app>.flyte... (no /v1)")
    ap.add_argument("--model", default="nemotron-omni", help="Served model id")
    ap.add_argument("--api-key", default=None, help="Bearer token if the app requires auth")
    ap.add_argument("--no-speak", action="store_true", help="Print replies instead of speaking them")
    ap.add_argument("--silence", type=float, default=0.8, help="Seconds of trailing silence to end a turn")
    args = ap.parse_args()

    base_url = args.endpoint.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"

    speaker = Speaker(enabled=not args.no_speak)
    client = httpx.Client()

    # We keep prior *assistant* turns as text context. User turns are audio and
    # are not re-sent each round (re-uploading old audio would balloon latency
    # and token cost); the running text history preserves enough context.
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Voice chat ready. Ctrl-C to quit.\n")
    try:
        while True:
            wav = record_utterance(silence_secs=args.silence)
            if wav is None:
                continue

            user_msg = {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": wav_to_data_url(wav)}},
                    {"type": "text", "text": "Respond to what I just said."},
                ],
            }
            messages = [*history, user_msg]

            reply = stream_reply(client, base_url, args.model, messages, args.api_key, speaker)

            # Keep a lightweight text-only record of this turn for context.
            history.append({"role": "user", "content": "[spoken audio]"})
            history.append({"role": "assistant", "content": reply})
            # Cap history so context stays small and latency stays low.
            if len(history) > 13:  # system + 6 turns
                history = [history[0], *history[-12:]]
    except KeyboardInterrupt:
        print("\n👋 Bye.")


if __name__ == "__main__":
    main()
