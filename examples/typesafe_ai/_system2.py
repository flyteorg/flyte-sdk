"""System 2 — open-ended LLMs (Qwen 3.8 27B / Claude Sonnet / Claude Opus).

System 2 is the "think hard, generate anything" model.  It is reached through a
Union-hosted model gateway using the demo virtual keys.  The gateway is
deployment-specific, so ``System2Client`` resolves a working base URL the first
time it is used by probing a candidate list (overridable with
``LLM_GATEWAY_BASE_URL``) and caching the first endpoint that answers.  Both the
OpenAI-compatible (``/v1/chat/completions``) and Anthropic (``/v1/messages``)
wire formats are supported.

Each call captures wall-clock latency and token usage (``prompt``/``completion``)
so the benchmark can compare throughput across the matrix.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx
from _config import (
    GATEWAY_BASE_URL_CANDIDATES,
    LLM_GATEWAY_BASE_URL_ENV,
    SYSTEM2_MAX_RETRIES,
    SYSTEM2_PROVIDERS,
    SYSTEM2_TIMEOUT_S,
)

# module-level cache: provider_key -> (base_url, api_style, model) that worked
_RESOLVED: dict[str, tuple[str, str, str]] = {}


@dataclass
class ChatResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    error: str | None = None


class System2Error(RuntimeError):
    pass


def _server_error(resp) -> str:
    """Best-effort extraction of a gateway's error message from an HTTP response."""
    try:
        err = resp.json().get("error", {})
        if isinstance(err, dict):
            return err.get("message") or str(err)
        return str(err)
    except Exception:
        return f"HTTP {resp.status_code}"


def _candidates():
    env = os.environ.get(LLM_GATEWAY_BASE_URL_ENV, "").strip().rstrip("/")
    if env:
        return [env, *[c for c in GATEWAY_BASE_URL_CANDIDATES if c != env]]
    return list(GATEWAY_BASE_URL_CANDIDATES)


def _probe_openai(base: str, model: str, key: str) -> bool:
    """Return True if {base} speaks the OpenAI-compatible HTTP API.

    Any HTTP response (2xx, 4xx, ...) means the gateway endpoint exists; per-model
    errors (e.g. a 4xx because a model is unavailable) are surfaced at call time.
    """
    try:
        r = httpx.post(
            f"{base}/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "max_tokens": 4, "messages": [{"role": "user", "content": "ping"}]},
            timeout=20.0,
        )
        return r.status_code < 500
    except Exception:
        return False


def _probe_anthropic(base: str, model: str, key: str) -> bool:
    try:
        r = httpx.post(
            f"{base}/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={"model": model, "max_tokens": 4, "messages": [{"role": "user", "content": "ping"}]},
            timeout=20.0,
        )
        return r.status_code < 500
    except Exception:
        return False


def _resolve(provider_key: str) -> tuple[str, str, str]:
    """Return (base_url, api_style, model) for a provider, resolving on first use."""
    if provider_key in _RESOLVED:
        return _RESOLVED[provider_key]
    cfg = SYSTEM2_PROVIDERS[provider_key]
    model = cfg["model"]
    api_style = cfg["api_style"]
    key = os.environ.get(cfg["env_var"], "")
    if not key:
        raise System2Error(f"System 2 provider {provider_key}: API key not mounted")

    if os.environ.get(LLM_GATEWAY_BASE_URL_ENV):
        base = _candidates()[0]
        _RESOLVED[provider_key] = (base, api_style, model)
        return _RESOLVED[provider_key]

    # Probe candidates; prefer the matching wire format, fall back to the other.
    probe = _probe_openai if api_style == "openai" else _probe_anthropic
    for base in _candidates():
        if probe(base, model, key):
            _RESOLVED[provider_key] = (base, api_style, model)
            return _RESOLVED[provider_key]
        # try the other wire format against this host too
        other = _probe_anthropic if api_style == "openai" else _probe_openai
        other_style = "anthropic" if api_style == "openai" else "openai"
        if other(base, model, key):
            _RESOLVED[provider_key] = (base, other_style, model)
            return _RESOLVED[provider_key]

    raise System2Error(
        f"System 2 provider {provider_key} ('{cfg['label']}'): no gateway base URL responded "
        f"among {_candidates()}. Set {LLM_GATEWAY_BASE_URL_ENV} to a working endpoint."
    )


class System2Client:
    """Async OpenAI/Anthropic-compatible chat client bound to a gateway + model."""

    def __init__(self, provider_key: str):
        self.provider_key = provider_key
        self.provider = SYSTEM2_PROVIDERS[provider_key]
        self._client = httpx.AsyncClient(timeout=SYSTEM2_TIMEOUT_S)

    async def __aenter__(self):
        self.base_url, self.api_style, self.model = _resolve(self.provider_key)
        return self

    async def __aexit__(self, *exc):
        await self._client.aclose()

    async def _chat_openai(self, messages, max_tokens: int = 1024) -> dict:
        resp = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ[self.provider['env_var']]}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "messages": messages, "max_tokens": max_tokens},
        )
        if resp.status_code >= 400:
            return {"error": _server_error(resp)}
        data = resp.json()
        usage = data.get("usage", {})
        return {
            "text": (data["choices"][0]["message"].get("content") or "") if data.get("choices") else "",
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        }

    async def _chat_anthropic(self, messages, max_tokens: int = 1024) -> dict:
        system, convo = _split_anthropic(messages)
        body = {"model": self.model, "max_tokens": max_tokens, "messages": convo}
        if system:
            body["system"] = system
        resp = await self._client.post(
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": os.environ[self.provider["env_var"]],
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=body,
        )
        if resp.status_code >= 400:
            return {"error": _server_error(resp)}
        data = resp.json()
        text = "".join(blk.get("text", "") for blk in data.get("content", []) if blk.get("type") == "text")
        usage = data.get("usage", {})
        return {
            "text": text,
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
        }

    async def chat(self, system: str, user: str, max_tokens: int = 1024) -> ChatResult:
        """Run one chat completion.  Retries transient errors up to a configured count."""
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": user})
        last_err: str | None = None
        for attempt in range(max(1, SYSTEM2_MAX_RETRIES)):
            t0 = time.perf_counter()
            try:
                if self.api_style == "anthropic":
                    out = await self._chat_anthropic(messages, max_tokens)
                else:
                    out = await self._chat_openai(messages, max_tokens)
                if "error" in out:
                    return ChatResult(
                        text="",
                        model=self.model,
                        input_tokens=0,
                        output_tokens=0,
                        latency_s=time.perf_counter() - t0,
                        error=out["error"],
                    )
                return ChatResult(
                    text=out["text"],
                    model=self.model,
                    input_tokens=out["input_tokens"],
                    output_tokens=out["output_tokens"],
                    latency_s=time.perf_counter() - t0,
                )
            except Exception as e:
                last_err = str(e)
                if attempt + 1 < max(1, SYSTEM2_MAX_RETRIES):
                    await asyncio_sleep(1.5 * (attempt + 1))
        return ChatResult(
            text="",
            model=self.model,
            input_tokens=0,
            output_tokens=0,
            latency_s=0.0,
            error=last_err,
        )


def asyncio_sleep(s: float):
    import asyncio

    return asyncio.sleep(s)


def _split_anthropic(messages):
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    convo = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    return system, convo
