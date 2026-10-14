"""System 2 — open-ended LLMs (Qwen 3.8 27B / Claude Sonnet / Claude Opus).

System 2 is the "think hard, generate anything" model.  It is reached through a
Union-hosted model gateway using the demo virtual keys.  The gateway is
deployment-specific, so ``System2Client`` resolves a working base URL the first
time it is used by probing a candidate list (overridable with
``LLM_GATEWAY_BASE_URL``) and caching the first endpoint that answers.  Both the
OpenAI-compatible (``/v1/chat/completions``) and Anthropic (``/v1/messages``)
wire formats are supported.

Each call captures wall-clock latency and token usage (``prompt``/``completion``)
so the benchmark can compare throughput across the matrix, plus whether the model
ran into ``max_tokens`` mid-answer — a truncated battery and a refused one look
identical in the parsed output and must not be scored the same way.

Sampling is pinned at ``SYSTEM2_TEMPERATURE`` (0 by default) because the report
measures run-to-run decision stability; leaving the baseline at a gateway's
default temperature would charge sampling noise to the model.

Transient gateway failures are expected — the self-hosted Qwen backend restarts
and scales from zero — so every call retries with exponential backoff and jitter
(see ``_backoff_delay``), honours ``Retry-After`` when the gateway sends one, and
re-probes for a base URL when the status looks like a stale route.

Timeout, retry budget and concurrency are per-provider (see ``SYSTEM2_PROVIDERS``):
a hosted frontier model and one self-hosted GPU fail in different ways, and a
single shared timeout starves the slower one.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass

import httpx
from _config import (
    GATEWAY_BASE_URL_CANDIDATES,
    LLM_GATEWAY_BASE_URL_ENV,
    SYSTEM2_MAX_RETRIES,
    SYSTEM2_PROVIDERS,
    SYSTEM2_RETRY_BASE_S,
    SYSTEM2_RETRY_BUDGET_S,
    SYSTEM2_RETRY_CAP_S,
    SYSTEM2_TEMPERATURE,
    SYSTEM2_TIMEOUT_S,
)

# HTTP statuses worth a second attempt. 429/5xx are the usual suspects; 405 and
# 404 are here because this gateway returns them while a backend is restarting or
# scaling from zero — a transient routing state, not a malformed request. Without
# this the first blip kills the unit, which is what has been eating the
# self-hosted arm's units in every run.
RETRYABLE_STATUS = frozenset({404, 405, 408, 409, 425, 429, 500, 502, 503, 504, 529})

# A subset that usually means "the endpoint you cached is not where the model is
# any more" rather than "the model is busy": the gateway re-routes on restart, so
# the fix is to re-probe for a base URL, not just to wait.
ROUTING_STATUS = frozenset({404, 405, 502, 503})


def _attempts() -> int:
    return max(1, SYSTEM2_MAX_RETRIES)


def _backoff_delay(attempt: int, retry_after: float | None = None) -> float:
    """Seconds to wait before attempt ``attempt + 1`` (0-indexed).

    Exponential — 1s, 2s, 4s, 8s ... — capped, and jittered across half the
    window so that sixteen units retrying a restarting backend do not all come
    back in lockstep and knock it over again. A numeric ``Retry-After`` from the
    gateway wins, because it knows better than we do, but is still capped.
    """
    if retry_after is not None:
        return min(max(0.0, retry_after), SYSTEM2_RETRY_CAP_S)
    window = min(SYSTEM2_RETRY_CAP_S, SYSTEM2_RETRY_BASE_S * (2**attempt))
    return random.uniform(window / 2, window)


def _retry_after(resp) -> float | None:
    """The gateway's own advice, when it sends a numeric one (the date form is rare)."""
    raw = resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


# module-level cache: provider_key -> (base_url, api_style, model) that worked
_RESOLVED: dict[str, tuple[str, str, str]] = {}

# Per-provider, per-event-loop concurrency gates. Keyed by loop id because a
# semaphore belongs to the loop that created it.
_PROVIDER_GATES: dict[tuple[str, int], asyncio.Semaphore] = {}


class _NoGate:
    """Stand-in for a provider that does not cap concurrency."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _provider_gate(provider_key: str):
    limit = SYSTEM2_PROVIDERS[provider_key].get("max_concurrency")
    if not limit:
        return _NoGate()
    key = (provider_key, id(asyncio.get_running_loop()))
    gate = _PROVIDER_GATES.get(key)
    if gate is None:
        gate = _PROVIDER_GATES[key] = asyncio.Semaphore(int(limit))
    return gate


@dataclass
class ChatResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    error: str | None = None
    attempts: int = 1  # how many HTTP attempts it took; > 1 means the gateway blipped
    # True when the model hit `max_tokens` mid-answer. Without this, a battery cut
    # off at field 60 of 89 is scored identically to a model that simply refused to
    # answer the last 29 — one is a harness ceiling, the other is a model property,
    # and a benchmark that cannot tell them apart is measuring its own `max_tokens`.
    truncated: bool = False


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


def _optimistic(provider_key: str) -> tuple[str, str, str]:
    """The endpoint to try first, chosen without touching the network.

    Probing to *find* an endpoint costs a full request against a model that may
    be cold. That is worth paying when a call actually fails, and not before.
    """
    cfg = SYSTEM2_PROVIDERS[provider_key]
    return _candidates()[0], str(cfg["api_style"]), str(cfg["model"])


def _probe_timeout(provider_key: str) -> float:
    """Probes get their own ceiling: long enough for a slow box, short enough to move on."""
    return min(float(SYSTEM2_PROVIDERS[provider_key].get("timeout_s") or SYSTEM2_TIMEOUT_S), 60.0)


def _probe_openai(base: str, model: str, key: str, timeout: float = 20.0) -> bool:
    """Return True if {base} speaks the OpenAI-compatible HTTP API.

    Any HTTP response (2xx, 4xx, ...) means the gateway endpoint exists; per-model
    errors (e.g. a 4xx because a model is unavailable) are surfaced at call time.
    """
    try:
        r = httpx.post(
            f"{base}/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "max_tokens": 4, "messages": [{"role": "user", "content": "ping"}]},
            timeout=timeout,
        )
        return r.status_code < 500
    except Exception:
        return False


def _probe_anthropic(base: str, model: str, key: str, timeout: float = 20.0) -> bool:
    try:
        r = httpx.post(
            f"{base}/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={"model": model, "max_tokens": 4, "messages": [{"role": "user", "content": "ping"}]},
            timeout=timeout,
        )
        return r.status_code < 500
    except Exception:
        return False


def _resolve(provider_key: str) -> tuple[str, str, str]:
    """Return (base_url, api_style, model) for a provider, resolving on first use."""
    if provider_key in _RESOLVED:
        return _RESOLVED[provider_key]
    cfg = SYSTEM2_PROVIDERS[provider_key]
    model = str(cfg["model"])
    api_style = str(cfg["api_style"])
    key = os.environ.get(str(cfg["env_var"]), "")
    if not key:
        raise System2Error(f"System 2 provider {provider_key}: API key not mounted")

    if os.environ.get(LLM_GATEWAY_BASE_URL_ENV):
        base = _candidates()[0]
        _RESOLVED[provider_key] = (base, api_style, model)
        return _RESOLVED[provider_key]

    # Probe candidates; prefer the matching wire format, fall back to the other.
    probe = _probe_openai if api_style == "openai" else _probe_anthropic
    timeout = _probe_timeout(provider_key)
    for base in _candidates():
        if probe(base, model, key, timeout):
            _RESOLVED[provider_key] = (base, api_style, model)
            return _RESOLVED[provider_key]
        # try the other wire format against this host too
        other = _probe_anthropic if api_style == "openai" else _probe_openai
        other_style = "anthropic" if api_style == "openai" else "openai"
        if other(base, model, key, timeout):
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
        self._timeout_s = float(self.provider.get("timeout_s") or SYSTEM2_TIMEOUT_S)
        self._retry_budget_s = float(self.provider.get("retry_budget_s") or SYSTEM2_RETRY_BUDGET_S)
        self._client = httpx.AsyncClient(timeout=self._timeout_s)

    async def __aenter__(self):
        """Adopt an endpoint without touching the network.

        Entering used to resolve by probing, which charged *every* unit a live
        request against the model — including the escalated ones that abstain and
        never generate anything. Against the slow self-hosted box that was ~20s
        per unit of pure measurement artifact, and it is the whole `Jev x Qwen`
        latency bar in run u4jw7dmf4mkv2c8h748m. Now the probe only runs if a
        real call comes back with a routing error.
        """
        if not os.environ.get(str(self.provider["env_var"]), ""):
            raise System2Error(f"System 2 provider {self.provider_key}: API key not mounted")
        self.base_url, self.api_style, self.model = _RESOLVED.get(self.provider_key) or _optimistic(self.provider_key)
        return self

    async def _reresolve(self) -> bool:
        """Drop the cached endpoint and probe again. False if nothing answered."""
        _RESOLVED.pop(self.provider_key, None)
        try:
            self.base_url, self.api_style, self.model = await asyncio.to_thread(_resolve, self.provider_key)
        except System2Error:
            return False  # keep the old endpoint; the next attempt may still land
        return True

    async def _wait(self, attempt: int, started: float, retry_after: float | None = None) -> bool:
        """Back off before the next attempt. False means the retry budget is spent.

        Without a budget, a gateway that hangs rather than refuses could hold one
        unit for attempts x timeout, which at benchmark concurrency is how a cell
        turns into a stall instead of a failure.
        """
        delay = _backoff_delay(attempt, retry_after)
        if time.perf_counter() - started + delay > self._retry_budget_s:
            return False
        await asyncio.sleep(delay)
        return True

    async def __aexit__(self, *exc):
        await self._client.aclose()

    async def _chat_openai(self, messages, max_tokens: int = 1024, temperature: float = SYSTEM2_TEMPERATURE) -> dict:
        resp = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ[self.provider['env_var']]}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        if resp.status_code >= 400:
            return {
                "error": _server_error(resp),
                "status": resp.status_code,
                "retry_after": _retry_after(resp),
            }
        data = resp.json()
        usage = data.get("usage", {})
        choices = data.get("choices") or []
        return {
            "text": (choices[0]["message"].get("content") or "") if choices else "",
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "truncated": bool(choices) and choices[0].get("finish_reason") == "length",
        }

    async def _chat_anthropic(self, messages, max_tokens: int = 1024, temperature: float = SYSTEM2_TEMPERATURE) -> dict:
        system, convo = _split_anthropic(messages)
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": convo,
            "temperature": temperature,
        }
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
            return {
                "error": _server_error(resp),
                "status": resp.status_code,
                "retry_after": _retry_after(resp),
            }
        data = resp.json()
        text = "".join(blk.get("text", "") for blk in data.get("content", []) if blk.get("type") == "text")
        usage = data.get("usage", {})
        return {
            "text": text,
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "truncated": data.get("stop_reason") == "max_tokens",
        }

    async def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = SYSTEM2_TEMPERATURE,
    ) -> ChatResult:
        """Run one chat completion, retrying transient gateway failures.

        Three things get retried, each with exponential backoff and jitter:
        raised exceptions (timeout, connection reset), HTTP error responses with
        a retryable status, and — for statuses that smell like re-routing — the
        endpoint itself, which is re-probed before the next attempt. Before this,
        one 405 from a restarting Qwen backend lost the unit outright, which is
        what emptied the self-hosted column of run u57vnk9qqhxqszzb9j7m.
        """
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": user})
        attempts = _attempts()
        last_err: str | None = None
        started = time.perf_counter()

        for attempt in range(attempts):
            t0 = time.perf_counter()
            try:
                async with _provider_gate(self.provider_key):
                    if self.api_style == "anthropic":
                        out = await self._chat_anthropic(messages, max_tokens, temperature)
                    else:
                        out = await self._chat_openai(messages, max_tokens, temperature)
            except Exception as e:  # timeout, connection reset, DNS — always transient enough to retry
                last_err = f"{type(e).__name__}: {e}"
                if attempt + 1 >= attempts or not await self._wait(attempt, started):
                    break
                continue

            if "error" not in out:
                return ChatResult(
                    text=out["text"],
                    model=self.model,
                    input_tokens=out["input_tokens"],
                    output_tokens=out["output_tokens"],
                    latency_s=time.perf_counter() - t0,
                    attempts=attempt + 1,
                    truncated=bool(out.get("truncated")),
                )

            status = out.get("status")
            last_err = f"HTTP {status}: {out['error']}" if status else str(out["error"])
            if status not in RETRYABLE_STATUS or attempt + 1 >= attempts:
                return ChatResult(
                    text="",
                    model=self.model,
                    input_tokens=0,
                    output_tokens=0,
                    latency_s=time.perf_counter() - t0,
                    error=_gave_up(last_err, attempt + 1),
                    attempts=attempt + 1,
                )
            if status in ROUTING_STATUS:
                await self._reresolve()
            if not await self._wait(attempt, started, out.get("retry_after")):
                break

        return ChatResult(
            text="",
            model=self.model,
            input_tokens=0,
            output_tokens=0,
            latency_s=0.0,
            error=_gave_up(last_err, attempt + 1),
            attempts=attempt + 1,
        )


def _gave_up(err: str | None, attempts: int) -> str:
    """Error text that says how hard we tried, so a report never hides a retry storm."""
    base = err or "unknown error"
    return base if attempts <= 1 else f"{base} (gave up after {attempts} attempts)"


def _split_anthropic(messages):
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    convo = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    return system, convo
