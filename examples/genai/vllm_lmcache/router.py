"""
OpenAI-compatible router in front of the LMCache vLLM workers.

For each request it tokenizes the prompt, asks ``PrefixRouter`` for a ranked list of
workers, and proxies to the first one that answers (streaming included). Response headers
show the decision:

- ``x-routed-worker``: the worker that served the request
- ``x-route-reason``: ``session``, ``prefix``, ``least_loaded``, ``spill``, ``roundrobin``,
  ``random`` or ``fallback`` (Valkey unavailable)
- ``x-matched-tokens``: prefix tokens the router believes are in that worker's L0/L1

The routing mode defaults to ``ROUTE_MODE``. A request can override it with the
``x-route-mode`` header, which is how ``bench.py`` compares modes on one deployment.
``x-route-worker: <name>`` sends a request to one specific worker (reason ``pinned``).
``bench.py`` uses it for the L2 probe.

Configuration (env vars, set by ``serve.py``):

- ``WORKER_APPS``: comma-separated worker app names, resolved to in-cluster endpoints
- ``WORKER_URLS``: optional comma-separated URLs that override ``WORKER_APPS`` (local testing)
- ``VALKEY_URL``: ``resp://``, ``redis://`` or ``valkey://`` URL of the shared Valkey
- ``TOKENIZER``: Hugging Face id of the tokenizer; must match the served model
- ``ROUTER_NAMESPACE``: key prefix for the routing index (tenant and model)
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prefix_router import ROUTE_MODES, Decision, PrefixRouter, RouterConfig, block_hashes
from starlette.background import BackgroundTask

logger = logging.getLogger("lmcache-router")

INTERNAL_APP_ENDPOINT_PATTERN = "INTERNAL_APP_ENDPOINT_PATTERN"
HEALTH_INTERVAL_S = 5.0
# Prometheus series scraped from each worker for /stats. vLLM's prefix-cache counters are
# the L0 hit rate; LMCache's token counters cover L1 and L2.
METRIC_PREFIXES = (
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "lmcache:num_requested_tokens",
    "lmcache:num_hit_tokens",
    "lmcache:num_remote_read_bytes",
    "lmcache:num_remote_write_bytes",
)


class _State:
    workers: dict[str, str]  # worker name -> base URL
    healthy: list[str]
    router: PrefixRouter
    tokenizer: Any
    http: Any
    valkey: Any
    default_mode: str
    reasons: collections.Counter
    health_task: asyncio.Task | None = None


S = _State()


def _worker_urls() -> dict[str, str]:
    urls = [u.strip().rstrip("/") for u in os.getenv("WORKER_URLS", "").split(",") if u.strip()]
    if urls:
        return {f"worker-{i}": u for i, u in enumerate(urls)}
    names = [n.strip() for n in os.environ["WORKER_APPS"].split(",") if n.strip()]
    pattern = os.environ[INTERNAL_APP_ENDPOINT_PATTERN]
    return {n: pattern.format(app_fqdn=n).rstrip("/") for n in names}


def _valkey_url() -> str:
    # The workers take LMCache's resp:// scheme; the Python client speaks the same protocol.
    url = os.environ["VALKEY_URL"]
    for scheme in ("resp://", "valkey://"):
        if url.startswith(scheme):
            return "redis://" + url[len(scheme) :]
    return url


async def _check_health() -> None:
    async def one(name: str, url: str) -> bool:
        try:
            r = await S.http.get(f"{url}/health", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    while True:
        results = await asyncio.gather(*(one(n, u) for n, u in S.workers.items()))
        S.healthy = [n for n, ok in zip(S.workers, results) if ok]
        await asyncio.sleep(HEALTH_INTERVAL_S)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    import httpx
    import valkey.asyncio as valkey
    from transformers import AutoTokenizer

    S.workers = _worker_urls()
    S.healthy = list(S.workers)
    S.default_mode = os.getenv("ROUTE_MODE", "prefix")
    S.reasons = collections.Counter()
    S.tokenizer = AutoTokenizer.from_pretrained(os.environ["TOKENIZER"])
    S.valkey = valkey.from_url(_valkey_url(), socket_connect_timeout=2, socket_timeout=1)
    S.router = PrefixRouter(
        S.valkey,
        list(S.workers),
        RouterConfig(namespace=os.getenv("ROUTER_NAMESPACE", "default")),
    )
    # No read timeout: generations can run for minutes. Connect failures still fail fast,
    # which is what moves a request on to the next candidate.
    S.http = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0), limits=httpx.Limits(max_connections=512))
    S.health_task = asyncio.create_task(_check_health())
    logger.info("routing to %s", S.workers)
    yield
    S.health_task.cancel()
    await S.http.aclose()
    await S.valkey.aclose()


app = FastAPI(title="LMCache prefix-aware router", lifespan=lifespan)


def _text(content: Any) -> str:
    # OpenAI content may be a list of parts; only text parts affect the prefix.
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def _tokens(path: str, body: dict) -> list[int]:
    if path.endswith("chat/completions"):
        messages = [{**m, "content": _text(m.get("content"))} for m in body.get("messages", [])]
        out = S.tokenizer.apply_chat_template(
            messages, tools=body.get("tools"), add_generation_prompt=True, tokenize=True
        )
        # transformers 5 returns a BatchEncoding here; 4.x returned a plain list.
        return list(out["input_ids"] if hasattr(out, "keys") else out)
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        if prompt and isinstance(prompt[0], int):
            return prompt
        prompt = prompt[0] if prompt else ""
    return S.tokenizer.encode(prompt, add_special_tokens=False)


def block_hashes_for(tokens: list[int]) -> list[str]:
    return block_hashes(tokens, S.router.cfg.block_tokens, S.router.cfg.max_blocks)


async def _proxy(request: Request, path: str) -> Response:
    raw = await request.body()
    body = json.loads(raw or b"{}")
    mode = request.headers.get("x-route-mode", S.default_mode)
    if mode not in ROUTE_MODES:
        return JSONResponse({"error": f"x-route-mode must be one of {ROUTE_MODES}"}, status_code=400)
    session_id = request.headers.get("x-session-id") or body.get("user")

    tokens = await asyncio.to_thread(_tokens, path, body)
    pinned = request.headers.get("x-route-worker")
    if pinned in S.workers:
        # Debug override used by bench.py's L2 probe to reach a specific worker.
        decision = Decision(candidates=[pinned], reason="pinned", block_hashes=block_hashes_for(tokens))
    else:
        decision = await S.router.pick(tokens, session_id=session_id, mode=mode, healthy=S.healthy)
    S.reasons[decision.reason] += 1

    last_error: Exception | str = "no candidates"
    for worker in decision.candidates:
        req = S.http.build_request(
            "POST", f"{S.workers[worker]}/{path}", content=raw, headers={"content-type": "application/json"}
        )
        await S.router.acquire(worker)
        try:
            upstream = await S.http.send(req, stream=True)
        except Exception as e:  # connect failure or reset: try the next candidate
            await S.router.release(worker)
            last_error = e
            continue
        if upstream.status_code >= 500:
            await upstream.aclose()
            await S.router.release(worker)
            last_error = f"{worker} returned {upstream.status_code}"
            continue

        await S.router.record(worker, decision.block_hashes, session_id)
        headers = {
            "x-routed-worker": worker,
            "x-route-reason": decision.reason if worker == decision.candidates[0] else "retry",
            "x-route-mode": mode,
            "x-matched-tokens": str(decision.matched_tokens if worker == decision.candidates[0] else 0),
            "content-type": upstream.headers.get("content-type", "application/json"),
        }

        async def done(w: str = worker, u: Any = upstream) -> None:
            await u.aclose()
            await S.router.release(w)

        return StreamingResponse(
            upstream.aiter_raw(), status_code=upstream.status_code, headers=headers, background=BackgroundTask(done)
        )

    return JSONResponse({"error": f"all candidates failed: {last_error}"}, status_code=503)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    return await _proxy(request, "v1/chat/completions")


@app.post("/v1/completions")
async def completions(request: Request) -> Response:
    return await _proxy(request, "v1/completions")


@app.get("/v1/models")
async def models() -> Response:
    worker = (S.healthy or list(S.workers))[0]
    r = await S.http.get(f"{S.workers[worker]}/v1/models")
    return Response(r.content, status_code=r.status_code, media_type="application/json")


@app.get("/health")
async def health() -> dict:
    return {"healthy_workers": S.healthy}


def _parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = collections.defaultdict(float)
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for p in METRIC_PREFIXES:
            if line.startswith(p):
                # Sum across label sets (e.g. one series per model or engine).
                try:
                    out[p] += float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return dict(out)


def _rates(m: dict[str, float]) -> dict[str, float | None]:
    def ratio(a: str, b: str) -> float | None:
        return round(m[a] / m[b], 4) if m.get(b) else None

    return {
        "l0_hit_rate": ratio("vllm:prefix_cache_hits", "vllm:prefix_cache_queries"),
        "lmcache_hit_rate": ratio("lmcache:num_hit_tokens", "lmcache:num_requested_tokens"),
    }


@app.get("/stats")
async def stats() -> dict:
    """Per-worker load and cache counters, plus this router replica's routing reasons."""

    async def scrape(name: str, url: str) -> tuple[str, dict]:
        try:
            r = await S.http.get(f"{url}/metrics", timeout=5.0)
            m = _parse_metrics(r.text)
            return name, {**m, **_rates(m)}
        except Exception as e:
            return name, {"error": repr(e)}

    try:
        loads = await S.router.loads()
    except Exception as e:
        loads = {"error": repr(e)}
    workers = dict(await asyncio.gather(*(scrape(n, u) for n, u in S.workers.items())))
    total: dict[str, float] = collections.defaultdict(float)
    for w in workers.values():
        for k in METRIC_PREFIXES:
            total[k] += w.get(k, 0.0) or 0.0
    return {
        "default_mode": S.default_mode,
        "healthy": S.healthy,
        "inflight": loads,
        "route_reasons": dict(S.reasons),
        "workers": workers,
        "total": {**total, **_rates(total)},
    }
